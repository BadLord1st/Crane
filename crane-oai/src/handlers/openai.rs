//! OpenAI-compatible API handlers.
//!
//! Endpoints:
//! - `POST /v1/chat/completions`
//! - `POST /v1/completions`
//! - `GET  /v1/models`
//! - `GET  /v1/models/:model_id`
//! - `POST /v1/tokenize`
//! - `POST /v1/detokenize`

use std::sync::Arc;
use std::sync::OnceLock;
use std::{
    collections::{HashMap, HashSet},
    fmt::Write as _,
};

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json, Response,
    },
};

use crate::engine::{types::GenerationParams, types::MultimodalInputs, EngineResponse};
use crate::openai_api::*;
use crate::{make_error, now_epoch, AppState};
use serde_json::{json, Value};
use std::convert::Infallible;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::{timeout, Duration};
use tracing::{info, warn};

use super::sse;
use super::vlm;

fn effective_include_reasoning(state: &AppState, requested: bool) -> bool {
    crate::engine::policies::output_policy::effective_include_reasoning(
        state.model_spec.output_strategy,
        requested,
    )
}

fn log_sampling_resolution(
    endpoint: &str,
    request_id: &str,
    requested_temperature: Option<f64>,
    requested_top_p: Option<f64>,
    requested_top_k: Option<usize>,
    effective_temperature: Option<f64>,
    effective_top_p: Option<f64>,
    effective_top_k: Option<usize>,
) {
    let log_one = |name: &str, requested: String, effective: String| {
        if requested == effective {
            info!(
                endpoint,
                id = %request_id,
                parameter = name,
                requested = %requested,
                effective = %effective,
                "sampling parameter applied"
            );
        } else {
            info!(
                endpoint,
                id = %request_id,
                "detected {}={} (fallback to {}={})",
                name,
                requested,
                name,
                effective
            );
        }
    };

    log_one(
        "temperature",
        requested_temperature
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "None".to_string()),
        effective_temperature
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "None".to_string()),
    );
    log_one(
        "top_p",
        requested_top_p
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "None".to_string()),
        effective_top_p
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "None".to_string()),
    );
    log_one(
        "top_k",
        requested_top_k
            .map(|v| v.to_string())
            .unwrap_or_else(|| "None".to_string()),
        effective_top_k
            .map(|v| v.to_string())
            .unwrap_or_else(|| "None".to_string()),
    );
}

fn log_sampling_notes(endpoint: &str, request_id: &str, notes: &[String]) {
    for note in notes {
        info!(endpoint, id = %request_id, "{}", note);
    }
}

// ─────────────────────────────────────────────────────────────
//  Chat Completions
// ─────────────────────────────────────────────────────────────

/// `POST /v1/chat/completions` — streaming and non-streaming.
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    // If VLM model is loaded, delegate to VLM handler.
    if state.vlm_tx.is_some() {
        return vlm::vlm_chat_completions(state, req).await;
    }

    if req.has_multimodal_inputs() {
        let candidate_inputs = collect_multimodal_inputs(&req.messages);
        validate_multimodal_inputs(&state, &candidate_inputs)?;
        if matches!(state.output_mode, crate::gemma4_output::OutputMode::Gemma4) {
            warn!(
                "Gemma4 received multimodal content; using placeholder-only prompt path (vision/audio towers are not integrated in text backend yet)"
            );
        }
    }

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let include_usage = req
        .stream_options
        .as_ref()
        .is_some_and(|so| so.include_usage);
    let output_mode = state.output_mode;
    let include_reasoning = effective_include_reasoning(&state, req.include_reasoning);
    if include_reasoning && !req.include_reasoning {
        warn!("Gemma4: include_reasoning=false requested, forcing reasoning=true");
    }
    let sampling = crate::engine::policies::sampling_policy::resolve_sampling(
        state.model_spec.sampling_defaults,
        req.temperature,
        req.top_p,
        req.top_k,
    );
    let temperature = sampling.temperature;
    let top_p = sampling.top_p;
    let top_k = sampling.top_k;
    log_sampling_resolution(
        "v1/chat/completions",
        &request_id,
        req.temperature,
        req.top_p,
        req.top_k,
        temperature,
        top_p,
        top_k,
    );
    log_sampling_notes("v1/chat/completions", &request_id, &sampling.notes);

    let engine = state.engine.as_ref().ok_or_else(|| {
        make_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Text engine not available (VLM model loaded)",
        )
    })?;

    if req.stream {
        let has_tools = req.tools.as_ref().is_some_and(|v| !v.is_empty());
        if has_tools {
            let stream = make_chat_sse_stream_with_tool_runtime(
                state.clone(),
                req,
                request_id,
                include_usage,
                output_mode,
                include_reasoning,
            );
            return Ok(Sse::new(stream)
                .keep_alive(KeepAlive::default())
                .into_response());
        }

        let formatted = state.chat_template.apply(&req.messages).map_err(|e| {
            make_error(
                StatusCode::BAD_REQUEST,
                &format!("Chat template failed: {e}"),
            )
        })?;

        let input_ids = state
            .tokenizer
            .encode(formatted.as_str(), true)
            .map_err(|e| make_error(StatusCode::BAD_REQUEST, &format!("Tokenize failed: {e}")))?
            .get_ids()
            .to_vec();

        let multimodal_inputs = collect_multimodal_inputs(&req.messages);
        let response_rx = engine
            .submit_with_multimodal(
                request_id.clone(),
                input_ids,
                multimodal_inputs,
                GenerationParams {
                    max_tokens: req.max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    repetition_penalty: req.repetition_penalty.unwrap_or(1.05),
                    eos_token_id: state.eos_token_id.clone(),
                },
            )
            .map_err(|e| make_error(StatusCode::SERVICE_UNAVAILABLE, &e.to_string()))?;

        let model_name = state.model_name.clone();
        let stream = sse::make_chat_sse_stream(
            request_id,
            model_name,
            response_rx,
            include_usage,
            output_mode,
            include_reasoning,
        );
        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        let declared_tools = req.tools.clone().unwrap_or_default();
        let allow_tool_runtime = is_tool_runtime_enabled(&req, &declared_tools);
        let mut allowed_tool_names: HashSet<String> = declared_tools
            .iter()
            .filter(|t| t.kind == "function")
            .map(|t| t.function.name.clone())
            .collect();

        if let Some(forced) = requested_tool_name(&req) {
            if allowed_tool_names.contains(&forced) {
                allowed_tool_names.retain(|n| n == &forced);
            }
        }

        let mut loop_messages = req.messages.clone();
        let max_rounds = std::env::var("CRANE_TOOL_MAX_ROUNDS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4);

        for round in 0..=max_rounds {
            let formatted = state.chat_template.apply(&loop_messages).map_err(|e| {
                make_error(
                    StatusCode::BAD_REQUEST,
                    &format!("Chat template failed: {e}"),
                )
            })?;

            let input_ids = state
                .tokenizer
                .encode(formatted.as_str(), true)
                .map_err(|e| make_error(StatusCode::BAD_REQUEST, &format!("Tokenize failed: {e}")))?
                .get_ids()
                .to_vec();

            let multimodal_inputs = collect_multimodal_inputs(&loop_messages);
            let response_rx = engine
                .submit_with_multimodal(
                    request_id.clone(),
                    input_ids,
                    multimodal_inputs,
                    GenerationParams {
                        max_tokens: req.max_tokens,
                        temperature,
                        top_p,
                        top_k,
                        repetition_penalty: req.repetition_penalty.unwrap_or(1.05),
                        eos_token_id: state.eos_token_id.clone(),
                    },
                )
                .map_err(|e| make_error(StatusCode::SERVICE_UNAVAILABLE, &e.to_string()))?;

            let collected =
                collect_response_details(response_rx, output_mode, include_reasoning).await?;

            if allow_tool_runtime {
                let tool_calls =
                    extract_tool_calls_from_raw(&collected.raw_text, &allowed_tool_names);
                if !tool_calls.is_empty() {
                    if round >= max_rounds {
                        return Err(make_error(
                            StatusCode::BAD_REQUEST,
                            "tool runtime exceeded max rounds (CRANE_TOOL_MAX_ROUNDS)",
                        ));
                    }

                    let response = ChatCompletionResponse {
                        id: request_id.clone(),
                        object: "chat.completion".into(),
                        created: now_epoch(),
                        model: state.model_name.clone(),
                        choices: vec![ChatChoice {
                            index: 0,
                            message: ChatCompletionMessage {
                                role: "assistant".into(),
                                content: None,
                                tool_calls: Some(tool_calls.clone()),
                            },
                            finish_reason: Some("tool_calls".into()),
                        }],
                        usage: Usage {
                            prompt_tokens: collected.prompt_tokens,
                            completion_tokens: collected.completion_tokens,
                            total_tokens: collected.prompt_tokens + collected.completion_tokens,
                        },
                    };

                    // Server-side runtime loop: execute built-ins and feed outputs back.
                    loop_messages.push(ChatMessage {
                        role: "assistant".into(),
                        content: ChatMessageContent::Text(collected.sanitized_text.clone()),
                    });

                    for call in tool_calls {
                        let args = serde_json::from_str::<Value>(&call.function.arguments)
                            .unwrap_or_else(|_| json!({ "raw": call.function.arguments }));
                        let tool_result = execute_server_tool(&call.function.name, &args).await;

                        let mut payload = String::new();
                        let _ = write!(
                            payload,
                            "Tool {} (id={}) result:\n{}",
                            call.function.name, call.id, tool_result
                        );

                        loop_messages.push(ChatMessage {
                            role: "system".into(),
                            content: ChatMessageContent::Text(payload),
                        });
                    }

                    if std::env::var("CRANE_TOOL_RETURN_FIRST_CALL")
                        .ok()
                        .as_deref()
                        == Some("1")
                    {
                        return Ok(Json(response).into_response());
                    }

                    continue;
                }
            }

            let response = ChatCompletionResponse {
                id: request_id,
                object: "chat.completion".into(),
                created: now_epoch(),
                model: state.model_name.clone(),
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatCompletionMessage {
                        role: "assistant".into(),
                        content: Some(collected.sanitized_text),
                        tool_calls: None,
                    },
                    finish_reason: Some(collected.finish_reason),
                }],
                usage: Usage {
                    prompt_tokens: collected.prompt_tokens,
                    completion_tokens: collected.completion_tokens,
                    total_tokens: collected.prompt_tokens + collected.completion_tokens,
                },
            };
            return Ok(Json(response).into_response());
        }

        Err(make_error(
            StatusCode::BAD_REQUEST,
            "tool runtime exhausted rounds without final assistant text",
        ))
    }
}

fn make_chat_sse_stream_with_tool_runtime(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
    request_id: String,
    include_usage: bool,
    output_mode: crate::gemma4_output::OutputMode,
    include_reasoning: bool,
) -> impl futures::Stream<Item = Result<Event, Infallible>> {
    let created = now_epoch();

    async_stream::stream! {
        let model_name = state.model_name.clone();

        let first_chunk = ChatCompletionChunk {
            id: request_id.clone(),
            object: "chat.completion.chunk".into(),
            created,
            model: model_name.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: Some("assistant".into()),
                    content: None,
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };
        yield Ok(Event::default().json_data(&first_chunk).unwrap());

        let declared_tools = req.tools.clone().unwrap_or_default();
        let allow_tool_runtime = is_tool_runtime_enabled(&req, &declared_tools);
        let mut allowed_tool_names: HashSet<String> = declared_tools
            .iter()
            .filter(|t| t.kind == "function")
            .map(|t| t.function.name.clone())
            .collect();
        if let Some(forced) = requested_tool_name(&req) {
            if allowed_tool_names.contains(&forced) {
                allowed_tool_names.retain(|n| n == &forced);
            }
        }

        let max_rounds = std::env::var("CRANE_TOOL_MAX_ROUNDS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4);

        let mut loop_messages = req.messages.clone();

        for round in 0..=max_rounds {
            let formatted = match state.chat_template.apply(&loop_messages) {
                Ok(s) => s,
                Err(e) => {
                    yield Ok(Event::default().data(format!("error: Chat template failed: {e}")));
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
            };

            let input_ids = match state
                .tokenizer
                .encode(formatted.as_str(), true)
                .map(|e| e.get_ids().to_vec())
            {
                Ok(ids) => ids,
                Err(e) => {
                    yield Ok(Event::default().data(format!("error: Tokenize failed: {e}")));
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
            };

            let multimodal_inputs = collect_multimodal_inputs(&loop_messages);
            let Some(engine) = state.engine.as_ref() else {
                yield Ok(Event::default().data("error: Text engine not available"));
                yield Ok(Event::default().data("[DONE]"));
                return;
            };
            let sampling = crate::engine::policies::sampling_policy::resolve_sampling(
                state.model_spec.sampling_defaults,
                req.temperature,
                req.top_p,
                req.top_k,
            );
            let temperature = sampling.temperature;
            let top_p = sampling.top_p;
            let top_k = sampling.top_k;
            log_sampling_resolution(
                "v1/chat/completions(stream-tool-runtime)",
                &request_id,
                req.temperature,
                req.top_p,
                req.top_k,
                temperature,
                top_p,
                top_k,
            );
            log_sampling_notes(
                "v1/chat/completions(stream-tool-runtime)",
                &request_id,
                &sampling.notes,
            );

            let response_rx = match engine.submit_with_multimodal(
                request_id.clone(),
                input_ids,
                multimodal_inputs,
                GenerationParams {
                    max_tokens: req.max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    repetition_penalty: req.repetition_penalty.unwrap_or(1.05),
                    eos_token_id: state.eos_token_id.clone(),
                },
            ) {
                Ok(rx) => rx,
                Err(e) => {
                    yield Ok(Event::default().data(format!("error: {e}")));
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
            };

            let collected = match collect_response_details(response_rx, output_mode, include_reasoning).await {
                Ok(c) => c,
                Err((_, e)) => {
                    yield Ok(Event::default().data(format!("error: {}", e.error.message)));
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
            };

            if allow_tool_runtime {
                let tool_calls = extract_tool_calls_from_raw(&collected.raw_text, &allowed_tool_names);
                if !tool_calls.is_empty() {
                    for (i, call) in tool_calls.iter().enumerate() {
                        let chunk = ChatCompletionChunk {
                            id: request_id.clone(),
                            object: "chat.completion.chunk".into(),
                            created,
                            model: model_name.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: None,
                                    content: None,
                                    tool_calls: Some(vec![ChunkToolCall {
                                        index: i,
                                        id: Some(call.id.clone()),
                                        kind: Some("function".into()),
                                        function: Some(ChunkToolCallFunction {
                                            name: Some(call.function.name.clone()),
                                            arguments: Some(call.function.arguments.clone()),
                                        }),
                                    }]),
                                },
                                finish_reason: None,
                            }],
                            usage: None,
                        };
                        yield Ok(Event::default().json_data(&chunk).unwrap());
                    }

                    let finish = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role: None,
                                content: None,
                                tool_calls: None,
                            },
                            finish_reason: Some("tool_calls".into()),
                        }],
                        usage: None,
                    };
                    yield Ok(Event::default().json_data(&finish).unwrap());

                    if include_usage {
                        let usage_chunk = ChatCompletionChunk {
                            id: request_id.clone(),
                            object: "chat.completion.chunk".into(),
                            created,
                            model: model_name.clone(),
                            choices: vec![],
                            usage: Some(Usage {
                                prompt_tokens: collected.prompt_tokens,
                                completion_tokens: collected.completion_tokens,
                                total_tokens: collected.prompt_tokens + collected.completion_tokens,
                            }),
                        };
                        yield Ok(Event::default().json_data(&usage_chunk).unwrap());
                    }

                    loop_messages.push(ChatMessage {
                        role: "assistant".into(),
                        content: ChatMessageContent::Text(collected.sanitized_text.clone()),
                    });

                    for call in tool_calls {
                        let args = serde_json::from_str::<Value>(&call.function.arguments)
                            .unwrap_or_else(|_| json!({ "raw": call.function.arguments }));
                        let tool_result = execute_server_tool(&call.function.name, &args).await;

                        let mut payload = String::new();
                        let _ = write!(
                            payload,
                            "Tool {} (id={}) result:\n{}",
                            call.function.name,
                            call.id,
                            tool_result
                        );

                        loop_messages.push(ChatMessage {
                            role: "system".into(),
                            content: ChatMessageContent::Text(payload),
                        });
                    }

                    if round >= max_rounds
                        || std::env::var("CRANE_TOOL_RETURN_FIRST_CALL").ok().as_deref() == Some("1")
                    {
                        yield Ok(Event::default().data("[DONE]"));
                        return;
                    }

                    continue;
                }
            }

            if !collected.sanitized_text.is_empty() {
                let text_chunk = ChatCompletionChunk {
                    id: request_id.clone(),
                    object: "chat.completion.chunk".into(),
                    created,
                    model: model_name.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: ChunkDelta {
                            role: None,
                            content: Some(collected.sanitized_text),
                            tool_calls: None,
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                };
                yield Ok(Event::default().json_data(&text_chunk).unwrap());
            }

            let finish = ChatCompletionChunk {
                id: request_id.clone(),
                object: "chat.completion.chunk".into(),
                created,
                model: model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: None,
                        content: None,
                        tool_calls: None,
                    },
                    finish_reason: Some(collected.finish_reason),
                }],
                usage: None,
            };
            yield Ok(Event::default().json_data(&finish).unwrap());

            if include_usage {
                let usage_chunk = ChatCompletionChunk {
                    id: request_id.clone(),
                    object: "chat.completion.chunk".into(),
                    created,
                    model: model_name.clone(),
                    choices: vec![],
                    usage: Some(Usage {
                        prompt_tokens: collected.prompt_tokens,
                        completion_tokens: collected.completion_tokens,
                        total_tokens: collected.prompt_tokens + collected.completion_tokens,
                    }),
                };
                yield Ok(Event::default().json_data(&usage_chunk).unwrap());
            }

            yield Ok(Event::default().data("[DONE]"));
            return;
        }

        yield Ok(Event::default().data("[DONE]"));
    }
}

// ─────────────────────────────────────────────────────────────
//  Text Completions
// ─────────────────────────────────────────────────────────────

/// `POST /v1/completions` — text completion (no chat template).
pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let prompt = req.prompt.as_string();
    let include_usage = req
        .stream_options
        .as_ref()
        .is_some_and(|so| so.include_usage);

    let input_ids = state
        .tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| make_error(StatusCode::BAD_REQUEST, &format!("Tokenize failed: {e}")))?
        .get_ids()
        .to_vec();

    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let output_mode = state.output_mode;
    let include_reasoning = effective_include_reasoning(&state, req.include_reasoning);
    if include_reasoning && !req.include_reasoning {
        warn!("Gemma4: include_reasoning=false requested, forcing reasoning=true");
    }
    let sampling = crate::engine::policies::sampling_policy::resolve_sampling(
        state.model_spec.sampling_defaults,
        req.temperature,
        req.top_p,
        req.top_k,
    );
    let temperature = sampling.temperature;
    let top_p = sampling.top_p;
    let top_k = sampling.top_k;
    log_sampling_resolution(
        "v1/completions",
        &request_id,
        req.temperature,
        req.top_p,
        req.top_k,
        temperature,
        top_p,
        top_k,
    );
    log_sampling_notes("v1/completions", &request_id, &sampling.notes);

    let engine = state.engine.as_ref().ok_or_else(|| {
        make_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Text engine not available (VLM model loaded)",
        )
    })?;

    let response_rx = engine
        .submit(
            request_id.clone(),
            input_ids,
            GenerationParams {
                max_tokens: req.max_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty: req.repetition_penalty.unwrap_or(1.05),
                eos_token_id: state.eos_token_id.clone(),
            },
        )
        .map_err(|e| make_error(StatusCode::SERVICE_UNAVAILABLE, &e.to_string()))?;

    if req.stream {
        let model_name = state.model_name.clone();
        let stream = sse::make_completion_sse_stream(
            request_id,
            model_name,
            response_rx,
            include_usage,
            output_mode,
            include_reasoning,
        );
        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        let (full_text, prompt_tokens, completion_tokens, finish_reason) =
            collect_response(response_rx, output_mode, include_reasoning).await?;

        let response = CompletionResponse {
            id: request_id,
            object: "text_completion".into(),
            created: now_epoch(),
            model: state.model_name.clone(),
            choices: vec![CompletionChoice {
                index: 0,
                text: full_text,
                finish_reason: Some(finish_reason),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        };
        Ok(Json(response).into_response())
    }
}

// ─────────────────────────────────────────────────────────────
//  Models
// ─────────────────────────────────────────────────────────────

/// `GET /v1/models` — list available models.
pub async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(ModelList {
        object: "list".into(),
        data: vec![make_model_info(&state)],
    })
}

/// `GET /v1/models/:model_id` — retrieve a specific model.
pub async fn retrieve_model(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Result<Json<ModelInfo>, (StatusCode, Json<ErrorResponse>)> {
    if model_id == state.model_name {
        Ok(Json(make_model_info(&state)))
    } else {
        Err(make_error(
            StatusCode::NOT_FOUND,
            &format!(
                "Model '{model_id}' not found. Available: {}",
                state.model_name
            ),
        ))
    }
}

fn make_model_info(state: &AppState) -> ModelInfo {
    ModelInfo {
        id: state.model_name.clone(),
        object: "model".into(),
        created: state.server_start_time,
        owned_by: "crane".into(),
        max_model_len: None,
        permission: None,
    }
}

// ─────────────────────────────────────────────────────────────
//  Tokenize / Detokenize
// ─────────────────────────────────────────────────────────────

/// `POST /v1/tokenize` or `POST /tokenize`
pub async fn tokenize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TokenizeRequest>,
) -> Result<Json<TokenizeResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Determine the text to tokenize.
    let text = if let Some(messages) = &req.messages {
        // Apply chat template first.
        state.chat_template.apply(messages).map_err(|e| {
            make_error(
                StatusCode::BAD_REQUEST,
                &format!("Chat template failed: {e}"),
            )
        })?
    } else if let Some(text) = &req.text {
        text.clone()
    } else {
        return Err(make_error(
            StatusCode::BAD_REQUEST,
            "Either 'text' or 'messages' must be provided",
        ));
    };

    let encoding = state
        .tokenizer
        .encode(text.as_str(), req.add_special_tokens)
        .map_err(|e| make_error(StatusCode::BAD_REQUEST, &format!("Tokenize failed: {e}")))?;

    let tokens = encoding.get_ids().to_vec();
    let count = tokens.len();

    Ok(Json(TokenizeResponse { tokens, count }))
}

/// `POST /v1/detokenize` or `POST /detokenize`
pub async fn detokenize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DetokenizeRequest>,
) -> Result<Json<DetokenizeResponse>, (StatusCode, Json<ErrorResponse>)> {
    let text = state
        .tokenizer
        .decode(&req.tokens, true)
        .map_err(|e| make_error(StatusCode::BAD_REQUEST, &format!("Detokenize failed: {e}")))?;

    Ok(Json(DetokenizeResponse { text }))
}

// ─────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────

fn collect_multimodal_inputs(messages: &[ChatMessage]) -> MultimodalInputs {
    messages
        .iter()
        .fold(MultimodalInputs::default(), |mut acc, msg| {
            acc.image_urls.extend(msg.image_urls());
            acc.audio_urls.extend(msg.audio_urls());
            acc
        })
}

fn validate_multimodal_inputs(
    state: &AppState,
    inputs: &MultimodalInputs,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if !inputs.image_urls.is_empty() && !state.accepts_image_inputs {
        return Err(make_error(
            StatusCode::BAD_REQUEST,
            "Loaded model does not support image inputs",
        ));
    }
    if !inputs.audio_urls.is_empty() && !state.accepts_audio_inputs {
        return Err(make_error(
            StatusCode::BAD_REQUEST,
            "Loaded model does not support audio inputs",
        ));
    }
    Ok(())
}

struct CollectedResponse {
    raw_text: String,
    sanitized_text: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    finish_reason: String,
}

/// Collect all response chunks and return both raw and sanitized texts.
async fn collect_response_details(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<EngineResponse>,
    output_mode: crate::gemma4_output::OutputMode,
    include_reasoning: bool,
) -> Result<CollectedResponse, (StatusCode, Json<ErrorResponse>)> {
    let mut full_text = String::new();
    let mut prompt_tokens = 0usize;
    let mut completion_tokens = 0usize;
    let mut finish_reason = "length".to_string();

    while let Some(resp) = rx.recv().await {
        match resp {
            EngineResponse::Token { text, .. } => {
                full_text.push_str(&text);
            }
            EngineResponse::Finished {
                full_text: ft,
                prompt_tokens: pt,
                completion_tokens: ct,
                finish_reason: fr,
            } => {
                full_text = ft;
                prompt_tokens = pt;
                completion_tokens = ct;
                finish_reason = fr;
                break;
            }
            EngineResponse::Error(e) => {
                return Err(make_error(StatusCode::INTERNAL_SERVER_ERROR, &e));
            }
        }
    }

    let mut sanitized = output_mode.sanitize_text(&full_text, include_reasoning);

    // Gemma4 base models may place almost all useful content in the reasoning
    // channel. If reasoning is hidden and the visible answer looks unusable,
    // fall back to including reasoning text instead of returning garbage.
    if !include_reasoning
        && output_mode == crate::gemma4_output::OutputMode::Gemma4
        && looks_unusable_answer(&sanitized)
    {
        sanitized = output_mode.sanitize_text(&full_text, true);
    }

    Ok(CollectedResponse {
        raw_text: full_text,
        sanitized_text: sanitized,
        prompt_tokens,
        completion_tokens,
        finish_reason,
    })
}

/// Backward-compatible view for completion endpoints.
async fn collect_response(
    rx: tokio::sync::mpsc::UnboundedReceiver<EngineResponse>,
    output_mode: crate::gemma4_output::OutputMode,
    include_reasoning: bool,
) -> Result<(String, usize, usize, String), (StatusCode, Json<ErrorResponse>)> {
    let c = collect_response_details(rx, output_mode, include_reasoning).await?;
    Ok((
        c.sanitized_text,
        c.prompt_tokens,
        c.completion_tokens,
        c.finish_reason,
    ))
}

fn is_tool_runtime_enabled(req: &ChatCompletionRequest, declared_tools: &[ToolSpec]) -> bool {
    if declared_tools.is_empty() {
        return false;
    }
    !matches!(&req.tool_choice, Some(ToolChoice::Mode(mode)) if mode.eq_ignore_ascii_case("none"))
}

fn requested_tool_name(req: &ChatCompletionRequest) -> Option<String> {
    match &req.tool_choice {
        Some(ToolChoice::Function { r#type, function }) if r#type == "function" => {
            Some(function.name.clone())
        }
        _ => None,
    }
}

fn extract_tool_calls_from_raw(raw: &str, allowed_tool_names: &HashSet<String>) -> Vec<ToolCall> {
    const OPEN: &str = "<|tool_call>";
    const CLOSE: &str = "<tool_call|>";

    let mut out = Vec::new();
    let mut cursor = 0usize;
    while let Some(start) = raw[cursor..].find(OPEN) {
        let start_abs = cursor + start + OPEN.len();
        let Some(end_rel) = raw[start_abs..].find(CLOSE) else {
            break;
        };
        let end_abs = start_abs + end_rel;
        let payload = raw[start_abs..end_abs].trim();
        if let Some((name, args)) = parse_tool_call_payload(payload) {
            if allowed_tool_names.contains(&name) {
                out.push(ToolCall {
                    id: format!("call_{}", uuid::Uuid::new_v4()),
                    kind: "function".into(),
                    function: ToolCallFunction {
                        name,
                        arguments: args.to_string(),
                    },
                });
            }
        }
        cursor = end_abs + CLOSE.len();
    }
    out
}

fn parse_tool_call_payload(payload: &str) -> Option<(String, Value)> {
    if let Ok(v) = serde_json::from_str::<Value>(payload) {
        if let Some(obj) = v.as_object() {
            if let Some(name) = obj.get("name").and_then(|x| x.as_str()) {
                let args = obj.get("arguments").cloned().unwrap_or_else(|| json!({}));
                return Some((name.to_string(), args));
            }
            if let Some(func) = obj.get("function").and_then(|x| x.as_object()) {
                if let Some(name) = func.get("name").and_then(|x| x.as_str()) {
                    let args = func.get("arguments").cloned().unwrap_or_else(|| json!({}));
                    return Some((name.to_string(), args));
                }
            }
        }
    }

    // Fallback format from some Gemma outputs: call:name{...json...}
    if let Some(rest) = payload.strip_prefix("call:") {
        let rest = rest.trim();
        if let Some(brace) = rest.find('{') {
            let name = rest[..brace].trim();
            let args_text = &rest[brace..];
            let args = serde_json::from_str::<Value>(args_text)
                .unwrap_or_else(|_| json!({ "raw": args_text }));
            if !name.is_empty() {
                return Some((name.to_string(), args));
            }
        } else if !rest.is_empty() {
            return Some((rest.to_string(), json!({})));
        }
    }

    None
}

#[derive(Clone, Default)]
struct ToolRegistry {
    command_tools: HashMap<String, String>,
    webhook_tools: HashMap<String, String>,
}

static TOOL_REGISTRY: OnceLock<ToolRegistry> = OnceLock::new();
static TOOL_HTTP: OnceLock<reqwest::Client> = OnceLock::new();

fn tool_registry() -> &'static ToolRegistry {
    TOOL_REGISTRY.get_or_init(|| {
        let command_tools = std::env::var("CRANE_TOOL_COMMANDS")
            .ok()
            .and_then(|s| serde_json::from_str::<HashMap<String, String>>(&s).ok())
            .unwrap_or_default();
        let webhook_tools = std::env::var("CRANE_TOOL_WEBHOOKS")
            .ok()
            .and_then(|s| serde_json::from_str::<HashMap<String, String>>(&s).ok())
            .unwrap_or_default();
        ToolRegistry {
            command_tools,
            webhook_tools,
        }
    })
}

fn tool_http_client() -> &'static reqwest::Client {
    TOOL_HTTP.get_or_init(reqwest::Client::new)
}

async fn execute_server_tool(name: &str, args: &Value) -> Value {
    let registry = tool_registry();

    if let Some(cmd) = registry.command_tools.get(name) {
        return exec_command_tool(name, cmd, args).await;
    }
    if let Some(url) = registry.webhook_tools.get(name) {
        return exec_webhook_tool(name, url, args).await;
    }

    match name {
        "get_time" => json!({
            "ok": true,
            "epoch": now_epoch(),
        }),
        "echo" => json!({
            "ok": true,
            "args": args,
        }),
        "ping" => json!({
            "ok": true,
            "pong": true,
        }),
        _ => json!({
            "ok": false,
            "error": format!("tool '{name}' is not implemented in server runtime"),
            "args": args,
        }),
    }
}

async fn exec_command_tool(name: &str, command: &str, args: &Value) -> Value {
    let timeout_ms = std::env::var("CRANE_TOOL_CMD_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(10_000);

    let mut child = match Command::new("sh")
        .arg("-lc")
        .arg(command)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            return json!({
                "ok": false,
                "error": format!("failed to spawn command tool '{name}': {e}"),
                "tool": name,
            });
        }
    };

    if let Some(mut stdin) = child.stdin.take() {
        let payload = args.to_string();
        if let Err(e) = stdin.write_all(payload.as_bytes()).await {
            return json!({
                "ok": false,
                "error": format!("failed to write tool args to stdin for '{name}': {e}"),
                "tool": name,
            });
        }
    }

    let output = match timeout(Duration::from_millis(timeout_ms), child.wait_with_output()).await {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => {
            return json!({
                "ok": false,
                "error": format!("command tool '{name}' failed: {e}"),
                "tool": name,
            });
        }
        Err(_) => {
            return json!({
                "ok": false,
                "error": format!("command tool '{name}' timed out after {timeout_ms}ms"),
                "tool": name,
            });
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let parsed = serde_json::from_str::<Value>(&stdout).ok();

    json!({
        "ok": output.status.success(),
        "tool": name,
        "status": output.status.code(),
        "stdout": parsed.unwrap_or(Value::String(stdout)),
        "stderr": stderr,
    })
}

async fn exec_webhook_tool(name: &str, url: &str, args: &Value) -> Value {
    let timeout_ms = std::env::var("CRANE_TOOL_HTTP_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(10_000);

    let req = tool_http_client().post(url).json(&json!({
        "tool": name,
        "arguments": args,
    }));

    let resp = match timeout(Duration::from_millis(timeout_ms), req.send()).await {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            return json!({
                "ok": false,
                "tool": name,
                "error": format!("webhook request failed: {e}"),
            });
        }
        Err(_) => {
            return json!({
                "ok": false,
                "tool": name,
                "error": format!("webhook tool '{name}' timed out after {timeout_ms}ms"),
            });
        }
    };

    let status = resp.status();
    let body_text = match resp.text().await {
        Ok(s) => s,
        Err(e) => {
            return json!({
                "ok": false,
                "tool": name,
                "status": status.as_u16(),
                "error": format!("failed to read webhook response: {e}"),
            });
        }
    };

    let parsed = serde_json::from_str::<Value>(&body_text).unwrap_or(Value::String(body_text));

    json!({
        "ok": status.is_success(),
        "tool": name,
        "status": status.as_u16(),
        "response": parsed,
    })
}

fn looks_unusable_answer(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() {
        return true;
    }

    let alpha = t.chars().filter(|c| c.is_alphabetic()).count();
    let digits = t.chars().filter(|c| c.is_ascii_digit()).count();

    if alpha == 0 {
        return true;
    }
    if t.len() < 16 && alpha < 4 {
        return true;
    }
    if digits > 0 && alpha.saturating_mul(3) < digits {
        return true;
    }
    false
}
