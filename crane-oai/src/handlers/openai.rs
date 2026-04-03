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

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{KeepAlive, Sse},
        IntoResponse, Json, Response,
    },
};

use crate::engine::{types::MultimodalInputs, EngineResponse};
use crate::openai_api::*;
use crate::{make_error, now_epoch, AppState};
use tracing::warn;

use super::sse;
use super::vlm;

fn default_sampling_for_state(state: &AppState) -> (Option<f64>, Option<f64>, Option<usize>) {
    if matches!(state.output_mode, crate::gemma4_output::OutputMode::Gemma4) {
        (Some(1.0), Some(0.95), Some(64))
    } else {
        (Some(0.8), Some(0.95), Some(40))
    }
}

fn effective_include_reasoning(
    output_mode: crate::gemma4_output::OutputMode,
    requested: bool,
) -> bool {
    if matches!(output_mode, crate::gemma4_output::OutputMode::Gemma4) {
        true
    } else {
        requested
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

    // Apply chat template.
    let formatted = state.chat_template.apply(&req.messages).map_err(|e| {
        make_error(
            StatusCode::BAD_REQUEST,
            &format!("Chat template failed: {e}"),
        )
    })?;

    // Tokenize.
    let input_ids = state
        .tokenizer
        .encode(formatted.as_str(), true)
        .map_err(|e| make_error(StatusCode::BAD_REQUEST, &format!("Tokenize failed: {e}")))?
        .get_ids()
        .to_vec();

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let include_usage = req
        .stream_options
        .as_ref()
        .map_or(false, |so| so.include_usage);
    let output_mode = state.output_mode;
    let include_reasoning = effective_include_reasoning(output_mode, req.include_reasoning);
    if matches!(output_mode, crate::gemma4_output::OutputMode::Gemma4) && !req.include_reasoning {
        warn!("Gemma4: include_reasoning=false requested, forcing reasoning=true");
    }
    let multimodal_inputs = collect_multimodal_inputs(&req.messages);
    let (default_temperature, default_top_p, default_top_k) = default_sampling_for_state(&state);

    let engine = state.engine.as_ref().ok_or_else(|| {
        make_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Text engine not available (VLM model loaded)",
        )
    })?;

    let response_rx = engine
        .submit_with_multimodal(
            request_id.clone(),
            input_ids,
            multimodal_inputs,
            req.max_tokens,
            req.temperature.or(default_temperature),
            req.top_p.or(default_top_p),
            req.top_k.or(default_top_k),
            req.repetition_penalty.unwrap_or(1.05),
            state.eos_token_id.clone(),
        )
        .map_err(|e| make_error(StatusCode::SERVICE_UNAVAILABLE, &e.to_string()))?;

    if req.stream {
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
        let (full_text, prompt_tokens, completion_tokens, finish_reason) =
            collect_response(response_rx, output_mode, include_reasoning).await?;

        let response = ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".into(),
            created: now_epoch(),
            model: state.model_name.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: ChatMessageContent::Text(full_text),
                },
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
        .map_or(false, |so| so.include_usage);

    let input_ids = state
        .tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| make_error(StatusCode::BAD_REQUEST, &format!("Tokenize failed: {e}")))?
        .get_ids()
        .to_vec();

    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let output_mode = state.output_mode;
    let include_reasoning = effective_include_reasoning(output_mode, req.include_reasoning);
    if matches!(output_mode, crate::gemma4_output::OutputMode::Gemma4) && !req.include_reasoning {
        warn!("Gemma4: include_reasoning=false requested, forcing reasoning=true");
    }
    let (default_temperature, default_top_p, default_top_k) = default_sampling_for_state(&state);

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
            req.max_tokens,
            req.temperature.or(default_temperature),
            req.top_p.or(default_top_p),
            req.top_k.or(default_top_k),
            req.repetition_penalty.unwrap_or(1.05),
            state.eos_token_id.clone(),
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

/// Collect all response chunks into (full_text, prompt_tokens, completion_tokens, finish_reason).
async fn collect_response(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<EngineResponse>,
    output_mode: crate::gemma4_output::OutputMode,
    include_reasoning: bool,
) -> Result<(String, usize, usize, String), (StatusCode, Json<ErrorResponse>)> {
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
        && matches!(output_mode, crate::gemma4_output::OutputMode::Gemma4)
        && looks_unusable_answer(&sanitized)
    {
        sanitized = output_mode.sanitize_text(&full_text, true);
    }

    full_text = sanitized;

    Ok((full_text, prompt_tokens, completion_tokens, finish_reason))
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
