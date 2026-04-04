//! SGLang-compatible native API handlers.
//!
//! Endpoints:
//! - `POST /generate`        — native text generation
//! - `GET  /model_info`      — model metadata
//! - `GET  /server_info`     — server configuration + live stats
//! - `GET  /engine_info`     — runtime capabilities + policies
//! - `GET  /health_generate` — deep health check (1-token probe)
//! - `POST /flush_cache`     — flush model KV caches (no-op informational)
//! - `POST /abort_request`   — abort in-flight request (informational)

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{KeepAlive, Sse},
        IntoResponse, Json, Response,
    },
};
use serde_json::json;
use tracing::{info, warn};

use crate::engine::runtime::{ChatFormatStrategy, OutputStrategy};
use crate::engine::{types::GenerationParams, types::MultimodalInputs, EngineResponse};
use crate::openai_api::ErrorResponse;
use crate::sglang_api::*;
use crate::{make_error, AppState};

use super::sse;
use super::vlm;

fn log_sampling_resolution(
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
                endpoint = "generate",
                id = %request_id,
                parameter = name,
                requested = %requested,
                effective = %effective,
                "sampling parameter applied"
            );
        } else {
            info!(
                endpoint = "generate",
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

// ─────────────────────────────────────────────────────────────
//  /generate
// ─────────────────────────────────────────────────────────────

/// `POST /generate` — SGLang-style native generation.
///
/// Accepts either `text` (prompt string) or `input_ids` (pre-tokenized).
/// Returns generated text + meta_info, or SSE stream if `stream: true`.
pub async fn generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let req_t0 = Instant::now();

    // If VLM model is loaded, delegate to VLM handler.
    if state.vlm_tx.is_some() {
        return vlm::vlm_generate(state, req).await;
    }

    // Resolve input tokens.
    let (input_ids, input_source) = if let Some(ids) = req.input_ids {
        (ids, "input_ids")
    } else if let Some(text) = &req.text {
        let ids = state
            .tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| make_error(StatusCode::BAD_REQUEST, &format!("Tokenize failed: {e}")))?
            .get_ids()
            .to_vec();
        (ids, "text")
    } else {
        return Err(make_error(
            StatusCode::BAD_REQUEST,
            "Either 'text' or 'input_ids' must be provided",
        ));
    };

    let sp = &req.sampling_params;
    let sampling = crate::engine::policies::sampling_policy::resolve_sampling(
        state.model_spec.sampling_defaults,
        sp.temperature,
        sp.top_p,
        sp.top_k,
    );
    let temperature = sampling.temperature;
    let top_p = sampling.top_p;
    let top_k = sampling.top_k;
    let request_id = req
        .rid
        .unwrap_or_else(|| format!("gen-{}", uuid::Uuid::new_v4()));
    log_sampling_resolution(
        &request_id,
        sp.temperature,
        sp.top_p,
        sp.top_k,
        temperature,
        top_p,
        top_k,
    );
    for note in &sampling.notes {
        info!(endpoint = "generate", id = %request_id, "{}", note);
    }
    let multimodal_inputs = req
        .image_url
        .clone()
        .map(|url| MultimodalInputs {
            image_urls: vec![url],
            audio_urls: vec![],
        })
        .unwrap_or_default();

    if !multimodal_inputs.image_urls.is_empty() && !state.accepts_image_inputs {
        warn!(
            id = %request_id,
            image_count = multimodal_inputs.image_urls.len(),
            "Rejected request: image inputs are not supported by loaded model"
        );
        return Err(make_error(
            StatusCode::BAD_REQUEST,
            "Loaded model does not support image inputs",
        ));
    }

    info!(
        id = %request_id,
        stream = req.stream,
        input_source,
        input_tokens = input_ids.len(),
        image_inputs = multimodal_inputs.image_urls.len(),
        max_new_tokens = sp.max_new_tokens,
        temperature = ?temperature,
        top_p = ?top_p,
        top_k = ?top_k,
        repetition_penalty = sp.repetition_penalty,
        "SGLang generate request accepted"
    );
    info!(id = %request_id, elapsed_ms = req_t0.elapsed().as_millis(), "Request pre-processing complete");

    let engine = state.engine.as_ref().ok_or_else(|| {
        make_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Text engine not available (VLM model loaded)",
        )
    })?;

    let submit_t0 = Instant::now();
    let response_rx = engine
        .submit_with_multimodal(
            request_id.clone(),
            input_ids,
            multimodal_inputs,
            GenerationParams {
                max_tokens: sp.max_new_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty: sp.repetition_penalty,
                eos_token_id: state.eos_token_id.clone(),
            },
        )
        .map_err(|e| make_error(StatusCode::SERVICE_UNAVAILABLE, &e.to_string()))?;
    info!(
        id = %request_id,
        submit_ms = submit_t0.elapsed().as_millis(),
        "Engine request submitted"
    );

    if req.stream {
        info!(
            id = %request_id,
            elapsed_ms = req_t0.elapsed().as_millis(),
            "Starting streaming response"
        );
        let stream = sse::make_generate_sse_stream(request_id, response_rx);
        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        // Collect full response.
        let collect_t0 = Instant::now();
        let mut full_text = String::new();
        let mut prompt_tokens = 0usize;
        let mut completion_tokens = 0usize;
        let mut finish_reason = "length".to_string();
        let mut token_chunks = 0usize;

        let mut response_rx = response_rx;
        while let Some(resp) = response_rx.recv().await {
            match resp {
                EngineResponse::Token { text, .. } => {
                    token_chunks += 1;
                    full_text.push_str(&text)
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
                    warn!(
                        id = %request_id,
                        error = %e,
                        elapsed_ms = req_t0.elapsed().as_millis(),
                        "Generation failed"
                    );
                    return Err(make_error(StatusCode::INTERNAL_SERVER_ERROR, &e));
                }
            }
        }

        info!(
            id = %request_id,
            prompt_tokens,
            completion_tokens,
            finish_reason = %finish_reason,
            token_chunks,
            output_chars = full_text.chars().count(),
            collect_ms = collect_t0.elapsed().as_millis(),
            total_ms = req_t0.elapsed().as_millis(),
            "Generation finished"
        );

        let response = GenerateResponse {
            text: full_text,
            meta_info: GenerateMetaInfo {
                id: request_id,
                prompt_tokens,
                completion_tokens,
                finish_reason,
            },
        };

        Ok(Json(response).into_response())
    }
}

// ─────────────────────────────────────────────────────────────
//  /model_info
// ─────────────────────────────────────────────────────────────

/// `GET /model_info` — model metadata.
pub async fn model_info(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(ModelInfoResponse {
        model_path: state.model_path.clone(),
        model_type: state.model_type_name.clone(),
        is_generation: true,
        accepts_image_inputs: state.accepts_image_inputs,
        accepts_audio_inputs: state.accepts_audio_inputs,
        dtype: Some(state.dtype_name.clone()),
        device: Some(state.device_name.clone()),
        max_model_len: None,
    })
}

// ─────────────────────────────────────────────────────────────
//  /server_info
// ─────────────────────────────────────────────────────────────

/// `GET /server_info` — server configuration + live stats.
pub async fn server_info(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let stats = state
        .engine
        .as_ref()
        .map(|e| e.stats.snapshot())
        .unwrap_or_default();
    Json(ServerInfoResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
        model_path: state.model_path.clone(),
        model_type: state.model_type_name.clone(),
        placement_policy: state.placement_policy.clone(),
        host: state.host.clone(),
        port: state.port,
        max_concurrent: state.max_concurrent,
        decode_tokens_per_seq: state.decode_tokens_per_seq,
        max_seq_len: state.max_seq_len,
        gpu_memory_limit: state.gpu_memory_limit.clone(),
        runtime_offload_ops: stats.total_runtime_offload_ops,
        runtime_offload_units: stats.total_runtime_offload_units,
        kv_offload_ops: stats.total_kv_offload_ops,
        kv_offload_tensors: stats.total_kv_offload_tensors,
        kv_prefetch_ops: stats.total_kv_prefetch_ops,
        kv_prefetch_tensors: stats.total_kv_prefetch_tensors,
        stats,
    })
}

// ─────────────────────────────────────────────────────────────
//  /engine_info
// ─────────────────────────────────────────────────────────────

/// `GET /engine_info` — runtime capabilities + policies.
pub async fn engine_info(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let chat_format_strategy = match state.model_spec.chat_format_strategy {
        ChatFormatStrategy::AutoJinja => "auto_jinja",
        ChatFormatStrategy::Hunyuan => "hunyuan",
        ChatFormatStrategy::Gemma4 => "gemma4",
    }
    .to_string();

    let output_strategy = match state.model_spec.output_strategy {
        OutputStrategy::Plain => "plain",
        OutputStrategy::Gemma4 => "gemma4",
    }
    .to_string();

    let caps = state.model_spec.capabilities;
    let supports_kv_swap = state.model_spec.supports_kv_swap();
    let kv_swap_unavailable_reason = if supports_kv_swap {
        None
    } else {
        let reason = if state.model_type_name == "qwen25" {
            "KV swap unavailable: current qwen2/qwen2_moe backend in candle_transformers does not expose public KV get/set APIs"
        } else if state.model_type_name == "gemma4" {
            "KV swap unavailable: engine auto-detects memory pressure, but current Gemma4 backend does not yet expose KV/expert offload targets"
        } else {
            "KV swap unavailable for this model runtime backend"
        };
        Some(reason.to_string())
    };

    Json(EngineInfoResponse {
        placement_policy: state.placement_policy.clone(),
        chat_format_strategy,
        output_strategy,
        supports_batch_decode: state.model_spec.supports_batch_decode(),
        supports_kv_swap,
        kv_swap_unavailable_reason,
        capabilities: EngineCapabilitiesResponse {
            text: caps.text,
            multimodal: caps.multimodal,
            tool_call_tokens: caps.tool_call_tokens,
            batch_decode: caps.batch_decode,
            kv_swap: caps.kv_swap,
            accepts_image_inputs: caps.accepts_image_inputs,
            accepts_audio_inputs: caps.accepts_audio_inputs,
        },
    })
}

// ─────────────────────────────────────────────────────────────
//  /health_generate
// ─────────────────────────────────────────────────────────────

/// `GET /health_generate` — deep health check.
///
/// Runs a tiny 1-token generation through the full pipeline.
/// Returns 200 on success, 503 on failure/timeout.
pub async fn health_generate(
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let probe_tokens = state.eos_token_id.clone(); // minimal input (already a Vec)
    let request_id = format!("health-{}", uuid::Uuid::new_v4());

    let engine = state.engine.as_ref().ok_or_else(|| {
        make_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Text engine not available (VLM model loaded)",
        )
    })?;

    let response_rx = engine
        .submit(
            request_id,
            probe_tokens,
            GenerationParams {
                max_tokens: 1,
                temperature: Some(0.0),
                top_p: None,
                top_k: None,
                repetition_penalty: 1.0,
                eos_token_id: state.eos_token_id.clone(),
            },
        )
        .map_err(|e| {
            make_error(
                StatusCode::SERVICE_UNAVAILABLE,
                &format!("Health probe failed to submit: {e}"),
            )
        })?;

    // Wait with timeout.
    let mut response_rx = response_rx;
    let result = tokio::time::timeout(std::time::Duration::from_secs(30), async {
        while let Some(resp) = response_rx.recv().await {
            match resp {
                EngineResponse::Token { .. } => continue,
                EngineResponse::Finished { .. } => return Ok(()),
                EngineResponse::Error(e) => return Err(e),
            }
        }
        Err("No response received".to_string())
    })
    .await;

    match result {
        Ok(Ok(())) => Ok(Json(json!({"status": "ok"}))),
        Ok(Err(e)) => Err(make_error(
            StatusCode::SERVICE_UNAVAILABLE,
            &format!("Health probe generation failed: {e}"),
        )),
        Err(_) => Err(make_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Health probe timed out (30s)",
        )),
    }
}

// ─────────────────────────────────────────────────────────────
//  /flush_cache
// ─────────────────────────────────────────────────────────────

/// `POST|GET /flush_cache` — informational endpoint.
///
/// In the current architecture, KV caches are per-sequence and automatically
/// freed on completion. This endpoint is provided for API compatibility.
pub async fn flush_cache() -> impl IntoResponse {
    Json(FlushCacheResponse {
        success: true,
        message: "KV caches are managed per-sequence and auto-freed on completion.".into(),
    })
}

// ─────────────────────────────────────────────────────────────
//  /abort_request
// ─────────────────────────────────────────────────────────────

/// `POST /abort_request` — request cancellation.
///
/// Currently, clients can cancel by dropping their SSE connection (which the
/// engine detects automatically). This endpoint is provided for API compatibility;
/// explicit abort via engine control channel is a future enhancement.
pub async fn abort_request(Json(req): Json<AbortRequest>) -> impl IntoResponse {
    // TODO: Add explicit abort via engine control channel.
    Json(AbortResponse {
        success: true,
        message: format!(
            "Request '{}' marked for abort. \
             Note: clients can also cancel by closing the SSE connection.",
            req.rid
        ),
    })
}
