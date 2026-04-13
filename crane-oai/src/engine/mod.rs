//! Continuous-batching inference engine.
//!
//! # Architecture
//!
//! ```text
//! API handlers ──(request channel)──► Engine thread
//!       ◄──(per-request response channel)──┘
//!
//! Engine loop (each iteration = one "step"):
//!   1. Drain new requests from channel
//!   2. Detect & cancel disconnected clients
//!   3. Scheduler picks next batch (prefill > decode)
//!   4. Prefill step: run full prompt for ONE new sequence
//!   5. Decode step: batched or sequential forward for running sequences
//!   6. If idle → blocking wait for new request
//! ```
//!
//! # Module layout
//!
//! | Module          | Responsibility                                   |
//! |-----------------|--------------------------------------------------|
//! | `types`         | Public request/response types + `EngineHandle`   |
//! | `stats`         | Lock-free counters shared with API layer          |
//! | `sampling`      | Token sampling (top-k, top-p, Gumbel-max, etc.) |
//! | `scheduler`     | FIFO scheduler with prefill priority              |
//! | `sequence`      | Per-request lifecycle state                       |
//! | `runtime`       | Runtime model contracts + capabilities             |
//! | `model_factory` | Auto-detection and factory creation               |

pub mod adapters;
pub mod model_factory;
pub mod policies;
pub mod runtime;
pub mod sampling;
pub mod scheduler;
pub mod sequence;
pub mod stats;
pub mod types;

// Re-export commonly used items for convenience.
pub use stats::{EngineStats, StatsSnapshot};
pub use types::{EngineHandle, EngineRequest, EngineResponse};

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use candle_core::{Device, Tensor};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::engine::policies::placement_policy::PlacementPolicy;
use crane_core::utils::token_output_stream::TokenOutputStream;
use runtime::{BatchDecodeContext, RuntimeModel, RuntimeRequestContext, RuntimeStepContext};
use sampling::SamplingBuffers;
use scheduler::{Scheduler, SchedulerOutput};
use sequence::{Sequence, SequenceStatus};

// ─────────────────────────────────────────────────────────────
//  Memory configuration
// ─────────────────────────────────────────────────────────────

/// Configuration for GPU memory limits.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Maximum tokens per sequence (prompt + completion). 0 = unlimited.
    pub max_seq_len: usize,
    /// Optional guardrail for text-only prompt prefill when max_seq_len is unlimited.
    /// Parsed from `CRANE_TEXT_PREFILL_TOKEN_LIMIT`. `None` = disabled.
    pub text_prefill_token_limit: Option<usize>,
    /// Dynamic text-only prefill admission settings used when max_seq_len is unlimited.
    pub text_prefill_admission: TextPrefillAdmissionConfig,
    /// GPU memory limit in bytes. 0 = unlimited.
    /// This is an **absolute** limit on total GPU memory usage.
    pub gpu_memory_limit_bytes: u64,
    /// Baseline GPU memory recorded after model load + warmup.
    /// The memory gate compares `(current_used - baseline)` against
    /// `(gpu_memory_limit_bytes - baseline)` so that the limit represents
    /// the *total* allowed usage, not just KV-cache growth.
    pub baseline_gpu_bytes: u64,
}

impl MemoryConfig {
    /// Parse memory configuration from CLI arguments.
    ///
    /// `gpu_memory_limit` accepts:
    ///   - Absolute sizes: "5G", "8G", "5120M", "5368709120" (bytes)
    ///   - Utilization fraction: "0.7" (70% of total GPU memory)
    pub fn parse(max_seq_len: usize, gpu_memory_limit: Option<&str>, device: &Device) -> Self {
        let gpu_memory_limit_bytes = match gpu_memory_limit {
            Some(s) => Self::parse_memory_limit(s, device),
            None => 0,
        };
        let text_prefill_token_limit = Self::parse_text_prefill_token_limit_env(
            std::env::var("CRANE_TEXT_PREFILL_TOKEN_LIMIT").ok(),
        );
        let text_prefill_admission = TextPrefillAdmissionConfig::parse_from_env(device);
        Self {
            max_seq_len,
            text_prefill_token_limit,
            text_prefill_admission,
            gpu_memory_limit_bytes,
            baseline_gpu_bytes: 0,
        }
    }

    fn parse_text_prefill_token_limit_env(raw: Option<String>) -> Option<usize> {
        let Some(raw) = raw else {
            return None;
        };

        let raw = raw.trim();
        if raw.is_empty() || raw == "0" {
            return None;
        }

        match raw.parse::<usize>() {
            Ok(limit) if limit > 0 => Some(limit),
            _ => {
                tracing::warn!(
                    value = raw,
                    "Could not parse CRANE_TEXT_PREFILL_TOKEN_LIMIT as a positive integer, ignoring"
                );
                None
            }
        }
    }

    fn parse_memory_limit(s: &str, device: &Device) -> u64 {
        let s = s.trim();
        if s.is_empty() || s == "0" {
            return 0;
        }

        // Try absolute sizes: "5G", "8G", "5120M", "1024K"
        let upper = s.to_uppercase();
        if upper.ends_with('G') {
            if let Ok(n) = upper[..upper.len() - 1].trim().parse::<f64>() {
                return (n * (1u64 << 30) as f64) as u64;
            }
        }
        if upper.ends_with('M') {
            if let Ok(n) = upper[..upper.len() - 1].trim().parse::<f64>() {
                return (n * (1u64 << 20) as f64) as u64;
            }
        }

        // Try as a fraction (0.0 - 1.0)
        if let Ok(frac) = s.parse::<f64>() {
            if (0.0..=1.0).contains(&frac) {
                let total = Self::query_total_gpu_memory(device);
                if total > 0 {
                    return (frac * total as f64) as u64;
                }
            }
            // If > 1.0, treat as bytes
            if frac > 1.0 {
                return frac as u64;
            }
        }

        tracing::warn!("Could not parse gpu_memory_limit '{}', ignoring", s);
        0
    }

    /// Record baseline GPU memory (call after model load + warmup).
    pub fn record_baseline(&mut self, device: &Device) {
        let (used, _total) = query_gpu_memory_usage(device);
        self.baseline_gpu_bytes = used;
    }

    /// Query total GPU memory (bytes). Returns 0 if unavailable.
    fn query_total_gpu_memory(_device: &Device) -> u64 {
        #[cfg(feature = "cuda")]
        {
            if let Device::Cuda(_) = _device {
                if let Ok((_free, total)) =
                    candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
                {
                    return total as u64;
                }
            }
        }
        0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextPrefillAdmissionMode {
    Off,
    Dynamic,
}

impl TextPrefillAdmissionMode {
    fn parse_env(raw: Option<String>) -> Self {
        match raw.as_deref().map(str::trim) {
            None | Some("") | Some("dynamic") | Some("estimate") | Some("on") | Some("1") => {
                Self::Dynamic
            }
            Some("off") | Some("disabled") | Some("0") => Self::Off,
            Some(other) => {
                tracing::warn!(
                    value = other,
                    "Could not parse CRANE_TEXT_PREFILL_ADMISSION_MODE, defaulting to dynamic"
                );
                Self::Dynamic
            }
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Dynamic => "dynamic",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextPrefillAdmissionConfig {
    pub mode: TextPrefillAdmissionMode,
    pub reserve_bytes: u64,
    pub bytes_per_token: u64,
    pub running_penalty_tokens: usize,
    pub waiting_penalty_tokens: usize,
}

impl TextPrefillAdmissionConfig {
    const DEFAULT_RESERVE_BYTES: u64 = 2 * (1u64 << 30);
    const DEFAULT_BYTES_PER_TOKEN: u64 = 2 * (1u64 << 20);
    const DEFAULT_RUNNING_PENALTY_TOKENS: usize = 256;
    const DEFAULT_WAITING_PENALTY_TOKENS: usize = 128;

    fn parse_from_env(device: &Device) -> Self {
        Self {
            mode: TextPrefillAdmissionMode::parse_env(
                std::env::var("CRANE_TEXT_PREFILL_ADMISSION_MODE").ok(),
            ),
            reserve_bytes: Self::parse_u64_env(
                "CRANE_TEXT_PREFILL_ADMISSION_RESERVE_BYTES",
                Self::DEFAULT_RESERVE_BYTES,
            ),
            bytes_per_token: Self::parse_u64_env(
                "CRANE_TEXT_PREFILL_ADMISSION_BYTES_PER_TOKEN",
                Self::DEFAULT_BYTES_PER_TOKEN,
            )
            .max(1),
            running_penalty_tokens: Self::parse_usize_env(
                "CRANE_TEXT_PREFILL_ADMISSION_RUNNING_PENALTY_TOKENS",
                Self::DEFAULT_RUNNING_PENALTY_TOKENS,
            ),
            waiting_penalty_tokens: Self::parse_usize_env(
                "CRANE_TEXT_PREFILL_ADMISSION_WAITING_PENALTY_TOKENS",
                Self::DEFAULT_WAITING_PENALTY_TOKENS,
            ),
        }
        .with_fractional_reserve_if_requested(device)
    }

    fn with_fractional_reserve_if_requested(mut self, device: &Device) -> Self {
        let Some(raw) = std::env::var("CRANE_TEXT_PREFILL_ADMISSION_RESERVE").ok() else {
            return self;
        };

        let parsed = MemoryConfig::parse_memory_limit(&raw, device);
        if parsed == 0 && raw.trim() != "0" {
            tracing::warn!(
                value = raw,
                "Could not parse CRANE_TEXT_PREFILL_ADMISSION_RESERVE, keeping reserve_bytes"
            );
            return self;
        }
        self.reserve_bytes = parsed;
        self
    }

    fn parse_u64_env(name: &str, default: u64) -> u64 {
        match std::env::var(name) {
            Ok(raw) => match raw.trim().parse::<u64>() {
                Ok(value) => value,
                Err(_) => {
                    tracing::warn!(
                        value = raw,
                        variable = name,
                        "Could not parse env as u64, using default"
                    );
                    default
                }
            },
            Err(_) => default,
        }
    }

    fn parse_usize_env(name: &str, default: usize) -> usize {
        match std::env::var(name) {
            Ok(raw) => match raw.trim().parse::<usize>() {
                Ok(value) => value,
                Err(_) => {
                    tracing::warn!(
                        value = raw,
                        variable = name,
                        "Could not parse env as usize, using default"
                    );
                    default
                }
            },
            Err(_) => default,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TextPrefillAdmissionSnapshot {
    prompt_len: usize,
    gpu_used_bytes: u64,
    gpu_total_bytes: u64,
    tracked_kv_bytes: u64,
    running_sequences: usize,
    waiting_sequences: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TextPrefillAdmissionEstimate {
    allowed_prompt_len_now: usize,
    configured_limit_bytes: u64,
    headroom_budget_bytes: u64,
    accounted_growth_bytes: u64,
    available_growth_bytes: u64,
    reserve_bytes: u64,
    tracked_growth_bytes: u64,
    running_penalty_tokens: usize,
    waiting_penalty_tokens: usize,
    reason: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TextPrefillAdmissionRejection {
    HardCap {
        limit: usize,
    },
    Dynamic {
        estimate: TextPrefillAdmissionEstimate,
    },
}

fn estimate_text_prefill_allowed_prompt_len(
    memory_config: &MemoryConfig,
    snapshot: &TextPrefillAdmissionSnapshot,
) -> Option<TextPrefillAdmissionEstimate> {
    if memory_config.max_seq_len != 0
        || memory_config.text_prefill_admission.mode != TextPrefillAdmissionMode::Dynamic
    {
        return None;
    }

    let configured_limit_bytes = if memory_config.gpu_memory_limit_bytes > 0 {
        memory_config.gpu_memory_limit_bytes
    } else if snapshot.gpu_total_bytes > 0 {
        snapshot.gpu_total_bytes
    } else {
        return None;
    };

    let headroom_budget_bytes =
        configured_limit_bytes.saturating_sub(memory_config.baseline_gpu_bytes);
    let observed_growth_bytes = snapshot
        .gpu_used_bytes
        .saturating_sub(memory_config.baseline_gpu_bytes);
    let tracked_growth_bytes = snapshot
        .tracked_kv_bytes
        .saturating_mul(KV_GPU_OVERHEAD_FACTOR);
    let accounted_growth_bytes = observed_growth_bytes.max(tracked_growth_bytes);
    let reserve_bytes = memory_config
        .text_prefill_admission
        .reserve_bytes
        .min(headroom_budget_bytes);
    let available_growth_bytes = headroom_budget_bytes
        .saturating_sub(accounted_growth_bytes)
        .saturating_sub(reserve_bytes);

    let tokens_from_bytes = (available_growth_bytes
        / memory_config.text_prefill_admission.bytes_per_token)
        .try_into()
        .unwrap_or(usize::MAX);
    let running_penalty_tokens = snapshot
        .running_sequences
        .saturating_mul(memory_config.text_prefill_admission.running_penalty_tokens);
    let waiting_penalty_tokens = snapshot
        .waiting_sequences
        .saturating_mul(memory_config.text_prefill_admission.waiting_penalty_tokens);
    let allowed_prompt_len_now = tokens_from_bytes
        .saturating_sub(running_penalty_tokens)
        .saturating_sub(waiting_penalty_tokens);

    let reason = if available_growth_bytes == 0 {
        "gpu_headroom"
    } else if running_penalty_tokens > 0 || waiting_penalty_tokens > 0 {
        "gpu_headroom_and_queue_load"
    } else {
        "gpu_headroom_budget"
    };

    Some(TextPrefillAdmissionEstimate {
        allowed_prompt_len_now,
        configured_limit_bytes,
        headroom_budget_bytes,
        accounted_growth_bytes,
        available_growth_bytes,
        reserve_bytes,
        tracked_growth_bytes,
        running_penalty_tokens,
        waiting_penalty_tokens,
        reason,
    })
}

fn should_reject_text_prefill_request(
    memory_config: &MemoryConfig,
    snapshot: &TextPrefillAdmissionSnapshot,
    multimodal_inputs: &types::MultimodalInputs,
) -> Option<TextPrefillAdmissionRejection> {
    if memory_config.max_seq_len != 0 || !multimodal_inputs.is_empty() {
        return None;
    }

    if let Some(limit) = memory_config.text_prefill_token_limit {
        if snapshot.prompt_len > limit {
            return Some(TextPrefillAdmissionRejection::HardCap { limit });
        }
    }

    let estimate = estimate_text_prefill_allowed_prompt_len(memory_config, snapshot)?;
    if snapshot.prompt_len > estimate.allowed_prompt_len_now {
        return Some(TextPrefillAdmissionRejection::Dynamic { estimate });
    }

    None
}

fn is_probable_oom(err: &str) -> bool {
    let lower = err.to_ascii_lowercase();
    lower.contains("out of memory") || lower.contains("cuda_error_out_of_memory")
}

/// Query current GPU memory usage. Returns (used_bytes, total_bytes).
/// Returns (0, 0) if not on CUDA.
fn query_gpu_memory_usage(_device: &Device) -> (u64, u64) {
    #[cfg(feature = "cuda")]
    {
        if let Device::Cuda(_) = _device {
            if let Ok((free, total)) =
                candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
            {
                return ((total - free) as u64, total as u64);
            }
        }
    }
    (0, 0)
}

/// Format a byte count as a human-readable string (used in engine log messages).
fn format_bytes_engine(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.1}G", bytes as f64 / (1u64 << 30) as f64)
    } else if bytes >= 1 << 20 {
        format!("{:.0}M", bytes as f64 / (1u64 << 20) as f64)
    } else {
        format!("{}B", bytes)
    }
}

// ─────────────────────────────────────────────────────────────
//  InferenceEngine
// ─────────────────────────────────────────────────────────────

/// KV-to-GPU overhead factor.
///
/// `tracked_kv_bytes` only captures live per-sequence KV cache tensors, which
/// is roughly 15-20% of the *real* GPU memory consumed.  Batch-decode setup
/// creates padded copies, the CUDA caching allocator retains freed blocks,
/// and forward-pass intermediates add extra pressure.  Empirically the ratio
/// between actual GPU growth over baseline and tracked KV bytes is 5-8×.
///
/// We use 6× so that `kv_budget = (limit - baseline) / 6`.  This gives the
/// engine a realistic estimate of how much KV it can afford before the GPU
/// runs out of memory.
const KV_GPU_OVERHEAD_FACTOR: u64 = 6;

/// Continuous-batching inference engine.
///
/// Runs on a dedicated OS thread (model forward passes are synchronous).
/// Communicates with async API handlers via channels.
pub struct InferenceEngine {
    model: Box<dyn RuntimeModel>,
    sequences: HashMap<String, Sequence>,
    request_states: HashMap<String, runtime::request_state::RequestState>,
    token_streams: HashMap<String, TokenOutputStream>,
    scheduler: Scheduler,
    request_rx: mpsc::UnboundedReceiver<EngineRequest>,
    active_seq_id: Option<String>,
    num_layers: usize,
    stats: Arc<EngineStats>,
    /// How many tokens to decode for one sequence before switching.
    decode_tokens_per_seq: usize,
    /// Engine start time for uptime calculation.
    start_time: Instant,
    /// Step counter for periodic stats logging.
    step_counter: u64,
    sampling_buffers: SamplingBuffers,
    input_builder: runtime::input_builder::IncrementalInputBuilder,
    kv_manager: runtime::kv_manager::KvMemoryManager,
    placement_policy: PlacementPolicy,
    /// Memory configuration for VRAM limits.
    memory_config: MemoryConfig,
    /// Timestamp of last memory-limit warning (to throttle log spam).
    last_mem_warn: Instant,
    /// Steps remaining before cuMemGetInfo checks are re-enabled after eviction.
    /// The CUDA caching allocator doesn't instantly reflect freed memory, so we
    /// grant a short cooldown after preemption to avoid a deadlock where
    /// cuMemGetInfo always reports over-limit.
    eviction_cooldown: u32,
}

impl InferenceEngine {
    /// Create the engine and return a handle for submitting requests.
    pub fn new(
        model: Box<dyn RuntimeModel>,
        max_concurrent: usize,
        decode_tokens_per_seq: usize,
        placement_policy: PlacementPolicy,
        memory_config: MemoryConfig,
    ) -> (Self, EngineHandle) {
        let (request_tx, request_rx) = mpsc::unbounded_channel();
        let num_layers = model.num_layers();

        // Cap max_concurrent to 1 for models without KV cache swapping.
        let effective_max = if model.supports_kv_swap() {
            max_concurrent
        } else {
            1.min(max_concurrent)
        };
        if effective_max != max_concurrent {
            info!(
                "Model does not support KV swap — limiting max_concurrent to {}",
                effective_max
            );
        }

        let stats = Arc::new(EngineStats::new());
        let engine = Self {
            model,
            sequences: HashMap::new(),
            request_states: HashMap::new(),
            token_streams: HashMap::new(),
            scheduler: Scheduler::new(effective_max),
            request_rx,
            active_seq_id: None,
            num_layers,
            stats: stats.clone(),
            decode_tokens_per_seq: decode_tokens_per_seq.max(1),
            start_time: Instant::now(),
            step_counter: 0,
            sampling_buffers: SamplingBuffers::new(),
            input_builder: runtime::input_builder::IncrementalInputBuilder::default(),
            kv_manager: runtime::kv_manager::KvMemoryManager::default(),
            placement_policy,
            memory_config,
            last_mem_warn: Instant::now() - std::time::Duration::from_secs(60),
            eviction_cooldown: 0,
        };
        let handle = EngineHandle { request_tx, stats };
        (engine, handle)
    }

    // ─────────────────────────────────────────────────────────
    //  Main loop
    // ─────────────────────────────────────────────────────────

    /// Run the engine loop (blocking — call from a dedicated thread).
    pub fn run(mut self) {
        // Log effective memory budget.
        let baseline = self.memory_config.baseline_gpu_bytes;
        let limit = self.memory_config.gpu_memory_limit_bytes;
        if limit > 0 {
            let kv_budget = self.kv_budget_bytes();
            if kv_budget == 0 || limit <= baseline {
                warn!(
                    "gpu_memory_limit ({}) <= model baseline ({}). \
                     KV-cache budget is 0 — all sequences will be immediately preempted.",
                    format_bytes_engine(limit),
                    format_bytes_engine(baseline),
                );
            } else {
                info!(
                    "Memory budget: total_limit={}, model_baseline={}, kv_budget={} (overhead={}x, also checked by cuMemGetInfo)",
                    format_bytes_engine(limit),
                    format_bytes_engine(baseline),
                    format_bytes_engine(kv_budget),
                    KV_GPU_OVERHEAD_FACTOR,
                );
            }
        }
        info!(
            "Engine started (max_concurrent={}, decode_tokens_per_seq={}, max_seq_len={}, text_prefill_token_limit={}, text_prefill_admission_mode={}, text_prefill_admission_reserve={}, text_prefill_admission_bytes_per_token={}, text_prefill_running_penalty_tokens={}, text_prefill_waiting_penalty_tokens={})",
            self.scheduler.max_running,
            self.decode_tokens_per_seq,
            if self.memory_config.max_seq_len == 0 {
                "unlimited".to_string()
            } else {
                self.memory_config.max_seq_len.to_string()
            },
            self.memory_config
                .text_prefill_token_limit
                .map(|v| v.to_string())
                .unwrap_or_else(|| "disabled".to_string()),
            self.memory_config.text_prefill_admission.mode.as_str(),
            format_bytes_engine(self.memory_config.text_prefill_admission.reserve_bytes),
            self.memory_config.text_prefill_admission.bytes_per_token,
            self.memory_config
                .text_prefill_admission
                .running_penalty_tokens,
            self.memory_config
                .text_prefill_admission
                .waiting_penalty_tokens,
        );

        loop {
            self.drain_requests();
            self.check_cancelled();
            if matches!(self.placement_policy, PlacementPolicy::OffloadReady) {
                self.model.on_device_policy_tick();
            }

            // Decrement eviction cooldown (cuMemGetInfo grace period).
            self.eviction_cooldown = self.eviction_cooldown.saturating_sub(1);

            self.stats
                .active_sequences
                .store(self.scheduler.running.len() as u64, Ordering::Relaxed);
            self.stats
                .waiting_sequences
                .store(self.scheduler.waiting.len() as u64, Ordering::Relaxed);

            let output = self.scheduler.schedule();

            match output {
                Some(output) => {
                    debug!(
                        is_prefill = output.is_prefill,
                        batch_size = output.batch.len(),
                        running = self.scheduler.running.len(),
                        waiting = self.scheduler.waiting.len(),
                        "Scheduler emitted batch"
                    );
                    // KV cache budget gate: if a prefill is scheduled but we're
                    // over the KV budget, first try to evict (preempt) the
                    // largest running sequence to make room. If still over,
                    // defer the prefill and drain existing sequences.
                    if output.is_prefill && self.is_over_kv_budget() {
                        if matches!(self.placement_policy, PlacementPolicy::OffloadReady) {
                            let runtime_offloaded = self.model.offload_under_memory_pressure();
                            if runtime_offloaded > 0 {
                                self.stats
                                    .total_runtime_offload_ops
                                    .fetch_add(1, Ordering::Relaxed);
                                self.stats
                                    .total_runtime_offload_units
                                    .fetch_add(runtime_offloaded as u64, Ordering::Relaxed);
                                info!(
                                    runtime_offloaded,
                                    "Runtime backend offloaded memory-managed units under pressure"
                                );
                            }

                            if self.model.supports_kv_swap() {
                                self.offload_inactive_kv_to_cpu();
                            }
                        }

                        // Attempt eviction before deferring.
                        self.evict_if_needed();

                        if self.is_over_kv_budget() && !self.scheduler.running.is_empty() {
                            // Still over budget and have running sequences to drain.
                            for seq_id in &output.batch {
                                self.scheduler.waiting.push_front(seq_id.clone());
                            }
                            let decode_batch: Vec<String> =
                                self.scheduler.running.iter().cloned().collect();
                            let decode_output = SchedulerOutput {
                                batch: decode_batch,
                                is_prefill: false,
                            };
                            self.execute_step(decode_output);
                        } else {
                            // Budget OK after eviction (or nothing running) — proceed.
                            self.execute_step(output);
                        }
                    } else {
                        self.execute_step(output);
                    }
                    self.step_counter += 1;

                    if self.step_counter.is_multiple_of(50) {
                        self.log_stats();
                    }
                }
                None => match self.request_rx.blocking_recv() {
                    Some(req) => self.accept_request(req),
                    None => {
                        info!("Engine channel closed, shutting down");
                        self.log_stats();
                        return;
                    }
                },
            }
        }
    }

    fn log_stats(&self) {
        let snap = self.stats.snapshot();
        let uptime = self.start_time.elapsed().as_secs();
        let (gpu_used, gpu_total) = query_gpu_memory_usage(self.model.device());
        let budget = self.kv_budget_bytes();
        let budget_info = if budget < u64::MAX {
            format!(" kv_budget: {}", format_bytes_engine(budget))
        } else {
            String::new()
        };
        let gpu_info = if gpu_total > 0 {
            format!(
                " | gpu_mem: {:.1}G/{:.1}G ({:.0}%) | kv_cache: {}{}",
                gpu_used as f64 / (1u64 << 30) as f64,
                gpu_total as f64 / (1u64 << 30) as f64,
                gpu_used as f64 / gpu_total as f64 * 100.0,
                format_bytes_engine(self.kv_manager.tracked_kv_bytes()),
                budget_info,
            )
        } else {
            format!(
                " | kv_cache: {}{}",
                format_bytes_engine(self.kv_manager.tracked_kv_bytes()),
                budget_info
            )
        };
        info!(
            "Engine stats | uptime={}s | requests: total={} completed={} cancelled={} failed={} | \
             sequences: active={} waiting={} | \
             tokens: prompt={} completion={} | \
             kv_swaps={} runtime_offload_ops={} runtime_offload_units={} kv_offload_ops={} kv_offload_tensors={} kv_prefetch_ops={} kv_prefetch_tensors={} | \
             speed: prefill={:.1} tok/s decode={:.1} tok/s{}",
            uptime,
            snap.total_requests,
            snap.completed_requests,
            snap.cancelled_requests,
            snap.failed_requests,
            snap.active_sequences,
            snap.waiting_sequences,
            snap.total_prompt_tokens,
            snap.total_completion_tokens,
            snap.total_kv_swaps,
            snap.total_runtime_offload_ops,
            snap.total_runtime_offload_units,
            snap.total_kv_offload_ops,
            snap.total_kv_offload_tensors,
            snap.total_kv_prefetch_ops,
            snap.total_kv_prefetch_tensors,
            snap.avg_prefill_tokens_per_sec,
            snap.avg_decode_tokens_per_sec,
            gpu_info,
        );
    }

    // ─────────────────────────────────────────────────────────
    //  Memory management
    // ─────────────────────────────────────────────────────────

    /// Recount `tracked_kv_bytes` from all sequences.
    /// For the active sequence, bytes are in the model (uses runtime `kv_bytes`).
    /// For other sequences, bytes are stored in `seq.kv_caches`.
    fn recount_kv_bytes(&mut self) {
        let mut total: u64 = 0;
        for (id, seq) in &self.sequences {
            if self.active_seq_id.as_deref() == Some(id.as_str()) {
                total += self.model.kv_bytes();
            } else {
                total += sequence::kv_cache_bytes(&seq.kv_caches);
            }
        }
        self.kv_manager.set_tracked_kv_bytes(total);
    }

    /// KV cache budget **in KV-cache bytes** (not raw GPU bytes).
    ///
    /// Each byte of live KV cache costs roughly `KV_GPU_OVERHEAD_FACTOR` bytes
    /// of real GPU memory (due to padded batch copies, CUDA pool bloat, and
    /// forward-pass intermediates).  The budget is therefore:
    ///
    /// ```text
    /// kv_budget = (gpu_limit - baseline) / KV_GPU_OVERHEAD_FACTOR
    /// ```
    ///
    /// Returns `u64::MAX` when no limit is configured.
    fn kv_budget_bytes(&self) -> u64 {
        let limit = self.memory_config.gpu_memory_limit_bytes;
        if limit == 0 {
            return u64::MAX;
        }
        let raw = limit.saturating_sub(self.memory_config.baseline_gpu_bytes);
        raw / KV_GPU_OVERHEAD_FACTOR
    }

    /// Check whether the engine should block new prefills due to memory
    /// pressure.  Two complementary checks:
    ///
    /// 1. **KV budget** — `tracked_kv_bytes > kv_budget_bytes()`.  This is the
    ///    primary admission control, using an overhead factor to estimate real
    ///    GPU cost from the tracked KV cache bytes.
    ///
    /// 2. **cuMemGetInfo hard safety** — if actual GPU memory (as reported by
    ///    the driver) exceeds the configured limit, block prefills.  This
    ///    catches cases where the overhead factor underestimates.  The check
    ///    is skipped during `eviction_cooldown` to avoid a deadlock (the CUDA
    ///    caching allocator doesn't instantly reflect freed memory).
    fn is_over_kv_budget(&mut self) -> bool {
        let limit = self.memory_config.gpu_memory_limit_bytes;
        if limit == 0 {
            return false;
        }

        let budget = self.kv_budget_bytes();
        if budget == 0 {
            return true; // limit <= baseline
        }

        // Check 1: tracked KV bytes vs overhead-adjusted budget.
        if self.kv_manager.tracked_kv_bytes() > budget {
            let now = Instant::now();
            if now.duration_since(self.last_mem_warn).as_secs() >= 5 {
                self.last_mem_warn = now;
                warn!(
                    "KV budget exceeded: kv_used={} > kv_budget={} (limit={} baseline={} overhead={}x)",
                    format_bytes_engine(self.kv_manager.tracked_kv_bytes()),
                    format_bytes_engine(budget),
                    format_bytes_engine(limit),
                    format_bytes_engine(self.memory_config.baseline_gpu_bytes),
                    KV_GPU_OVERHEAD_FACTOR,
                );
            }
            return true;
        }

        // Check 2: cuMemGetInfo hard safety (skip during cooldown).
        if self.eviction_cooldown == 0 {
            let (gpu_used, _) = query_gpu_memory_usage(self.model.device());
            if gpu_used > 0 && gpu_used > limit {
                let now = Instant::now();
                if now.duration_since(self.last_mem_warn).as_secs() >= 5 {
                    self.last_mem_warn = now;
                    warn!(
                        "GPU memory hard limit exceeded: gpu_used={} > limit={} (kv_tracked={})",
                        format_bytes_engine(gpu_used),
                        format_bytes_engine(limit),
                        format_bytes_engine(self.kv_manager.tracked_kv_bytes()),
                    );
                }
                return true;
            }
        }

        false
    }

    /// Preempt (evict) running sequences until KV usage is within budget.
    ///
    /// Eviction policy: **longest-output-first** — the sequence that has
    /// generated the most tokens (and therefore holds the largest KV cache)
    /// is evicted first. Its KV cache is dropped and it is moved back to
    /// the waiting queue for later re-prefill.
    ///
    /// This mirrors sglang's retraction strategy.
    fn evict_if_needed(&mut self) {
        let budget = self.kv_budget_bytes();
        if budget == u64::MAX {
            return;
        }

        while self.kv_manager.tracked_kv_bytes() > budget && !self.scheduler.running.is_empty() {
            // Find the running sequence with the most generated tokens (largest KV).
            let victim_id = self
                .scheduler
                .running
                .iter()
                .filter_map(|id| {
                    self.sequences
                        .get(id)
                        .map(|seq| (id.clone(), seq.tokens.len()))
                })
                .max_by_key(|(_, len)| *len)
                .map(|(id, _)| id);

            let victim_id = match victim_id {
                Some(id) => id,
                None => break,
            };

            // Compute bytes being freed.
            let freed = self
                .sequences
                .get(&victim_id)
                .map(|seq| sequence::kv_cache_bytes(&seq.kv_caches))
                .unwrap_or(0);

            info!(
                id = %victim_id,
                freed_bytes = %format_bytes_engine(freed),
                kv_used = %format_bytes_engine(self.kv_manager.tracked_kv_bytes()),
                kv_budget = %format_bytes_engine(budget),
                "Preempting sequence (KV cache eviction) — will re-prefill later",
            );

            // If this sequence's KV is currently loaded in the model, clear it.
            if self.active_seq_id.as_deref() == Some(&victim_id) {
                self.model.clear_kv_cache();
                self.active_seq_id = None;
            }

            // Drop KV caches and reset sequence state to Waiting.
            if let Some(seq) = self.sequences.get_mut(&victim_id) {
                seq.kv_caches = vec![None; self.num_layers];
                seq.status = SequenceStatus::Waiting;
                // Reset tokens to just the prompt to allow re-prefill.
                seq.tokens.truncate(seq.prompt_len);
                if let Some(state) = self.request_states.get_mut(&victim_id) {
                    state.pending_input_ids = seq.tokens.iter().copied().collect();
                    state.processed_tokens = 0;
                }
            }

            self.kv_manager
                .set_tracked_kv_bytes(self.kv_manager.tracked_kv_bytes().saturating_sub(freed));

            // Move from running back to waiting (back, not front — avoid
            // immediate re-prefill which would cause thrashing).
            self.scheduler.running.retain(|id| id != &victim_id);
            self.scheduler.waiting.push_back(victim_id);
        }

        // Cap effective max_running to the post-eviction running count.
        // This prevents the scheduler from admitting new sequences that
        // would immediately exceed the budget again (eviction thrashing).
        // The cap is lifted when a sequence finishes naturally.
        let post_eviction_running = self.scheduler.running.len();
        self.scheduler.effective_max_running = Some(post_eviction_running);
        info!(
            "Eviction complete: capping concurrent sequences at {} (was {})",
            post_eviction_running, self.scheduler.max_running,
        );

        // Grant a cooldown period so the cuMemGetInfo hard-safety check
        // doesn't immediately re-trigger (CUDA pool retains freed blocks).
        self.eviction_cooldown = 5;
    }

    fn offload_inactive_kv_to_cpu(&mut self) {
        if matches!(self.model.device(), Device::Cpu) {
            return;
        }

        let mut offloaded_tensors = 0usize;
        for (seq_id, seq) in self.sequences.iter_mut() {
            if self.active_seq_id.as_deref() == Some(seq_id.as_str()) {
                continue;
            }
            offloaded_tensors += self.model.offload_kv_caches(&mut seq.kv_caches);
        }

        if offloaded_tensors > 0 {
            self.recount_kv_bytes();
            self.stats
                .total_kv_offload_ops
                .fetch_add(1, Ordering::Relaxed);
            self.stats
                .total_kv_offload_tensors
                .fetch_add(offloaded_tensors as u64, Ordering::Relaxed);
            info!(
                offloaded_tensors,
                kv_gpu_used = %format_bytes_engine(self.kv_manager.tracked_kv_bytes()),
                "Offloaded inactive KV caches to CPU",
            );
        }
    }

    /// Effective max_tokens for a request, taking server-level max_seq_len into account.
    fn effective_max_tokens(&self, prompt_len: usize, requested_max_tokens: usize) -> usize {
        if self.memory_config.max_seq_len == 0 {
            return requested_max_tokens;
        }
        let remaining = self.memory_config.max_seq_len.saturating_sub(prompt_len);
        requested_max_tokens.min(remaining)
    }

    // ─────────────────────────────────────────────────────────
    //  Request handling
    // ─────────────────────────────────────────────────────────

    fn drain_requests(&mut self) {
        while let Ok(req) = self.request_rx.try_recv() {
            self.accept_request(req);
        }
    }

    fn accept_request(&mut self, req: EngineRequest) {
        let prompt_len = req.tokens.len();
        let tokenizer = self.model.tokenizer().clone();
        let request_id = req.id.clone();
        let request_tokens = req.tokens.clone();
        let request_multimodal = req.multimodal_inputs.clone();
        let (gpu_used_bytes, gpu_total_bytes) = query_gpu_memory_usage(self.model.device());
        let admission_snapshot = TextPrefillAdmissionSnapshot {
            prompt_len,
            gpu_used_bytes,
            gpu_total_bytes,
            tracked_kv_bytes: self.kv_manager.tracked_kv_bytes(),
            running_sequences: self.scheduler.running.len(),
            waiting_sequences: self.scheduler.waiting.len(),
        };
        let dynamic_estimate =
            estimate_text_prefill_allowed_prompt_len(&self.memory_config, &admission_snapshot);

        if let Some(estimate) = dynamic_estimate.as_ref() {
            info!(
                id = %req.id,
                prompt_len,
                allowed_prompt_len_now = estimate.allowed_prompt_len_now,
                admission_reason = estimate.reason,
                gpu_used = %format_bytes_engine(admission_snapshot.gpu_used_bytes),
                gpu_total = %format_bytes_engine(admission_snapshot.gpu_total_bytes),
                configured_limit = %format_bytes_engine(estimate.configured_limit_bytes),
                headroom_budget = %format_bytes_engine(estimate.headroom_budget_bytes),
                accounted_growth = %format_bytes_engine(estimate.accounted_growth_bytes),
                available_growth = %format_bytes_engine(estimate.available_growth_bytes),
                reserve = %format_bytes_engine(estimate.reserve_bytes),
                tracked_kv = %format_bytes_engine(admission_snapshot.tracked_kv_bytes),
                tracked_growth = %format_bytes_engine(estimate.tracked_growth_bytes),
                running = admission_snapshot.running_sequences,
                waiting = admission_snapshot.waiting_sequences,
                running_penalty_tokens = estimate.running_penalty_tokens,
                waiting_penalty_tokens = estimate.waiting_penalty_tokens,
                hard_cap = ?self.memory_config.text_prefill_token_limit,
                "Dynamic text-prefill admission evaluated",
            );
        }

        if let Some(rejection) = should_reject_text_prefill_request(
            &self.memory_config,
            &admission_snapshot,
            &req.multimodal_inputs,
        ) {
            match rejection {
                TextPrefillAdmissionRejection::HardCap { limit } => {
                    warn!(
                        id = %req.id,
                        prompt_len,
                        text_prefill_token_limit = limit,
                        queue_waiting = self.scheduler.waiting.len(),
                        queue_running = self.scheduler.running.len(),
                        image_inputs = req.multimodal_inputs.image_urls.len(),
                        audio_inputs = req.multimodal_inputs.audio_urls.len(),
                        "Prompt exceeds text-only hard prefill cap, rejecting request",
                    );
                    let _ = req.response_tx.send(EngineResponse::Error(format!(
                        "Prompt length ({prompt_len}) exceeds CRANE_TEXT_PREFILL_TOKEN_LIMIT ({limit}) for text-only requests while server max_seq_len is unlimited"
                    )));
                }
                TextPrefillAdmissionRejection::Dynamic { estimate } => {
                    warn!(
                        id = %req.id,
                        prompt_len,
                        allowed_prompt_len_now = estimate.allowed_prompt_len_now,
                        admission_reason = estimate.reason,
                        gpu_used = %format_bytes_engine(admission_snapshot.gpu_used_bytes),
                        configured_limit = %format_bytes_engine(estimate.configured_limit_bytes),
                        reserve = %format_bytes_engine(estimate.reserve_bytes),
                        tracked_kv = %format_bytes_engine(admission_snapshot.tracked_kv_bytes),
                        queue_waiting = admission_snapshot.waiting_sequences,
                        queue_running = admission_snapshot.running_sequences,
                        image_inputs = req.multimodal_inputs.image_urls.len(),
                        audio_inputs = req.multimodal_inputs.audio_urls.len(),
                        "Prompt exceeds dynamic text-prefill budget, rejecting request",
                    );
                    let _ = req.response_tx.send(EngineResponse::Error(format!(
                        "Prompt length ({prompt_len}) exceeds the current dynamic text-prefill budget: allowed prompt_len_now≈{} tokens (reason={}, gpu_used={}, configured_limit={}, reserve={}, tracked_kv={}, running={}, waiting={})",
                        estimate.allowed_prompt_len_now,
                        estimate.reason,
                        format_bytes_engine(admission_snapshot.gpu_used_bytes),
                        format_bytes_engine(estimate.configured_limit_bytes),
                        format_bytes_engine(estimate.reserve_bytes),
                        format_bytes_engine(admission_snapshot.tracked_kv_bytes),
                        admission_snapshot.running_sequences,
                        admission_snapshot.waiting_sequences,
                    )));
                }
            }
            self.stats.failed_requests.fetch_add(1, Ordering::Relaxed);
            return;
        }

        // Reject prompts that already exceed max_seq_len.
        if self.memory_config.max_seq_len > 0 && prompt_len > self.memory_config.max_seq_len {
            warn!(
                id = %req.id,
                prompt_len,
                max_seq_len = self.memory_config.max_seq_len,
                "Prompt exceeds max_seq_len, rejecting request",
            );
            let _ = req.response_tx.send(EngineResponse::Error(format!(
                "Prompt length ({}) exceeds server max_seq_len ({})",
                prompt_len, self.memory_config.max_seq_len,
            )));
            self.stats.failed_requests.fetch_add(1, Ordering::Relaxed);
            return;
        }

        // Cap max_tokens to respect max_seq_len.
        let effective_max_tokens = self.effective_max_tokens(prompt_len, req.max_tokens);

        info!(
            id = %req.id,
            prompt_len,
            image_inputs = req.multimodal_inputs.image_urls.len(),
            audio_inputs = req.multimodal_inputs.audio_urls.len(),
            max_tokens = effective_max_tokens,
            temp = ?req.temperature,
            top_p = ?req.top_p,
            top_k = ?req.top_k,
            rep_penalty = req.repetition_penalty,
            "New request accepted (queue: waiting={} running={})",
            self.scheduler.waiting.len() + 1,
            self.scheduler.running.len(),
        );

        self.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_prompt_tokens
            .fetch_add(prompt_len as u64, Ordering::Relaxed);

        let seq = Sequence {
            id: req.id.clone(),
            status: SequenceStatus::Waiting,
            tokens: req.tokens,
            multimodal_inputs: req.multimodal_inputs,
            prompt_len,
            kv_caches: vec![None; self.num_layers],
            logits_processor: candle_transformers::generation::LogitsProcessor::new(
                sampling::rand_seed(),
                req.temperature,
                req.top_p,
            ),
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            max_tokens: effective_max_tokens,
            eos_token_id: req.eos_token_id,
            repetition_penalty: req.repetition_penalty,
            repeat_last_n: 64,
            response_tx: req.response_tx,
        };

        let stream = TokenOutputStream::new(tokenizer);
        self.sequences.insert(req.id.clone(), seq);
        self.request_states.insert(
            request_id,
            runtime::request_state::RequestState {
                id: req.id.clone(),
                pending_input_ids: request_tokens.into_iter().collect(),
                processed_tokens: 0,
                multimodal_inputs: request_multimodal,
            },
        );
        self.token_streams.insert(req.id.clone(), stream);
        self.scheduler.add(req.id);
    }

    // ─────────────────────────────────────────────────────────
    //  Cancellation detection
    // ─────────────────────────────────────────────────────────

    fn check_cancelled(&mut self) {
        let cancelled: Vec<String> = self
            .sequences
            .iter()
            .filter(|(_, seq)| seq.response_tx.is_closed())
            .map(|(id, _)| id.clone())
            .collect();

        for id in cancelled {
            warn!(id = %id, "Client disconnected, cancelling sequence");
            self.stats
                .cancelled_requests
                .fetch_add(1, Ordering::Relaxed);
            self.cleanup_sequence(&id);
        }
    }

    // ─────────────────────────────────────────────────────────
    //  Step execution dispatch
    // ─────────────────────────────────────────────────────────

    fn execute_step(&mut self, output: SchedulerOutput) {
        if output.is_prefill {
            debug_assert_eq!(output.batch.len(), 1);
            let seq_id = &output.batch[0];
            self.step_prefill(seq_id.clone());
        } else if self.model.supports_batch_decode() && output.batch.len() > 1 {
            // True batched decode only when there are multiple sequences.
            // For a single sequence the sequential path is far cheaper: it
            // keeps the KV cache resident in the model and avoids the
            // extract→pad→stack→extract GPU-copy cycle that batch decode
            // performs every scheduling round.
            self.step_decode_batch(output.batch);
        } else {
            self.step_decode_sequential(output.batch);
        }
    }

    // ─────────────────────────────────────────────────────────
    //  Prefill
    // ─────────────────────────────────────────────────────────

    fn step_prefill(&mut self, seq_id: String) {
        let t0 = Instant::now();

        if !self.swap_in(&seq_id) {
            return;
        }

        let (input_ids, start_pos) = {
            let seq = self.sequences.get(&seq_id).unwrap();
            let prompt_len = seq.prompt_len;
            let input_ids = self
                .request_states
                .get_mut(&seq_id)
                .map(|state| state.pop_incremental(prompt_len))
                .filter(|ids| !ids.is_empty())
                .unwrap_or_else(|| seq.next_input_ids().to_vec());
            (input_ids, seq.start_pos())
        };
        let multimodal_inputs = self
            .sequences
            .get(&seq_id)
            .map(|seq| seq.multimodal_inputs.clone())
            .unwrap_or_default();

        let prompt_len = input_ids.len();
        debug!(
            id = %seq_id,
            input_len = prompt_len,
            start_pos,
            image_inputs = multimodal_inputs.image_urls.len(),
            audio_inputs = multimodal_inputs.audio_urls.len(),
            "Prefill step inputs prepared"
        );

        let logits = match self.model.prefill(RuntimeRequestContext {
            input_ids,
            start_pos,
            multimodal_inputs: multimodal_inputs.clone(),
        }) {
            Ok(step) => step.logits,
            Err(e) => {
                let err = e.to_string();
                let oom_context =
                    self.prefill_oom_context(prompt_len, start_pos, &multimodal_inputs);
                if is_probable_oom(&err) {
                    error!(
                        id = %seq_id,
                        prompt_len,
                        start_pos,
                        queue_waiting = self.scheduler.waiting.len(),
                        queue_running = self.scheduler.running.len(),
                        image_inputs = multimodal_inputs.image_urls.len(),
                        audio_inputs = multimodal_inputs.audio_urls.len(),
                        oom_context = %oom_context,
                        error = %err,
                        "Prefill forward hit OOM"
                    );
                    self.send_error(
                        &seq_id,
                        &format!("Prefill forward failed: {err} [{oom_context}]"),
                    );
                } else {
                    self.send_error(&seq_id, &format!("Prefill forward failed: {err}"));
                }
                return;
            }
        };

        let next_token = {
            let seq = self.sequences.get_mut(&seq_id).unwrap();
            match sampling::sample(&seq_id, seq, &logits, &mut self.sampling_buffers) {
                Ok(t) => t,
                Err(e) => {
                    self.send_error(&seq_id, &format!("Sampling failed: {e}"));
                    return;
                }
            }
        };

        self.swap_out(&seq_id);

        let prefill_us = t0.elapsed().as_micros() as u64;
        self.stats
            .total_prefill_time_us
            .fetch_add(prefill_us, Ordering::Relaxed);

        let prefill_tok_s = if prefill_us > 0 {
            (prompt_len as f64) / (prefill_us as f64 / 1_000_000.0)
        } else {
            0.0
        };

        {
            let seq = self.sequences.get_mut(&seq_id).unwrap();
            seq.tokens.push(next_token);
            seq.status = SequenceStatus::Running;
        }
        if let Some(state) = self.request_states.get_mut(&seq_id) {
            state.pending_input_ids.push_back(next_token);
        }

        info!(
            id = %seq_id,
            prompt_len,
            prefill_ms = prefill_us / 1000,
            prefill_tok_s = format!("{:.1}", prefill_tok_s),
            "Prefill complete, first token generated",
        );

        self.send_token(&seq_id, next_token);

        if self.sequences.get(&seq_id).unwrap().should_stop() {
            self.finish_sequence(&seq_id);
        } else {
            self.scheduler.promote_to_running(seq_id);
        }
    }

    // ─────────────────────────────────────────────────────────
    //  Batched decode
    // ─────────────────────────────────────────────────────────

    /// Decode step for all running sequences — TRUE BATCHED forward.
    ///
    /// Uses **lazy eviction**: when a sequence completes or is cancelled
    /// mid-loop, it stays in the batch tensor (wasting trivial compute)
    /// rather than triggering an expensive extract→re-setup cycle.
    fn step_decode_batch(&mut self, batch: Vec<String>) {
        let t0 = Instant::now();

        // Filter cancelled sequences.
        let cancelled: Vec<String> = batch
            .iter()
            .filter(|id| {
                self.sequences
                    .get(id.as_str())
                    .is_none_or(|s| s.response_tx.is_closed())
            })
            .cloned()
            .collect();
        for id in &cancelled {
            warn!(id = %id, "Client disconnected before decode batch");
            self.stats
                .cancelled_requests
                .fetch_add(1, Ordering::Relaxed);
            self.cleanup_sequence(id);
        }
        let batch: Vec<String> = batch
            .into_iter()
            .filter(|id| !cancelled.contains(id))
            .collect();
        if batch.is_empty() {
            return;
        }

        debug!(batch_size = batch.len(), batch = ?batch, "Starting batched decode");

        if matches!(self.placement_policy, PlacementPolicy::OffloadReady) {
            let mut prefetched_tensors = 0usize;
            for seq_id in &batch {
                if let Some(seq) = self.sequences.get_mut(seq_id) {
                    prefetched_tensors += self.model.prefetch_kv_caches(&mut seq.kv_caches);
                }
            }
            if prefetched_tensors > 0 {
                self.stats
                    .total_kv_prefetch_ops
                    .fetch_add(1, Ordering::Relaxed);
                self.stats
                    .total_kv_prefetch_tensors
                    .fetch_add(prefetched_tensors as u64, Ordering::Relaxed);
            }
        }

        let batch_size = batch.len();

        // Flush model's internal KV cache state.
        if let Some(ref prev_id) = self.active_seq_id.take() {
            if self.sequences.contains_key(prev_id) {
                match self.model.kv_extract() {
                    Ok(caches) => {
                        if let Some(seq) = self.sequences.get_mut(prev_id) {
                            seq.kv_caches = caches;
                        }
                    }
                    Err(err) => {
                        self.model.clear_kv_cache();
                        self.recount_kv_bytes();
                        self.send_error(prev_id, &format!("KV export failed: {err}"));
                        return;
                    }
                }
            }
            self.model.clear_kv_cache();
        }
        self.recount_kv_bytes();

        // Collect KV caches and setup batched decode.
        let kv_caches: Vec<runtime::LayerKvCaches> = batch
            .iter()
            .map(|id| self.sequences.get(id).unwrap().kv_caches.clone())
            .collect();

        let (kv_lens, original_max_kv) = match self
            .model
            .setup_batch_decode(&kv_caches, self.decode_tokens_per_seq)
        {
            Ok(r) => r,
            Err(e) => {
                error!("Batch decode setup failed: {e}");
                for seq_id in &batch {
                    self.send_error(seq_id, &format!("Batch decode setup failed: {e}"));
                }
                return;
            }
        };
        drop(kv_caches);

        // Now that setup_batch_decode has consumed the KV views (building its
        // own padded buffer), drop the per-sequence cache references.  With
        // zero-copy narrow views from get_kv_caches(), these still pin the
        // old pre-allocated buffers — clearing them here lets CUDA free that
        // VRAM before the decode loop allocates intermediates.
        for seq_id in &batch {
            if let Some(seq) = self.sequences.get_mut(seq_id) {
                seq.kv_caches = vec![None; self.num_layers];
            }
        }

        let t_setup = t0.elapsed();

        // Pre-build attention mask.
        let max_total_width = original_max_kv + self.decode_tokens_per_seq;
        let full_mask =
            match self
                .model
                .build_batch_decode_mask(&kv_lens, original_max_kv, max_total_width)
            {
                Ok(m) => m,
                Err(e) => {
                    error!("Mask build failed: {e}");
                    self.model.clear_kv_cache();
                    return;
                }
            };

        // Multi-round decode loop with lazy eviction.
        let mut total_tokens_this_step = 0u64;
        let mut rounds_done = 0usize;
        let mut alive = vec![true; batch.len()];
        let mut pending_finish: Vec<String> = Vec::new();
        let mut pending_cancel: Vec<String> = Vec::new();

        let mut positions: Vec<usize> = batch
            .iter()
            .map(|id| self.sequences.get(id).unwrap().start_pos())
            .collect();

        let mut last_tokens: Vec<u32> = batch
            .iter()
            .map(|id| *self.sequences.get(id).unwrap().tokens.last().unwrap())
            .collect();

        for round in 0..self.decode_tokens_per_seq {
            if alive.iter().all(|a| !a) {
                break;
            }

            let tokens: Vec<u32> = (0..batch.len())
                .map(|i| {
                    if alive[i] {
                        *self
                            .sequences
                            .get(&batch[i])
                            .unwrap()
                            .tokens
                            .last()
                            .unwrap()
                    } else {
                        last_tokens[i]
                    }
                })
                .collect();

            let input_ids =
                match crane_core::fused_ops::copy_from_slice_u32(&tokens, self.model.device())
                    .and_then(|t| t.reshape((batch_size, 1)))
                {
                    Ok(t) => t,
                    Err(e) => {
                        error!("Decode input_ids upload failed: {e}");
                        self.model.clear_kv_cache();
                        return;
                    }
                };

            let mask_width = original_max_kv + round + 1;
            let mask_for_round = match &full_mask {
                Some(full) => full.narrow(3, 0, mask_width).ok(),
                None => None,
            };

            let logits = match self.model.batch_decode(BatchDecodeContext {
                input_ids: &input_ids,
                positions: &positions,
                attention_mask: mask_for_round.as_ref(),
                batch_kv_info: Some((&kv_lens, original_max_kv)),
            }) {
                Ok(l) => l,
                Err(e) => {
                    error!("Batched decode forward failed (round {round}): {e}");
                    for (i, seq_id) in batch.iter().enumerate() {
                        if alive[i] {
                            self.send_error(seq_id, &format!("Batched decode failed: {e}"));
                        }
                    }
                    self.model.clear_kv_cache();
                    return;
                }
            };

            rounds_done += 1;

            for (i, seq_id) in batch.iter().enumerate() {
                if !alive[i] {
                    continue;
                }

                let seq_logits = match logits.narrow(0, i, 1) {
                    Ok(l) => l,
                    Err(e) => {
                        self.send_error(seq_id, &format!("Logits extraction failed: {e}"));
                        alive[i] = false;
                        continue;
                    }
                };

                let next_token = {
                    let seq = self.sequences.get_mut(seq_id).unwrap();
                    match sampling::sample(seq_id, seq, &seq_logits, &mut self.sampling_buffers) {
                        Ok(t) => t,
                        Err(e) => {
                            self.send_error(seq_id, &format!("Sampling failed: {e}"));
                            alive[i] = false;
                            continue;
                        }
                    }
                };

                if let Some(seq) = self.sequences.get_mut(seq_id) {
                    seq.tokens.push(next_token);
                }
                last_tokens[i] = next_token;

                total_tokens_this_step += 1;
                self.stats
                    .total_decode_steps
                    .fetch_add(1, Ordering::Relaxed);

                self.send_token(seq_id, next_token);

                if self.sequences.get(seq_id).is_none_or(|s| s.should_stop()) {
                    alive[i] = false;
                    pending_finish.push(seq_id.clone());
                } else if self
                    .sequences
                    .get(seq_id)
                    .is_none_or(|s| s.response_tx.is_closed())
                {
                    warn!(id = %seq_id, "Client disconnected mid-batch-decode");
                    alive[i] = false;
                    pending_cancel.push(seq_id.clone());
                }
            }

            for p in positions.iter_mut() {
                *p += 1;
            }
        }

        // Extract per-sequence KV caches.
        if rounds_done > 0 {
            match self
                .model
                .extract_batch_kv(&kv_lens, original_max_kv, rounds_done)
            {
                Ok(extracted) => {
                    for (i, seq_id) in batch.iter().enumerate() {
                        if alive[i] {
                            if let Some(seq) = self.sequences.get_mut(seq_id) {
                                if i < extracted.len() {
                                    seq.kv_caches = extracted[i].clone();
                                }
                            }
                        }
                    }
                    // KV caches changed for multiple sequences — recount.
                    self.recount_kv_bytes();
                }
                Err(e) => {
                    error!("Final KV extraction failed: {e}");
                    self.model.clear_kv_cache();
                    self.recount_kv_bytes();
                }
            }
        }

        for id in &pending_finish {
            self.finish_sequence(id);
        }
        for id in &pending_cancel {
            self.stats
                .cancelled_requests
                .fetch_add(1, Ordering::Relaxed);
            self.cleanup_sequence(id);
        }

        let decode_us = t0.elapsed().as_micros() as u64;
        self.stats
            .total_decode_time_us
            .fetch_add(decode_us, Ordering::Relaxed);

        if total_tokens_this_step > 0 {
            let tok_s = if decode_us > 0 {
                (total_tokens_this_step as f64) / (decode_us as f64 / 1_000_000.0)
            } else {
                0.0
            };
            debug!(
                batch_size,
                tokens = total_tokens_this_step,
                rounds = rounds_done,
                finished = pending_finish.len(),
                setup_ms = t_setup.as_millis() as u64,
                decode_ms = decode_us / 1000,
                tok_s = format!("{:.1}", tok_s),
                "Batched decode step complete",
            );
        }

        self.drain_requests();
        self.check_cancelled();
    }

    // ─────────────────────────────────────────────────────────
    //  Sequential decode
    // ─────────────────────────────────────────────────────────

    /// Sequential decode for backends without batch decode support.
    fn step_decode_sequential(&mut self, batch: Vec<String>) {
        let t0 = Instant::now();
        let mut total_tokens: u64 = 0;
        let mut swap_in_us: u64 = 0;
        let mut forward_us: u64 = 0;
        let mut sample_us: u64 = 0;
        let mut swap_out_us: u64 = 0;

        debug!(batch_size = batch.len(), batch = ?batch, "Starting sequential decode");

        for seq_id in &batch {
            if self
                .sequences
                .get(seq_id)
                .is_none_or(|s| s.response_tx.is_closed())
            {
                self.stats
                    .cancelled_requests
                    .fetch_add(1, Ordering::Relaxed);
                self.cleanup_sequence(seq_id);
                continue;
            }

            let t_swap_in = Instant::now();
            if !self.swap_in(seq_id) {
                continue;
            }
            swap_in_us += t_swap_in.elapsed().as_micros() as u64;

            debug!(id = %seq_id, "Sequential decode sequence activated");

            for _round in 0..self.decode_tokens_per_seq {
                let (input_ids, start_pos) = {
                    let seq = match self.sequences.get(seq_id) {
                        Some(s) => s,
                        None => break,
                    };
                    let input_ids = self
                        .request_states
                        .get_mut(seq_id)
                        .map(|state| self.input_builder.next_decode_chunk(state))
                        .filter(|ids| !ids.is_empty())
                        .unwrap_or_else(|| seq.next_input_ids().to_vec());
                    (input_ids, seq.start_pos())
                };

                let t_forward = Instant::now();
                let logits = match self.model.decode(RuntimeStepContext {
                    input_ids,
                    start_pos,
                }) {
                    Ok(step) => step.logits,
                    Err(e) => {
                        self.send_error(seq_id, &format!("Decode forward failed: {e}"));
                        break;
                    }
                };
                forward_us += t_forward.elapsed().as_micros() as u64;

                let t_sample = Instant::now();
                let next_token = {
                    let seq = self.sequences.get_mut(seq_id).unwrap();
                    match sampling::sample(seq_id, seq, &logits, &mut self.sampling_buffers) {
                        Ok(t) => t,
                        Err(e) => {
                            self.send_error(seq_id, &format!("Sampling failed: {e}"));
                            break;
                        }
                    }
                };
                sample_us += t_sample.elapsed().as_micros() as u64;

                if let Some(seq) = self.sequences.get_mut(seq_id) {
                    seq.tokens.push(next_token);
                }
                if let Some(state) = self.request_states.get_mut(seq_id) {
                    state.pending_input_ids.push_back(next_token);
                }

                total_tokens += 1;
                self.stats
                    .total_decode_steps
                    .fetch_add(1, Ordering::Relaxed);

                self.send_token(seq_id, next_token);

                if self.sequences.get(seq_id).is_none_or(|s| s.should_stop()) {
                    self.finish_sequence(seq_id);
                    break;
                }

                if self
                    .sequences
                    .get(seq_id)
                    .is_none_or(|s| s.response_tx.is_closed())
                {
                    warn!(id = %seq_id, "Client disconnected mid-decode");
                    self.stats
                        .cancelled_requests
                        .fetch_add(1, Ordering::Relaxed);
                    self.cleanup_sequence(seq_id);
                    break;
                }
            }

            let t_swap_out = Instant::now();
            self.swap_out(seq_id);
            swap_out_us += t_swap_out.elapsed().as_micros() as u64;
        }

        let decode_us = t0.elapsed().as_micros() as u64;
        self.stats
            .total_decode_time_us
            .fetch_add(decode_us, Ordering::Relaxed);

        if total_tokens > 0 {
            let tok_s = if decode_us > 0 {
                (total_tokens as f64) / (decode_us as f64 / 1_000_000.0)
            } else {
                0.0
            };
            debug!(
                tokens = total_tokens,
                decode_ms = decode_us / 1000,
                tok_s = format!("{:.1}", tok_s),
                "Sequential decode step complete",
            );
            info!(
                tokens = total_tokens,
                decode_ms = decode_us as f64 / 1_000.0,
                tok_s,
                swap_in_us,
                forward_us,
                sample_us,
                swap_out_us,
                seq_count = batch.len(),
                "decode_perf_sequential",
            );
        }

        self.drain_requests();
        self.check_cancelled();
    }

    // ─────────────────────────────────────────────────────────
    //  KV cache management
    // ─────────────────────────────────────────────────────────

    fn swap_in(&mut self, seq_id: &str) -> bool {
        if self.active_seq_id.as_deref() == Some(seq_id) {
            return true;
        }

        debug!(
            next_id = %seq_id,
            prev_active = ?self.active_seq_id,
            kv_swap = self.model.supports_kv_swap(),
            "swap_in begin"
        );

        if !self.model.supports_kv_swap() {
            if self.active_seq_id.as_deref() != Some(seq_id) {
                self.model.clear_kv_cache();
                self.active_seq_id = Some(seq_id.to_string());
            }
            return true;
        }

        // Save previous active sequence's KV cache from the model.
        if let Some(ref prev_id) = self.active_seq_id.clone() {
            match self.model.kv_extract() {
                Ok(caches) => {
                    if let Some(prev_seq) = self.sequences.get_mut(prev_id) {
                        prev_seq.kv_caches = caches;
                    }
                }
                Err(err) => {
                    self.model.clear_kv_cache();
                    self.recount_kv_bytes();
                    self.send_error(prev_id, &format!("KV export failed: {err}"));
                }
            }
        }

        // Load new sequence's KV cache into the model.
        let mut caches = self
            .sequences
            .get(seq_id)
            .map(|s| s.kv_caches.clone())
            .unwrap_or_else(|| vec![None; self.num_layers]);
        if matches!(self.placement_policy, PlacementPolicy::OffloadReady) {
            let prefetched_tensors = self.model.prefetch_kv_caches(&mut caches);
            if prefetched_tensors > 0 {
                self.stats
                    .total_kv_prefetch_ops
                    .fetch_add(1, Ordering::Relaxed);
                self.stats
                    .total_kv_prefetch_tensors
                    .fetch_add(prefetched_tensors as u64, Ordering::Relaxed);
            }
        }
        if let Err(err) = self.model.kv_restore(caches) {
            self.model.clear_kv_cache();
            self.recount_kv_bytes();
            self.send_error(seq_id, &format!("KV restore failed: {err}"));
            return false;
        }
        self.active_seq_id = Some(seq_id.to_string());

        self.recount_kv_bytes();
        self.stats
            .total_kv_swap_count
            .fetch_add(1, Ordering::Relaxed);
        debug!(id = %seq_id, "swap_in complete");
        true
    }

    /// Mark that the model finished processing `seq_id` for this scheduling
    /// round.  Instead of extracting full KV caches (expensive GPU copies),
    /// we only update byte tracking from the model's internal state.
    /// The actual KV tensors remain in the model and are saved lazily by
    /// `swap_in` when switching to a different sequence.
    fn swap_out(&mut self, seq_id: &str) {
        if !self.model.supports_kv_swap() {
            return;
        }
        if self.active_seq_id.as_deref() != Some(seq_id) {
            return;
        }
        // Drop stale seq cache references (from the last swap_in) to free
        // GPU memory.  swap_in will extract fresh caches from the model
        // when switching to a different sequence.
        if let Some(seq) = self.sequences.get_mut(seq_id) {
            if seq.kv_caches.iter().any(|c| c.is_some()) {
                seq.kv_caches = vec![None; seq.kv_caches.len()];
            }
        }
        self.recount_kv_bytes();
        debug!(id = %seq_id, "swap_out complete");
    }

    // ─────────────────────────────────────────────────────────
    //  Response sending
    // ─────────────────────────────────────────────────────────

    fn send_token(&mut self, seq_id: &str, token_id: u32) {
        let text = if let Some(stream) = self.token_streams.get_mut(seq_id) {
            match stream.next_token(token_id) {
                Ok(Some(t)) => t,
                Ok(None) => return,
                Err(e) => {
                    warn!(id = %seq_id, "Token decode error: {e}");
                    return;
                }
            }
        } else {
            return;
        };

        if let Some(seq) = self.sequences.get(seq_id) {
            if seq
                .response_tx
                .send(EngineResponse::Token { text, token_id })
                .is_err()
            {
                debug!(id = %seq_id, "Response channel closed (client disconnected)");
            }
        }
    }

    fn send_error(&mut self, seq_id: &str, msg: &str) {
        error!(id = %seq_id, "Engine error: {msg}");
        if let Some(seq) = self.sequences.get(seq_id) {
            let _ = seq.response_tx.send(EngineResponse::Error(msg.to_string()));
        }
        self.stats.failed_requests.fetch_add(1, Ordering::Relaxed);
        self.cleanup_sequence(seq_id);
    }

    fn prefill_oom_context(
        &self,
        prompt_len: usize,
        start_pos: usize,
        multimodal_inputs: &types::MultimodalInputs,
    ) -> String {
        let (gpu_used, gpu_total) = query_gpu_memory_usage(self.model.device());
        let mut fields = vec![
            format!("prompt_len={prompt_len}"),
            format!("start_pos={start_pos}"),
            format!("queue_waiting={}", self.scheduler.waiting.len()),
            format!("queue_running={}", self.scheduler.running.len()),
            format!("image_inputs={}", multimodal_inputs.image_urls.len()),
            format!("audio_inputs={}", multimodal_inputs.audio_urls.len()),
            format!(
                "text_prefill_token_limit={}",
                self.memory_config
                    .text_prefill_token_limit
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "disabled".to_string())
            ),
            format!(
                "max_seq_len={}",
                if self.memory_config.max_seq_len == 0 {
                    "unlimited".to_string()
                } else {
                    self.memory_config.max_seq_len.to_string()
                }
            ),
        ];

        if gpu_total > 0 {
            fields.push(format!("gpu_used={}", format_bytes_engine(gpu_used)));
            fields.push(format!("gpu_total={}", format_bytes_engine(gpu_total)));
            fields.push(format!(
                "gpu_baseline={}",
                format_bytes_engine(self.memory_config.baseline_gpu_bytes)
            ));
            fields.push(format!(
                "gpu_limit={}",
                if self.memory_config.gpu_memory_limit_bytes == 0 {
                    "unlimited".to_string()
                } else {
                    format_bytes_engine(self.memory_config.gpu_memory_limit_bytes)
                }
            ));
        }

        fields.join(", ")
    }

    fn finish_sequence(&mut self, seq_id: &str) {
        let remaining = self
            .token_streams
            .get(seq_id)
            .and_then(|s| s.decode_rest().ok().flatten())
            .unwrap_or_default();

        if !remaining.is_empty() {
            if let Some(seq) = self.sequences.get(seq_id) {
                let _ = seq.response_tx.send(EngineResponse::Token {
                    text: remaining,
                    token_id: 0,
                });
            }
        }

        if let Some(seq) = self.sequences.get(seq_id) {
            let generated_ids = &seq.tokens[seq.prompt_len..];
            let completion_tokens = seq.num_generated();
            let full_text = self
                .model
                .tokenizer()
                .decode(generated_ids, true)
                .unwrap_or_default();

            let finish_reason = seq.finish_reason().to_string();

            info!(
                id = %seq_id,
                prompt_tokens = seq.prompt_len,
                completion_tokens,
                finish_reason = %finish_reason,
                "Sequence finished",
            );

            let _ = seq.response_tx.send(EngineResponse::Finished {
                full_text,
                prompt_tokens: seq.prompt_len,
                completion_tokens,
                finish_reason,
            });

            self.stats
                .total_completion_tokens
                .fetch_add(completion_tokens as u64, Ordering::Relaxed);
            self.stats
                .completed_requests
                .fetch_add(1, Ordering::Relaxed);
        }

        self.cleanup_sequence(seq_id);
    }

    fn cleanup_sequence(&mut self, seq_id: &str) {
        // Subtract this sequence's KV bytes from the tracked total.
        // If active, bytes are in the model (not in seq.kv_caches).
        let freed = if self.active_seq_id.as_deref() == Some(seq_id) {
            self.model.kv_bytes()
        } else if let Some(seq) = self.sequences.get(seq_id) {
            sequence::kv_cache_bytes(&seq.kv_caches)
        } else {
            0
        };
        self.kv_manager
            .set_tracked_kv_bytes(self.kv_manager.tracked_kv_bytes().saturating_sub(freed));

        debug!(
            id = %seq_id,
            freed_bytes = %format_bytes_engine(freed),
            tracked_kv_after = %format_bytes_engine(self.kv_manager.tracked_kv_bytes()),
            "Cleaning up sequence resources"
        );

        self.sequences.remove(seq_id);
        self.request_states.remove(seq_id);
        self.token_streams.remove(seq_id);
        self.scheduler.remove(seq_id);

        if self.active_seq_id.as_deref() == Some(seq_id) {
            self.active_seq_id = None;
        }
        self.model.clear_kv_cache();

        // Only lift the eviction cap when the system has drained all
        // waiting sequences. Under sustained load, keeping the cap prevents
        // repeated eviction-readmit cycles (e.g., cap=6 → finish → admit 7th →
        // evict → cap=6 → repeat). Once the load subsides and all waiting
        // sequences are served, we reset so the next burst can try full
        // concurrency again.
        if self.scheduler.effective_max_running.is_some() && self.scheduler.waiting.is_empty() {
            debug!("Eviction cap lifted (no waiting sequences, load subsided)");
            self.scheduler.effective_max_running = None;
        }

        debug!(id = %seq_id, "Sequence cleaned up");
    }
}

#[cfg(test)]
mod tests {
    use super::{
        estimate_text_prefill_allowed_prompt_len, is_probable_oom,
        should_reject_text_prefill_request, MemoryConfig, TextPrefillAdmissionConfig,
        TextPrefillAdmissionMode, TextPrefillAdmissionRejection, TextPrefillAdmissionSnapshot,
    };
    use crate::engine::runtime::{LayerKvCaches, RuntimeModel, RuntimeStepOutput};
    use crate::engine::sequence::{Sequence, SequenceStatus};
    use crate::engine::types::EngineResponse;
    use crate::engine::types::MultimodalInputs;
    use candle_core::{DType, Device, Tensor};
    use candle_transformers::generation::LogitsProcessor;
    use tokenizers::models::bpe::BPE;
    use tokio::sync::mpsc;

    struct MockRuntimeModel {
        device: Device,
        tokenizer: tokenizers::Tokenizer,
        extract_err: Option<String>,
        restore_err: Option<String>,
    }

    impl MockRuntimeModel {
        fn new(extract_err: Option<&str>, restore_err: Option<&str>) -> Self {
            Self {
                device: Device::Cpu,
                tokenizer: tokenizers::Tokenizer::new(BPE::default()),
                extract_err: extract_err.map(str::to_string),
                restore_err: restore_err.map(str::to_string),
            }
        }
    }

    impl RuntimeModel for MockRuntimeModel {
        fn prefill(
            &mut self,
            _ctx: crate::engine::runtime::RuntimeRequestContext,
        ) -> anyhow::Result<RuntimeStepOutput> {
            panic!("not used in swap_in tests")
        }

        fn decode(
            &mut self,
            _ctx: crate::engine::runtime::RuntimeStepContext,
        ) -> anyhow::Result<RuntimeStepOutput> {
            panic!("not used in swap_in tests")
        }

        fn batch_decode(
            &mut self,
            _ctx: crate::engine::runtime::BatchDecodeContext<'_>,
        ) -> candle_core::Result<Tensor> {
            panic!("not used in swap_in tests")
        }

        fn clear_kv_cache(&mut self) {}
        fn num_layers(&self) -> usize {
            1
        }
        fn device(&self) -> &Device {
            &self.device
        }
        fn dtype(&self) -> DType {
            DType::BF16
        }
        fn tokenizer(&self) -> &tokenizers::Tokenizer {
            &self.tokenizer
        }
        fn eos_token_id(&self) -> Vec<u32> {
            vec![0]
        }
        fn warmup(&mut self) {}
        fn supports_kv_swap(&self) -> bool {
            true
        }
        fn kv_extract(&self) -> anyhow::Result<LayerKvCaches> {
            match &self.extract_err {
                Some(err) => Err(anyhow::anyhow!(err.clone())),
                None => Ok(vec![None]),
            }
        }
        fn kv_restore(&mut self, _caches: LayerKvCaches) -> anyhow::Result<()> {
            match &self.restore_err {
                Some(err) => Err(anyhow::anyhow!(err.clone())),
                None => Ok(()),
            }
        }
        fn kv_bytes(&self) -> u64 {
            0
        }
    }

    fn make_test_sequence(id: &str) -> (Sequence, mpsc::UnboundedReceiver<EngineResponse>) {
        let (response_tx, response_rx) = mpsc::unbounded_channel();
        (
            Sequence {
                id: id.to_string(),
                status: SequenceStatus::Running,
                tokens: vec![1, 2],
                multimodal_inputs: MultimodalInputs::default(),
                prompt_len: 1,
                kv_caches: vec![None],
                logits_processor: LogitsProcessor::new(42, Some(0.8), Some(0.95)),
                temperature: Some(0.8),
                top_p: Some(0.95),
                top_k: Some(40),
                max_tokens: 16,
                eos_token_id: vec![0],
                repetition_penalty: 1.0,
                repeat_last_n: 64,
                response_tx,
            },
            response_rx,
        )
    }

    #[test]
    fn parse_text_prefill_token_limit_env_accepts_positive_integer() {
        assert_eq!(
            MemoryConfig::parse_text_prefill_token_limit_env(Some("4096".to_string())),
            Some(4096)
        );
    }

    #[test]
    fn parse_text_prefill_token_limit_env_rejects_zero_and_invalid_values() {
        assert_eq!(
            MemoryConfig::parse_text_prefill_token_limit_env(Some("0".to_string())),
            None
        );
        assert_eq!(
            MemoryConfig::parse_text_prefill_token_limit_env(Some("abc".to_string())),
            None
        );
    }

    #[test]
    fn text_prefill_hard_cap_only_applies_to_unlimited_text_only_requests() {
        let memory_config = MemoryConfig {
            max_seq_len: 0,
            text_prefill_token_limit: Some(2048),
            text_prefill_admission: TextPrefillAdmissionConfig {
                mode: TextPrefillAdmissionMode::Off,
                reserve_bytes: 0,
                bytes_per_token: 1,
                running_penalty_tokens: 0,
                waiting_penalty_tokens: 0,
            },
            gpu_memory_limit_bytes: 0,
            baseline_gpu_bytes: 0,
        };
        let snapshot = TextPrefillAdmissionSnapshot {
            prompt_len: 4096,
            gpu_used_bytes: 0,
            gpu_total_bytes: 0,
            tracked_kv_bytes: 0,
            running_sequences: 0,
            waiting_sequences: 0,
        };

        assert_eq!(
            should_reject_text_prefill_request(
                &memory_config,
                &snapshot,
                &MultimodalInputs::default()
            ),
            Some(TextPrefillAdmissionRejection::HardCap { limit: 2048 })
        );

        assert_eq!(
            should_reject_text_prefill_request(
                &MemoryConfig {
                    max_seq_len: 8192,
                    ..memory_config.clone()
                },
                &snapshot,
                &MultimodalInputs::default(),
            ),
            None
        );

        assert_eq!(
            should_reject_text_prefill_request(
                &memory_config,
                &snapshot,
                &MultimodalInputs {
                    image_urls: vec!["https://example.test/image.png".to_string()],
                    audio_urls: vec![],
                },
            ),
            None
        );
    }

    #[test]
    fn dynamic_text_prefill_estimator_reduces_budget_with_headroom_and_load() {
        let memory_config = MemoryConfig {
            max_seq_len: 0,
            text_prefill_token_limit: None,
            text_prefill_admission: TextPrefillAdmissionConfig {
                mode: TextPrefillAdmissionMode::Dynamic,
                reserve_bytes: 2_000,
                bytes_per_token: 10,
                running_penalty_tokens: 200,
                waiting_penalty_tokens: 100,
            },
            gpu_memory_limit_bytes: 20_000,
            baseline_gpu_bytes: 8_000,
        };
        let snapshot = TextPrefillAdmissionSnapshot {
            prompt_len: 900,
            gpu_used_bytes: 12_500,
            gpu_total_bytes: 24_000,
            tracked_kv_bytes: 500,
            running_sequences: 2,
            waiting_sequences: 1,
        };

        let estimate = estimate_text_prefill_allowed_prompt_len(&memory_config, &snapshot).unwrap();
        assert_eq!(estimate.headroom_budget_bytes, 12_000);
        assert_eq!(estimate.accounted_growth_bytes, 4_500);
        assert_eq!(estimate.available_growth_bytes, 5_500);
        assert_eq!(estimate.running_penalty_tokens, 400);
        assert_eq!(estimate.waiting_penalty_tokens, 100);
        assert_eq!(estimate.allowed_prompt_len_now, 50);
        assert_eq!(estimate.reason, "gpu_headroom_and_queue_load");
    }

    #[test]
    fn dynamic_text_prefill_rejects_when_prompt_exceeds_estimated_budget() {
        let memory_config = MemoryConfig {
            max_seq_len: 0,
            text_prefill_token_limit: None,
            text_prefill_admission: TextPrefillAdmissionConfig {
                mode: TextPrefillAdmissionMode::Dynamic,
                reserve_bytes: 2_000,
                bytes_per_token: 10,
                running_penalty_tokens: 200,
                waiting_penalty_tokens: 100,
            },
            gpu_memory_limit_bytes: 20_000,
            baseline_gpu_bytes: 8_000,
        };
        let snapshot = TextPrefillAdmissionSnapshot {
            prompt_len: 900,
            gpu_used_bytes: 12_500,
            gpu_total_bytes: 24_000,
            tracked_kv_bytes: 500,
            running_sequences: 2,
            waiting_sequences: 1,
        };

        match should_reject_text_prefill_request(
            &memory_config,
            &snapshot,
            &MultimodalInputs::default(),
        ) {
            Some(TextPrefillAdmissionRejection::Dynamic { estimate }) => {
                assert_eq!(estimate.allowed_prompt_len_now, 50);
                assert_eq!(estimate.reason, "gpu_headroom_and_queue_load");
            }
            other => panic!("expected dynamic rejection, got {other:?}"),
        }
    }

    #[test]
    fn dynamic_text_prefill_skips_when_gpu_budget_is_not_observable() {
        let memory_config = MemoryConfig {
            max_seq_len: 0,
            text_prefill_token_limit: None,
            text_prefill_admission: TextPrefillAdmissionConfig {
                mode: TextPrefillAdmissionMode::Dynamic,
                reserve_bytes: 2_000,
                bytes_per_token: 10,
                running_penalty_tokens: 0,
                waiting_penalty_tokens: 0,
            },
            gpu_memory_limit_bytes: 0,
            baseline_gpu_bytes: 8_000,
        };
        let snapshot = TextPrefillAdmissionSnapshot {
            prompt_len: 9_000,
            gpu_used_bytes: 0,
            gpu_total_bytes: 0,
            tracked_kv_bytes: 0,
            running_sequences: 0,
            waiting_sequences: 0,
        };

        assert_eq!(
            estimate_text_prefill_allowed_prompt_len(&memory_config, &snapshot),
            None
        );
        assert_eq!(
            should_reject_text_prefill_request(
                &memory_config,
                &snapshot,
                &MultimodalInputs::default()
            ),
            None
        );
    }

    #[test]
    fn detects_probable_oom_messages() {
        assert!(is_probable_oom(
            "DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")"
        ));
        assert!(is_probable_oom(
            "CUDA out of memory while allocating tensor"
        ));
        assert!(!is_probable_oom(
            "dtype mismatch in mul, lhs: BF16, rhs: F32"
        ));
    }

    #[test]
    fn swap_in_restore_failure_sends_error_and_cleans_sequence() {
        let model = Box::new(MockRuntimeModel::new(
            None,
            Some("corrupt int8_rowwise_kv payload"),
        ));
        let (mut engine, _handle) = super::InferenceEngine::new(
            model,
            1,
            1,
            super::PlacementPolicy::KeepOnDevice,
            MemoryConfig::parse(0, None, &Device::Cpu),
        );
        let (seq, mut rx) = make_test_sequence("seq-restore");
        engine.sequences.insert(seq.id.clone(), seq);

        assert!(!engine.swap_in("seq-restore"));
        assert!(engine.active_seq_id.is_none());
        assert!(!engine.sequences.contains_key("seq-restore"));
        match rx.try_recv().unwrap() {
            EngineResponse::Error(message) => {
                assert!(message.contains("KV restore failed"));
                assert!(message.contains("corrupt int8_rowwise_kv payload"));
            }
            other => panic!("expected error response, got {other:?}"),
        }
    }

    #[test]
    fn swap_in_export_failure_only_fails_previous_sequence() {
        let model = Box::new(MockRuntimeModel::new(Some("export decode failure"), None));
        let (mut engine, _handle) = super::InferenceEngine::new(
            model,
            1,
            1,
            super::PlacementPolicy::KeepOnDevice,
            MemoryConfig::parse(0, None, &Device::Cpu),
        );
        let (prev_seq, mut prev_rx) = make_test_sequence("seq-prev");
        let (next_seq, mut next_rx) = make_test_sequence("seq-next");
        engine.sequences.insert(prev_seq.id.clone(), prev_seq);
        engine.sequences.insert(next_seq.id.clone(), next_seq);
        engine.active_seq_id = Some("seq-prev".to_string());

        assert!(engine.swap_in("seq-next"));
        assert_eq!(engine.active_seq_id.as_deref(), Some("seq-next"));
        assert!(!engine.sequences.contains_key("seq-prev"));
        assert!(engine.sequences.contains_key("seq-next"));
        match prev_rx.try_recv().unwrap() {
            EngineResponse::Error(message) => {
                assert!(message.contains("KV export failed"));
                assert!(message.contains("export decode failure"));
            }
            other => panic!("expected error response, got {other:?}"),
        }
        assert!(next_rx.try_recv().is_err());
    }
}
