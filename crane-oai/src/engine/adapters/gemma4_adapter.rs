use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::sync::Arc;
use tracing::{info, warn};

use crane_core::models::gemma4::modeling::DecodePrefixKvSource;

use crate::engine::runtime::{
    make_kv_backend, BatchDecodeContext, DenseLayerKv, KvBackendConfig, KvCacheBackend,
    KvCacheMode, KvLayerEnvelope, LayerKvCaches, RuntimeModel, RuntimeRequestContext,
    RuntimeStateDelta, RuntimeStepContext, RuntimeStepOutput, SequenceKvCaches,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TurboQuantPathReason {
    SupportedNarrowPath,
    MixedLayerRestoreUnsupported,
    MissingStoredKv,
    BackendNoCompressedKScores,
    BackendBatchDecodeUnsupported,
    SharedKvUnsupported,
    SlidingWindowUnsupported,
    BatchSizeUnsupported,
    QueryLenUnsupported,
    KvGroupsUnsupported,
    HeadLayoutUnsupported,
}

impl TurboQuantPathReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::SupportedNarrowPath => "supported_narrow_path",
            Self::MixedLayerRestoreUnsupported => "mixed_layer_restore_unsupported",
            Self::MissingStoredKv => "missing_stored_kv",
            Self::BackendNoCompressedKScores => "backend_no_compressed_k_scores",
            Self::BackendBatchDecodeUnsupported => "backend_batch_decode_unsupported",
            Self::SharedKvUnsupported => "shared_kv_unsupported",
            Self::SlidingWindowUnsupported => "sliding_window_unsupported",
            Self::BatchSizeUnsupported => "batch_size_unsupported",
            Self::QueryLenUnsupported => "query_len_unsupported",
            Self::KvGroupsUnsupported => "kv_groups_unsupported",
            Self::HeadLayoutUnsupported => "head_layout_unsupported",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TurboQuantLayerDecision {
    use_decode_prefix: bool,
    reason: TurboQuantPathReason,
}

impl TurboQuantLayerDecision {
    fn supported() -> Self {
        Self {
            use_decode_prefix: true,
            reason: TurboQuantPathReason::SupportedNarrowPath,
        }
    }

    fn fallback(reason: TurboQuantPathReason) -> Self {
        Self {
            use_decode_prefix: false,
            reason,
        }
    }

    fn decision_label(self) -> &'static str {
        if self.use_decode_prefix {
            "turboquant_narrow_path"
        } else {
            "dense_fallback"
        }
    }
}

fn classify_turboquant_restore_layer(
    backend_supports_compressed_k_scores: bool,
    has_shared_kv_layers: bool,
    layer_uses_sliding_window: bool,
    has_stored_kv: bool,
) -> TurboQuantLayerDecision {
    if !has_stored_kv {
        return TurboQuantLayerDecision::fallback(TurboQuantPathReason::MissingStoredKv);
    }
    if !backend_supports_compressed_k_scores {
        return TurboQuantLayerDecision::fallback(TurboQuantPathReason::BackendNoCompressedKScores);
    }
    if has_shared_kv_layers {
        return TurboQuantLayerDecision::fallback(TurboQuantPathReason::SharedKvUnsupported);
    }
    if layer_uses_sliding_window {
        return TurboQuantLayerDecision::fallback(TurboQuantPathReason::SlidingWindowUnsupported);
    }
    TurboQuantLayerDecision::supported()
}

fn apply_turboquant_restore_support_boundary(
    decisions: &[TurboQuantLayerDecision],
) -> Vec<TurboQuantLayerDecision> {
    let has_unsupported_stored_layer = decisions.iter().any(|decision| {
        !decision.use_decode_prefix && decision.reason != TurboQuantPathReason::MissingStoredKv
    });
    if !has_unsupported_stored_layer {
        return decisions.to_vec();
    }

    decisions
        .iter()
        .map(|decision| {
            if decision.use_decode_prefix {
                TurboQuantLayerDecision::fallback(
                    TurboQuantPathReason::MixedLayerRestoreUnsupported,
                )
            } else {
                *decision
            }
        })
        .collect()
}

fn classify_turboquant_decode_shape(
    batch: usize,
    q_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
) -> std::result::Result<(), TurboQuantPathReason> {
    if batch != 1 {
        return Err(TurboQuantPathReason::BatchSizeUnsupported);
    }
    if q_len == 0 {
        return Err(TurboQuantPathReason::QueryLenUnsupported);
    }
    if num_kv_groups == 0 {
        return Err(TurboQuantPathReason::KvGroupsUnsupported);
    }
    if num_heads != num_kv_heads * num_kv_groups {
        return Err(TurboQuantPathReason::HeadLayoutUnsupported);
    }
    Ok(())
}

fn classify_turboquant_batch_decode_support(
    backend_mode: KvCacheMode,
    has_shared_kv_layers: bool,
    has_sliding_window_layers: bool,
) -> std::result::Result<(), TurboQuantPathReason> {
    if backend_mode != KvCacheMode::TurboQuant {
        return Err(TurboQuantPathReason::BackendBatchDecodeUnsupported);
    }
    if has_shared_kv_layers {
        return Err(TurboQuantPathReason::SharedKvUnsupported);
    }
    if has_sliding_window_layers {
        return Err(TurboQuantPathReason::SlidingWindowUnsupported);
    }
    Ok(())
}

struct Gemma4TurboQuantDecodePrefix {
    backend: Arc<dyn KvCacheBackend>,
    caches: Vec<Option<KvLayerEnvelope>>,
    enabled_layers: Vec<bool>,
}

impl std::fmt::Debug for Gemma4TurboQuantDecodePrefix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gemma4TurboQuantDecodePrefix")
            .field("cache_count", &self.caches.len())
            .field("enabled_layers", &self.enabled_layers)
            .finish()
    }
}

impl DecodePrefixKvSource for Gemma4TurboQuantDecodePrefix {
    fn score_prefix_keys(
        &self,
        layer_idx: usize,
        query_states: &Tensor,
        num_kv_heads: usize,
        num_kv_groups: usize,
    ) -> candle_core::Result<Option<Tensor>> {
        if !self.enabled_layers.get(layer_idx).copied().unwrap_or(false) {
            return Ok(None);
        }

        let Some(stored) = self
            .caches
            .get(layer_idx)
            .and_then(|stored| stored.as_ref())
        else {
            return Ok(None);
        };

        let (batch, num_heads, q_len, head_dim) = query_states.dims4()?;
        if let Err(reason) =
            classify_turboquant_decode_shape(batch, q_len, num_heads, num_kv_heads, num_kv_groups)
        {
            warn!(
                event = "gemma4_turboquant_decode_prefix_fallback",
                decision = "dense_fallback",
                reason = reason.as_str(),
                backend = self.backend.backend_id(),
                layer_idx,
                batch,
                q_len,
                num_heads,
                num_kv_heads,
                num_kv_groups,
                "Gemma4 TurboQuant decode-prefix fell back to dense attention"
            );
            return Ok(None);
        }

        let query_rows = query_states
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .reshape((num_heads, q_len, head_dim))?
            .to_vec3::<f32>()?;
        let mut flat_query_rows = Vec::with_capacity(num_heads * q_len * head_dim);
        for head_rows in &query_rows {
            for row in head_rows {
                flat_query_rows.extend_from_slice(row);
            }
        }

        let grouped_query =
            Tensor::from_vec(flat_query_rows, (num_heads * q_len, head_dim), &Device::Cpu)?;
        let Some(base_scores) = self
            .backend
            .score_query_against_stored_keys(&grouped_query, stored)
            .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        else {
            return Ok(None);
        };

        let prefix_len = stored.seq_len;
        let base_scores = base_scores.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let base_scores = base_scores.to_vec2::<f32>()?;
        let mut per_head_scores = Vec::with_capacity(num_heads * q_len * prefix_len);
        for head_idx in 0..num_heads {
            let kv_head_idx = head_idx / num_kv_groups;
            let start = kv_head_idx * prefix_len;
            let end = start + prefix_len;
            for q_idx in 0..q_len {
                let query_row_idx = head_idx * q_len + q_idx;
                per_head_scores.extend_from_slice(&base_scores[query_row_idx][start..end]);
            }
        }
        Ok(Some(Tensor::from_vec(
            per_head_scores,
            (1, num_heads, q_len, prefix_len),
            &Device::Cpu,
        )?))
    }

    fn value_prefix(
        &self,
        layer_idx: usize,
        target_device: &Device,
        target_dtype: DType,
    ) -> candle_core::Result<Option<Tensor>> {
        if !self.enabled_layers.get(layer_idx).copied().unwrap_or(false) {
            return Ok(None);
        }

        let Some(stored) = self.caches.get(layer_idx).cloned().flatten() else {
            return Ok(None);
        };
        let Some((_, value)) = self
            .backend
            .import_layer(layer_idx, Some(stored), target_device, target_dtype)
            .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        else {
            return Ok(None);
        };
        Ok(Some(value))
    }

    fn weighted_value_prefix(
        &self,
        layer_idx: usize,
        attn_weights: &Tensor,
        num_kv_heads: usize,
        num_kv_groups: usize,
        target_device: &Device,
        target_dtype: DType,
    ) -> candle_core::Result<Option<Tensor>> {
        if !self.enabled_layers.get(layer_idx).copied().unwrap_or(false) {
            return Ok(None);
        }

        let Some(stored) = self
            .caches
            .get(layer_idx)
            .and_then(|stored| stored.as_ref())
        else {
            return Ok(None);
        };

        self.backend
            .weighted_value_prefix(
                attn_weights,
                stored,
                num_kv_heads,
                num_kv_groups,
                target_device,
                target_dtype,
            )
            .map_err(|err| candle_core::Error::Msg(err.to_string()))
    }
}

pub struct Gemma4RuntimeAdapter {
    model: crane_core::models::gemma4::Model,
    kv_backend: Arc<dyn KvCacheBackend>,
    decode_prefix: Option<Arc<Gemma4TurboQuantDecodePrefix>>,
}

impl Gemma4RuntimeAdapter {
    pub fn new(
        model_path: &str,
        device: &Device,
        dtype: &DType,
        kv_config: KvBackendConfig,
    ) -> Result<Self> {
        let load_dtype = match device {
            Device::Cpu => *dtype,
            _ => DType::BF16,
        };
        let model = crane_core::models::gemma4::Model::new(model_path, device, &load_dtype)?;
        let kv_backend: Arc<dyn KvCacheBackend> = Arc::from(make_kv_backend(kv_config)?);
        Ok(Self {
            model,
            kv_backend,
            decode_prefix: None,
        })
    }

    fn clear_decode_prefix(&mut self) {
        self.decode_prefix = None;
        self.model.set_decode_prefix_kv_source(None);
    }

    fn maybe_enable_decode_prefix(
        &mut self,
        caches: &[Option<KvLayerEnvelope>],
        enabled_layers: Vec<bool>,
    ) {
        if !enabled_layers.iter().any(|enabled| *enabled) {
            self.clear_decode_prefix();
            return;
        }

        let prefix = Arc::new(Gemma4TurboQuantDecodePrefix {
            backend: Arc::clone(&self.kv_backend),
            caches: caches.to_vec(),
            enabled_layers,
        });
        self.model.set_decode_prefix_kv_source(Some(prefix.clone()));
        self.decode_prefix = Some(prefix);
    }

    fn turboquant_restore_layer_decision(
        &self,
        layer_idx: usize,
        stored: &Option<KvLayerEnvelope>,
    ) -> TurboQuantLayerDecision {
        classify_turboquant_restore_layer(
            self.kv_backend.supports_compressed_k_scores(),
            self.model.has_shared_kv_layers(),
            self.model.layer_uses_sliding_window(layer_idx),
            stored.is_some(),
        )
    }

    fn log_turboquant_restore_decisions(&self, decisions: &[TurboQuantLayerDecision]) {
        let stored_layers = decisions
            .iter()
            .filter(|decision| decision.reason != TurboQuantPathReason::MissingStoredKv)
            .count();
        let active_layers = decisions
            .iter()
            .filter(|decision| decision.use_decode_prefix)
            .count();
        let fallback_layers = stored_layers.saturating_sub(active_layers);
        info!(
            event = "gemma4_turboquant_restore_summary",
            backend = self.kv_backend.backend_id(),
            decision = if active_layers > 0 {
                "turboquant_narrow_path_active"
            } else {
                "dense_fallback_only"
            },
            reason = if active_layers > 0 {
                "supported_layers_present"
            } else if stored_layers == 0 {
                "missing_stored_kv"
            } else {
                "no_supported_layers"
            },
            active_layers,
            fallback_layers,
            stored_layers,
            total_layers = decisions.len(),
            shared_kv = self.model.has_shared_kv_layers(),
            "Gemma4 TurboQuant restore decisions evaluated"
        );

        for (layer_idx, decision) in decisions.iter().enumerate() {
            if decision.reason == TurboQuantPathReason::MissingStoredKv {
                continue;
            }
            info!(
                event = "gemma4_turboquant_restore_layer",
                backend = self.kv_backend.backend_id(),
                layer_idx,
                decision = decision.decision_label(),
                reason = decision.reason.as_str(),
                sliding_window = self.model.layer_uses_sliding_window(layer_idx),
                shared_kv = self.model.has_shared_kv_layers(),
                "Gemma4 TurboQuant restore layer decision"
            );
        }
    }

    fn has_sliding_window_layers(&self) -> bool {
        (0..self.model.num_layers())
            .any(|layer_idx| self.model.layer_uses_sliding_window(layer_idx))
    }

    fn batch_decode_support_reason(&self) -> std::result::Result<(), TurboQuantPathReason> {
        classify_turboquant_batch_decode_support(
            self.kv_backend.mode(),
            self.model.has_shared_kv_layers(),
            self.has_sliding_window_layers(),
        )
    }

    fn ensure_batch_decode_supported(&self) -> candle_core::Result<()> {
        if let Err(reason) = self.batch_decode_support_reason() {
            candle_core::bail!(
                "gemma4_batch_decode_unsupported backend={} reason={} decision=reject",
                self.kv_backend.backend_id(),
                reason.as_str()
            )
        }
        Ok(())
    }

    fn import_batch_decode_dense_caches(
        &self,
        seq_kv_caches: &[LayerKvCaches],
    ) -> candle_core::Result<Vec<Vec<Option<(Tensor, Tensor)>>>> {
        seq_kv_caches
            .iter()
            .map(|layers| {
                layers
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(layer_idx, stored)| {
                        self.kv_backend
                            .import_layer(layer_idx, stored, self.device(), self.dtype())
                            .map_err(|err| candle_core::Error::Msg(err.to_string()))
                    })
                    .collect::<candle_core::Result<Vec<_>>>()
            })
            .collect::<candle_core::Result<Vec<_>>>()
    }

    fn merged_extract_layer(
        &self,
        layer_idx: usize,
        dense_local: DenseLayerKv,
    ) -> Result<Option<KvLayerEnvelope>> {
        let Some(prefix) = self.decode_prefix.as_ref() else {
            return self.kv_backend.export_layer(layer_idx, dense_local);
        };
        let Some(stored_prefix) = prefix.caches.get(layer_idx).cloned().flatten() else {
            return self.kv_backend.export_layer(layer_idx, dense_local);
        };
        if !prefix
            .enabled_layers
            .get(layer_idx)
            .copied()
            .unwrap_or(false)
        {
            return self.kv_backend.export_layer(layer_idx, dense_local);
        }

        let prefix_dense = self.kv_backend.import_layer(
            layer_idx,
            Some(stored_prefix),
            self.device(),
            self.dtype(),
        )?;
        let merged_dense = match (prefix_dense, dense_local) {
            (Some((prefix_k, prefix_v)), Some((local_k, local_v))) => Some((
                Tensor::cat(&[&prefix_k, &local_k], 2)?,
                Tensor::cat(&[&prefix_v, &local_v], 2)?,
            )),
            (Some(prefix), None) => Some(prefix),
            (None, local) => local,
        };
        self.kv_backend.export_layer(layer_idx, merged_dense)
    }
}

impl RuntimeModel for Gemma4RuntimeAdapter {
    fn prefill(&mut self, ctx: RuntimeRequestContext) -> Result<RuntimeStepOutput> {
        self.clear_decode_prefix();
        let logits = self.model.forward_step_with_multimodal(
            &ctx.input_ids,
            ctx.start_pos,
            &ctx.multimodal_inputs.image_urls,
            &ctx.multimodal_inputs.audio_urls,
        )?;
        Ok(RuntimeStepOutput {
            logits,
            state_delta: RuntimeStateDelta {
                consumed_tokens: ctx.input_ids.len(),
            },
        })
    }

    fn decode(&mut self, ctx: RuntimeStepContext) -> Result<RuntimeStepOutput> {
        let logits = self.model.forward_step(&ctx.input_ids, ctx.start_pos)?;
        Ok(RuntimeStepOutput {
            logits,
            state_delta: RuntimeStateDelta {
                consumed_tokens: ctx.input_ids.len(),
            },
        })
    }

    fn batch_decode(&mut self, ctx: BatchDecodeContext<'_>) -> candle_core::Result<Tensor> {
        self.ensure_batch_decode_supported()?;
        self.clear_decode_prefix();

        let (batch_size, q_len) = ctx.input_ids.dims2()?;
        if q_len != 1 {
            candle_core::bail!(
                "gemma4_batch_decode_unsupported backend={} reason={} decision=reject batch={} q_len={}",
                self.kv_backend.backend_id(),
                TurboQuantPathReason::QueryLenUnsupported.as_str(),
                batch_size,
                q_len
            )
        }
        if ctx.positions.len() != batch_size {
            candle_core::bail!(
                "gemma4_batch_decode_unsupported backend={} reason=positions_len_mismatch decision=reject batch={} positions_len={}",
                self.kv_backend.backend_id(),
                batch_size,
                ctx.positions.len()
            )
        }

        self.model.step_batch_decode_with_input_ids(
            ctx.input_ids,
            ctx.positions,
            ctx.attention_mask,
            ctx.batch_kv_info,
        )
    }

    fn clear_kv_cache(&mut self) {
        self.clear_decode_prefix();
        self.model.clear_kv_cache();
    }

    fn num_layers(&self) -> usize {
        self.model.num_layers()
    }

    fn device(&self) -> &Device {
        &self.model.device
    }

    fn dtype(&self) -> DType {
        self.model.dtype
    }

    fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.model.tokenizer.tokenizer
    }

    fn eos_token_id(&self) -> Vec<u32> {
        self.model.eos_token_ids.clone()
    }

    fn warmup(&mut self) {
        self.model.warmup();
    }

    fn supports_kv_swap(&self) -> bool {
        true
    }

    fn kv_extract(&self) -> Result<LayerKvCaches> {
        self.model
            .get_kv_caches()
            .into_iter()
            .enumerate()
            .map(|(layer_idx, dense)| {
                self.merged_extract_layer(layer_idx, dense).map_err(|err| {
                    anyhow::anyhow!(
                        "Gemma4 KV export failed for layer {} with backend '{}': {err}",
                        layer_idx,
                        self.kv_backend.backend_id()
                    )
                })
            })
            .collect()
    }

    fn kv_restore(&mut self, caches: LayerKvCaches) -> Result<()> {
        let raw_layer_decisions: Vec<_> = caches
            .iter()
            .enumerate()
            .map(|(layer_idx, stored)| self.turboquant_restore_layer_decision(layer_idx, stored))
            .collect();
        let layer_decisions = apply_turboquant_restore_support_boundary(&raw_layer_decisions);
        let enabled_layers: Vec<bool> = layer_decisions
            .iter()
            .map(|decision| decision.use_decode_prefix)
            .collect();
        let dense = caches
            .iter()
            .enumerate()
            .map(|(layer_idx, stored)| {
                let use_decode_prefix = layer_decisions[layer_idx].use_decode_prefix;
                let dense = if use_decode_prefix {
                    Ok(None)
                } else {
                    self.kv_backend.import_layer(
                        layer_idx,
                        stored.clone(),
                        self.device(),
                        self.dtype(),
                    )
                };
                dense.map_err(|err| {
                    anyhow::anyhow!(
                        "Gemma4 KV restore failed for layer {} with backend '{}': {err}",
                        layer_idx,
                        self.kv_backend.backend_id()
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.model.set_kv_caches(dense);
        self.log_turboquant_restore_decisions(&layer_decisions);
        self.maybe_enable_decode_prefix(&caches, enabled_layers);
        Ok(())
    }

    fn kv_bytes(&self) -> u64 {
        let prefix_bytes = self
            .decode_prefix
            .as_ref()
            .map(|prefix| {
                prefix
                    .caches
                    .iter()
                    .map(|stored| self.kv_backend.stored_bytes(stored))
                    .sum::<u64>()
            })
            .unwrap_or(0);
        self.model.active_kv_cache_bytes() + prefix_bytes
    }

    fn offload_under_memory_pressure(&mut self) -> usize {
        self.model.offload_experts_to_cpu()
    }

    fn supports_batch_decode(&self) -> bool {
        self.batch_decode_support_reason().is_ok()
    }

    fn setup_batch_decode(
        &mut self,
        seq_kv_caches: &[LayerKvCaches],
        extra_room: usize,
    ) -> candle_core::Result<(Vec<usize>, usize)> {
        self.ensure_batch_decode_supported()?;
        self.clear_decode_prefix();
        let dense_caches = self.import_batch_decode_dense_caches(seq_kv_caches)?;
        self.model.setup_batch_decode(&dense_caches, extra_room)
    }

    fn extract_batch_kv(
        &mut self,
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
    ) -> candle_core::Result<SequenceKvCaches> {
        self.ensure_batch_decode_supported()?;
        self.model
            .extract_batch_kv(kv_lens, original_max_kv, rounds_done)?
            .into_iter()
            .map(|layers| {
                layers
                    .into_iter()
                    .enumerate()
                    .map(|(layer_idx, dense)| {
                        self.kv_backend
                            .export_layer(layer_idx, dense)
                            .map_err(|err| candle_core::Error::Msg(err.to_string()))
                    })
                    .collect::<candle_core::Result<Vec<_>>>()
            })
            .collect()
    }

    fn build_batch_decode_mask(
        &self,
        kv_lens: &[usize],
        original_max_kv: usize,
        max_total_width: usize,
    ) -> candle_core::Result<Option<Tensor>> {
        self.ensure_batch_decode_supported()?;
        crane_core::models::gemma4::modeling::build_batch_decode_mask(
            kv_lens,
            original_max_kv,
            max_total_width,
            self.device(),
            self.dtype(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::runtime::kv_backend::{KvLayerPayload, TurboQuantValuePayload};
    use crate::engine::runtime::KvCacheMode;
    use crate::engine::types::MultimodalInputs;
    use candle_nn::{Activation, VarBuilder};
    use crane_core::models::gemma4::modeling::{Config, Gemma4TextModel};
    use crane_core::models::gemma4::Model as Gemma4Model;
    use safetensors::{serialize_to_file, Dtype as SafeDtype, View};
    use std::borrow::Cow;
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;
    use tempfile::TempDir;

    fn dense_scores(query_rows: &[Vec<f32>], key_rows: &[Vec<f32>]) -> Vec<Vec<f32>> {
        query_rows
            .iter()
            .map(|query| {
                key_rows
                    .iter()
                    .map(|key| {
                        query
                            .iter()
                            .zip(key.iter())
                            .map(|(lhs, rhs)| lhs * rhs)
                            .sum::<f32>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn turboquant_prefix_fixture() -> (Gemma4TurboQuantDecodePrefix, Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let backend: Arc<dyn KvCacheBackend> = Arc::from(
            make_kv_backend(KvBackendConfig {
                mode: KvCacheMode::TurboQuant,
            })
            .expect("turboquant backend"),
        );
        let key_rows = vec![
            vec![-0.78_f32, -0.82, 0.75, 0.79, 0.11, -0.08, 0.09, -0.12],
            vec![0.84_f32, 0.81, -0.77, -0.74, -0.09, 0.07, -0.10, 0.08],
        ];
        let value_rows = vec![
            vec![0.1_f32, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4],
            vec![0.5_f32, 0.6, 0.7, 0.8, -0.5, -0.6, -0.7, -0.8],
        ];
        let key = Tensor::from_vec(key_rows.concat(), (1, 1, 2, 8), &Device::Cpu).expect("key");
        let value =
            Tensor::from_vec(value_rows.concat(), (1, 1, 2, 8), &Device::Cpu).expect("value");
        let stored = backend
            .export_layer(0, Some((key, value)))
            .expect("export layer")
            .expect("stored layer");

        (
            Gemma4TurboQuantDecodePrefix {
                backend,
                caches: vec![Some(stored)],
                enabled_layers: vec![true],
            },
            key_rows,
            value_rows,
        )
    }

    #[derive(Debug)]
    struct CountingDecodePrefixKvSource {
        inner: Arc<Gemma4TurboQuantDecodePrefix>,
        score_calls_by_layer: Arc<Mutex<Vec<usize>>>,
        weighted_value_calls_by_layer: Arc<Mutex<Vec<usize>>>,
        dense_value_calls_by_layer: Arc<Mutex<Vec<usize>>>,
    }

    impl DecodePrefixKvSource for CountingDecodePrefixKvSource {
        fn score_prefix_keys(
            &self,
            layer_idx: usize,
            query_states: &Tensor,
            num_kv_heads: usize,
            num_kv_groups: usize,
        ) -> candle_core::Result<Option<Tensor>> {
            let mut score_calls = self
                .score_calls_by_layer
                .lock()
                .expect("score calls mutex poisoned");
            *score_calls
                .get_mut(layer_idx)
                .expect("layer index should exist") += 1;
            drop(score_calls);

            self.inner
                .score_prefix_keys(layer_idx, query_states, num_kv_heads, num_kv_groups)
        }

        fn value_prefix(
            &self,
            layer_idx: usize,
            target_device: &Device,
            target_dtype: DType,
        ) -> candle_core::Result<Option<Tensor>> {
            let mut dense_value_calls = self
                .dense_value_calls_by_layer
                .lock()
                .expect("dense value calls mutex poisoned");
            *dense_value_calls
                .get_mut(layer_idx)
                .expect("layer index should exist") += 1;
            drop(dense_value_calls);

            self.inner
                .value_prefix(layer_idx, target_device, target_dtype)
        }

        fn weighted_value_prefix(
            &self,
            layer_idx: usize,
            attn_weights: &Tensor,
            num_kv_heads: usize,
            num_kv_groups: usize,
            target_device: &Device,
            target_dtype: DType,
        ) -> candle_core::Result<Option<Tensor>> {
            let mut weighted_value_calls = self
                .weighted_value_calls_by_layer
                .lock()
                .expect("weighted value calls mutex poisoned");
            *weighted_value_calls
                .get_mut(layer_idx)
                .expect("layer index should exist") += 1;
            drop(weighted_value_calls);

            self.inner.weighted_value_prefix(
                layer_idx,
                attn_weights,
                num_kv_heads,
                num_kv_groups,
                target_device,
                target_dtype,
            )
        }
    }

    fn tiny_decode_logit_config(num_hidden_layers: usize) -> Config {
        Config {
            attention_bias: false,
            attention_k_eq_v: false,
            head_dim: 2,
            global_head_dim: None,
            hidden_activation: Activation::Silu,
            hidden_size: 4,
            hidden_size_per_layer_input: None,
            intermediate_size: 8,
            num_attention_heads: 2,
            num_hidden_layers,
            num_key_value_heads: 1,
            num_global_key_value_heads: None,
            rms_norm_eps: 1e-6,
            vocab_size: 8,
            max_position_embeddings: 16,
            sliding_window: 8,
            layer_types: vec!["full_attention".to_string(); num_hidden_layers],
            final_logit_softcapping: None,
            rope_parameters: None,
            num_kv_shared_layers: None,
            enable_moe_block: false,
            num_experts: None,
            top_k_experts: None,
            moe_intermediate_size: None,
            expert_intermediate_size: None,
            use_double_wide_mlp: None,
            eos_token_id: None,
        }
    }

    fn tiny_decode_logit_tensor_map(
        num_hidden_layers: usize,
    ) -> candle_core::Result<HashMap<String, Tensor>> {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();

        tensors.insert(
            "embed_tokens.weight".to_string(),
            Tensor::from_vec(
                vec![
                    0.0_f32, 0.0, 0.0, 0.0, // token 0
                    1.0, 0.2, 0.0, 0.0, // token 1
                    -0.6, 0.8, 0.0, 0.0, // token 2
                    0.4, 0.9, 0.0, 0.0, // token 3
                    0.7, -0.3, 0.0, 0.0, // token 4
                    -0.5, -0.4, 0.0, 0.0, // token 5
                    0.1, 0.6, 0.0, 0.0, // token 6
                    -0.2, 0.3, 0.0, 0.0, // token 7
                ],
                (8, 4),
                &device,
            )?,
        );

        let q_proj_weights = [
            vec![
                1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            ],
            vec![
                0.9_f32, 0.1, 0.0, 0.0, -0.1, 1.0, 0.0, 0.0, 1.0, 0.0, 0.1, 0.0, 0.0, 0.8, 0.0, 0.2,
            ],
            vec![
                1.0_f32, 0.0, 0.1, 0.0, 0.0, 0.95, 0.0, 0.1, 0.9, 0.1, 0.0, 0.0, 0.0, 0.85, 0.1,
                0.0,
            ],
        ];
        let kv_proj_weights = [
            vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            vec![0.9_f32, 0.1, 0.0, 0.0, -0.1, 1.0, 0.0, 0.0],
            vec![1.0_f32, 0.0, 0.1, 0.0, 0.0, 0.85, 0.0, 0.15],
        ];
        let o_proj_weights = [
            vec![
                1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            vec![
                0.95_f32, 0.0, 0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.1,
                0.9,
            ],
            vec![
                0.9_f32, 0.1, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.1, 0.0, 0.95, 0.0, 0.0, 0.1, 0.0,
                0.95,
            ],
        ];

        for layer_idx in 0..num_hidden_layers {
            let variant_idx = layer_idx.min(2);
            tensors.insert(
                format!("layers.{layer_idx}.self_attn.q_proj.weight"),
                Tensor::from_vec(q_proj_weights[variant_idx].clone(), (4, 4), &device)?,
            );
            tensors.insert(
                format!("layers.{layer_idx}.self_attn.k_proj.weight"),
                Tensor::from_vec(kv_proj_weights[variant_idx].clone(), (2, 4), &device)?,
            );
            tensors.insert(
                format!("layers.{layer_idx}.self_attn.v_proj.weight"),
                Tensor::from_vec(kv_proj_weights[variant_idx].clone(), (2, 4), &device)?,
            );
            tensors.insert(
                format!("layers.{layer_idx}.self_attn.o_proj.weight"),
                Tensor::from_vec(o_proj_weights[variant_idx].clone(), (4, 4), &device)?,
            );
            tensors.insert(
                format!("layers.{layer_idx}.self_attn.q_norm.weight"),
                Tensor::ones(2, DType::F32, &device)?,
            );
            tensors.insert(
                format!("layers.{layer_idx}.self_attn.k_norm.weight"),
                Tensor::ones(2, DType::F32, &device)?,
            );
            tensors.insert(
                format!("layers.{layer_idx}.mlp.gate_proj.weight"),
                Tensor::zeros((8, 4), DType::F32, &device)?,
            );
            tensors.insert(
                format!("layers.{layer_idx}.mlp.up_proj.weight"),
                Tensor::zeros((8, 4), DType::F32, &device)?,
            );
            tensors.insert(
                format!("layers.{layer_idx}.mlp.down_proj.weight"),
                Tensor::zeros((4, 8), DType::F32, &device)?,
            );
            for name in [
                "input_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "post_feedforward_layernorm.weight",
                "post_attention_layernorm.weight",
            ] {
                tensors.insert(
                    format!("layers.{layer_idx}.{name}"),
                    Tensor::ones(4, DType::F32, &device)?,
                );
            }
            tensors.insert(
                format!("layers.{layer_idx}.layer_scalar"),
                Tensor::from_vec(vec![1.0_f32], 1, &device)?,
            );
        }
        tensors.insert(
            "norm.weight".to_string(),
            Tensor::ones(4, DType::F32, &device)?,
        );

        Ok(tensors)
    }

    fn tiny_decode_logit_model(num_hidden_layers: usize) -> Gemma4TextModel {
        let cfg = tiny_decode_logit_config(num_hidden_layers);
        let vb = VarBuilder::from_tensors(
            tiny_decode_logit_tensor_map(num_hidden_layers).expect("tiny decode tensor map"),
            DType::F32,
            &Device::Cpu,
        );
        Gemma4TextModel::new(&cfg, vb).expect("tiny Gemma4TextModel")
    }

    fn argmax(values: &[f32]) -> usize {
        values
            .iter()
            .enumerate()
            .max_by(|(_, lhs), (_, rhs)| lhs.partial_cmp(rhs).expect("finite logits"))
            .map(|(idx, _)| idx)
            .expect("non-empty logits")
    }

    fn max_abs_diff(lhs: &[f32], rhs: &[f32]) -> f32 {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0_f32, f32::max)
    }

    fn top_k_indices(values: &[f32], k: usize) -> Vec<usize> {
        let mut indexed = values.iter().copied().enumerate().collect::<Vec<_>>();
        indexed.sort_by(|(lhs_idx, lhs), (rhs_idx, rhs)| {
            rhs.partial_cmp(lhs)
                .expect("finite logits")
                .then_with(|| lhs_idx.cmp(rhs_idx))
        });
        indexed.into_iter().take(k).map(|(idx, _)| idx).collect()
    }

    fn tensor_to_vec_f32(tensor: &Tensor) -> Vec<f32> {
        tensor
            .flatten_all()
            .expect("flatten tensor")
            .to_dtype(DType::F32)
            .expect("tensor to f32")
            .to_vec1::<f32>()
            .expect("tensor vec")
    }

    #[derive(Clone)]
    struct OwnedTensorView {
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl View for OwnedTensorView {
        fn dtype(&self) -> SafeDtype {
            SafeDtype::F32
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.data)
        }

        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    fn owned_tensor_view(tensor: &Tensor) -> OwnedTensorView {
        let shape = tensor.shape().dims().to_vec();
        let values = tensor
            .flatten_all()
            .expect("flatten checkpoint tensor")
            .to_dtype(DType::F32)
            .expect("checkpoint tensor to f32")
            .to_vec1::<f32>()
            .expect("checkpoint tensor vec");
        let mut data = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        OwnedTensorView { shape, data }
    }

    fn tiny_decode_logit_tokenizer_json() -> serde_json::Value {
        serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": { "type": "Whitespace" },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "<unk>": 0,
                    "tok1": 1,
                    "tok2": 2,
                    "tok3": 3,
                    "tok4": 4,
                    "tok5": 5,
                    "tok6": 6,
                    "tok7": 7
                },
                "unk_token": "<unk>"
            }
        })
    }

    fn tiny_decode_logit_config_json_with_layout(
        layer_types: &[&str],
        sliding_window: usize,
    ) -> serde_json::Value {
        serde_json::json!({
            "text_config": {
                "attention_bias": false,
                "attention_k_eq_v": false,
                "head_dim": 2,
                "hidden_activation": "silu",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_attention_heads": 2,
                "num_hidden_layers": layer_types.len(),
                "num_key_value_heads": 1,
                "rms_norm_eps": 1e-6,
                "vocab_size": 8,
                "max_position_embeddings": 16,
                "sliding_window": sliding_window,
                "layer_types": layer_types,
                "enable_moe_block": false
            },
            "eos_token_id": [7]
        })
    }

    fn tiny_decode_logit_config_json(num_hidden_layers: usize) -> serde_json::Value {
        let layer_types = vec!["full_attention"; num_hidden_layers];
        tiny_decode_logit_config_json_with_layout(&layer_types, 8)
    }

    fn write_json(path: &Path, value: &serde_json::Value) {
        std::fs::write(
            path,
            serde_json::to_vec_pretty(value).expect("serialize json fixture"),
        )
        .expect("write json fixture");
    }

    fn write_tiny_gemma4_checkpoint_fixture(num_hidden_layers: usize) -> TempDir {
        // This is the closest repo-backed runtime fixture available: a tiny on-disk Gemma4
        // checkpoint exercising Model::new + safetensors/tokenizer/config loading, not a
        // production Gemma4 checkpoint snapshot.
        let dir = tempfile::tempdir().expect("checkpoint tempdir");
        write_json(
            &dir.path().join("tokenizer.json"),
            &tiny_decode_logit_tokenizer_json(),
        );
        write_json(
            &dir.path().join("config.json"),
            &tiny_decode_logit_config_json(num_hidden_layers),
        );

        let mut entries = tiny_decode_logit_tensor_map(num_hidden_layers)
            .expect("tiny decode tensor map")
            .into_iter()
            .map(|(name, tensor)| {
                (
                    format!("model.language_model.{name}"),
                    owned_tensor_view(&tensor),
                )
            })
            .collect::<Vec<_>>();
        entries.sort_by(|(lhs, _), (rhs, _)| lhs.cmp(rhs));
        serialize_to_file(entries, &None, &dir.path().join("model.safetensors"))
            .expect("serialize safetensors checkpoint");
        dir
    }

    fn write_tiny_gemma4_checkpoint_fixture_with_layout(
        layer_types: &[&str],
        sliding_window: usize,
    ) -> TempDir {
        let dir = tempfile::tempdir().expect("checkpoint tempdir");
        write_json(
            &dir.path().join("tokenizer.json"),
            &tiny_decode_logit_tokenizer_json(),
        );
        write_json(
            &dir.path().join("config.json"),
            &tiny_decode_logit_config_json_with_layout(layer_types, sliding_window),
        );

        let mut entries = tiny_decode_logit_tensor_map(layer_types.len())
            .expect("tiny decode tensor map")
            .into_iter()
            .map(|(name, tensor)| {
                (
                    format!("model.language_model.{name}"),
                    owned_tensor_view(&tensor),
                )
            })
            .collect::<Vec<_>>();
        entries.sort_by(|(lhs, _), (rhs, _)| lhs.cmp(rhs));
        serialize_to_file(entries, &None, &dir.path().join("model.safetensors"))
            .expect("serialize safetensors checkpoint");
        dir
    }

    struct RuntimeCheckpointFixture {
        _dir: TempDir,
        model_path: PathBuf,
        prompt_text: &'static str,
    }

    const DEFAULT_REAL_GEMMA4_CHECKPOINT_PATH: &str =
        "/models/google/gemma-4-26B-A4B-it/1db3cff1840c2ae59759d8e842ff37831cf8cb63";
    const DEFAULT_REAL_GEMMA4_PARITY_PROMPT: &str = "Write one short sentence about cranes.";
    const DEFAULT_REAL_GEMMA4_PARITY_STEPS: usize = 4;

    impl RuntimeCheckpointFixture {
        fn new(num_hidden_layers: usize) -> Self {
            let dir = write_tiny_gemma4_checkpoint_fixture(num_hidden_layers);
            let model_path = dir.path().to_path_buf();
            Self {
                _dir: dir,
                model_path,
                prompt_text: "tok1 tok2",
            }
        }

        fn with_layout(
            layer_types: &[&str],
            sliding_window: usize,
            prompt_text: &'static str,
        ) -> Self {
            let dir = write_tiny_gemma4_checkpoint_fixture_with_layout(layer_types, sliding_window);
            let model_path = dir.path().to_path_buf();
            Self {
                _dir: dir,
                model_path,
                prompt_text,
            }
        }

        fn path_str(&self) -> &str {
            self.model_path.to_str().expect("utf8 temp path")
        }
    }

    struct RealCheckpointHarnessConfig {
        model_path: PathBuf,
        prompt_text: String,
        decode_steps: usize,
    }

    impl RealCheckpointHarnessConfig {
        fn from_env() -> Self {
            let model_path = std::env::var("CRANE_GEMMA4_REAL_CHECKPOINT")
                .unwrap_or_else(|_| DEFAULT_REAL_GEMMA4_CHECKPOINT_PATH.to_string());
            let prompt_text = std::env::var("CRANE_GEMMA4_REAL_PARITY_PROMPT")
                .unwrap_or_else(|_| DEFAULT_REAL_GEMMA4_PARITY_PROMPT.to_string());
            let decode_steps = std::env::var("CRANE_GEMMA4_REAL_PARITY_STEPS")
                .ok()
                .and_then(|raw| raw.parse::<usize>().ok())
                .filter(|steps| *steps > 0)
                .unwrap_or(DEFAULT_REAL_GEMMA4_PARITY_STEPS);
            Self {
                model_path: PathBuf::from(model_path),
                prompt_text,
                decode_steps,
            }
        }

        fn path_str(&self) -> &str {
            self.model_path.to_str().expect("utf8 real checkpoint path")
        }

        fn require_accessible_path(&self) {
            assert!(
                self.model_path.exists(),
                "real Gemma4 checkpoint path '{}' is not accessible. Set CRANE_GEMMA4_REAL_CHECKPOINT to a mounted snapshot path before running this ignored parity harness.",
                self.model_path.display()
            );
        }
    }

    struct DenseParityReference {
        prompt_ids: Vec<u32>,
        prefill_logits: Vec<f32>,
        teacher_forced_tokens: Vec<u32>,
        decode_logits_by_step: Vec<Vec<f32>>,
    }

    #[derive(Debug, Clone)]
    struct OpenLoopParityTrace {
        generated_tokens: Vec<u32>,
        logits_by_step: Vec<Vec<f32>>,
    }

    #[derive(Debug, Clone)]
    struct OpenLoopParityComparison {
        dense: OpenLoopParityTrace,
        turbo: OpenLoopParityTrace,
        first_divergence_step: Option<usize>,
    }

    fn runtime_prefill_ctx(input_ids: Vec<u32>) -> RuntimeRequestContext {
        RuntimeRequestContext {
            input_ids,
            start_pos: 0,
            multimodal_inputs: MultimodalInputs::default(),
        }
    }

    fn runtime_decode_ctx(input_id: u32, start_pos: usize) -> RuntimeStepContext {
        RuntimeStepContext {
            input_ids: vec![input_id],
            start_pos,
        }
    }

    fn encoded_prompt_ids(model: &dyn RuntimeModel, prompt: &str) -> Vec<u32> {
        model
            .tokenizer()
            .encode(prompt, false)
            .expect("encode prompt")
            .get_ids()
            .to_vec()
    }

    fn assert_top_k_match(lhs: &[f32], rhs: &[f32], k: usize) {
        assert_eq!(top_k_indices(lhs, k), top_k_indices(rhs, k));
    }

    fn parity_harness_device() -> Device {
        crane_core::utils::select_device(false).expect("select parity harness device")
    }

    fn parity_harness_dtype(device: &Device) -> DType {
        match device {
            Device::Cpu => DType::F32,
            _ => DType::BF16,
        }
    }

    fn collect_dense_parity_reference(
        config: &RealCheckpointHarnessConfig,
        device: &Device,
        dtype: DType,
    ) -> DenseParityReference {
        let mut dense = Gemma4RuntimeAdapter::new(
            config.path_str(),
            device,
            &dtype,
            KvBackendConfig {
                mode: KvCacheMode::Bf16Dense,
            },
        )
        .expect("dense parity adapter");
        let prompt_ids = encoded_prompt_ids(&dense, &config.prompt_text);
        assert!(
            !prompt_ids.is_empty(),
            "real checkpoint parity prompt must tokenize to at least one token"
        );
        let prefill = dense
            .prefill(runtime_prefill_ctx(prompt_ids.clone()))
            .expect("dense parity prefill");
        let prefill_logits = tensor_to_vec_f32(&prefill.logits);

        let mut previous_logits = prefill_logits.clone();
        let mut teacher_forced_tokens = Vec::with_capacity(config.decode_steps);
        let mut decode_logits_by_step = Vec::with_capacity(config.decode_steps);
        for step in 0..config.decode_steps {
            let token = argmax(&previous_logits) as u32;
            teacher_forced_tokens.push(token);
            let decode = dense
                .decode(runtime_decode_ctx(token, prompt_ids.len() + step))
                .expect("dense parity decode");
            let decode_logits = tensor_to_vec_f32(&decode.logits);
            previous_logits = decode_logits.clone();
            decode_logits_by_step.push(decode_logits);
        }

        DenseParityReference {
            prompt_ids,
            prefill_logits,
            teacher_forced_tokens,
            decode_logits_by_step,
        }
    }

    fn collect_open_loop_trace(
        model: &mut Gemma4RuntimeAdapter,
        prompt_len: usize,
        initial_logits: &[f32],
        decode_steps: usize,
    ) -> OpenLoopParityTrace {
        let mut previous_logits = initial_logits.to_vec();
        let mut generated_tokens = Vec::with_capacity(decode_steps);
        let mut logits_by_step = Vec::with_capacity(decode_steps);
        for step in 0..decode_steps {
            let token = argmax(&previous_logits) as u32;
            generated_tokens.push(token);
            let decode = model
                .decode(runtime_decode_ctx(token, prompt_len + step))
                .expect("open-loop decode");
            let decode_logits = tensor_to_vec_f32(&decode.logits);
            previous_logits = decode_logits.clone();
            logits_by_step.push(decode_logits);
        }

        OpenLoopParityTrace {
            generated_tokens,
            logits_by_step,
        }
    }

    fn compare_open_loop_generation(
        dense: &mut Gemma4RuntimeAdapter,
        restored_turbo: &mut Gemma4RuntimeAdapter,
        prompt_len: usize,
        dense_prefill_logits: &[f32],
        turbo_prefill_logits: &[f32],
        decode_steps: usize,
    ) -> OpenLoopParityComparison {
        let dense_trace =
            collect_open_loop_trace(dense, prompt_len, dense_prefill_logits, decode_steps);
        let turbo_trace = collect_open_loop_trace(
            restored_turbo,
            prompt_len,
            turbo_prefill_logits,
            decode_steps,
        );
        let first_divergence_step = dense_trace
            .generated_tokens
            .iter()
            .zip(turbo_trace.generated_tokens.iter())
            .position(|(dense_token, turbo_token)| dense_token != turbo_token);

        OpenLoopParityComparison {
            dense: dense_trace,
            turbo: turbo_trace,
            first_divergence_step,
        }
    }

    #[test]
    fn turboquant_decode_prefix_scores_match_dense_prefix_with_bounded_drift() {
        let (prefix, key_rows, value_rows) = turboquant_prefix_fixture();
        let query_rows = vec![
            vec![-0.74_f32, -0.79, 0.72, 0.76, 0.10, -0.07, 0.08, -0.11],
            vec![0.80_f32, 0.78, -0.74, -0.71, -0.07, 0.05, -0.08, 0.06],
        ];
        let query_states = Tensor::from_vec(query_rows.concat(), (1, 2, 1, 8), &Device::Cpu)
            .expect("query states");

        let prefix_scores = prefix
            .score_prefix_keys(0, &query_states, 1, 2)
            .expect("prefix scores")
            .expect("supported turboquant prefix path");
        let actual = prefix_scores
            .flatten_all()
            .expect("flatten scores")
            .to_vec1::<f32>()
            .expect("scores vec");

        let expected = dense_scores(&query_rows, &key_rows);
        assert_eq!(actual.len(), 4);
        for head_idx in 0..2 {
            for pos_idx in 0..2 {
                let flat_idx = head_idx * 2 + pos_idx;
                assert!(
                    (actual[flat_idx] - expected[head_idx][pos_idx]).abs() <= 0.35,
                    "head={head_idx} pos={pos_idx} actual={} expected={}",
                    actual[flat_idx],
                    expected[head_idx][pos_idx]
                );
            }
        }

        let prefix_values = prefix
            .value_prefix(0, &Device::Cpu, DType::F32)
            .expect("prefix values")
            .expect("dense fallback V");
        let restored = prefix_values
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected = value_rows.concat();
        assert_eq!(restored.len(), expected.len());
        for (idx, (actual, expected)) in restored.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*actual - *expected).abs() <= 0.01,
                "idx={idx} actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn turboquant_decode_prefix_supports_multi_token_queries_for_batch_one() {
        let (prefix, ..) = turboquant_prefix_fixture();

        let multi_token_query = Tensor::from_vec(
            vec![
                -0.74_f32, -0.79, 0.72, 0.76, 0.10, -0.07, 0.08, -0.11, -0.30_f32, -0.25, 0.22,
                0.26, 0.04, -0.03, 0.02, -0.01, 0.80_f32, 0.78, -0.74, -0.71, -0.07, 0.05, -0.08,
                0.06, 0.18_f32, 0.15, -0.12, -0.10, -0.01, 0.02, -0.03, 0.04,
            ],
            (1, 2, 2, 8),
            &Device::Cpu,
        )
        .expect("multi-token query");
        let actual = prefix
            .score_prefix_keys(0, &multi_token_query, 1, 2)
            .expect("multi-token query should not error")
            .expect("batch=1 multi-token query should stay on turboquant path")
            .flatten_all()
            .expect("flatten scores")
            .to_vec1::<f32>()
            .expect("scores vec");
        let expected = dense_scores(
            &[
                vec![-0.74_f32, -0.79, 0.72, 0.76, 0.10, -0.07, 0.08, -0.11],
                vec![-0.30_f32, -0.25, 0.22, 0.26, 0.04, -0.03, 0.02, -0.01],
                vec![0.80_f32, 0.78, -0.74, -0.71, -0.07, 0.05, -0.08, 0.06],
                vec![0.18_f32, 0.15, -0.12, -0.10, -0.01, 0.02, -0.03, 0.04],
            ],
            &[
                vec![-0.78_f32, -0.82, 0.75, 0.79, 0.11, -0.08, 0.09, -0.12],
                vec![0.84_f32, 0.81, -0.77, -0.74, -0.09, 0.07, -0.10, 0.08],
            ],
        )
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*actual - *expected).abs() <= 0.35,
                "idx={idx} actual={actual} expected={expected}"
            );
        }

        let batch_query =
            Tensor::zeros((2, 2, 1, 8), DType::F32, &Device::Cpu).expect("batch query");
        assert!(prefix
            .score_prefix_keys(0, &batch_query, 1, 2)
            .expect("batch query should not error")
            .is_none());
    }

    #[test]
    fn turboquant_restore_layer_decisions_are_bounded_and_explicit() {
        assert_eq!(
            classify_turboquant_restore_layer(true, false, false, true),
            TurboQuantLayerDecision::supported()
        );
        assert_eq!(
            classify_turboquant_restore_layer(false, false, false, true),
            TurboQuantLayerDecision::fallback(TurboQuantPathReason::BackendNoCompressedKScores)
        );
        assert_eq!(
            classify_turboquant_restore_layer(true, true, false, true),
            TurboQuantLayerDecision::fallback(TurboQuantPathReason::SharedKvUnsupported)
        );
        assert_eq!(
            classify_turboquant_restore_layer(true, false, true, true),
            TurboQuantLayerDecision::fallback(TurboQuantPathReason::SlidingWindowUnsupported)
        );
        assert_eq!(
            classify_turboquant_restore_layer(true, false, false, false),
            TurboQuantLayerDecision::fallback(TurboQuantPathReason::MissingStoredKv)
        );
    }

    #[test]
    fn turboquant_restore_support_boundary_rejects_partial_hybrid_restore() {
        assert_eq!(
            apply_turboquant_restore_support_boundary(&[
                TurboQuantLayerDecision::supported(),
                TurboQuantLayerDecision::fallback(TurboQuantPathReason::SlidingWindowUnsupported),
                TurboQuantLayerDecision::supported(),
                TurboQuantLayerDecision::fallback(TurboQuantPathReason::MissingStoredKv),
            ]),
            vec![
                TurboQuantLayerDecision::fallback(
                    TurboQuantPathReason::MixedLayerRestoreUnsupported
                ),
                TurboQuantLayerDecision::fallback(TurboQuantPathReason::SlidingWindowUnsupported),
                TurboQuantLayerDecision::fallback(
                    TurboQuantPathReason::MixedLayerRestoreUnsupported
                ),
                TurboQuantLayerDecision::fallback(TurboQuantPathReason::MissingStoredKv),
            ]
        );
    }

    #[test]
    fn turboquant_decode_shape_fallback_reasons_are_bounded_and_explicit() {
        assert_eq!(classify_turboquant_decode_shape(1, 1, 4, 2, 2), Ok(()));
        assert_eq!(classify_turboquant_decode_shape(1, 2, 4, 2, 2), Ok(()));
        assert_eq!(
            classify_turboquant_decode_shape(2, 1, 4, 2, 2),
            Err(TurboQuantPathReason::BatchSizeUnsupported)
        );
        assert_eq!(
            classify_turboquant_decode_shape(1, 0, 4, 2, 2),
            Err(TurboQuantPathReason::QueryLenUnsupported)
        );
        assert_eq!(
            classify_turboquant_decode_shape(1, 1, 4, 2, 0),
            Err(TurboQuantPathReason::KvGroupsUnsupported)
        );
        assert_eq!(
            classify_turboquant_decode_shape(1, 1, 3, 2, 2),
            Err(TurboQuantPathReason::HeadLayoutUnsupported)
        );
    }

    #[test]
    fn turboquant_batch_decode_support_is_bounded_and_explicit() {
        assert_eq!(
            classify_turboquant_batch_decode_support(KvCacheMode::TurboQuant, false, false),
            Ok(())
        );
        assert_eq!(
            classify_turboquant_batch_decode_support(KvCacheMode::Bf16Dense, false, false),
            Err(TurboQuantPathReason::BackendBatchDecodeUnsupported)
        );
        assert_eq!(
            classify_turboquant_batch_decode_support(KvCacheMode::TurboQuant, true, false),
            Err(TurboQuantPathReason::SharedKvUnsupported)
        );
        assert_eq!(
            classify_turboquant_batch_decode_support(KvCacheMode::TurboQuant, false, true),
            Err(TurboQuantPathReason::SlidingWindowUnsupported)
        );
    }

    #[test]
    fn gemma4_turboquant_batch_decode_matches_dense_baseline_with_mixed_positions() {
        let fixture = RuntimeCheckpointFixture::new(1);
        let prompt_a = vec![1_u32, 2_u32];
        let prompt_b = vec![1_u32, 2_u32, 3_u32];
        let decode_tokens = vec![3_u32, 4_u32];

        let mut dense_a = Gemma4RuntimeAdapter::new(
            fixture.path_str(),
            &Device::Cpu,
            &DType::F32,
            KvBackendConfig {
                mode: KvCacheMode::Bf16Dense,
            },
        )
        .expect("dense adapter A");
        dense_a
            .prefill(RuntimeRequestContext {
                input_ids: prompt_a.clone(),
                start_pos: 0,
                multimodal_inputs: MultimodalInputs::default(),
            })
            .expect("dense prefill A");
        let dense_logits_a = dense_a
            .decode(RuntimeStepContext {
                input_ids: vec![decode_tokens[0]],
                start_pos: prompt_a.len(),
            })
            .expect("dense decode A")
            .logits;

        let mut dense_b = Gemma4RuntimeAdapter::new(
            fixture.path_str(),
            &Device::Cpu,
            &DType::F32,
            KvBackendConfig {
                mode: KvCacheMode::Bf16Dense,
            },
        )
        .expect("dense adapter B");
        dense_b
            .prefill(RuntimeRequestContext {
                input_ids: prompt_b.clone(),
                start_pos: 0,
                multimodal_inputs: MultimodalInputs::default(),
            })
            .expect("dense prefill B");
        let dense_logits_b = dense_b
            .decode(RuntimeStepContext {
                input_ids: vec![decode_tokens[1]],
                start_pos: prompt_b.len(),
            })
            .expect("dense decode B")
            .logits;

        let mut turbo_a = Gemma4RuntimeAdapter::new(
            fixture.path_str(),
            &Device::Cpu,
            &DType::F32,
            KvBackendConfig {
                mode: KvCacheMode::TurboQuant,
            },
        )
        .expect("turbo adapter A");
        turbo_a
            .prefill(RuntimeRequestContext {
                input_ids: prompt_a.clone(),
                start_pos: 0,
                multimodal_inputs: MultimodalInputs::default(),
            })
            .expect("turbo prefill A");
        let caches_a = turbo_a.kv_extract().expect("turbo caches A");

        let mut turbo_b = Gemma4RuntimeAdapter::new(
            fixture.path_str(),
            &Device::Cpu,
            &DType::F32,
            KvBackendConfig {
                mode: KvCacheMode::TurboQuant,
            },
        )
        .expect("turbo adapter B");
        turbo_b
            .prefill(RuntimeRequestContext {
                input_ids: prompt_b.clone(),
                start_pos: 0,
                multimodal_inputs: MultimodalInputs::default(),
            })
            .expect("turbo prefill B");
        let caches_b = turbo_b.kv_extract().expect("turbo caches B");

        let mut batch_adapter = Gemma4RuntimeAdapter::new(
            fixture.path_str(),
            &Device::Cpu,
            &DType::F32,
            KvBackendConfig {
                mode: KvCacheMode::TurboQuant,
            },
        )
        .expect("turbo batch adapter");
        assert!(batch_adapter.supports_batch_decode());

        let (kv_lens, original_max_kv) = batch_adapter
            .setup_batch_decode(&[caches_a.clone(), caches_b.clone()], 1)
            .expect("setup batch decode");
        assert_eq!(kv_lens, vec![prompt_a.len(), prompt_b.len()]);
        assert_eq!(original_max_kv, prompt_b.len());

        let mask = batch_adapter
            .build_batch_decode_mask(&kv_lens, original_max_kv, original_max_kv + 1)
            .expect("build batch mask");
        assert!(mask.is_some());

        let input_ids = Tensor::new(decode_tokens.as_slice(), &Device::Cpu)
            .expect("decode tokens")
            .reshape((2, 1))
            .expect("batch input ids");
        let batch_logits = batch_adapter
            .batch_decode(BatchDecodeContext {
                input_ids: &input_ids,
                positions: &[prompt_a.len(), prompt_b.len()],
                attention_mask: mask.as_ref(),
                batch_kv_info: Some((&kv_lens, original_max_kv)),
            })
            .expect("batch decode logits");
        let batch_logits_a = batch_logits.narrow(0, 0, 1).expect("row A logits");
        let batch_logits_b = batch_logits.narrow(0, 1, 1).expect("row B logits");

        let dense_logits_a = tensor_to_vec_f32(&dense_logits_a);
        let dense_logits_b = tensor_to_vec_f32(&dense_logits_b);
        let batch_logits_a = tensor_to_vec_f32(&batch_logits_a);
        let batch_logits_b = tensor_to_vec_f32(&batch_logits_b);

        assert_eq!(argmax(&dense_logits_a), argmax(&batch_logits_a));
        assert_eq!(argmax(&dense_logits_b), argmax(&batch_logits_b));
        assert!(
            max_abs_diff(&dense_logits_a, &batch_logits_a) <= 0.75,
            "dense_a={dense_logits_a:?} batch_a={batch_logits_a:?}"
        );
        assert!(
            max_abs_diff(&dense_logits_b, &batch_logits_b) <= 0.75,
            "dense_b={dense_logits_b:?} batch_b={batch_logits_b:?}"
        );

        let extracted = batch_adapter
            .extract_batch_kv(&kv_lens, original_max_kv, 1)
            .expect("extract batch kv");
        assert_eq!(extracted.len(), 2);
        assert_eq!(extracted[0].len(), 1);
        assert_eq!(extracted[1].len(), 1);
        assert_eq!(
            extracted[0][0].as_ref().expect("seq A layer 0").seq_len,
            prompt_a.len() + 1
        );
        assert_eq!(
            extracted[1][0].as_ref().expect("seq B layer 0").seq_len,
            prompt_b.len() + 1
        );
        assert_eq!(
            extracted[0][0].as_ref().expect("seq A format").format,
            KvCacheMode::TurboQuant
        );
        assert_eq!(
            extracted[1][0].as_ref().expect("seq B format").format,
            KvCacheMode::TurboQuant
        );
    }

    #[test]
    fn gemma4_batch_decode_rejections_stay_explicit_and_grep_friendly() {
        let fixture = RuntimeCheckpointFixture::new(1);
        let mut turbo = Gemma4RuntimeAdapter::new(
            fixture.path_str(),
            &Device::Cpu,
            &DType::F32,
            KvBackendConfig {
                mode: KvCacheMode::TurboQuant,
            },
        )
        .expect("turbo adapter");
        let bad_input_ids = Tensor::zeros((2, 2), DType::U32, &Device::Cpu).expect("bad input ids");

        let err = turbo
            .batch_decode(BatchDecodeContext {
                input_ids: &bad_input_ids,
                positions: &[0, 1],
                attention_mask: None,
                batch_kv_info: None,
            })
            .expect_err("multi-token batch decode should stay bounded");
        let message = err.to_string();
        assert!(message.contains("gemma4_batch_decode_unsupported"));
        assert!(message.contains("reason=query_len_unsupported"));
        assert!(message.contains("decision=reject"));

        let mut dense = Gemma4RuntimeAdapter::new(
            fixture.path_str(),
            &Device::Cpu,
            &DType::F32,
            KvBackendConfig {
                mode: KvCacheMode::Bf16Dense,
            },
        )
        .expect("dense adapter");
        assert!(!dense.supports_batch_decode());
        let dense_input_ids =
            Tensor::zeros((2, 1), DType::U32, &Device::Cpu).expect("dense input ids");
        let err = dense
            .batch_decode(BatchDecodeContext {
                input_ids: &dense_input_ids,
                positions: &[0, 1],
                attention_mask: None,
                batch_kv_info: None,
            })
            .expect_err("bf16_dense should stay out of batch surface");
        let message = err.to_string();
        assert!(message.contains("gemma4_batch_decode_unsupported"));
        assert!(message.contains("reason=backend_batch_decode_unsupported"));
        assert!(message.contains("decision=reject"));
    }

    #[test]
    fn turboquant_decode_prefix_value_payload_is_backend_owned_and_restores_for_fallback() {
        let (prefix, ..) = turboquant_prefix_fixture();
        let stored = prefix.caches[0].as_ref().expect("stored cache");

        let KvLayerPayload::TurboQuant { value, .. } = &stored.payload else {
            panic!("expected turboquant payload")
        };
        assert!(matches!(value, TurboQuantValuePayload::RowwiseInt8 { .. }));

        let imported = prefix
            .value_prefix(0, &Device::Cpu, DType::BF16)
            .expect("value prefix")
            .expect("decoded value prefix");
        assert_eq!(imported.dtype(), DType::BF16);
        assert_eq!(imported.dims4().expect("value dims"), (1, 1, 2, 8));
    }

    #[test]
    fn turboquant_decode_prefix_logits_track_dense_baseline_with_bounded_drift() {
        let cfg = tiny_decode_logit_config(1);
        let mut prefix_model = tiny_decode_logit_model(1);
        let prefix_input = Tensor::new(&[1_u32, 2_u32], &Device::Cpu)
            .expect("prefix ids")
            .unsqueeze(0)
            .expect("batch prefix ids");
        prefix_model
            .forward(&prefix_input, 0, cfg.sliding_window)
            .expect("prefix prefill");
        let dense_prefix = prefix_model.get_kv_caches();

        let mut dense_model = tiny_decode_logit_model(1);
        dense_model.set_kv_caches(
            dense_prefix
                .iter()
                .map(|layer| {
                    layer
                        .as_ref()
                        .map(|(key, value)| (key.clone(), value.clone()))
                })
                .collect(),
        );

        let backend: Arc<dyn KvCacheBackend> = Arc::from(
            make_kv_backend(KvBackendConfig {
                mode: KvCacheMode::TurboQuant,
            })
            .expect("turboquant backend"),
        );
        let stored_prefix = dense_prefix
            .iter()
            .enumerate()
            .map(|(layer_idx, dense)| {
                backend
                    .export_layer(
                        layer_idx,
                        dense
                            .as_ref()
                            .map(|(key, value)| (key.clone(), value.clone())),
                    )
                    .expect("export turboquant prefix")
            })
            .collect();
        let prefix = Arc::new(Gemma4TurboQuantDecodePrefix {
            backend,
            caches: stored_prefix,
            enabled_layers: vec![true],
        });

        let mut turbo_model = tiny_decode_logit_model(1);
        turbo_model.set_decode_prefix_kv_source(Some(prefix));

        let decode_input = Tensor::new(&[3_u32], &Device::Cpu)
            .expect("decode ids")
            .unsqueeze(0)
            .expect("batch decode ids");
        let dense_logits = dense_model
            .forward(&decode_input, 2, cfg.sliding_window)
            .expect("dense decode logits")
            .flatten_all()
            .expect("flatten dense logits")
            .to_dtype(DType::F32)
            .expect("dense logits f32")
            .to_vec1::<f32>()
            .expect("dense logits vec");
        let turbo_logits = turbo_model
            .forward(&decode_input, 2, cfg.sliding_window)
            .expect("turboquant decode logits")
            .flatten_all()
            .expect("flatten turbo logits")
            .to_dtype(DType::F32)
            .expect("turbo logits f32")
            .to_vec1::<f32>()
            .expect("turbo logits vec");

        assert_eq!(dense_logits.len(), turbo_logits.len());
        assert_eq!(argmax(&dense_logits), argmax(&turbo_logits));
        assert!(
            max_abs_diff(&dense_logits, &turbo_logits) <= 0.75,
            "dense={dense_logits:?} turbo={turbo_logits:?}"
        );
    }

    #[test]
    fn turboquant_decode_prefix_multi_layer_logits_track_dense_baseline_with_bounded_drift() {
        let layer_count = 3;
        let cfg = tiny_decode_logit_config(layer_count);
        let mut prefix_model = tiny_decode_logit_model(layer_count);
        let prefix_input = Tensor::new(&[1_u32, 2_u32], &Device::Cpu)
            .expect("prefix ids")
            .unsqueeze(0)
            .expect("batch prefix ids");
        prefix_model
            .forward(&prefix_input, 0, cfg.sliding_window)
            .expect("prefix prefill");
        let dense_prefix = prefix_model.get_kv_caches();
        assert_eq!(dense_prefix.len(), layer_count);
        assert!(dense_prefix.iter().all(|layer| layer.is_some()));

        let mut dense_model = tiny_decode_logit_model(layer_count);
        dense_model.set_kv_caches(
            dense_prefix
                .iter()
                .map(|layer| {
                    layer
                        .as_ref()
                        .map(|(key, value)| (key.clone(), value.clone()))
                })
                .collect(),
        );

        let backend: Arc<dyn KvCacheBackend> = Arc::from(
            make_kv_backend(KvBackendConfig {
                mode: KvCacheMode::TurboQuant,
            })
            .expect("turboquant backend"),
        );
        let stored_prefix = dense_prefix
            .iter()
            .enumerate()
            .map(|(layer_idx, dense)| {
                backend
                    .export_layer(
                        layer_idx,
                        dense
                            .as_ref()
                            .map(|(key, value)| (key.clone(), value.clone())),
                    )
                    .expect("export turboquant prefix")
            })
            .collect();
        let inner_prefix = Arc::new(Gemma4TurboQuantDecodePrefix {
            backend,
            caches: stored_prefix,
            enabled_layers: vec![true; layer_count],
        });
        let score_calls_by_layer = Arc::new(Mutex::new(vec![0_usize; layer_count]));
        let weighted_value_calls_by_layer = Arc::new(Mutex::new(vec![0_usize; layer_count]));
        let dense_value_calls_by_layer = Arc::new(Mutex::new(vec![0_usize; layer_count]));
        let prefix = Arc::new(CountingDecodePrefixKvSource {
            inner: inner_prefix,
            score_calls_by_layer: score_calls_by_layer.clone(),
            weighted_value_calls_by_layer: weighted_value_calls_by_layer.clone(),
            dense_value_calls_by_layer: dense_value_calls_by_layer.clone(),
        });

        let mut turbo_model = tiny_decode_logit_model(layer_count);
        turbo_model.set_decode_prefix_kv_source(Some(prefix));

        let decode_input = Tensor::new(&[3_u32], &Device::Cpu)
            .expect("decode ids")
            .unsqueeze(0)
            .expect("batch decode ids");
        let dense_logits = dense_model
            .forward(&decode_input, 2, cfg.sliding_window)
            .expect("dense decode logits")
            .flatten_all()
            .expect("flatten dense logits")
            .to_dtype(DType::F32)
            .expect("dense logits f32")
            .to_vec1::<f32>()
            .expect("dense logits vec");
        let turbo_logits = turbo_model
            .forward(&decode_input, 2, cfg.sliding_window)
            .expect("turboquant decode logits")
            .flatten_all()
            .expect("flatten turbo logits")
            .to_dtype(DType::F32)
            .expect("turbo logits f32")
            .to_vec1::<f32>()
            .expect("turbo logits vec");

        assert_eq!(dense_logits.len(), turbo_logits.len());
        assert_eq!(argmax(&dense_logits), argmax(&turbo_logits));
        assert!(
            max_abs_diff(&dense_logits, &turbo_logits) <= 0.35,
            "dense={dense_logits:?} turbo={turbo_logits:?}"
        );
        assert_eq!(
            *score_calls_by_layer
                .lock()
                .expect("score calls mutex poisoned"),
            vec![1, 1, 1]
        );
        assert_eq!(
            *weighted_value_calls_by_layer
                .lock()
                .expect("weighted value calls mutex poisoned"),
            vec![1, 1, 1]
        );
        assert_eq!(
            *dense_value_calls_by_layer
                .lock()
                .expect("dense value calls mutex poisoned"),
            vec![0, 0, 0]
        );
    }

    #[test]
    fn turboquant_decode_prefix_multi_token_logits_track_dense_baseline_with_bounded_drift() {
        let cfg = tiny_decode_logit_config(1);
        let mut prefix_model = tiny_decode_logit_model(1);
        let prefix_input = Tensor::new(&[1_u32, 2_u32], &Device::Cpu)
            .expect("prefix ids")
            .unsqueeze(0)
            .expect("batch prefix ids");
        prefix_model
            .forward(&prefix_input, 0, cfg.sliding_window)
            .expect("prefix prefill");
        let dense_prefix = prefix_model.get_kv_caches();

        let mut dense_model = tiny_decode_logit_model(1);
        dense_model.set_kv_caches(
            dense_prefix
                .iter()
                .map(|layer| {
                    layer
                        .as_ref()
                        .map(|(key, value)| (key.clone(), value.clone()))
                })
                .collect(),
        );

        let backend: Arc<dyn KvCacheBackend> = Arc::from(
            make_kv_backend(KvBackendConfig {
                mode: KvCacheMode::TurboQuant,
            })
            .expect("turboquant backend"),
        );
        let stored_prefix = dense_prefix
            .iter()
            .enumerate()
            .map(|(layer_idx, dense)| {
                backend
                    .export_layer(
                        layer_idx,
                        dense
                            .as_ref()
                            .map(|(key, value)| (key.clone(), value.clone())),
                    )
                    .expect("export turboquant prefix")
            })
            .collect();
        let prefix = Arc::new(Gemma4TurboQuantDecodePrefix {
            backend,
            caches: stored_prefix,
            enabled_layers: vec![true],
        });

        let mut turbo_model = tiny_decode_logit_model(1);
        turbo_model.set_decode_prefix_kv_source(Some(prefix));

        let decode_input = Tensor::new(&[3_u32, 4_u32], &Device::Cpu)
            .expect("decode ids")
            .unsqueeze(0)
            .expect("batch decode ids");
        let dense_logits = dense_model
            .forward(&decode_input, 2, cfg.sliding_window)
            .expect("dense decode logits")
            .flatten_all()
            .expect("flatten dense logits")
            .to_dtype(DType::F32)
            .expect("dense logits f32")
            .to_vec1::<f32>()
            .expect("dense logits vec");
        let turbo_logits = turbo_model
            .forward(&decode_input, 2, cfg.sliding_window)
            .expect("turboquant decode logits")
            .flatten_all()
            .expect("flatten turbo logits")
            .to_dtype(DType::F32)
            .expect("turbo logits f32")
            .to_vec1::<f32>()
            .expect("turbo logits vec");

        assert_eq!(dense_logits.len(), turbo_logits.len());
        assert_eq!(argmax(&dense_logits), argmax(&turbo_logits));
        assert!(
            max_abs_diff(&dense_logits, &turbo_logits) <= 0.75,
            "dense={dense_logits:?} turbo={turbo_logits:?}"
        );
    }

    #[test]
    fn turboquant_runtime_checkpoint_fixture_matches_dense_baseline_on_prefill_and_restore() {
        let fixture = RuntimeCheckpointFixture::new(3);
        let mut dense = Gemma4RuntimeAdapter::new(
            fixture.path_str(),
            &Device::Cpu,
            &DType::F32,
            KvBackendConfig {
                mode: KvCacheMode::Bf16Dense,
            },
        )
        .expect("dense adapter");
        let mut turbo_prefill = Gemma4RuntimeAdapter::new(
            fixture.path_str(),
            &Device::Cpu,
            &DType::F32,
            KvBackendConfig {
                mode: KvCacheMode::TurboQuant,
            },
        )
        .expect("turbo adapter");

        let prompt_ids = encoded_prompt_ids(&dense, fixture.prompt_text);
        assert_eq!(prompt_ids, vec![1, 2]);

        let dense_prefill = dense
            .prefill(runtime_prefill_ctx(prompt_ids.clone()))
            .expect("dense prefill");
        let turbo_prefill_logits = turbo_prefill
            .prefill(runtime_prefill_ctx(prompt_ids.clone()))
            .expect("turbo prefill")
            .logits;
        let dense_prefill_logits = tensor_to_vec_f32(&dense_prefill.logits);
        let turbo_prefill_logits = tensor_to_vec_f32(&turbo_prefill_logits);

        assert_eq!(argmax(&dense_prefill_logits), argmax(&turbo_prefill_logits));
        assert_top_k_match(&dense_prefill_logits, &turbo_prefill_logits, 3);
        assert!(
            max_abs_diff(&dense_prefill_logits, &turbo_prefill_logits) <= 1e-5,
            "dense={dense_prefill_logits:?} turbo={turbo_prefill_logits:?}"
        );

        let turbo_caches = turbo_prefill.kv_extract().expect("extract turbo caches");
        assert_eq!(turbo_caches.len(), 3);
        assert!(turbo_caches.iter().all(|stored| {
            matches!(
                stored.as_ref().map(|layer| &layer.payload),
                Some(KvLayerPayload::TurboQuant { .. })
            )
        }));

        let mut restored_turbo = Gemma4RuntimeAdapter::new(
            fixture.path_str(),
            &Device::Cpu,
            &DType::F32,
            KvBackendConfig {
                mode: KvCacheMode::TurboQuant,
            },
        )
        .expect("restored turbo adapter");
        restored_turbo
            .kv_restore(turbo_caches)
            .expect("restore turbo caches");

        let first_token = argmax(&dense_prefill_logits) as u32;
        let dense_decode = dense
            .decode(runtime_decode_ctx(first_token, prompt_ids.len()))
            .expect("dense decode after restore checkpoint");
        let turbo_decode = restored_turbo
            .decode(runtime_decode_ctx(first_token, prompt_ids.len()))
            .expect("turbo decode after restore checkpoint");
        let dense_decode_logits = tensor_to_vec_f32(&dense_decode.logits);
        let turbo_decode_logits = tensor_to_vec_f32(&turbo_decode.logits);

        assert_eq!(argmax(&dense_decode_logits), argmax(&turbo_decode_logits));
        assert_top_k_match(&dense_decode_logits, &turbo_decode_logits, 3);
        assert!(
            max_abs_diff(&dense_decode_logits, &turbo_decode_logits) <= 0.35,
            "dense={dense_decode_logits:?} turbo={turbo_decode_logits:?}"
        );
    }

    #[test]
    fn turboquant_runtime_checkpoint_fixture_keeps_teacher_forced_multi_step_stream_stable() {
        let fixture = RuntimeCheckpointFixture::new(3);
        let mut dense = Gemma4RuntimeAdapter::new(
            fixture.path_str(),
            &Device::Cpu,
            &DType::F32,
            KvBackendConfig {
                mode: KvCacheMode::Bf16Dense,
            },
        )
        .expect("dense adapter");
        let mut turbo_prefill = Gemma4RuntimeAdapter::new(
            fixture.path_str(),
            &Device::Cpu,
            &DType::F32,
            KvBackendConfig {
                mode: KvCacheMode::TurboQuant,
            },
        )
        .expect("turbo adapter");

        let prompt_ids = encoded_prompt_ids(&dense, fixture.prompt_text);
        let dense_prefill = dense
            .prefill(runtime_prefill_ctx(prompt_ids.clone()))
            .expect("dense prefill");
        turbo_prefill
            .prefill(runtime_prefill_ctx(prompt_ids.clone()))
            .expect("turbo prefill");

        let turbo_caches = turbo_prefill.kv_extract().expect("extract turbo caches");
        let mut restored_turbo = Gemma4RuntimeAdapter::new(
            fixture.path_str(),
            &Device::Cpu,
            &DType::F32,
            KvBackendConfig {
                mode: KvCacheMode::TurboQuant,
            },
        )
        .expect("restored turbo adapter");
        restored_turbo
            .kv_restore(turbo_caches)
            .expect("restore turbo caches");

        let teacher_forced_stream = [2_u32, 2, 1, 2];
        let mut dense_next_tokens = vec![argmax(&tensor_to_vec_f32(&dense_prefill.logits)) as u32];
        let mut turbo_next_tokens = Vec::new();

        for (step, token) in teacher_forced_stream.into_iter().enumerate() {
            let turbo_step = restored_turbo
                .decode(runtime_decode_ctx(token, prompt_ids.len() + step))
                .expect("turbo multi-step decode");
            let dense_step = dense
                .decode(runtime_decode_ctx(token, prompt_ids.len() + step))
                .expect("dense multi-step decode");

            let turbo_logits = tensor_to_vec_f32(&turbo_step.logits);
            let dense_logits = tensor_to_vec_f32(&dense_step.logits);
            let dense_argmax = argmax(&dense_logits) as u32;
            let turbo_argmax = argmax(&turbo_logits) as u32;
            dense_next_tokens.push(dense_argmax);
            turbo_next_tokens.push(turbo_argmax);

            assert_eq!(
                dense_argmax, turbo_argmax,
                "step={step} dense={dense_logits:?} turbo={turbo_logits:?}"
            );
            assert_top_k_match(&dense_logits, &turbo_logits, 3);
            assert!(
                max_abs_diff(&dense_logits, &turbo_logits) <= 0.45,
                "step={step} dense={dense_logits:?} turbo={turbo_logits:?}"
            );
        }

        assert_eq!(dense_next_tokens[1..], turbo_next_tokens);
    }

    #[test]
    fn turboquant_runtime_checkpoint_fixture_records_open_loop_divergence_boundary() {
        let fixture = RuntimeCheckpointFixture::new(3);
        let mut dense = Gemma4RuntimeAdapter::new(
            fixture.path_str(),
            &Device::Cpu,
            &DType::F32,
            KvBackendConfig {
                mode: KvCacheMode::Bf16Dense,
            },
        )
        .expect("dense adapter");
        let mut turbo_prefill = Gemma4RuntimeAdapter::new(
            fixture.path_str(),
            &Device::Cpu,
            &DType::F32,
            KvBackendConfig {
                mode: KvCacheMode::TurboQuant,
            },
        )
        .expect("turbo adapter");

        let prompt_ids = encoded_prompt_ids(&dense, fixture.prompt_text);
        let dense_prefill_logits = tensor_to_vec_f32(
            &dense
                .prefill(runtime_prefill_ctx(prompt_ids.clone()))
                .expect("dense prefill")
                .logits,
        );
        let turbo_prefill_logits = tensor_to_vec_f32(
            &turbo_prefill
                .prefill(runtime_prefill_ctx(prompt_ids.clone()))
                .expect("turbo prefill")
                .logits,
        );
        let turbo_caches = turbo_prefill.kv_extract().expect("extract turbo caches");

        let mut restored_turbo = Gemma4RuntimeAdapter::new(
            fixture.path_str(),
            &Device::Cpu,
            &DType::F32,
            KvBackendConfig {
                mode: KvCacheMode::TurboQuant,
            },
        )
        .expect("restored turbo adapter");
        restored_turbo
            .kv_restore(turbo_caches)
            .expect("restore turbo caches");

        let comparison = compare_open_loop_generation(
            &mut dense,
            &mut restored_turbo,
            prompt_ids.len(),
            &dense_prefill_logits,
            &turbo_prefill_logits,
            4,
        );

        assert_eq!(comparison.first_divergence_step, Some(3));
        assert_eq!(
            &comparison.dense.generated_tokens[..3],
            &comparison.turbo.generated_tokens[..3]
        );
        assert_eq!(comparison.dense.generated_tokens, vec![2, 2, 2, 2]);
        assert_eq!(comparison.turbo.generated_tokens, vec![2, 2, 2, 3]);
        for (step, (dense_logits, turbo_logits)) in comparison
            .dense
            .logits_by_step
            .iter()
            .zip(comparison.turbo.logits_by_step.iter())
            .enumerate()
        {
            if step + 1 < comparison.first_divergence_step.unwrap_or(usize::MAX) {
                assert_eq!(argmax(dense_logits), argmax(turbo_logits));
                assert!(
                    max_abs_diff(dense_logits, turbo_logits) <= 0.45,
                    "open-loop step={step} dense={dense_logits:?} turbo={turbo_logits:?}"
                );
            }
        }
    }

    #[test]
    fn dense_runtime_checkpoint_restore_keeps_sliding_window_history_semantically_aligned() {
        let fixture = RuntimeCheckpointFixture::with_layout(
            &["full_attention", "sliding_attention", "sliding_attention"],
            2,
            "tok1 tok2 tok3 tok4",
        );
        let mut dense = Gemma4RuntimeAdapter::new(
            fixture.path_str(),
            &Device::Cpu,
            &DType::F32,
            KvBackendConfig {
                mode: KvCacheMode::Bf16Dense,
            },
        )
        .expect("dense adapter");

        let prompt_ids = encoded_prompt_ids(&dense, fixture.prompt_text);
        dense
            .prefill(runtime_prefill_ctx(prompt_ids.clone()))
            .expect("dense prefill");
        let dense_caches = dense.kv_extract().expect("extract dense caches");

        let mut restored_dense = Gemma4RuntimeAdapter::new(
            fixture.path_str(),
            &Device::Cpu,
            &DType::F32,
            KvBackendConfig {
                mode: KvCacheMode::Bf16Dense,
            },
        )
        .expect("restored dense adapter");
        restored_dense
            .kv_restore(dense_caches)
            .expect("restore dense caches");

        for (step, token) in [2_u32, 2, 1, 2].into_iter().enumerate() {
            let dense_step = dense
                .decode(runtime_decode_ctx(token, prompt_ids.len() + step))
                .expect("dense decode");
            let restored_step = restored_dense
                .decode(runtime_decode_ctx(token, prompt_ids.len() + step))
                .expect("restored dense decode");
            let dense_logits = tensor_to_vec_f32(&dense_step.logits);
            let restored_logits = tensor_to_vec_f32(&restored_step.logits);

            assert_eq!(
                argmax(&dense_logits),
                argmax(&restored_logits),
                "step={step} dense={dense_logits:?} restored={restored_logits:?}"
            );
            assert!(
                max_abs_diff(&dense_logits, &restored_logits) <= 1e-5,
                "step={step} dense={dense_logits:?} restored={restored_logits:?}"
            );
        }
    }

    #[test]
    fn turboquant_runtime_checkpoint_fixture_falls_back_to_dense_restore_when_some_layers_are_unsupported(
    ) {
        let fixture = RuntimeCheckpointFixture::with_layout(
            &["full_attention", "sliding_attention", "sliding_attention"],
            2,
            "tok1 tok2 tok3 tok4",
        );
        let mut dense = Gemma4RuntimeAdapter::new(
            fixture.path_str(),
            &Device::Cpu,
            &DType::F32,
            KvBackendConfig {
                mode: KvCacheMode::Bf16Dense,
            },
        )
        .expect("dense adapter");
        let mut turbo_prefill = Gemma4RuntimeAdapter::new(
            fixture.path_str(),
            &Device::Cpu,
            &DType::F32,
            KvBackendConfig {
                mode: KvCacheMode::TurboQuant,
            },
        )
        .expect("turbo adapter");

        let prompt_ids = encoded_prompt_ids(&dense, fixture.prompt_text);
        dense
            .prefill(runtime_prefill_ctx(prompt_ids.clone()))
            .expect("dense prefill");
        turbo_prefill
            .prefill(runtime_prefill_ctx(prompt_ids.clone()))
            .expect("turbo prefill");
        let turbo_caches = turbo_prefill.kv_extract().expect("extract turbo caches");

        let expected_enabled_layers = apply_turboquant_restore_support_boundary(
            &(0..turbo_caches.len())
                .map(|layer_idx| {
                    classify_turboquant_restore_layer(
                        true,
                        false,
                        layer_idx > 0,
                        turbo_caches[layer_idx].is_some(),
                    )
                })
                .collect::<Vec<_>>(),
        )
        .into_iter()
        .map(|decision| decision.use_decode_prefix)
        .collect::<Vec<_>>();

        let mut restored_turbo = Gemma4RuntimeAdapter::new(
            fixture.path_str(),
            &Device::Cpu,
            &DType::F32,
            KvBackendConfig {
                mode: KvCacheMode::TurboQuant,
            },
        )
        .expect("restored turbo adapter");
        restored_turbo
            .kv_restore(turbo_caches)
            .expect("restore turbo caches");

        let enabled_layers = restored_turbo
            .decode_prefix
            .as_ref()
            .map(|prefix| prefix.enabled_layers.clone())
            .unwrap_or_else(|| vec![false; restored_turbo.num_layers()]);
        assert_eq!(enabled_layers, expected_enabled_layers);
        assert_eq!(enabled_layers, vec![false, false, false]);

        for (step, token) in [2_u32, 2, 1, 2].into_iter().enumerate() {
            let dense_step = dense
                .decode(runtime_decode_ctx(token, prompt_ids.len() + step))
                .expect("dense decode");
            let turbo_step = restored_turbo
                .decode(runtime_decode_ctx(token, prompt_ids.len() + step))
                .expect("turbo decode");
            let dense_logits = tensor_to_vec_f32(&dense_step.logits);
            let turbo_logits = tensor_to_vec_f32(&turbo_step.logits);

            assert_eq!(
                argmax(&dense_logits),
                argmax(&turbo_logits),
                "step={step} dense={dense_logits:?} turbo={turbo_logits:?}"
            );
            assert_top_k_match(&dense_logits, &turbo_logits, 3);
            assert!(
                max_abs_diff(&dense_logits, &turbo_logits) <= 0.45,
                "step={step} dense={dense_logits:?} turbo={turbo_logits:?}"
            );
        }
    }

    #[test]
    fn tiny_runtime_checkpoint_fixture_loads_through_model_new() {
        let fixture = RuntimeCheckpointFixture::new(3);
        let model = Gemma4Model::new(fixture.path_str(), &Device::Cpu, &DType::F32)
            .expect("load tiny runtime checkpoint through Model::new");
        let prompt_ids = model
            .tokenizer
            .tokenizer
            .encode(fixture.prompt_text, false)
            .expect("encode prompt from on-disk tokenizer")
            .get_ids()
            .to_vec();
        assert_eq!(prompt_ids, vec![1, 2]);
        assert_eq!(model.num_layers(), 3);
    }

    #[test]
    #[ignore = "manual integration: requires GPU-accessible real Gemma4 checkpoint mount"]
    fn gemma4_real_checkpoint_turboquant_restore_parity_harness() {
        let config = RealCheckpointHarnessConfig::from_env();
        config.require_accessible_path();

        let device = parity_harness_device();
        let dtype = parity_harness_dtype(&device);
        let dense_reference = collect_dense_parity_reference(&config, &device, dtype);

        let mut turbo_prefill = Gemma4RuntimeAdapter::new(
            config.path_str(),
            &device,
            &dtype,
            KvBackendConfig {
                mode: KvCacheMode::TurboQuant,
            },
        )
        .expect("turbo parity adapter");
        let turbo_prefill_logits = tensor_to_vec_f32(
            &turbo_prefill
                .prefill(runtime_prefill_ctx(dense_reference.prompt_ids.clone()))
                .expect("turbo parity prefill")
                .logits,
        );

        let prefill_dense_top1 = argmax(&dense_reference.prefill_logits) as u32;
        let prefill_turbo_top1 = argmax(&turbo_prefill_logits) as u32;
        eprintln!(
            "[gemma4-real-parity] device={device:?} dtype={dtype:?} path={} prompt_tokens={} decode_steps={} prefill_top1_dense={} prefill_top1_turbo={} prefill_max_abs_diff={:.6}",
            config.model_path.display(),
            dense_reference.prompt_ids.len(),
            config.decode_steps,
            prefill_dense_top1,
            prefill_turbo_top1,
            max_abs_diff(&dense_reference.prefill_logits, &turbo_prefill_logits),
        );
        assert_eq!(
            prefill_dense_top1, prefill_turbo_top1,
            "prefill top-1 drifted before restore path"
        );

        let turbo_caches = turbo_prefill.kv_extract().expect("extract turbo caches");
        let turbo_caches_for_open_loop = turbo_caches.clone();
        assert_eq!(turbo_caches.len(), turbo_prefill.num_layers());
        assert!(turbo_caches.iter().all(|stored| {
            matches!(
                stored.as_ref().map(|layer| &layer.payload),
                Some(KvLayerPayload::TurboQuant { .. })
            )
        }));
        drop(turbo_prefill);

        let mut restored_turbo = Gemma4RuntimeAdapter::new(
            config.path_str(),
            &device,
            &dtype,
            KvBackendConfig {
                mode: KvCacheMode::TurboQuant,
            },
        )
        .expect("restored turbo parity adapter");
        restored_turbo
            .kv_restore(turbo_caches)
            .expect("restore turbo caches");

        let expected_enabled_layers = apply_turboquant_restore_support_boundary(
            &turbo_caches_for_open_loop
                .iter()
                .enumerate()
                .map(|(layer_idx, stored)| {
                    classify_turboquant_restore_layer(
                        restored_turbo.kv_backend.supports_compressed_k_scores(),
                        restored_turbo.model.has_shared_kv_layers(),
                        restored_turbo.model.layer_uses_sliding_window(layer_idx),
                        stored.is_some(),
                    )
                })
                .collect::<Vec<_>>(),
        )
        .into_iter()
        .map(|decision| decision.use_decode_prefix)
        .collect::<Vec<_>>();
        let actual_enabled_layers = restored_turbo
            .decode_prefix
            .as_ref()
            .map(|prefix| prefix.enabled_layers.clone())
            .unwrap_or_else(|| vec![false; restored_turbo.num_layers()]);
        assert_eq!(actual_enabled_layers, expected_enabled_layers);
        eprintln!(
            "[gemma4-real-parity] decode_prefix_enabled_layers={}/{} shared_kv={} sliding_layers={}",
            actual_enabled_layers.iter().filter(|enabled| **enabled).count(),
            actual_enabled_layers.len(),
            restored_turbo.model.has_shared_kv_layers(),
            (0..restored_turbo.num_layers())
                .filter(|layer_idx| restored_turbo.model.layer_uses_sliding_window(*layer_idx))
                .count(),
        );

        let mut dense_open_loop = Gemma4RuntimeAdapter::new(
            config.path_str(),
            &device,
            &dtype,
            KvBackendConfig {
                mode: KvCacheMode::Bf16Dense,
            },
        )
        .expect("dense open-loop parity adapter");
        let mut restored_turbo_open_loop = Gemma4RuntimeAdapter::new(
            config.path_str(),
            &device,
            &dtype,
            KvBackendConfig {
                mode: KvCacheMode::TurboQuant,
            },
        )
        .expect("restored turbo open-loop parity adapter");
        restored_turbo_open_loop
            .kv_restore(turbo_caches_for_open_loop)
            .expect("restore turbo caches for open-loop parity");
        let dense_open_loop_prefill_logits = tensor_to_vec_f32(
            &dense_open_loop
                .prefill(runtime_prefill_ctx(dense_reference.prompt_ids.clone()))
                .expect("dense open-loop prefill")
                .logits,
        );
        let open_loop = compare_open_loop_generation(
            &mut dense_open_loop,
            &mut restored_turbo_open_loop,
            dense_reference.prompt_ids.len(),
            &dense_open_loop_prefill_logits,
            &turbo_prefill_logits,
            config.decode_steps,
        );
        eprintln!(
            "[gemma4-real-parity] open_loop_dense_tokens={:?} open_loop_turbo_tokens={:?} first_divergence_step={:?}",
            open_loop.dense.generated_tokens,
            open_loop.turbo.generated_tokens,
            open_loop.first_divergence_step,
        );
        for (step, (dense_logits, turbo_logits)) in open_loop
            .dense
            .logits_by_step
            .iter()
            .zip(open_loop.turbo.logits_by_step.iter())
            .enumerate()
        {
            let dense_top1 = argmax(dense_logits) as u32;
            let turbo_top1 = argmax(turbo_logits) as u32;
            let dense_top5 = top_k_indices(dense_logits, 5);
            let turbo_top5 = top_k_indices(turbo_logits, 5);
            let drift = max_abs_diff(dense_logits, turbo_logits);
            eprintln!(
                "[gemma4-real-parity] open_loop_step={step} dense_top1={dense_top1} turbo_top1={turbo_top1} max_abs_diff={drift:.6} dense_top5={dense_top5:?} turbo_top5={turbo_top5:?}"
            );
        }
        assert_eq!(
            open_loop.first_divergence_step,
            None,
            "real checkpoint open-loop token-stream divergence begins at step {:?}: dense={:?} turbo={:?}",
            open_loop.first_divergence_step,
            open_loop.dense.generated_tokens,
            open_loop.turbo.generated_tokens,
        );

        for (step, token) in dense_reference
            .teacher_forced_tokens
            .iter()
            .copied()
            .enumerate()
        {
            let turbo_step = restored_turbo
                .decode(runtime_decode_ctx(
                    token,
                    dense_reference.prompt_ids.len() + step,
                ))
                .expect("turbo parity decode");
            let turbo_logits = tensor_to_vec_f32(&turbo_step.logits);
            let dense_logits = &dense_reference.decode_logits_by_step[step];
            let dense_top1 = argmax(dense_logits) as u32;
            let turbo_top1 = argmax(&turbo_logits) as u32;
            let dense_top5 = top_k_indices(dense_logits, 5);
            let turbo_top5 = top_k_indices(&turbo_logits, 5);
            let drift = max_abs_diff(dense_logits, &turbo_logits);
            eprintln!(
                "[gemma4-real-parity] step={step} input_token={token} dense_top1={dense_top1} turbo_top1={turbo_top1} max_abs_diff={drift:.6} dense_top5={dense_top5:?} turbo_top5={turbo_top5:?}"
            );
            assert_eq!(
                dense_top1, turbo_top1,
                "real checkpoint turboquant top-1 drift at decode step {step}"
            );
            assert!(
                drift.is_finite(),
                "real checkpoint turboquant drift must stay finite at decode step {step}"
            );
        }
    }
}
