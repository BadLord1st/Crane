use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::sync::Arc;

use crane_core::models::gemma4::modeling::DecodePrefixKvSource;

use crate::engine::runtime::{
    make_kv_backend, DenseLayerKv, KvBackendConfig, KvCacheBackend, KvLayerEnvelope, LayerKvCaches,
    RuntimeModel, RuntimeRequestContext, RuntimeStateDelta, RuntimeStepContext, RuntimeStepOutput,
};

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
        if batch != 1
            || q_len != 1
            || num_kv_groups == 0
            || num_heads != num_kv_heads * num_kv_groups
        {
            return Ok(None);
        }

        let query_rows = query_states
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .reshape((num_heads, head_dim))?
            .to_vec2::<f32>()?;
        let mut flat_query_rows = Vec::with_capacity(num_heads * head_dim);
        for row in &query_rows {
            flat_query_rows.extend_from_slice(row);
        }

        let grouped_query = Tensor::from_vec(flat_query_rows, (num_heads, head_dim), &Device::Cpu)?;
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
        let mut per_head_scores = Vec::with_capacity(num_heads * prefix_len);
        for head_idx in 0..num_heads {
            let kv_head_idx = head_idx / num_kv_groups;
            let start = kv_head_idx * prefix_len;
            let end = start + prefix_len;
            per_head_scores.extend_from_slice(&base_scores[head_idx][start..end]);
        }
        Ok(Some(Tensor::from_vec(
            per_head_scores,
            (1, num_heads, 1, prefix_len),
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

    fn turboquant_decode_layer_enabled(&self, layer_idx: usize) -> bool {
        self.kv_backend.supports_compressed_k_scores()
            && !self.model.has_shared_kv_layers()
            && !self.model.layer_uses_sliding_window(layer_idx)
    }

    fn maybe_enable_decode_prefix(&mut self, caches: &[Option<KvLayerEnvelope>]) {
        if !self.kv_backend.supports_compressed_k_scores() || self.model.has_shared_kv_layers() {
            self.clear_decode_prefix();
            return;
        }

        let enabled_layers: Vec<bool> = caches
            .iter()
            .enumerate()
            .map(|(layer_idx, stored)| {
                stored.is_some() && self.turboquant_decode_layer_enabled(layer_idx)
            })
            .collect();
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

    fn batch_decode(
        &mut self,
        _ctx: crate::engine::runtime::BatchDecodeContext<'_>,
    ) -> candle_core::Result<Tensor> {
        candle_core::bail!("Batch decode not supported by Gemma4RuntimeAdapter")
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
        let dense = caches
            .iter()
            .enumerate()
            .map(|(layer_idx, stored)| {
                let use_decode_prefix =
                    stored.is_some() && self.turboquant_decode_layer_enabled(layer_idx);
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
        self.maybe_enable_decode_prefix(&caches);
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

    fn tiny_decode_logit_config_json(num_hidden_layers: usize) -> serde_json::Value {
        serde_json::json!({
            "text_config": {
                "attention_bias": false,
                "attention_k_eq_v": false,
                "head_dim": 2,
                "hidden_activation": "silu",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_attention_heads": 2,
                "num_hidden_layers": num_hidden_layers,
                "num_key_value_heads": 1,
                "rms_norm_eps": 1e-6,
                "vocab_size": 8,
                "max_position_embeddings": 16,
                "sliding_window": 8,
                "layer_types": vec!["full_attention"; num_hidden_layers],
                "enable_moe_block": false
            },
            "eos_token_id": [7]
        })
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
    fn turboquant_decode_prefix_returns_none_for_unsupported_decode_shapes() {
        let (prefix, ..) = turboquant_prefix_fixture();

        let multi_token_query =
            Tensor::zeros((1, 2, 2, 8), DType::F32, &Device::Cpu).expect("multi-token query");
        assert!(prefix
            .score_prefix_keys(0, &multi_token_query, 1, 2)
            .expect("multi-token query should not error")
            .is_none());

        let batch_query =
            Tensor::zeros((2, 2, 1, 8), DType::F32, &Device::Cpu).expect("batch query");
        assert!(prefix
            .score_prefix_keys(0, &batch_query, 1, 2)
            .expect("batch query should not error")
            .is_none());
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
            max_abs_diff(&dense_logits, &turbo_logits) <= 0.25,
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

        let expected_enabled_layers: Vec<bool> = (0..restored_turbo.num_layers())
            .map(|layer_idx| {
                !restored_turbo.model.has_shared_kv_layers()
                    && !restored_turbo.model.layer_uses_sliding_window(layer_idx)
            })
            .collect();
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
