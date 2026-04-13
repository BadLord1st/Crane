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
    use candle_nn::{Activation, VarBuilder};
    use crane_core::models::gemma4::modeling::{Config, Gemma4TextModel};
    use std::collections::HashMap;
    use std::sync::Mutex;

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
            self.inner
                .value_prefix(layer_idx, target_device, target_dtype)
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
        assert_eq!(
            prefix_values
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            value_rows.concat()
        );
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
    fn turboquant_decode_prefix_value_import_stays_dense_for_supported_layers() {
        let (prefix, ..) = turboquant_prefix_fixture();
        let stored = prefix.caches[0].as_ref().expect("stored cache");

        let KvLayerPayload::TurboQuant { value, .. } = &stored.payload else {
            panic!("expected turboquant payload")
        };
        assert!(matches!(value, TurboQuantValuePayload::Dense(_)));

        let imported = prefix
            .value_prefix(0, &Device::Cpu, DType::BF16)
            .expect("value prefix")
            .expect("dense value prefix");
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
        let prefix = Arc::new(CountingDecodePrefixKvSource {
            inner: inner_prefix,
            score_calls_by_layer: score_calls_by_layer.clone(),
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
    }
}
