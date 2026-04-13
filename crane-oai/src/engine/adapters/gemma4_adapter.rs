use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::engine::runtime::{
    make_kv_backend, KvBackendConfig, KvCacheBackend, LayerKvCaches, RuntimeModel,
    RuntimeRequestContext, RuntimeStateDelta, RuntimeStepContext, RuntimeStepOutput,
};

pub struct Gemma4RuntimeAdapter {
    model: crane_core::models::gemma4::Model,
    kv_backend: Box<dyn KvCacheBackend>,
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
        let kv_backend = make_kv_backend(kv_config)?;
        Ok(Self { model, kv_backend })
    }
}

impl RuntimeModel for Gemma4RuntimeAdapter {
    fn prefill(&mut self, ctx: RuntimeRequestContext) -> Result<RuntimeStepOutput> {
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

    fn kv_extract(&self) -> LayerKvCaches {
        self.model
            .get_kv_caches()
            .into_iter()
            .enumerate()
            .map(|(layer_idx, dense)| {
                self.kv_backend
                    .export_layer(layer_idx, dense)
                    .unwrap_or_else(|err| {
                        panic!(
                            "Gemma4 KV export failed for layer {} with backend '{}': {err}",
                            layer_idx,
                            self.kv_backend.backend_id()
                        )
                    })
            })
            .collect()
    }

    fn kv_restore(&mut self, caches: LayerKvCaches) {
        let dense = caches
            .into_iter()
            .enumerate()
            .map(|(layer_idx, stored)| {
                self.kv_backend
                    .import_layer(layer_idx, stored, self.device(), self.dtype())
                    .unwrap_or_else(|err| {
                        panic!(
                            "Gemma4 KV restore failed for layer {} with backend '{}': {err}",
                            layer_idx,
                            self.kv_backend.backend_id()
                        )
                    })
            })
            .collect();
        self.model.set_kv_caches(dense);
    }

    fn kv_bytes(&self) -> u64 {
        self.model.active_kv_cache_bytes()
    }

    fn offload_under_memory_pressure(&mut self) -> usize {
        self.model.offload_experts_to_cpu()
    }
}
