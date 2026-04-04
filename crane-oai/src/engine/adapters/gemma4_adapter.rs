use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::engine::runtime::model_contract::LayerKvCaches;

use crate::engine::runtime::{
    RuntimeModel, RuntimeRequestContext, RuntimeStateDelta, RuntimeStepContext, RuntimeStepOutput,
};

pub struct Gemma4RuntimeAdapter {
    model: crane_core::models::gemma4::Model,
}

impl Gemma4RuntimeAdapter {
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        let load_dtype = match device {
            Device::Cpu => *dtype,
            _ => DType::BF16,
        };
        let model = crane_core::models::gemma4::Model::new(model_path, device, &load_dtype)?;
        Ok(Self { model })
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
        self.model.get_kv_caches()
    }

    fn kv_restore(&mut self, caches: LayerKvCaches) {
        self.model.set_kv_caches(caches);
    }

    fn kv_bytes(&self) -> u64 {
        self.model.active_kv_cache_bytes()
    }

    fn offload_under_memory_pressure(&mut self) -> usize {
        self.model.offload_experts_to_cpu()
    }
}
