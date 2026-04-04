use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::engine::runtime::{
    BatchDecodeContext, RuntimeModel, RuntimeRequestContext, RuntimeStateDelta, RuntimeStepContext,
    RuntimeStepOutput,
};

pub struct Qwen25RuntimeAdapter {
    model: crane_core::models::qwen25::Model,
    dtype: DType,
}

impl Qwen25RuntimeAdapter {
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        let model = crane_core::models::qwen25::Model::new(model_path, device, dtype)?;
        Ok(Self {
            model,
            dtype: *dtype,
        })
    }
}

impl RuntimeModel for Qwen25RuntimeAdapter {
    fn prefill(&mut self, ctx: RuntimeRequestContext) -> Result<RuntimeStepOutput> {
        let logits = self.model.forward_step(&ctx.input_ids, ctx.start_pos)?;
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

    fn batch_decode(&mut self, _ctx: BatchDecodeContext<'_>) -> candle_core::Result<Tensor> {
        candle_core::bail!("Batch decode not supported by Qwen25RuntimeAdapter")
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
        self.dtype
    }

    fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.model.tokenizer.tokenizer
    }

    fn eos_token_id(&self) -> Vec<u32> {
        self.model
            .tokenizer
            .tokenizer
            .token_to_id("<|endoftext|>")
            .or_else(|| self.model.tokenizer.tokenizer.token_to_id("<|im_end|>"))
            .map(|id| vec![id])
            .unwrap_or_else(|| vec![151643])
    }

    fn warmup(&mut self) {
        self.model.warmup();
    }
}
