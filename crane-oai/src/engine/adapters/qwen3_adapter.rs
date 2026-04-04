use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::engine::runtime::model_contract::{LayerKvCaches, SequenceKvCaches};
use crate::engine::runtime::{
    BatchDecodeContext, RuntimeModel, RuntimeRequestContext, RuntimeStateDelta, RuntimeStepContext,
    RuntimeStepOutput,
};

pub struct Qwen3RuntimeAdapter {
    model: crane_core::models::qwen3::Model,
}

impl Qwen3RuntimeAdapter {
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        let model = crane_core::models::qwen3::Model::new(model_path, device, dtype)?;
        Ok(Self { model })
    }
}

impl RuntimeModel for Qwen3RuntimeAdapter {
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

    fn batch_decode(&mut self, ctx: BatchDecodeContext<'_>) -> candle_core::Result<Tensor> {
        self.model.step_batch_decode_with_input_ids(
            ctx.input_ids,
            ctx.positions,
            ctx.attention_mask,
            ctx.batch_kv_info,
        )
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
        let tok = &self.model.tokenizer.tokenizer;
        let mut ids = Vec::new();
        if let Some(id) = tok.token_to_id("<|im_end|>") {
            ids.push(id);
        }
        if let Some(id) = tok.token_to_id("<|endoftext|>") {
            ids.push(id);
        }
        if ids.is_empty() {
            ids.push(151645);
            ids.push(151643);
        }
        ids
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

    fn offload_kv_caches(&self, caches: &mut LayerKvCaches) -> usize {
        if matches!(self.device(), Device::Cpu) {
            return 0;
        }

        let mut moved = 0usize;
        for cache in caches {
            let Some((k, v)) = cache else {
                continue;
            };
            if matches!(k.device(), Device::Cpu) {
                continue;
            }
            let Ok(new_k) = k.to_device(&Device::Cpu) else {
                continue;
            };
            let Ok(new_v) = v.to_device(&Device::Cpu) else {
                continue;
            };
            *k = new_k;
            *v = new_v;
            moved += 2;
        }
        moved
    }

    fn prefetch_kv_caches(&self, caches: &mut LayerKvCaches) -> usize {
        if matches!(self.device(), Device::Cpu) {
            return 0;
        }

        let mut moved = 0usize;
        for cache in caches {
            let Some((k, v)) = cache else {
                continue;
            };
            if !matches!(k.device(), Device::Cpu) {
                continue;
            }
            let Ok(new_k) = k.to_device(self.device()) else {
                continue;
            };
            let Ok(new_v) = v.to_device(self.device()) else {
                continue;
            };
            *k = new_k;
            *v = new_v;
            moved += 2;
        }
        moved
    }

    fn supports_batch_decode(&self) -> bool {
        true
    }

    fn setup_batch_decode(
        &mut self,
        seq_kv_caches: &[LayerKvCaches],
        extra_room: usize,
    ) -> candle_core::Result<(Vec<usize>, usize)> {
        self.model.setup_batch_decode(seq_kv_caches, extra_room)
    }

    fn extract_batch_kv(
        &mut self,
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
    ) -> candle_core::Result<SequenceKvCaches> {
        self.model
            .extract_batch_kv(kv_lens, original_max_kv, rounds_done)
    }

    fn build_batch_decode_mask(
        &self,
        kv_lens: &[usize],
        original_max_kv: usize,
        max_total_width: usize,
    ) -> candle_core::Result<Option<Tensor>> {
        crane_core::models::qwen3::modeling::build_batch_decode_mask(
            kv_lens,
            original_max_kv,
            max_total_width,
            self.device(),
            self.dtype(),
        )
    }
}
