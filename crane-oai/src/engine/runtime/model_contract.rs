use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::engine::types::MultimodalInputs;

pub type LayerKv = Option<(Tensor, Tensor)>;
pub type LayerKvCaches = Vec<LayerKv>;
pub type SequenceKvCaches = Vec<LayerKvCaches>;

#[derive(Debug, Clone)]
pub struct RuntimeRequestContext {
    pub input_ids: Vec<u32>,
    pub start_pos: usize,
    pub multimodal_inputs: MultimodalInputs,
}

#[derive(Debug, Clone)]
pub struct RuntimeStepContext {
    pub input_ids: Vec<u32>,
    pub start_pos: usize,
}

pub struct BatchDecodeContext<'a> {
    pub input_ids: &'a Tensor,
    pub positions: &'a [usize],
    pub attention_mask: Option<&'a Tensor>,
    pub batch_kv_info: Option<(&'a [usize], usize)>,
}

#[derive(Debug, Clone, Copy, Default)]
#[allow(dead_code)]
pub struct RuntimeStateDelta {
    pub consumed_tokens: usize,
}

#[allow(dead_code)]
pub struct RuntimeStepOutput {
    pub logits: Tensor,
    pub state_delta: RuntimeStateDelta,
}

pub trait RuntimeModel: Send + 'static {
    fn prefill(&mut self, ctx: RuntimeRequestContext) -> Result<RuntimeStepOutput>;

    fn decode(&mut self, ctx: RuntimeStepContext) -> Result<RuntimeStepOutput>;

    fn batch_decode(&mut self, ctx: BatchDecodeContext<'_>) -> candle_core::Result<Tensor>;

    fn clear_kv_cache(&mut self);
    fn num_layers(&self) -> usize;
    fn device(&self) -> &Device;
    fn dtype(&self) -> DType;
    fn tokenizer(&self) -> &tokenizers::Tokenizer;
    fn eos_token_id(&self) -> Vec<u32>;
    fn warmup(&mut self);

    fn supports_kv_swap(&self) -> bool {
        false
    }

    fn kv_extract(&self) -> LayerKvCaches {
        vec![]
    }

    fn kv_restore(&mut self, _caches: LayerKvCaches) {}

    fn kv_bytes(&self) -> u64 {
        0
    }

    /// Called by the engine when memory pressure is detected.
    /// Runtime backend decides what to offload (experts, blocks, caches, etc.).
    /// Returns number of offloaded units.
    fn offload_under_memory_pressure(&mut self) -> usize {
        0
    }

    /// Move inactive sequence KV caches away from model device.
    /// Returns number of tensors moved.
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

    /// Move sequence KV caches onto model device before restore/decode.
    /// Returns number of tensors moved.
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
        false
    }

    fn setup_batch_decode(
        &mut self,
        _seq_kv_caches: &[LayerKvCaches],
        _extra_room: usize,
    ) -> candle_core::Result<(Vec<usize>, usize)> {
        candle_core::bail!("Batch decode not supported by this runtime")
    }

    fn extract_batch_kv(
        &mut self,
        _kv_lens: &[usize],
        _original_max_kv: usize,
        _rounds_done: usize,
    ) -> candle_core::Result<SequenceKvCaches> {
        candle_core::bail!("Batch decode not supported by this runtime")
    }

    fn build_batch_decode_mask(
        &self,
        _kv_lens: &[usize],
        _original_max_kv: usize,
        _max_total_width: usize,
    ) -> candle_core::Result<Option<Tensor>> {
        candle_core::bail!("Batch decode not supported by this runtime")
    }

    #[allow(dead_code)]
    fn on_device_policy_tick(&mut self) {}
}
