use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::engine::backend::ModelBackend;
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
pub struct RuntimeStateDelta {
    pub consumed_tokens: usize,
}

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

    fn on_device_policy_tick(&mut self) {}
}

pub struct BackendRuntimeShim {
    backend: Box<dyn ModelBackend>,
}

impl BackendRuntimeShim {
    pub fn new(backend: Box<dyn ModelBackend>) -> Self {
        Self { backend }
    }

    pub fn into_backend(self) -> Box<dyn ModelBackend> {
        self.backend
    }
}

impl RuntimeModel for BackendRuntimeShim {
    fn prefill(&mut self, ctx: RuntimeRequestContext) -> Result<RuntimeStepOutput> {
        let logits = self.backend.forward_prefill(
            &ctx.input_ids,
            ctx.start_pos,
            &ctx.multimodal_inputs,
        )?;
        Ok(RuntimeStepOutput {
            logits,
            state_delta: RuntimeStateDelta {
                consumed_tokens: ctx.input_ids.len(),
            },
        })
    }

    fn decode(&mut self, ctx: RuntimeStepContext) -> Result<RuntimeStepOutput> {
        let logits = self.backend.forward_step(&ctx.input_ids, ctx.start_pos)?;
        Ok(RuntimeStepOutput {
            logits,
            state_delta: RuntimeStateDelta {
                consumed_tokens: ctx.input_ids.len(),
            },
        })
    }

    fn batch_decode(&mut self, ctx: BatchDecodeContext<'_>) -> candle_core::Result<Tensor> {
        self.backend.step_batch_decode(
            ctx.input_ids,
            ctx.positions,
            ctx.attention_mask,
            ctx.batch_kv_info,
        )
    }

    fn clear_kv_cache(&mut self) {
        self.backend.clear_kv_cache();
    }

    fn num_layers(&self) -> usize {
        self.backend.num_layers()
    }

    fn device(&self) -> &Device {
        self.backend.device()
    }

    fn dtype(&self) -> DType {
        self.backend.dtype()
    }

    fn tokenizer(&self) -> &tokenizers::Tokenizer {
        self.backend.tokenizer()
    }

    fn eos_token_id(&self) -> Vec<u32> {
        self.backend.eos_token_id()
    }

    fn warmup(&mut self) {
        self.backend.warmup();
    }

    fn supports_kv_swap(&self) -> bool {
        self.backend.supports_kv_swap()
    }

    fn kv_extract(&self) -> LayerKvCaches {
        self.backend.get_kv_caches()
    }

    fn kv_restore(&mut self, caches: LayerKvCaches) {
        self.backend.set_kv_caches(caches);
    }

    fn kv_bytes(&self) -> u64 {
        self.backend.active_kv_cache_bytes()
    }

    fn supports_batch_decode(&self) -> bool {
        self.backend.supports_batch_decode()
    }

    fn setup_batch_decode(
        &mut self,
        seq_kv_caches: &[LayerKvCaches],
        extra_room: usize,
    ) -> candle_core::Result<(Vec<usize>, usize)> {
        self.backend.setup_batch_decode(seq_kv_caches, extra_room)
    }

    fn extract_batch_kv(
        &mut self,
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
    ) -> candle_core::Result<SequenceKvCaches> {
        self.backend
            .extract_batch_kv(kv_lens, original_max_kv, rounds_done)
    }

    fn build_batch_decode_mask(
        &self,
        kv_lens: &[usize],
        original_max_kv: usize,
        max_total_width: usize,
    ) -> candle_core::Result<Option<Tensor>> {
        self.backend
            .build_batch_decode_mask(kv_lens, original_max_kv, max_total_width)
    }
}
