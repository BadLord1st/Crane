use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::engine::runtime::{
    BatchDecodeContext, Bf16PassthroughBackend, KvCacheBackend, LayerKvCaches, RuntimeModel,
    RuntimeRequestContext, RuntimeStateDelta, RuntimeStepContext, RuntimeStepOutput,
    SequenceKvCaches,
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

    fn kv_extract(&self) -> Result<LayerKvCaches> {
        let backend = Bf16PassthroughBackend;
        self.model
            .get_kv_caches()
            .into_iter()
            .enumerate()
            .map(|(layer_idx, dense)| {
                backend.export_layer(layer_idx, dense).map_err(|err| {
                    anyhow::anyhow!("Qwen3 KV export failed for layer {layer_idx}: {err}")
                })
            })
            .collect()
    }

    fn kv_restore(&mut self, caches: LayerKvCaches) -> Result<()> {
        let backend = Bf16PassthroughBackend;
        let dense = caches
            .into_iter()
            .enumerate()
            .map(|(layer_idx, stored)| {
                backend
                    .import_layer(layer_idx, stored, self.device(), self.dtype())
                    .map_err(|err| {
                        anyhow::anyhow!("Qwen3 KV restore failed for layer {layer_idx}: {err}")
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        self.model.set_kv_caches(dense);
        Ok(())
    }

    fn kv_bytes(&self) -> u64 {
        self.model.active_kv_cache_bytes()
    }

    fn supports_batch_decode(&self) -> bool {
        true
    }

    fn setup_batch_decode(
        &mut self,
        seq_kv_caches: &[LayerKvCaches],
        extra_room: usize,
    ) -> candle_core::Result<(Vec<usize>, usize)> {
        let backend = Bf16PassthroughBackend;
        let dense_caches = seq_kv_caches
            .iter()
            .map(|layers| {
                layers
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(layer_idx, stored)| {
                        backend
                            .import_layer(layer_idx, stored, self.device(), self.dtype())
                            .map_err(|err| candle_core::Error::Msg(err.to_string()))
                    })
                    .collect::<candle_core::Result<Vec<_>>>()
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        self.model.setup_batch_decode(&dense_caches, extra_room)
    }

    fn extract_batch_kv(
        &mut self,
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
    ) -> candle_core::Result<SequenceKvCaches> {
        let backend = Bf16PassthroughBackend;
        self.model
            .extract_batch_kv(kv_lens, original_max_kv, rounds_done)?
            .into_iter()
            .map(|layers| {
                layers
                    .into_iter()
                    .enumerate()
                    .map(|(layer_idx, dense)| {
                        backend
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
        crane_core::models::qwen3::modeling::build_batch_decode_mask(
            kv_lens,
            original_max_kv,
            max_total_width,
            self.device(),
            self.dtype(),
        )
    }
}
