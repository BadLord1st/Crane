use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::engine::runtime::{
    BatchDecodeContext, Bf16PassthroughBackend, KvCacheBackend, LayerKvCaches, RuntimeModel,
    RuntimeRequestContext, RuntimeStateDelta, RuntimeStepContext, RuntimeStepOutput,
    SequenceKvCaches,
};

pub struct HunyuanRuntimeAdapter {
    model: crane_core::models::hunyuan_dense::Model,
}

impl HunyuanRuntimeAdapter {
    pub fn new(
        model_path: &str,
        device: &Device,
        dtype: &DType,
        format: crane_core::models::hunyuan_dense::ModelFormat,
    ) -> Result<Self> {
        let model = crane_core::models::hunyuan_dense::Model::new_with_format(
            model_path, device, dtype, format,
        )?;
        Ok(Self { model })
    }
}

impl RuntimeModel for HunyuanRuntimeAdapter {
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
        vec![120020]
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
                    anyhow::anyhow!("Hunyuan KV export failed for layer {layer_idx}: {err}")
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
                        anyhow::anyhow!("Hunyuan KV restore failed for layer {layer_idx}: {err}")
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
        crane_core::models::hunyuan_dense::modeling::build_batch_decode_mask(
            kv_lens,
            original_max_kv,
            max_total_width,
            self.device(),
            self.dtype(),
        )
    }
}
