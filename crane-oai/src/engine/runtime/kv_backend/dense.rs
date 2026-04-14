use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};

use super::config::KvCacheMode;
use super::turboquant::TurboQuantBackend;
use super::types::{
    DenseLayerKv, KvCacheBackend, KvLayerEnvelope, KvLayerPayload, TurboQuantValuePayload,
};

#[derive(Debug, Default, Clone, Copy)]
pub struct Bf16PassthroughBackend;

impl Bf16PassthroughBackend {
    pub(crate) fn dense_shapes(
        key: &Tensor,
        value: &Tensor,
    ) -> Result<(Vec<usize>, Vec<usize>, usize)> {
        let key_shape = key.shape().dims().to_vec();
        let value_shape = value.shape().dims().to_vec();
        if key_shape.len() < 3 || value_shape.len() < 3 {
            bail!(
                "Dense KV tensors must have rank >= 3, got key_rank={} value_rank={}",
                key_shape.len(),
                value_shape.len()
            );
        }

        let key_seq_len = key_shape[key_shape.len() - 2];
        let value_seq_len = value_shape[value_shape.len() - 2];
        if key_seq_len != value_seq_len {
            bail!(
                "Dense KV sequence length mismatch: key_seq_len={} value_seq_len={}",
                key_seq_len,
                value_seq_len
            );
        }

        Ok((key_shape, value_shape, key_seq_len))
    }
}

impl KvCacheBackend for Bf16PassthroughBackend {
    fn mode(&self) -> KvCacheMode {
        KvCacheMode::Bf16Dense
    }

    fn export_layer(
        &self,
        layer_idx: usize,
        dense: DenseLayerKv,
    ) -> Result<Option<KvLayerEnvelope>> {
        let Some((key, value)) = dense else {
            return Ok(None);
        };

        let (key_shape, value_shape, seq_len) = Self::dense_shapes(&key, &value)?;
        let dtype = key.dtype();
        if value.dtype() != dtype {
            bail!(
                "Dense KV dtype mismatch on export for layer {layer_idx}: key={:?} value={:?}",
                dtype,
                value.dtype()
            );
        }

        Ok(Some(KvLayerEnvelope {
            format: self.mode(),
            layer_idx,
            seq_len,
            key_shape,
            value_shape,
            dtype,
            payload: KvLayerPayload::Dense { key, value },
        }))
    }

    fn import_layer(
        &self,
        layer_idx: usize,
        stored: Option<KvLayerEnvelope>,
        target_device: &Device,
        target_dtype: DType,
    ) -> Result<DenseLayerKv> {
        let Some(stored) = stored else {
            return Ok(None);
        };

        self.validate_layer(layer_idx, &stored)?;
        match stored.payload {
            KvLayerPayload::Dense { key, value } => {
                let key = key.to_device(target_device)?.to_dtype(target_dtype)?;
                let value = value.to_device(target_device)?.to_dtype(target_dtype)?;
                Ok(Some((key, value)))
            }
            KvLayerPayload::Encoded { .. } => {
                bail!(
                    "KV backend '{}' cannot restore encoded payloads",
                    self.backend_id()
                )
            }
            KvLayerPayload::TurboQuant { .. } => bail!(
                "KV backend '{}' cannot restore turboquant payloads",
                self.backend_id()
            ),
        }
    }

    fn stored_bytes(&self, stored: &Option<KvLayerEnvelope>) -> u64 {
        match stored {
            Some(KvLayerEnvelope {
                payload: KvLayerPayload::Dense { key, value },
                ..
            }) => {
                (key.elem_count() as u64 * key.dtype().size_in_bytes() as u64)
                    + (value.elem_count() as u64 * value.dtype().size_in_bytes() as u64)
            }
            Some(KvLayerEnvelope {
                payload: KvLayerPayload::Encoded { bytes, .. },
                ..
            }) => bytes.len() as u64,
            Some(KvLayerEnvelope {
                payload: KvLayerPayload::TurboQuant { key, value },
                ..
            }) => {
                let value_dense_bytes = match value {
                    TurboQuantValuePayload::Dense(value) => {
                        value.elem_count() as u64 * value.dtype().size_in_bytes() as u64
                    }
                    TurboQuantValuePayload::RowwiseInt8 { scales, bytes, .. } => {
                        (scales.len() as u64 * std::mem::size_of::<f32>() as u64)
                            + bytes.len() as u64
                    }
                };
                TurboQuantBackend::key_payload_metadata_bytes(key) + value_dense_bytes
            }
            None => 0,
        }
    }

    fn validate_layer(&self, layer_idx: usize, stored: &KvLayerEnvelope) -> Result<()> {
        if stored.format != self.mode() {
            bail!(
                "KV layer {} uses format '{}' but backend '{}' was selected",
                layer_idx,
                stored.format,
                self.backend_id()
            );
        }

        if stored.layer_idx != layer_idx {
            bail!(
                "KV layer index mismatch: expected {}, stored {}",
                layer_idx,
                stored.layer_idx
            );
        }

        match &stored.payload {
            KvLayerPayload::Dense { key, value } => {
                let (key_shape, value_shape, seq_len) = Self::dense_shapes(key, value)?;
                if key_shape != stored.key_shape || value_shape != stored.value_shape {
                    bail!("Stored KV shape metadata mismatch for layer {}", layer_idx);
                }
                if seq_len != stored.seq_len {
                    bail!(
                        "Stored KV seq_len mismatch for layer {}: expected {} from tensors, envelope recorded {}",
                        layer_idx,
                        seq_len,
                        stored.seq_len
                    );
                }
                if key.dtype() != stored.dtype || value.dtype() != stored.dtype {
                    bail!(
                        "Stored KV dtype mismatch for layer {}: envelope={:?} key={:?} value={:?}",
                        layer_idx,
                        stored.dtype,
                        key.dtype(),
                        value.dtype()
                    );
                }
            }
            KvLayerPayload::Encoded { .. } => bail!(
                "KV backend '{}' cannot validate encoded payloads yet",
                self.backend_id()
            ),
            KvLayerPayload::TurboQuant { .. } => bail!(
                "KV backend '{}' cannot validate turboquant payloads",
                self.backend_id()
            ),
        }

        Ok(())
    }
}
