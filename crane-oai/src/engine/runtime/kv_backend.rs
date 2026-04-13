use std::fmt;
use std::str::FromStr;

use anyhow::{anyhow, bail, Result};
use candle_core::{DType, Device, Tensor};

pub type DenseLayerKv = Option<(Tensor, Tensor)>;
pub type LayerKvCaches = Vec<Option<KvLayerEnvelope>>;
pub type SequenceKvCaches = Vec<LayerKvCaches>;

const SUPPORTED_KV_CACHE_MODES: &[&str] = &["bf16_dense", "turboquant"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvCacheMode {
    #[default]
    Bf16Dense,
    TurboQuant,
}

impl KvCacheMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bf16Dense => "bf16_dense",
            Self::TurboQuant => "turboquant",
        }
    }

    fn parse_err(value: &str) -> anyhow::Error {
        anyhow!(
            "Unsupported KV cache mode '{value}'. Supported modes: {}",
            SUPPORTED_KV_CACHE_MODES.join("|")
        )
    }

    pub fn resolve(cli_value: Option<&str>, env_value: Option<&str>) -> Result<Self> {
        match cli_value.or(env_value) {
            Some(value) => Self::from_str(value),
            None => Ok(Self::default()),
        }
    }
}

impl fmt::Display for KvCacheMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for KvCacheMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "bf16_dense" => Ok(Self::Bf16Dense),
            "turboquant" => Ok(Self::TurboQuant),
            other => Err(Self::parse_err(other)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KvBackendConfig {
    pub mode: KvCacheMode,
}

impl KvBackendConfig {
    pub fn resolve(cli_value: Option<&str>, env_value: Option<&str>) -> Result<Self> {
        Ok(Self {
            mode: KvCacheMode::resolve(cli_value, env_value)?,
        })
    }
}

#[derive(Debug, Clone)]
pub enum KvLayerPayload {
    Dense { key: Tensor, value: Tensor },
    Encoded { bytes: Vec<u8> },
}

#[derive(Debug, Clone)]
pub struct KvLayerEnvelope {
    pub format: KvCacheMode,
    pub layer_idx: usize,
    pub seq_len: usize,
    pub key_shape: Vec<usize>,
    pub value_shape: Vec<usize>,
    pub dtype: DType,
    pub payload: KvLayerPayload,
}

pub trait KvCacheBackend: Send + Sync + 'static {
    fn mode(&self) -> KvCacheMode;

    fn backend_id(&self) -> &'static str {
        self.mode().as_str()
    }

    fn export_layer(
        &self,
        layer_idx: usize,
        dense: DenseLayerKv,
    ) -> Result<Option<KvLayerEnvelope>>;

    fn import_layer(
        &self,
        layer_idx: usize,
        stored: Option<KvLayerEnvelope>,
        target_device: &Device,
        target_dtype: DType,
    ) -> Result<DenseLayerKv>;

    fn stored_bytes(&self, stored: &Option<KvLayerEnvelope>) -> u64;

    fn validate_layer(&self, layer_idx: usize, stored: &KvLayerEnvelope) -> Result<()>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Bf16PassthroughBackend;

impl Bf16PassthroughBackend {
    fn dense_shapes(key: &Tensor, value: &Tensor) -> Result<(Vec<usize>, Vec<usize>, usize)> {
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
            KvLayerPayload::Encoded { .. } => bail!(
                "KV backend '{}' cannot restore encoded payloads",
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
                payload: KvLayerPayload::Encoded { bytes },
                ..
            }) => bytes.len() as u64,
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
            KvLayerPayload::Encoded { .. } => {
                bail!(
                    "KV backend '{}' cannot validate encoded payloads yet",
                    self.backend_id()
                );
            }
        }

        Ok(())
    }
}

pub fn make_kv_backend(config: KvBackendConfig) -> Result<Box<dyn KvCacheBackend>> {
    match config.mode {
        KvCacheMode::Bf16Dense => Ok(Box::new(Bf16PassthroughBackend)),
        KvCacheMode::TurboQuant => bail!(
            "KV cache mode 'turboquant' is reserved for future support and unsupported in this build"
        ),
    }
}

pub fn move_kv_caches_to_device(caches: &mut LayerKvCaches, target_device: &Device) -> usize {
    let mut moved = 0usize;
    for cache in caches {
        let Some(stored) = cache.as_mut() else {
            continue;
        };
        let KvLayerPayload::Dense { key, value } = &mut stored.payload else {
            continue;
        };
        let already_on_target = matches!((key.device(), target_device), (Device::Cpu, Device::Cpu))
            || (!matches!(key.device(), Device::Cpu) && !matches!(target_device, Device::Cpu));
        if already_on_target {
            continue;
        }

        let Ok(new_key) = key.to_device(target_device) else {
            continue;
        };
        let Ok(new_value) = value.to_device(target_device) else {
            continue;
        };

        *key = new_key;
        *value = new_value;
        moved += 2;
    }
    moved
}

pub fn stored_kv_cache_bytes(caches: &[Option<KvLayerEnvelope>]) -> u64 {
    caches
        .iter()
        .filter_map(|cache| cache.as_ref())
        .map(|stored| match &stored.payload {
            KvLayerPayload::Dense { key, value } => {
                let key_bytes = if matches!(key.device(), Device::Cpu) {
                    0
                } else {
                    key.elem_count() as u64 * key.dtype().size_in_bytes() as u64
                };
                let value_bytes = if matches!(value.device(), Device::Cpu) {
                    0
                } else {
                    value.elem_count() as u64 * value.dtype().size_in_bytes() as u64
                };
                key_bytes + value_bytes
            }
            KvLayerPayload::Encoded { .. } => 0,
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_cache_mode_round_trip() {
        assert_eq!(
            KvCacheMode::from_str("bf16_dense").unwrap(),
            KvCacheMode::Bf16Dense
        );
        assert_eq!(
            KvCacheMode::from_str("turboquant").unwrap(),
            KvCacheMode::TurboQuant
        );
        assert_eq!(KvCacheMode::Bf16Dense.to_string(), "bf16_dense");
        assert_eq!(KvCacheMode::TurboQuant.to_string(), "turboquant");
    }

    #[test]
    fn kv_cache_mode_prefers_cli_over_env() {
        let config = KvBackendConfig::resolve(Some("bf16_dense"), Some("turboquant")).unwrap();
        assert_eq!(config.mode, KvCacheMode::Bf16Dense);
    }

    #[test]
    fn kv_cache_mode_rejects_invalid_value() {
        let err = KvBackendConfig::resolve(None, Some("not-a-mode")).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("Unsupported KV cache mode 'not-a-mode'"));
        assert!(message.contains("bf16_dense|turboquant"));
    }

    #[test]
    fn bf16_passthrough_round_trips_dense_payload() {
        let backend = Bf16PassthroughBackend;
        let key = Tensor::zeros((1, 2, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let value = Tensor::zeros((1, 2, 3, 4), DType::F32, &Device::Cpu).unwrap();

        let stored = backend.export_layer(2, Some((key, value))).unwrap();
        assert_eq!(backend.stored_bytes(&stored), 192);

        let restored = backend
            .import_layer(2, stored, &Device::Cpu, DType::F32)
            .unwrap()
            .unwrap();

        assert_eq!(restored.0.dims4().unwrap(), (1, 2, 3, 4));
        assert_eq!(restored.1.dims4().unwrap(), (1, 2, 3, 4));
        assert_eq!(restored.0.dtype(), DType::F32);
        assert_eq!(restored.1.dtype(), DType::F32);
    }
}
