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
    Dense {
        key: Tensor,
        value: Tensor,
    },
    Encoded {
        encoding: KvEncodedPayloadEncoding,
        bytes: Vec<u8>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvEncodedPayloadEncoding {
    TurboQuantInt8V1,
}

impl fmt::Display for KvEncodedPayloadEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TurboQuantInt8V1 => f.write_str("turboquant_int8_v1"),
        }
    }
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

#[derive(Debug, Default, Clone, Copy)]
pub struct TurboQuantBackend;

impl TurboQuantBackend {
    const MAGIC: [u8; 8] = *b"CRKVTQ01";
    const HEADER_LEN: usize = 8 + 4 + 4 + 4 + 4;

    fn supported_dense_dtype(dtype: DType) -> bool {
        matches!(dtype, DType::F16 | DType::BF16 | DType::F32)
    }

    fn validate_dense_metadata(
        &self,
        layer_idx: usize,
        dtype: DType,
        key_shape: &[usize],
        value_shape: &[usize],
        seq_len: usize,
    ) -> Result<()> {
        if !Self::supported_dense_dtype(dtype) {
            bail!(
                "TurboQuant layer {} only supports floating-point KV tensors, got {:?}",
                layer_idx,
                dtype
            );
        }

        if key_shape.len() < 3 || value_shape.len() < 3 {
            bail!(
                "TurboQuant KV tensors must have rank >= 3, got key_rank={} value_rank={}",
                key_shape.len(),
                value_shape.len()
            );
        }

        let key_seq_len = key_shape[key_shape.len() - 2];
        let value_seq_len = value_shape[value_shape.len() - 2];
        if key_seq_len != value_seq_len || key_seq_len != seq_len {
            bail!(
                "TurboQuant KV seq_len mismatch for layer {}: key_seq_len={} value_seq_len={} envelope_seq_len={}",
                layer_idx,
                key_seq_len,
                value_seq_len,
                seq_len
            );
        }

        Ok(())
    }

    fn flatten_to_f32(tensor: &Tensor) -> Result<Vec<f32>> {
        Ok(tensor
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?)
    }

    fn quantize_values(values: &[f32]) -> (f32, Vec<i8>) {
        let max_abs = values
            .iter()
            .fold(0.0_f32, |acc, value| acc.max(value.abs()));
        if max_abs == 0.0 {
            return (0.0, vec![0; values.len()]);
        }

        let scale = max_abs / 127.0;
        let quants = values
            .iter()
            .map(|value| {
                let q = (value / scale).round().clamp(-127.0, 127.0);
                q as i8
            })
            .collect();
        (scale, quants)
    }

    fn dequantize_values(scale: f32, bytes: &[u8]) -> Vec<f32> {
        if scale == 0.0 {
            return vec![0.0; bytes.len()];
        }

        bytes
            .iter()
            .map(|byte| (*byte as i8) as f32 * scale)
            .collect()
    }

    fn encode_payload(
        key_values: &[f32],
        value_values: &[f32],
    ) -> (KvEncodedPayloadEncoding, Vec<u8>) {
        let (key_scale, key_quants) = Self::quantize_values(key_values);
        let (value_scale, value_quants) = Self::quantize_values(value_values);

        let mut bytes =
            Vec::with_capacity(Self::HEADER_LEN + key_quants.len() + value_quants.len());
        bytes.extend_from_slice(&Self::MAGIC);
        bytes.extend_from_slice(&(key_quants.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(value_quants.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&key_scale.to_le_bytes());
        bytes.extend_from_slice(&value_scale.to_le_bytes());
        bytes.extend(key_quants.into_iter().map(|value| value as u8));
        bytes.extend(value_quants.into_iter().map(|value| value as u8));

        (KvEncodedPayloadEncoding::TurboQuantInt8V1, bytes)
    }

    fn parse_payload<'a>(bytes: &'a [u8]) -> Result<(f32, f32, &'a [u8], &'a [u8])> {
        if bytes.len() < Self::HEADER_LEN {
            bail!(
                "TurboQuant payload too short: got {} bytes, need at least {}",
                bytes.len(),
                Self::HEADER_LEN
            );
        }

        if bytes[..Self::MAGIC.len()] != Self::MAGIC {
            bail!("TurboQuant payload magic mismatch")
        }

        let key_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let value_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let key_scale = f32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let value_scale = f32::from_le_bytes(bytes[20..24].try_into().unwrap());
        let expected_len = Self::HEADER_LEN + key_len + value_len;
        if bytes.len() != expected_len {
            bail!(
                "TurboQuant payload length mismatch: header expects {} bytes, got {}",
                expected_len,
                bytes.len()
            );
        }

        let key_start = Self::HEADER_LEN;
        let value_start = key_start + key_len;
        Ok((
            key_scale,
            value_scale,
            &bytes[key_start..value_start],
            &bytes[value_start..],
        ))
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
                payload: KvLayerPayload::Encoded { bytes, .. },
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

impl KvCacheBackend for TurboQuantBackend {
    fn mode(&self) -> KvCacheMode {
        KvCacheMode::TurboQuant
    }

    fn export_layer(
        &self,
        layer_idx: usize,
        dense: DenseLayerKv,
    ) -> Result<Option<KvLayerEnvelope>> {
        let Some((key, value)) = dense else {
            return Ok(None);
        };

        let (key_shape, value_shape, seq_len) = Bf16PassthroughBackend::dense_shapes(&key, &value)?;
        let dtype = key.dtype();
        if value.dtype() != dtype {
            bail!(
                "Dense KV dtype mismatch on TurboQuant export for layer {layer_idx}: key={:?} value={:?}",
                dtype,
                value.dtype()
            );
        }
        self.validate_dense_metadata(layer_idx, dtype, &key_shape, &value_shape, seq_len)?;

        let key_values = Self::flatten_to_f32(&key)?;
        let value_values = Self::flatten_to_f32(&value)?;
        let (encoding, bytes) = Self::encode_payload(&key_values, &value_values);

        Ok(Some(KvLayerEnvelope {
            format: self.mode(),
            layer_idx,
            seq_len,
            key_shape,
            value_shape,
            dtype,
            payload: KvLayerPayload::Encoded { encoding, bytes },
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
            KvLayerPayload::Encoded { bytes, .. } => {
                let (key_scale, value_scale, key_bytes, value_bytes) = Self::parse_payload(&bytes)?;
                let key_values = Self::dequantize_values(key_scale, key_bytes);
                let value_values = Self::dequantize_values(value_scale, value_bytes);

                let key = Tensor::from_vec(key_values, stored.key_shape.as_slice(), &Device::Cpu)?
                    .to_device(target_device)?
                    .to_dtype(target_dtype)?;
                let value =
                    Tensor::from_vec(value_values, stored.value_shape.as_slice(), &Device::Cpu)?
                        .to_device(target_device)?
                        .to_dtype(target_dtype)?;
                Ok(Some((key, value)))
            }
            KvLayerPayload::Dense { .. } => bail!(
                "KV backend '{}' expected encoded TurboQuant payloads",
                self.backend_id()
            ),
        }
    }

    fn stored_bytes(&self, stored: &Option<KvLayerEnvelope>) -> u64 {
        match stored {
            Some(KvLayerEnvelope {
                payload: KvLayerPayload::Encoded { bytes, .. },
                ..
            }) => bytes.len() as u64,
            Some(KvLayerEnvelope {
                payload: KvLayerPayload::Dense { key, value },
                ..
            }) => {
                (key.elem_count() as u64 * key.dtype().size_in_bytes() as u64)
                    + (value.elem_count() as u64 * value.dtype().size_in_bytes() as u64)
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

        self.validate_dense_metadata(
            layer_idx,
            stored.dtype,
            &stored.key_shape,
            &stored.value_shape,
            stored.seq_len,
        )?;

        match &stored.payload {
            KvLayerPayload::Encoded { encoding, bytes } => {
                if *encoding != KvEncodedPayloadEncoding::TurboQuantInt8V1 {
                    bail!(
                        "TurboQuant layer {} uses unsupported encoding '{}'",
                        layer_idx,
                        encoding
                    );
                }

                let (_key_scale, _value_scale, key_bytes, value_bytes) =
                    Self::parse_payload(bytes)?;
                let expected_key_len = stored.key_shape.iter().product::<usize>();
                let expected_value_len = stored.value_shape.iter().product::<usize>();
                if key_bytes.len() != expected_key_len || value_bytes.len() != expected_value_len {
                    bail!(
                        "TurboQuant payload element-count mismatch for layer {}: key={} value={} expected_key={} expected_value={}",
                        layer_idx,
                        key_bytes.len(),
                        value_bytes.len(),
                        expected_key_len,
                        expected_value_len
                    );
                }
            }
            KvLayerPayload::Dense { .. } => bail!(
                "KV backend '{}' expected encoded TurboQuant payloads",
                self.backend_id()
            ),
        }

        Ok(())
    }
}

pub fn make_kv_backend(config: KvBackendConfig) -> Result<Box<dyn KvCacheBackend>> {
    match config.mode {
        KvCacheMode::Bf16Dense => Ok(Box::new(Bf16PassthroughBackend)),
        KvCacheMode::TurboQuant => Ok(Box::new(TurboQuantBackend)),
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

    #[test]
    fn turboquant_backend_factory_is_real() {
        let backend = make_kv_backend(KvBackendConfig {
            mode: KvCacheMode::TurboQuant,
        })
        .unwrap();
        assert_eq!(backend.backend_id(), "turboquant");
    }

    #[test]
    fn turboquant_round_trips_shapes_and_approximately_restores_values() {
        let backend = TurboQuantBackend;
        let key_values = vec![-1.0_f32, -0.5, 0.0, 0.5, 1.0, 0.25, -0.25, 0.75];
        let value_values = vec![0.9_f32, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2];
        let key = Tensor::from_vec(key_values.clone(), (1, 1, 2, 4), &Device::Cpu).unwrap();
        let value = Tensor::from_vec(value_values.clone(), (1, 1, 2, 4), &Device::Cpu).unwrap();

        let stored = backend.export_layer(1, Some((key, value))).unwrap();
        let stored = stored.unwrap();
        assert_eq!(stored.format, KvCacheMode::TurboQuant);
        match &stored.payload {
            KvLayerPayload::Encoded { encoding, bytes } => {
                assert_eq!(*encoding, KvEncodedPayloadEncoding::TurboQuantInt8V1);
                assert_eq!(bytes.len(), TurboQuantBackend::HEADER_LEN + 16);
            }
            KvLayerPayload::Dense { .. } => panic!("expected encoded payload"),
        }

        let restored = backend
            .import_layer(1, Some(stored), &Device::Cpu, DType::F32)
            .unwrap()
            .unwrap();

        assert_eq!(restored.0.dims4().unwrap(), (1, 1, 2, 4));
        assert_eq!(restored.1.dims4().unwrap(), (1, 1, 2, 4));

        let restored_key = restored.0.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let restored_value = restored.1.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let key_scale = 1.0_f32 / 127.0;
        let value_scale = 0.9_f32 / 127.0;
        for (expected, actual) in key_values.iter().zip(restored_key.iter()) {
            assert!((expected - actual).abs() <= key_scale + 1e-6);
        }
        for (expected, actual) in value_values.iter().zip(restored_value.iter()) {
            assert!((expected - actual).abs() <= value_scale + 1e-6);
        }
    }

    #[test]
    fn turboquant_validate_rejects_shape_metadata_drift() {
        let backend = TurboQuantBackend;
        let key = Tensor::zeros((1, 1, 2, 4), DType::BF16, &Device::Cpu).unwrap();
        let value = Tensor::zeros((1, 1, 2, 4), DType::BF16, &Device::Cpu).unwrap();
        let mut stored = backend
            .export_layer(3, Some((key, value)))
            .unwrap()
            .unwrap();
        stored.key_shape = vec![1, 1, 3, 4];

        let err = backend.validate_layer(3, &stored).unwrap_err();
        assert!(err.to_string().contains("element-count mismatch"));
    }
}
