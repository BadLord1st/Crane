use std::fmt;
use std::str::FromStr;

use anyhow::{anyhow, bail, Result};
use candle_core::{DType, Device, Tensor};

pub type DenseLayerKv = Option<(Tensor, Tensor)>;
pub type LayerKvCaches = Vec<Option<KvLayerEnvelope>>;
pub type SequenceKvCaches = Vec<LayerKvCaches>;

const SUPPORTED_KV_CACHE_MODES: &[&str] = &["bf16_dense", "int8_rowwise_kv", "turboquant"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvCacheMode {
    #[default]
    Bf16Dense,
    Int8RowwiseKv,
    TurboQuant,
}

impl KvCacheMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bf16Dense => "bf16_dense",
            Self::Int8RowwiseKv => "int8_rowwise_kv",
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
            "int8_rowwise_kv" => Ok(Self::Int8RowwiseKv),
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
    TurboQuant {
        key: TurboQuantKeyPayload,
        value: TurboQuantValuePayload,
    },
}

#[derive(Debug, Clone)]
pub struct TurboQuantKeyPayload {
    pub row_width: usize,
    pub sketch_dim: usize,
    pub magnitudes: Vec<f32>,
    pub sign_bits: Vec<u8>,
    pub residual_sketch: Vec<f32>,
    pub dense_fallback: Tensor,
}

#[derive(Debug, Clone)]
pub enum TurboQuantValuePayload {
    Dense(Tensor),
    RowwiseInt8 {
        row_width: usize,
        scales: Vec<f32>,
        bytes: Vec<u8>,
        prepared_rows_f32: Tensor,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvEncodedPayloadEncoding {
    LegacyTurboQuantInt8V1,
    Int8RowwiseKvV1,
}

impl fmt::Display for KvEncodedPayloadEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LegacyTurboQuantInt8V1 => f.write_str("legacy_turboquant_int8_v1"),
            Self::Int8RowwiseKvV1 => f.write_str("int8_rowwise_kv_v1"),
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

    fn supports_compressed_k_scores(&self) -> bool {
        false
    }

    fn score_query_against_stored_keys(
        &self,
        _query: &Tensor,
        _stored: &KvLayerEnvelope,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    fn weighted_value_prefix(
        &self,
        _attn_weights: &Tensor,
        _stored: &KvLayerEnvelope,
        _num_kv_heads: usize,
        _num_kv_groups: usize,
        _target_device: &Device,
        _target_dtype: DType,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }
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
pub struct Int8RowwiseKvBackend;

#[derive(Debug, Default, Clone, Copy)]
pub struct TurboQuantBackend;

impl Int8RowwiseKvBackend {
    const MAGIC: [u8; 8] = *b"CRKVTQ01";
    const ROWWISE_MAGIC: [u8; 8] = *b"CRKVRW01";
    const LEGACY_HEADER_LEN: usize = 8 + 4 + 4 + 4 + 4;
    const ROWWISE_HEADER_LEN: usize = 8 + 4 + 4 + 4 + 4;

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
                "int8_rowwise_kv layer {} only supports floating-point KV tensors, got {:?}",
                layer_idx,
                dtype
            );
        }

        if key_shape.len() < 3 || value_shape.len() < 3 {
            bail!(
                "int8_rowwise_kv KV tensors must have rank >= 3, got key_rank={} value_rank={}",
                key_shape.len(),
                value_shape.len()
            );
        }

        let key_seq_len = key_shape[key_shape.len() - 2];
        let value_seq_len = value_shape[value_shape.len() - 2];
        if key_seq_len != value_seq_len || key_seq_len != seq_len {
            bail!(
                "int8_rowwise_kv KV seq_len mismatch for layer {}: key_seq_len={} value_seq_len={} envelope_seq_len={}",
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

    fn quantize_rowwise(values: &[f32], row_width: usize) -> Result<(Vec<f32>, Vec<i8>)> {
        if row_width == 0 {
            bail!("int8_rowwise_kv row width must be > 0");
        }
        if values.len() % row_width != 0 {
            bail!(
                "int8_rowwise_kv rowwise quantization requires element_count={} to be divisible by row_width={}",
                values.len(),
                row_width
            );
        }

        let mut scales = Vec::with_capacity(values.len() / row_width);
        let mut quants = Vec::with_capacity(values.len());
        for row in values.chunks(row_width) {
            let (scale, row_quants) = Self::quantize_values(row);
            scales.push(scale);
            quants.extend(row_quants);
        }
        Ok((scales, quants))
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

    fn dequantize_rowwise(scales: &[f32], row_width: usize, bytes: &[u8]) -> Result<Vec<f32>> {
        if row_width == 0 {
            bail!("int8_rowwise_kv row width must be > 0");
        }
        if bytes.len() % row_width != 0 {
            bail!(
                "int8_rowwise_kv payload element_count={} is not divisible by row_width={}",
                bytes.len(),
                row_width
            );
        }
        let row_count = bytes.len() / row_width;
        if scales.len() != row_count {
            bail!(
                "int8_rowwise_kv scale count mismatch: got {} scales for {} rows",
                scales.len(),
                row_count
            );
        }

        let mut values = Vec::with_capacity(bytes.len());
        for (scale, row_bytes) in scales.iter().zip(bytes.chunks(row_width)) {
            values.extend(Self::dequantize_values(*scale, row_bytes));
        }
        Ok(values)
    }

    fn encode_payload(
        key_values: &[f32],
        key_row_width: usize,
        value_values: &[f32],
        value_row_width: usize,
    ) -> Result<(KvEncodedPayloadEncoding, Vec<u8>)> {
        let (key_scales, key_quants) = Self::quantize_rowwise(key_values, key_row_width)?;
        let (value_scales, value_quants) = Self::quantize_rowwise(value_values, value_row_width)?;

        let mut bytes = Vec::with_capacity(
            Self::ROWWISE_HEADER_LEN
                + (key_scales.len() + value_scales.len()) * std::mem::size_of::<f32>()
                + key_quants.len()
                + value_quants.len(),
        );
        bytes.extend_from_slice(&Self::ROWWISE_MAGIC);
        bytes.extend_from_slice(&(key_quants.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(value_quants.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(key_row_width as u32).to_le_bytes());
        bytes.extend_from_slice(&(value_row_width as u32).to_le_bytes());
        for scale in &key_scales {
            bytes.extend_from_slice(&scale.to_le_bytes());
        }
        for scale in &value_scales {
            bytes.extend_from_slice(&scale.to_le_bytes());
        }
        bytes.extend(key_quants.into_iter().map(|value| value as u8));
        bytes.extend(value_quants.into_iter().map(|value| value as u8));

        Ok((KvEncodedPayloadEncoding::Int8RowwiseKvV1, bytes))
    }

    fn parse_legacy_payload<'a>(bytes: &'a [u8]) -> Result<(f32, f32, &'a [u8], &'a [u8])> {
        if bytes.len() < Self::LEGACY_HEADER_LEN {
            bail!(
                "TurboQuant payload too short: got {} bytes, need at least {}",
                bytes.len(),
                Self::LEGACY_HEADER_LEN
            );
        }

        if bytes[..Self::MAGIC.len()] != Self::MAGIC {
            bail!("TurboQuant payload magic mismatch")
        }

        let key_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let value_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let key_scale = f32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let value_scale = f32::from_le_bytes(bytes[20..24].try_into().unwrap());
        let expected_len = Self::LEGACY_HEADER_LEN + key_len + value_len;
        if bytes.len() != expected_len {
            bail!(
                "TurboQuant payload length mismatch: header expects {} bytes, got {}",
                expected_len,
                bytes.len()
            );
        }

        let key_start = Self::LEGACY_HEADER_LEN;
        let value_start = key_start + key_len;
        Ok((
            key_scale,
            value_scale,
            &bytes[key_start..value_start],
            &bytes[value_start..],
        ))
    }

    fn read_u32(bytes: &[u8], start: usize) -> u32 {
        u32::from_le_bytes(bytes[start..start + 4].try_into().unwrap())
    }

    fn parse_scales(bytes: &[u8], offset: usize, count: usize) -> Result<(Vec<f32>, usize)> {
        let byte_len = count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow!("int8_rowwise_kv scale metadata overflow"))?;
        let end = offset
            .checked_add(byte_len)
            .ok_or_else(|| anyhow!("int8_rowwise_kv scale payload overflow"))?;
        if end > bytes.len() {
            bail!(
                "int8_rowwise_kv payload too short for {} row scales: need {} bytes, got {}",
                count,
                end,
                bytes.len()
            );
        }

        let mut scales = Vec::with_capacity(count);
        let mut cursor = offset;
        for _ in 0..count {
            scales.push(f32::from_le_bytes(
                bytes[cursor..cursor + 4].try_into().unwrap(),
            ));
            cursor += 4;
        }
        Ok((scales, cursor))
    }

    fn row_count(element_count: usize, row_width: usize) -> Result<usize> {
        if row_width == 0 {
            bail!("int8_rowwise_kv row width must be > 0");
        }
        if element_count % row_width != 0 {
            bail!(
                "int8_rowwise_kv element_count={} is not divisible by row_width={}",
                element_count,
                row_width
            );
        }
        Ok(element_count / row_width)
    }

    fn parse_rowwise_payload<'a>(
        bytes: &'a [u8],
    ) -> Result<(Vec<f32>, usize, Vec<f32>, usize, &'a [u8], &'a [u8])> {
        if bytes.len() < Self::ROWWISE_HEADER_LEN {
            bail!(
                "int8_rowwise_kv payload too short: got {} bytes, need at least {}",
                bytes.len(),
                Self::ROWWISE_HEADER_LEN
            );
        }

        if bytes[..Self::ROWWISE_MAGIC.len()] != Self::ROWWISE_MAGIC {
            bail!("int8_rowwise_kv payload magic mismatch")
        }

        let key_len = Self::read_u32(bytes, 8) as usize;
        let value_len = Self::read_u32(bytes, 12) as usize;
        let key_row_width = Self::read_u32(bytes, 16) as usize;
        let value_row_width = Self::read_u32(bytes, 20) as usize;
        let key_rows = Self::row_count(key_len, key_row_width)?;
        let value_rows = Self::row_count(value_len, value_row_width)?;

        let (key_scales, after_key_scales) =
            Self::parse_scales(bytes, Self::ROWWISE_HEADER_LEN, key_rows)?;
        let (value_scales, payload_start) =
            Self::parse_scales(bytes, after_key_scales, value_rows)?;
        let expected_len = payload_start
            .checked_add(key_len)
            .and_then(|n| n.checked_add(value_len))
            .ok_or_else(|| anyhow!("int8_rowwise_kv payload length overflow"))?;
        if bytes.len() != expected_len {
            bail!(
                "int8_rowwise_kv payload length mismatch: header expects {} bytes, got {}",
                expected_len,
                bytes.len()
            );
        }

        let key_start = payload_start;
        let value_start = key_start + key_len;
        Ok((
            key_scales,
            key_row_width,
            value_scales,
            value_row_width,
            &bytes[key_start..value_start],
            &bytes[value_start..],
        ))
    }
}

impl TurboQuantBackend {
    fn validate_dense_metadata(
        &self,
        layer_idx: usize,
        dtype: DType,
        key_shape: &[usize],
        value_shape: &[usize],
        seq_len: usize,
    ) -> Result<()> {
        Int8RowwiseKvBackend.validate_dense_metadata(
            layer_idx,
            dtype,
            key_shape,
            value_shape,
            seq_len,
        )
    }

    fn row_width(key_shape: &[usize]) -> Result<usize> {
        let row_width = *key_shape
            .last()
            .ok_or_else(|| anyhow!("turboquant key tensor must have rank >= 1"))?;
        if row_width == 0 {
            bail!("turboquant key row width must be > 0");
        }
        Ok(row_width)
    }

    fn row_count(key_shape: &[usize]) -> Result<usize> {
        let row_width = Self::row_width(key_shape)?;
        let element_count = key_shape.iter().product::<usize>();
        if element_count % row_width != 0 {
            bail!(
                "turboquant key element_count={} is not divisible by row_width={}",
                element_count,
                row_width
            );
        }
        Ok(element_count / row_width)
    }

    fn sketch_dim_for_row_width(row_width: usize) -> usize {
        (row_width / 2).clamp(1, 16)
    }

    fn packed_sign_bit_len(row_count: usize, row_width: usize) -> usize {
        (row_count * row_width).div_ceil(8)
    }

    fn sign_hash(sketch_idx: usize, dim_idx: usize) -> f32 {
        let mut x = (sketch_idx as u64 + 1)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add((dim_idx as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9));
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        if (x >> 63) == 0 {
            1.0
        } else {
            -1.0
        }
    }

    fn sign_bit(sign_bits: &[u8], bit_idx: usize) -> Result<f32> {
        let byte_idx = bit_idx / 8;
        let bit_mask = 1u8 << (bit_idx % 8);
        let byte = *sign_bits
            .get(byte_idx)
            .ok_or_else(|| anyhow!("turboquant sign bit index {} out of range", bit_idx))?;
        Ok(if byte & bit_mask == 0 { -1.0 } else { 1.0 })
    }

    fn pack_sign_bits(rows: &[f32], row_width: usize) -> Vec<u8> {
        let mut bits = vec![0u8; Self::packed_sign_bit_len(rows.len() / row_width, row_width)];
        for (idx, value) in rows.iter().enumerate() {
            if *value >= 0.0 {
                bits[idx / 8] |= 1u8 << (idx % 8);
            }
        }
        bits
    }

    fn project_vector(values: &[f32], sketch_dim: usize) -> Vec<f32> {
        let scale = 1.0 / (sketch_dim as f32).sqrt();
        (0..sketch_dim)
            .map(|sketch_idx| {
                values
                    .iter()
                    .enumerate()
                    .map(|(dim_idx, value)| Self::sign_hash(sketch_idx, dim_idx) * *value * scale)
                    .sum()
            })
            .collect()
    }

    fn encode_key_payload(
        &self,
        key: &Tensor,
        key_shape: &[usize],
    ) -> Result<TurboQuantKeyPayload> {
        let row_width = Self::row_width(key_shape)?;
        let key_values = Int8RowwiseKvBackend::flatten_to_f32(key)?;
        let row_count = key_values.len() / row_width;
        let sketch_dim = Self::sketch_dim_for_row_width(row_width);
        let sign_bits = Self::pack_sign_bits(&key_values, row_width);
        let mut magnitudes = Vec::with_capacity(row_count);
        let mut residual_sketch = Vec::with_capacity(row_count * sketch_dim);

        for row in key_values.chunks(row_width) {
            let magnitude = row.iter().map(|value| value.abs()).sum::<f32>() / row_width as f32;
            magnitudes.push(magnitude);

            let mut residual = Vec::with_capacity(row_width);
            for value in row.iter() {
                let sign = if *value >= 0.0 { 1.0 } else { -1.0 };
                let base = sign * magnitude;
                residual.push(*value - base);
            }
            residual_sketch.extend(Self::project_vector(&residual, sketch_dim));
        }

        Ok(TurboQuantKeyPayload {
            row_width,
            sketch_dim,
            magnitudes,
            sign_bits,
            residual_sketch,
            dense_fallback: key.clone(),
        })
    }

    fn encode_value_payload(
        &self,
        value: &Tensor,
        value_shape: &[usize],
    ) -> Result<TurboQuantValuePayload> {
        let row_width = *value_shape
            .last()
            .ok_or_else(|| anyhow!("turboquant value tensor must have rank >= 1"))?;
        let value_values = Int8RowwiseKvBackend::flatten_to_f32(value)?;
        let (scales, quants) = Int8RowwiseKvBackend::quantize_rowwise(&value_values, row_width)?;
        Ok(TurboQuantValuePayload::RowwiseInt8 {
            row_width,
            scales,
            bytes: quants.into_iter().map(|value| value as u8).collect(),
            prepared_rows_f32: value
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .reshape((value_shape[1], value_shape[2], row_width))?,
        })
    }

    fn decode_value_payload(
        value: &TurboQuantValuePayload,
        value_shape: &[usize],
    ) -> Result<Vec<f32>> {
        let values = match value {
            TurboQuantValuePayload::Dense(value) => Int8RowwiseKvBackend::flatten_to_f32(value),
            TurboQuantValuePayload::RowwiseInt8 {
                row_width,
                scales,
                bytes,
                ..
            } => Int8RowwiseKvBackend::dequantize_rowwise(scales, *row_width, bytes),
        }?;

        let expected_len = value_shape.iter().product::<usize>();
        if values.len() != expected_len {
            bail!(
                "turboquant value payload length mismatch: got {} values, expected {}",
                values.len(),
                expected_len
            );
        }
        Ok(values)
    }

    fn weighted_value_from_rowwise_payload_reference(
        &self,
        attn_weights: &Tensor,
        value_shape: &[usize],
        value: &TurboQuantValuePayload,
        num_kv_heads: usize,
        num_kv_groups: usize,
        target_device: &Device,
        target_dtype: DType,
    ) -> Result<Option<Tensor>> {
        let TurboQuantValuePayload::RowwiseInt8 {
            row_width,
            scales,
            bytes,
            ..
        } = value
        else {
            return Ok(None);
        };

        if value_shape.len() != 4 || value_shape[0] != 1 {
            return Ok(None);
        }

        let prefix_len = value_shape[2];
        let stored_kv_heads = value_shape[1];
        if stored_kv_heads != num_kv_heads || value_shape[3] != *row_width {
            return Ok(None);
        }

        let (batch, num_heads, q_len, weight_prefix_len) = attn_weights.dims4()?;
        if batch != 1
            || q_len == 0
            || num_heads != num_kv_heads * num_kv_groups
            || weight_prefix_len != prefix_len
        {
            return Ok(None);
        }

        let expected_rows = stored_kv_heads * prefix_len;
        if scales.len() != expected_rows || bytes.len() != expected_rows * row_width {
            bail!("turboquant rowwise V payload metadata mismatch");
        }

        let weights = attn_weights
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .reshape((num_heads, q_len, prefix_len))?
            .to_vec3::<f32>()?;
        let mut aggregated = vec![0.0_f32; num_heads * q_len * row_width];
        for head_idx in 0..num_heads {
            let kv_head_idx = head_idx / num_kv_groups;
            for q_idx in 0..q_len {
                let output_offset = (head_idx * q_len + q_idx) * row_width;
                let out_row = &mut aggregated[output_offset..output_offset + row_width];
                for pos_idx in 0..prefix_len {
                    let weight = weights[head_idx][q_idx][pos_idx];
                    if weight == 0.0 {
                        continue;
                    }
                    let row_idx = kv_head_idx * prefix_len + pos_idx;
                    let scale = scales[row_idx];
                    if scale == 0.0 {
                        continue;
                    }
                    let row_bytes = &bytes[row_idx * row_width..(row_idx + 1) * row_width];
                    for (dst, quantized) in out_row.iter_mut().zip(row_bytes.iter()) {
                        *dst += weight * ((*quantized as i8) as f32 * scale);
                    }
                }
            }
        }

        Ok(Some(
            Tensor::from_vec(aggregated, (1, num_heads, q_len, *row_width), &Device::Cpu)?
                .to_device(target_device)?
                .to_dtype(target_dtype)?,
        ))
    }

    fn weighted_value_from_rowwise_payload_hot_path(
        &self,
        attn_weights: &Tensor,
        value_shape: &[usize],
        value: &TurboQuantValuePayload,
        num_kv_heads: usize,
        num_kv_groups: usize,
        target_device: &Device,
        target_dtype: DType,
    ) -> Result<Option<Tensor>> {
        let TurboQuantValuePayload::RowwiseInt8 {
            row_width,
            prepared_rows_f32,
            ..
        } = value
        else {
            return Ok(None);
        };

        if value_shape.len() != 4 || value_shape[0] != 1 {
            return Ok(None);
        }

        let prefix_len = value_shape[2];
        let stored_kv_heads = value_shape[1];
        if stored_kv_heads != num_kv_heads || value_shape[3] != *row_width {
            return Ok(None);
        }

        let (batch, num_heads, q_len, weight_prefix_len) = attn_weights.dims4()?;
        if batch != 1
            || q_len == 0
            || num_heads != num_kv_heads * num_kv_groups
            || weight_prefix_len != prefix_len
        {
            return Ok(None);
        }

        let prepared_shape = prepared_rows_f32.shape().dims();
        if prepared_shape != [stored_kv_heads, prefix_len, *row_width] {
            bail!(
                "turboquant prepared V cache shape mismatch: got {:?}, expected [{}, {}, {}]",
                prepared_shape,
                stored_kv_heads,
                prefix_len,
                row_width
            );
        }

        let weights = attn_weights
            .to_device(target_device)?
            .to_dtype(DType::F32)?
            .reshape((num_kv_heads, num_kv_groups, q_len, prefix_len))?;
        let prepared_rows = prepared_rows_f32.to_device(target_device)?;

        let mut per_kv_outputs = Vec::with_capacity(num_kv_heads);
        for kv_head_idx in 0..num_kv_heads {
            let head_weights = weights
                .narrow(0, kv_head_idx, 1)?
                .reshape((num_kv_groups * q_len, prefix_len))?;
            let head_values = prepared_rows
                .narrow(0, kv_head_idx, 1)?
                .reshape((prefix_len, *row_width))?;
            per_kv_outputs.push(head_weights.matmul(&head_values)?.reshape((
                num_kv_groups,
                q_len,
                *row_width,
            ))?);
        }

        let per_kv_refs = per_kv_outputs.iter().collect::<Vec<_>>();
        Ok(Some(
            Tensor::cat(&per_kv_refs, 0)?
                .reshape((1, num_heads, q_len, *row_width))?
                .to_dtype(target_dtype)?,
        ))
    }

    fn query_rows(query: &Tensor, row_width: usize) -> Result<Vec<Vec<f32>>> {
        let query = query.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let shape = query.shape().dims();
        match shape {
            [width] if *width == row_width => Ok(vec![query.flatten_all()?.to_vec1::<f32>()?]),
            [_, width] if *width == row_width => Ok(query.to_vec2::<f32>()?),
            other => bail!(
                "turboquant query shape {:?} is incompatible with key row width {}",
                other,
                row_width
            ),
        }
    }

    fn score_query_rows(
        &self,
        query_rows: &[Vec<f32>],
        key: &TurboQuantKeyPayload,
    ) -> Result<Vec<f32>> {
        let row_count = key.magnitudes.len();
        let expected_sign_bits = Self::packed_sign_bit_len(row_count, key.row_width);
        if key.sign_bits.len() != expected_sign_bits {
            bail!(
                "turboquant sign-bit payload mismatch: got {} bytes, expected {}",
                key.sign_bits.len(),
                expected_sign_bits
            );
        }
        if key.residual_sketch.len() != row_count * key.sketch_dim {
            bail!(
                "turboquant residual sketch mismatch: got {} floats, expected {}",
                key.residual_sketch.len(),
                row_count * key.sketch_dim
            );
        }

        let mut scores = Vec::with_capacity(query_rows.len() * row_count);
        for query_row in query_rows {
            let query_sketch = Self::project_vector(query_row, key.sketch_dim);
            for row_idx in 0..row_count {
                let mut signed_sum = 0.0_f32;
                for (dim_idx, value) in query_row.iter().enumerate() {
                    let sign = Self::sign_bit(&key.sign_bits, row_idx * key.row_width + dim_idx)?;
                    signed_sum += *value * sign;
                }
                let base_score = key.magnitudes[row_idx] * signed_sum;
                let correction = query_sketch
                    .iter()
                    .zip(
                        key.residual_sketch
                            [row_idx * key.sketch_dim..(row_idx + 1) * key.sketch_dim]
                            .iter(),
                    )
                    .map(|(lhs, rhs)| lhs * rhs)
                    .sum::<f32>();
                scores.push(base_score + correction);
            }
        }
        Ok(scores)
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
                let key_dense_bytes = key.dense_fallback.elem_count() as u64
                    * key.dense_fallback.dtype().size_in_bytes() as u64;
                let value_dense_bytes = match value {
                    TurboQuantValuePayload::Dense(value) => {
                        value.elem_count() as u64 * value.dtype().size_in_bytes() as u64
                    }
                    TurboQuantValuePayload::RowwiseInt8 { scales, bytes, .. } => {
                        (scales.len() as u64 * std::mem::size_of::<f32>() as u64)
                            + bytes.len() as u64
                    }
                };
                key_dense_bytes
                    + value_dense_bytes
                    + (key.magnitudes.len() as u64 * std::mem::size_of::<f32>() as u64)
                    + (key.sign_bits.len() as u64)
                    + (key.residual_sketch.len() as u64 * std::mem::size_of::<f32>() as u64)
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
            KvLayerPayload::Encoded { .. } => {
                bail!(
                    "KV backend '{}' cannot validate encoded payloads yet",
                    self.backend_id()
                );
            }
            KvLayerPayload::TurboQuant { .. } => {
                bail!(
                    "KV backend '{}' cannot validate turboquant payloads",
                    self.backend_id()
                );
            }
        }

        Ok(())
    }
}

impl KvCacheBackend for Int8RowwiseKvBackend {
    fn mode(&self) -> KvCacheMode {
        KvCacheMode::Int8RowwiseKv
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
                "Dense KV dtype mismatch on int8_rowwise_kv export for layer {layer_idx}: key={:?} value={:?}",
                dtype,
                value.dtype()
            );
        }
        self.validate_dense_metadata(layer_idx, dtype, &key_shape, &value_shape, seq_len)?;

        let key_values = Self::flatten_to_f32(&key)?;
        let value_values = Self::flatten_to_f32(&value)?;
        let key_row_width = *key_shape.last().unwrap_or(&0);
        let value_row_width = *value_shape.last().unwrap_or(&0);
        let (encoding, bytes) =
            Self::encode_payload(&key_values, key_row_width, &value_values, value_row_width)?;

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
            KvLayerPayload::Encoded { encoding, bytes } => {
                let (key_values, value_values) = match encoding {
                    KvEncodedPayloadEncoding::LegacyTurboQuantInt8V1 => {
                        let (key_scale, value_scale, key_bytes, value_bytes) =
                            Self::parse_legacy_payload(&bytes)?;
                        (
                            Self::dequantize_values(key_scale, key_bytes),
                            Self::dequantize_values(value_scale, value_bytes),
                        )
                    }
                    KvEncodedPayloadEncoding::Int8RowwiseKvV1 => {
                        let (
                            key_scales,
                            key_row_width,
                            value_scales,
                            value_row_width,
                            key_bytes,
                            value_bytes,
                        ) = Self::parse_rowwise_payload(&bytes)?;
                        (
                            Self::dequantize_rowwise(&key_scales, key_row_width, key_bytes)?,
                            Self::dequantize_rowwise(&value_scales, value_row_width, value_bytes)?,
                        )
                    }
                };

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
                "KV backend '{}' expected encoded int8_rowwise_kv payloads",
                self.backend_id()
            ),
            KvLayerPayload::TurboQuant { .. } => bail!(
                "KV backend '{}' expected encoded int8_rowwise_kv payloads",
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
            Some(KvLayerEnvelope {
                payload: KvLayerPayload::TurboQuant { key, value },
                ..
            }) => {
                let key_dense_bytes = key.dense_fallback.elem_count() as u64
                    * key.dense_fallback.dtype().size_in_bytes() as u64;
                let value_dense_bytes = match value {
                    TurboQuantValuePayload::Dense(value) => {
                        value.elem_count() as u64 * value.dtype().size_in_bytes() as u64
                    }
                    TurboQuantValuePayload::RowwiseInt8 { scales, bytes, .. } => {
                        (scales.len() as u64 * std::mem::size_of::<f32>() as u64)
                            + bytes.len() as u64
                    }
                };
                key_dense_bytes
                    + value_dense_bytes
                    + (key.magnitudes.len() as u64 * std::mem::size_of::<f32>() as u64)
                    + (key.sign_bits.len() as u64)
                    + (key.residual_sketch.len() as u64 * std::mem::size_of::<f32>() as u64)
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
                let (key_bytes_len, value_bytes_len) = match encoding {
                    KvEncodedPayloadEncoding::LegacyTurboQuantInt8V1 => {
                        let (_key_scale, _value_scale, key_bytes, value_bytes) =
                            Self::parse_legacy_payload(bytes)?;
                        (key_bytes.len(), value_bytes.len())
                    }
                    KvEncodedPayloadEncoding::Int8RowwiseKvV1 => {
                        let (
                            key_scales,
                            key_row_width,
                            value_scales,
                            value_row_width,
                            key_bytes,
                            value_bytes,
                        ) = Self::parse_rowwise_payload(bytes)?;
                        let expected_key_row_width = *stored.key_shape.last().unwrap_or(&0);
                        let expected_value_row_width = *stored.value_shape.last().unwrap_or(&0);
                        if key_row_width != expected_key_row_width
                            || value_row_width != expected_value_row_width
                        {
                            bail!(
                                "int8_rowwise_kv row width mismatch for layer {}: key={} value={} expected_key={} expected_value={}",
                                layer_idx,
                                key_row_width,
                                value_row_width,
                                expected_key_row_width,
                                expected_value_row_width
                            );
                        }
                        if key_scales.len()
                            != stored.key_shape[..stored.key_shape.len() - 1]
                                .iter()
                                .product::<usize>()
                            || value_scales.len()
                                != stored.value_shape[..stored.value_shape.len() - 1]
                                    .iter()
                                    .product::<usize>()
                        {
                            bail!(
                                "int8_rowwise_kv row-scale metadata mismatch for layer {}",
                                layer_idx
                            );
                        }
                        (key_bytes.len(), value_bytes.len())
                    }
                };
                let expected_key_len = stored.key_shape.iter().product::<usize>();
                let expected_value_len = stored.value_shape.iter().product::<usize>();
                if key_bytes_len != expected_key_len || value_bytes_len != expected_value_len {
                    bail!(
                        "int8_rowwise_kv payload element-count mismatch for layer {}: key={} value={} expected_key={} expected_value={}",
                        layer_idx,
                        key_bytes_len,
                        value_bytes_len,
                        expected_key_len,
                        expected_value_len
                    );
                }
            }
            KvLayerPayload::Dense { .. } => bail!(
                "KV backend '{}' expected encoded int8_rowwise_kv payloads",
                self.backend_id()
            ),
            KvLayerPayload::TurboQuant { .. } => bail!(
                "KV backend '{}' expected encoded int8_rowwise_kv payloads",
                self.backend_id()
            ),
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
                "Dense KV dtype mismatch on turboquant export for layer {layer_idx}: key={:?} value={:?}",
                dtype,
                value.dtype()
            );
        }
        self.validate_dense_metadata(layer_idx, dtype, &key_shape, &value_shape, seq_len)?;

        Ok(Some(KvLayerEnvelope {
            format: self.mode(),
            layer_idx,
            seq_len,
            key_shape: key_shape.clone(),
            value_shape: value_shape.clone(),
            dtype,
            payload: KvLayerPayload::TurboQuant {
                key: self.encode_key_payload(&key, &key_shape)?,
                value: self.encode_value_payload(&value, &value_shape)?,
            },
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
            KvLayerPayload::TurboQuant { key, value } => {
                let key = key
                    .dense_fallback
                    .to_device(target_device)?
                    .to_dtype(target_dtype)?;
                let value = Tensor::from_vec(
                    Self::decode_value_payload(&value, &stored.value_shape)?,
                    stored.value_shape.as_slice(),
                    &Device::Cpu,
                )?
                .to_device(target_device)?
                .to_dtype(target_dtype)?;
                Ok(Some((key, value)))
            }
            _ => bail!(
                "KV backend '{}' expected turboquant payloads",
                self.backend_id()
            ),
        }
    }

    fn stored_bytes(&self, stored: &Option<KvLayerEnvelope>) -> u64 {
        match stored {
            Some(KvLayerEnvelope {
                payload: KvLayerPayload::TurboQuant { key, value },
                ..
            }) => {
                let key_dense_bytes = key.dense_fallback.elem_count() as u64
                    * key.dense_fallback.dtype().size_in_bytes() as u64;
                let value_dense_bytes = match value {
                    TurboQuantValuePayload::Dense(value) => {
                        value.elem_count() as u64 * value.dtype().size_in_bytes() as u64
                    }
                    TurboQuantValuePayload::RowwiseInt8 { scales, bytes, .. } => {
                        (scales.len() as u64 * std::mem::size_of::<f32>() as u64)
                            + bytes.len() as u64
                    }
                };
                key_dense_bytes
                    + value_dense_bytes
                    + (key.magnitudes.len() as u64 * std::mem::size_of::<f32>() as u64)
                    + (key.sign_bits.len() as u64)
                    + (key.residual_sketch.len() as u64 * std::mem::size_of::<f32>() as u64)
            }
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

        self.validate_dense_metadata(
            layer_idx,
            stored.dtype,
            &stored.key_shape,
            &stored.value_shape,
            stored.seq_len,
        )?;

        match &stored.payload {
            KvLayerPayload::TurboQuant { key, value } => {
                let row_width = Self::row_width(&stored.key_shape)?;
                let row_count = Self::row_count(&stored.key_shape)?;
                if key.row_width != row_width {
                    bail!(
                        "turboquant row width mismatch for layer {}: stored={} expected={}",
                        layer_idx,
                        key.row_width,
                        row_width
                    );
                }
                if key.sketch_dim != Self::sketch_dim_for_row_width(row_width) {
                    bail!(
                        "turboquant sketch dim mismatch for layer {}: stored={} expected={}",
                        layer_idx,
                        key.sketch_dim,
                        Self::sketch_dim_for_row_width(row_width)
                    );
                }
                if key.magnitudes.len() != row_count {
                    bail!(
                        "turboquant magnitude count mismatch for layer {}: stored={} expected={}",
                        layer_idx,
                        key.magnitudes.len(),
                        row_count
                    );
                }
                if key.sign_bits.len() != Self::packed_sign_bit_len(row_count, row_width) {
                    bail!(
                        "turboquant sign-bit payload length mismatch for layer {}",
                        layer_idx
                    );
                }
                if key.residual_sketch.len() != row_count * key.sketch_dim {
                    bail!(
                        "turboquant residual sketch length mismatch for layer {}",
                        layer_idx
                    );
                }

                let dense_key_shape = key.dense_fallback.shape().dims().to_vec();
                if dense_key_shape != stored.key_shape {
                    bail!(
                        "turboquant dense-fallback key shape mismatch for layer {}",
                        layer_idx
                    );
                }
                if key.dense_fallback.dtype() != stored.dtype {
                    bail!(
                        "turboquant dense-fallback key dtype mismatch for layer {}",
                        layer_idx
                    );
                }

                match value {
                    TurboQuantValuePayload::Dense(value) => {
                        let dense_value_shape = value.shape().dims().to_vec();
                        if dense_value_shape != stored.value_shape {
                            bail!(
                                "turboquant dense-fallback value shape mismatch for layer {}",
                                layer_idx
                            );
                        }
                        if value.dtype() != stored.dtype {
                            bail!(
                                "turboquant dense-fallback value dtype mismatch for layer {}",
                                layer_idx
                            );
                        }
                    }
                    TurboQuantValuePayload::RowwiseInt8 {
                        row_width,
                        scales,
                        bytes,
                        prepared_rows_f32,
                    } => {
                        let expected_row_width = *stored.value_shape.last().unwrap_or(&0);
                        if *row_width != expected_row_width {
                            bail!(
                                "turboquant V row width mismatch for layer {}: stored={} expected={}",
                                layer_idx,
                                row_width,
                                expected_row_width
                            );
                        }
                        let expected_rows = stored.value_shape[..stored.value_shape.len() - 1]
                            .iter()
                            .product::<usize>();
                        if scales.len() != expected_rows {
                            bail!(
                                "turboquant V scale count mismatch for layer {}: stored={} expected={}",
                                layer_idx,
                                scales.len(),
                                expected_rows
                            );
                        }
                        if bytes.len() != stored.value_shape.iter().product::<usize>() {
                            bail!("turboquant V byte length mismatch for layer {}", layer_idx);
                        }
                        let expected_prepared_shape =
                            [stored.value_shape[1], stored.value_shape[2], *row_width];
                        if prepared_rows_f32.shape().dims() != expected_prepared_shape {
                            bail!(
                                "turboquant prepared V cache shape mismatch for layer {}: stored={:?} expected={:?}",
                                layer_idx,
                                prepared_rows_f32.shape().dims(),
                                expected_prepared_shape
                            );
                        }
                        if prepared_rows_f32.dtype() != DType::F32 {
                            bail!(
                                "turboquant prepared V cache dtype mismatch for layer {}: stored={:?} expected={:?}",
                                layer_idx,
                                prepared_rows_f32.dtype(),
                                DType::F32
                            );
                        }
                    }
                }
            }
            _ => bail!(
                "KV backend '{}' expected turboquant payloads",
                self.backend_id()
            ),
        }

        Ok(())
    }

    fn supports_compressed_k_scores(&self) -> bool {
        true
    }

    fn score_query_against_stored_keys(
        &self,
        query: &Tensor,
        stored: &KvLayerEnvelope,
    ) -> Result<Option<Tensor>> {
        self.validate_layer(stored.layer_idx, stored)?;
        let KvLayerPayload::TurboQuant { key, .. } = &stored.payload else {
            bail!(
                "KV backend '{}' expected turboquant payloads",
                self.backend_id()
            );
        };
        let query_rows = Self::query_rows(query, key.row_width)?;
        let query_count = query_rows.len();
        let row_count = key.magnitudes.len();
        let scores = self.score_query_rows(&query_rows, key)?;
        Ok(Some(Tensor::from_vec(
            scores,
            (query_count, row_count),
            &Device::Cpu,
        )?))
    }

    fn weighted_value_prefix(
        &self,
        attn_weights: &Tensor,
        stored: &KvLayerEnvelope,
        num_kv_heads: usize,
        num_kv_groups: usize,
        target_device: &Device,
        target_dtype: DType,
    ) -> Result<Option<Tensor>> {
        self.validate_layer(stored.layer_idx, stored)?;
        let KvLayerPayload::TurboQuant { value, .. } = &stored.payload else {
            bail!(
                "KV backend '{}' expected turboquant payloads",
                self.backend_id()
            );
        };
        match self.weighted_value_from_rowwise_payload_hot_path(
            attn_weights,
            &stored.value_shape,
            value,
            num_kv_heads,
            num_kv_groups,
            target_device,
            target_dtype,
        )? {
            Some(weighted) => Ok(Some(weighted)),
            None => self.weighted_value_from_rowwise_payload_reference(
                attn_weights,
                &stored.value_shape,
                value,
                num_kv_heads,
                num_kv_groups,
                target_device,
                target_dtype,
            ),
        }
    }
}

pub fn make_kv_backend(config: KvBackendConfig) -> Result<Box<dyn KvCacheBackend>> {
    match config.mode {
        KvCacheMode::Bf16Dense => Ok(Box::new(Bf16PassthroughBackend)),
        KvCacheMode::Int8RowwiseKv => Ok(Box::new(Int8RowwiseKvBackend)),
        KvCacheMode::TurboQuant => Ok(Box::new(TurboQuantBackend)),
    }
}

pub fn move_kv_caches_to_device(caches: &mut LayerKvCaches, target_device: &Device) -> usize {
    let mut moved = 0usize;
    for cache in caches {
        let Some(stored) = cache.as_mut() else {
            continue;
        };
        let (key, value) = match &mut stored.payload {
            KvLayerPayload::Dense { key, value } => (key, value),
            KvLayerPayload::TurboQuant { key, value } => match value {
                TurboQuantValuePayload::Dense(value) => (&mut key.dense_fallback, value),
                TurboQuantValuePayload::RowwiseInt8 { .. } => continue,
            },
            KvLayerPayload::Encoded { .. } => continue,
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
            KvLayerPayload::TurboQuant { key, value } => {
                let key_bytes = if matches!(key.dense_fallback.device(), Device::Cpu) {
                    0
                } else {
                    key.dense_fallback.elem_count() as u64
                        * key.dense_fallback.dtype().size_in_bytes() as u64
                };
                let value_bytes = match value {
                    TurboQuantValuePayload::Dense(value) => {
                        if matches!(value.device(), Device::Cpu) {
                            0
                        } else {
                            value.elem_count() as u64 * value.dtype().size_in_bytes() as u64
                        }
                    }
                    TurboQuantValuePayload::RowwiseInt8 { .. } => 0,
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
            KvCacheMode::from_str("int8_rowwise_kv").unwrap(),
            KvCacheMode::Int8RowwiseKv
        );
        assert_eq!(
            KvCacheMode::from_str("turboquant").unwrap(),
            KvCacheMode::TurboQuant
        );
        assert_eq!(KvCacheMode::Bf16Dense.to_string(), "bf16_dense");
        assert_eq!(KvCacheMode::Int8RowwiseKv.to_string(), "int8_rowwise_kv");
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
        assert!(message.contains("bf16_dense|int8_rowwise_kv|turboquant"));
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
    fn int8_rowwise_kv_backend_factory_is_real() {
        let backend = make_kv_backend(KvBackendConfig {
            mode: KvCacheMode::Int8RowwiseKv,
        })
        .unwrap();
        assert_eq!(backend.backend_id(), "int8_rowwise_kv");
    }

    #[test]
    fn turboquant_backend_factory_is_real() {
        let backend = make_kv_backend(KvBackendConfig {
            mode: KvCacheMode::TurboQuant,
        })
        .unwrap();
        assert_eq!(backend.backend_id(), "turboquant");
        assert!(backend.supports_compressed_k_scores());
    }

    fn dense_scores(query_rows: &[Vec<f32>], key_rows: &[Vec<f32>]) -> Vec<f32> {
        query_rows
            .iter()
            .flat_map(|query| {
                key_rows.iter().map(|key| {
                    query
                        .iter()
                        .zip(key.iter())
                        .map(|(lhs, rhs)| lhs * rhs)
                        .sum::<f32>()
                })
            })
            .collect()
    }

    #[test]
    fn turboquant_k_payload_uses_distinct_compressed_representation() {
        let backend = TurboQuantBackend;
        let key = Tensor::from_vec(
            vec![
                -0.78_f32, -0.82, 0.75, 0.79, 0.11, -0.08, 0.09, -0.12, 0.84, 0.81, -0.77, -0.74,
                -0.09, 0.07, -0.1, 0.08,
            ],
            (1, 1, 2, 8),
            &Device::Cpu,
        )
        .unwrap();
        let value = Tensor::zeros((1, 1, 2, 8), DType::F32, &Device::Cpu).unwrap();

        let stored = backend
            .export_layer(0, Some((key.clone(), value)))
            .unwrap()
            .unwrap();

        let KvLayerPayload::TurboQuant { key, value } = &stored.payload else {
            panic!("expected turboquant payload")
        };
        assert_eq!(key.row_width, 8);
        assert_eq!(key.sketch_dim, 4);
        assert_eq!(key.magnitudes.len(), 2);
        assert_eq!(key.sign_bits.len(), 2);
        assert_eq!(key.residual_sketch.len(), 8);
        assert_eq!(key.dense_fallback.shape().dims(), &[1, 1, 2, 8]);
        assert!(matches!(value, TurboQuantValuePayload::RowwiseInt8 { .. }));
        assert_ne!(key.sign_bits.len(), key.dense_fallback.elem_count());
    }

    #[test]
    fn turboquant_weighted_value_prefix_consumes_backend_owned_v_path() {
        let backend = TurboQuantBackend;
        let key = Tensor::zeros((1, 1, 2, 8), DType::F32, &Device::Cpu).unwrap();
        let value_rows = vec![
            vec![0.1_f32, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4],
            vec![0.5_f32, 0.6, 0.7, 0.8, -0.5, -0.6, -0.7, -0.8],
        ];
        let value = Tensor::from_vec(value_rows.concat(), (1, 1, 2, 8), &Device::Cpu).unwrap();
        let stored = backend
            .export_layer(0, Some((key, value)))
            .unwrap()
            .unwrap();

        let attn_weights =
            Tensor::from_vec(vec![0.75_f32, 0.25, 0.10, 0.90], (1, 2, 1, 2), &Device::Cpu).unwrap();
        let aggregated = backend
            .weighted_value_prefix(&attn_weights, &stored, 1, 2, &Device::Cpu, DType::F32)
            .unwrap()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let expected = vec![
            value_rows[0]
                .iter()
                .zip(value_rows[1].iter())
                .map(|(lhs, rhs)| 0.75 * lhs + 0.25 * rhs)
                .collect::<Vec<_>>(),
            value_rows[0]
                .iter()
                .zip(value_rows[1].iter())
                .map(|(lhs, rhs)| 0.10 * lhs + 0.90 * rhs)
                .collect::<Vec<_>>(),
        ];

        for (actual_row, expected_row) in aggregated.chunks(8).zip(expected.iter()) {
            for (actual, expected) in actual_row.iter().zip(expected_row.iter()) {
                assert!(
                    (*actual - *expected).abs() <= 0.02_f32,
                    "actual={actual} expected={expected}"
                );
            }
        }

        let KvLayerPayload::TurboQuant { value, .. } = &stored.payload else {
            panic!("expected turboquant payload")
        };
        let TurboQuantValuePayload::RowwiseInt8 {
            prepared_rows_f32, ..
        } = value
        else {
            panic!("expected rowwise V payload")
        };
        assert_eq!(prepared_rows_f32.dims3().unwrap(), (1, 2, 8));
    }

    #[test]
    fn turboquant_weighted_value_prefix_hot_path_matches_prepared_dense_cache() {
        let backend = TurboQuantBackend;
        let key = Tensor::zeros((1, 2, 3, 8), DType::F32, &Device::Cpu).unwrap();
        let value = Tensor::from_vec(
            vec![
                0.10_f32, 0.20, 0.30, 0.40, -0.10, -0.20, -0.30, -0.40, 0.50_f32, 0.60, 0.70, 0.80,
                -0.50, -0.60, -0.70, -0.80, 0.90_f32, 1.00, 1.10, 1.20, -0.90, -1.00, -1.10, -1.20,
                -0.15_f32, -0.25, -0.35, -0.45, 0.15, 0.25, 0.35, 0.45, -0.55_f32, -0.65, -0.75,
                -0.85, 0.55, 0.65, 0.75, 0.85, -0.95_f32, -1.05, -1.15, -1.25, 0.95, 1.05, 1.15,
                1.25,
            ],
            (1, 2, 3, 8),
            &Device::Cpu,
        )
        .unwrap();
        let stored = backend
            .export_layer(0, Some((key, value)))
            .unwrap()
            .unwrap();
        let attn_weights = Tensor::from_vec(
            vec![
                0.70_f32, 0.20, 0.10, 0.05_f32, 0.15, 0.80, 0.60_f32, 0.25, 0.15, 0.20_f32, 0.30,
                0.50,
            ],
            (1, 4, 1, 3),
            &Device::Cpu,
        )
        .unwrap();

        let KvLayerPayload::TurboQuant { value, .. } = &stored.payload else {
            panic!("expected turboquant payload")
        };
        let hot = backend
            .weighted_value_from_rowwise_payload_hot_path(
                &attn_weights,
                &stored.value_shape,
                value,
                2,
                2,
                &Device::Cpu,
                DType::F32,
            )
            .unwrap()
            .unwrap();
        let prepared_rows_f32 = match value {
            TurboQuantValuePayload::RowwiseInt8 {
                prepared_rows_f32, ..
            } => prepared_rows_f32,
            TurboQuantValuePayload::Dense(_) => panic!("expected rowwise V payload"),
        };
        let dense_reference = attn_weights
            .reshape((2, 2, 3))
            .unwrap()
            .narrow(0, 0, 1)
            .unwrap()
            .reshape((2, 3))
            .unwrap()
            .matmul(
                &prepared_rows_f32
                    .narrow(0, 0, 1)
                    .unwrap()
                    .reshape((3, 8))
                    .unwrap(),
            )
            .unwrap();
        let dense_reference_1 = attn_weights
            .reshape((2, 2, 3))
            .unwrap()
            .narrow(0, 1, 1)
            .unwrap()
            .reshape((2, 3))
            .unwrap()
            .matmul(
                &prepared_rows_f32
                    .narrow(0, 1, 1)
                    .unwrap()
                    .reshape((3, 8))
                    .unwrap(),
            )
            .unwrap();
        let dense_reference = Tensor::cat(&[&dense_reference, &dense_reference_1], 0)
            .unwrap()
            .reshape((1, 4, 1, 8))
            .unwrap();

        let hot = hot.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let reference = dense_reference
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(hot.len(), reference.len());
        for (actual, expected) in hot.iter().zip(reference.iter()) {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn turboquant_weighted_value_prefix_supports_multi_token_decode_queries() {
        let backend = TurboQuantBackend;
        let key = Tensor::zeros((1, 2, 3, 8), DType::F32, &Device::Cpu).unwrap();
        let value = Tensor::from_vec(
            vec![
                0.10_f32, 0.20, 0.30, 0.40, -0.10, -0.20, -0.30, -0.40, 0.50_f32, 0.60, 0.70, 0.80,
                -0.50, -0.60, -0.70, -0.80, 0.90_f32, 1.00, 1.10, 1.20, -0.90, -1.00, -1.10, -1.20,
                -0.15_f32, -0.25, -0.35, -0.45, 0.15, 0.25, 0.35, 0.45, -0.55_f32, -0.65, -0.75,
                -0.85, 0.55, 0.65, 0.75, 0.85, -0.95_f32, -1.05, -1.15, -1.25, 0.95, 1.05, 1.15,
                1.25,
            ],
            (1, 2, 3, 8),
            &Device::Cpu,
        )
        .unwrap();
        let stored = backend
            .export_layer(0, Some((key, value)))
            .unwrap()
            .unwrap();
        let attn_weights = Tensor::from_vec(
            vec![
                0.70_f32, 0.20, 0.10, 0.05_f32, 0.15, 0.80, 0.10_f32, 0.30, 0.60, 0.45_f32, 0.35,
                0.20, 0.60_f32, 0.25, 0.15, 0.20_f32, 0.30, 0.50, 0.25_f32, 0.50, 0.25, 0.55_f32,
                0.15, 0.30,
            ],
            (1, 4, 2, 3),
            &Device::Cpu,
        )
        .unwrap();

        let KvLayerPayload::TurboQuant { value, .. } = &stored.payload else {
            panic!("expected turboquant payload")
        };
        let hot = backend
            .weighted_value_prefix(&attn_weights, &stored, 2, 2, &Device::Cpu, DType::F32)
            .unwrap()
            .unwrap();
        let prepared_rows_f32 = match value {
            TurboQuantValuePayload::RowwiseInt8 {
                prepared_rows_f32, ..
            } => prepared_rows_f32,
            TurboQuantValuePayload::Dense(_) => panic!("expected rowwise V payload"),
        };
        let weights = attn_weights.reshape((2, 2, 2, 3)).unwrap();
        let mut per_kv_outputs = Vec::new();
        for kv_head_idx in 0..2 {
            let dense_reference = weights
                .narrow(0, kv_head_idx, 1)
                .unwrap()
                .reshape((4, 3))
                .unwrap()
                .matmul(
                    &prepared_rows_f32
                        .narrow(0, kv_head_idx, 1)
                        .unwrap()
                        .reshape((3, 8))
                        .unwrap(),
                )
                .unwrap()
                .reshape((2, 2, 8))
                .unwrap();
            per_kv_outputs.push(dense_reference);
        }
        let per_kv_refs = per_kv_outputs.iter().collect::<Vec<_>>();
        let dense_reference = Tensor::cat(&per_kv_refs, 0)
            .unwrap()
            .reshape((1, 4, 2, 8))
            .unwrap();

        let hot = hot.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let reference = dense_reference
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(hot.len(), reference.len());
        for (actual, expected) in hot.iter().zip(reference.iter()) {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn turboquant_weighted_value_prefix_supports_multi_token_decode_queries_single_kv_head() {
        let backend = TurboQuantBackend;
        let key = Tensor::zeros((1, 1, 2, 8), DType::F32, &Device::Cpu).unwrap();
        let value_rows = vec![
            vec![0.1_f32, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4],
            vec![0.5_f32, 0.6, 0.7, 0.8, -0.5, -0.6, -0.7, -0.8],
        ];
        let value = Tensor::from_vec(value_rows.concat(), (1, 1, 2, 8), &Device::Cpu).unwrap();
        let stored = backend
            .export_layer(0, Some((key, value)))
            .unwrap()
            .unwrap();

        let attn_weights = Tensor::from_vec(
            vec![0.75_f32, 0.25, 0.10, 0.90, 0.30_f32, 0.70, 0.65_f32, 0.35],
            (1, 2, 2, 2),
            &Device::Cpu,
        )
        .unwrap();
        let aggregated = backend
            .weighted_value_prefix(&attn_weights, &stored, 1, 2, &Device::Cpu, DType::F32)
            .unwrap()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let expected = vec![
            value_rows[0]
                .iter()
                .zip(value_rows[1].iter())
                .map(|(lhs, rhs)| 0.75 * lhs + 0.25 * rhs)
                .collect::<Vec<_>>(),
            value_rows[0]
                .iter()
                .zip(value_rows[1].iter())
                .map(|(lhs, rhs)| 0.10 * lhs + 0.90 * rhs)
                .collect::<Vec<_>>(),
            value_rows[0]
                .iter()
                .zip(value_rows[1].iter())
                .map(|(lhs, rhs)| 0.30 * lhs + 0.70 * rhs)
                .collect::<Vec<_>>(),
            value_rows[0]
                .iter()
                .zip(value_rows[1].iter())
                .map(|(lhs, rhs)| 0.65 * lhs + 0.35 * rhs)
                .collect::<Vec<_>>(),
        ];

        for (actual_row, expected_row) in aggregated.chunks(8).zip(expected.iter()) {
            for (actual, expected) in actual_row.iter().zip(expected_row.iter()) {
                assert!(
                    (actual - expected).abs() <= 0.02_f32,
                    "actual={actual} expected={expected}"
                );
            }
        }
    }

    #[test]
    fn turboquant_reference_scores_consume_compressed_k_path() {
        let backend = TurboQuantBackend;
        let key_rows = vec![
            vec![-0.78_f32, -0.82, 0.75, 0.79, 0.11, -0.08, 0.09, -0.12],
            vec![0.84_f32, 0.81, -0.77, -0.74, -0.09, 0.07, -0.10, 0.08],
        ];
        let key = Tensor::from_vec(key_rows.concat(), (1, 1, 2, 8), &Device::Cpu).unwrap();
        let value = Tensor::zeros((1, 1, 2, 8), DType::F32, &Device::Cpu).unwrap();
        let stored = backend
            .export_layer(0, Some((key, value)))
            .unwrap()
            .unwrap();

        let query_rows = vec![
            vec![-0.74_f32, -0.79, 0.72, 0.76, 0.10, -0.07, 0.08, -0.11],
            vec![0.80_f32, 0.78, -0.74, -0.71, -0.07, 0.05, -0.08, 0.06],
        ];
        let query = Tensor::from_vec(query_rows.concat(), (2, 8), &Device::Cpu).unwrap();

        let compressed = backend
            .score_query_against_stored_keys(&query, &stored)
            .unwrap()
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let dense = dense_scores(&query_rows, &key_rows)
            .chunks(2)
            .map(|row| row.to_vec())
            .collect::<Vec<_>>();

        for (dense_row, compressed_row) in dense.iter().zip(compressed.iter()) {
            for (dense_score, compressed_score) in dense_row.iter().zip(compressed_row.iter()) {
                assert!((dense_score - compressed_score).abs() <= 0.35);
            }
        }
    }

    #[test]
    fn turboquant_import_restores_dense_key_fallback_and_decoded_v_for_safe_decode() {
        let backend = TurboQuantBackend;
        let key_values = vec![
            -0.78_f32, -0.82, 0.75, 0.79, 0.11, -0.08, 0.09, -0.12, 0.84, 0.81, -0.77, -0.74,
            -0.09, 0.07, -0.10, 0.08,
        ];
        let value_values = vec![
            0.1_f32, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4, 0.5, 0.6, 0.7, 0.8, -0.5, -0.6, -0.7,
            -0.8,
        ];
        let key = Tensor::from_vec(key_values.clone(), (1, 1, 2, 8), &Device::Cpu).unwrap();
        let value = Tensor::from_vec(value_values.clone(), (1, 1, 2, 8), &Device::Cpu).unwrap();

        let stored = backend.export_layer(1, Some((key, value))).unwrap();
        let restored = backend
            .import_layer(1, stored, &Device::Cpu, DType::F32)
            .unwrap()
            .unwrap();

        assert_eq!(
            restored.0.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            key_values
        );
        for (actual, expected) in restored
            .1
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .zip(value_values.iter())
        {
            assert!(
                (actual - expected).abs() <= 0.02,
                "actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn int8_rowwise_kv_round_trips_shapes_and_approximately_restores_values() {
        let backend = Int8RowwiseKvBackend;
        let key_values = vec![-1.0_f32, -0.5, 0.0, 0.5, 1.0, 0.25, -0.25, 0.75];
        let value_values = vec![0.9_f32, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2];
        let key = Tensor::from_vec(key_values.clone(), (1, 1, 2, 4), &Device::Cpu).unwrap();
        let value = Tensor::from_vec(value_values.clone(), (1, 1, 2, 4), &Device::Cpu).unwrap();

        let stored = backend.export_layer(1, Some((key, value))).unwrap();
        let stored = stored.unwrap();
        assert_eq!(stored.format, KvCacheMode::Int8RowwiseKv);
        match &stored.payload {
            KvLayerPayload::Encoded { encoding, bytes } => {
                assert_eq!(*encoding, KvEncodedPayloadEncoding::Int8RowwiseKvV1);
                let (
                    key_scales,
                    key_row_width,
                    value_scales,
                    value_row_width,
                    key_bytes,
                    value_bytes,
                ) = Int8RowwiseKvBackend::parse_rowwise_payload(bytes).unwrap();
                assert_eq!(key_row_width, 4);
                assert_eq!(value_row_width, 4);
                assert_eq!(key_scales.len(), 2);
                assert_eq!(value_scales.len(), 2);
                assert_eq!(key_bytes.len(), key_values.len());
                assert_eq!(value_bytes.len(), value_values.len());
            }
            KvLayerPayload::Dense { .. } => panic!("expected encoded payload"),
            KvLayerPayload::TurboQuant { .. } => panic!("expected encoded payload"),
        }

        let restored = backend
            .import_layer(1, Some(stored), &Device::Cpu, DType::F32)
            .unwrap()
            .unwrap();

        assert_eq!(restored.0.dims4().unwrap(), (1, 1, 2, 4));
        assert_eq!(restored.1.dims4().unwrap(), (1, 1, 2, 4));

        let restored_key = restored.0.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let restored_value = restored.1.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let key_row_scales = [1.0_f32 / 127.0, 0.75_f32 / 127.0];
        let value_row_scales = [0.9_f32 / 127.0, 0.5_f32 / 127.0];
        for (row_idx, (expected, actual)) in
            key_values.chunks(4).zip(restored_key.chunks(4)).enumerate()
        {
            let tolerance = key_row_scales[row_idx] + 1e-6;
            for (expected, actual) in expected.iter().zip(actual.iter()) {
                assert!((expected - actual).abs() <= tolerance);
            }
        }
        for (row_idx, (expected, actual)) in value_values
            .chunks(4)
            .zip(restored_value.chunks(4))
            .enumerate()
        {
            let tolerance = value_row_scales[row_idx] + 1e-6;
            for (expected, actual) in expected.iter().zip(actual.iter()) {
                assert!((expected - actual).abs() <= tolerance);
            }
        }
    }

    #[test]
    fn int8_rowwise_kv_payload_uses_distinct_row_scales() {
        let backend = Int8RowwiseKvBackend;
        let key = Tensor::from_vec(
            vec![0.01_f32, -0.02, 0.03, -0.04, 4.0, -5.0, 6.0, -7.0],
            (1, 1, 2, 4),
            &Device::Cpu,
        )
        .unwrap();
        let value = Tensor::from_vec(
            vec![0.1_f32, -0.1, 0.2, -0.2, 8.0, -8.0, 7.5, -7.5],
            (1, 1, 2, 4),
            &Device::Cpu,
        )
        .unwrap();

        let stored = backend
            .export_layer(0, Some((key, value)))
            .unwrap()
            .unwrap();
        let KvLayerPayload::Encoded { encoding, bytes } = stored.payload else {
            panic!("expected encoded payload")
        };
        assert_eq!(encoding, KvEncodedPayloadEncoding::Int8RowwiseKvV1);

        let (key_scales, key_row_width, value_scales, value_row_width, ..) =
            Int8RowwiseKvBackend::parse_rowwise_payload(&bytes).unwrap();
        assert_eq!(key_row_width, 4);
        assert_eq!(value_row_width, 4);
        assert_eq!(key_scales.len(), 2);
        assert_eq!(value_scales.len(), 2);
        assert!(key_scales[0] < key_scales[1]);
        assert!(value_scales[0] < value_scales[1]);
    }

    #[test]
    fn int8_rowwise_kv_validate_rejects_row_metadata_drift() {
        let backend = Int8RowwiseKvBackend;
        let key = Tensor::zeros((1, 1, 2, 4), DType::BF16, &Device::Cpu).unwrap();
        let value = Tensor::zeros((1, 1, 2, 4), DType::BF16, &Device::Cpu).unwrap();
        let mut stored = backend
            .export_layer(3, Some((key, value)))
            .unwrap()
            .unwrap();
        if let KvLayerPayload::Encoded { bytes, .. } = &mut stored.payload {
            bytes[16..20].copy_from_slice(&(3u32).to_le_bytes());
        }

        let err = backend.validate_layer(3, &stored).unwrap_err();
        assert!(err.to_string().contains("not divisible by row_width"));
    }
}
