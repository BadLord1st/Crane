use anyhow::{anyhow, bail, Result};
use candle_core::{DType, Device, Tensor};

use super::config::KvCacheMode;
use super::dense::Bf16PassthroughBackend;
use super::turboquant::TurboQuantBackend;
use super::types::{
    DenseLayerKv, KvCacheBackend, KvEncodedPayloadEncoding, KvLayerEnvelope, KvLayerPayload,
    TurboQuantValuePayload,
};

#[derive(Debug, Default, Clone, Copy)]
pub struct Int8RowwiseKvBackend;

impl Int8RowwiseKvBackend {
    const MAGIC: [u8; 8] = *b"CRKVTQ01";
    const ROWWISE_MAGIC: [u8; 8] = *b"CRKVRW01";
    const LEGACY_HEADER_LEN: usize = 8 + 4 + 4 + 4 + 4;
    const ROWWISE_HEADER_LEN: usize = 8 + 4 + 4 + 4 + 4;

    fn supported_dense_dtype(dtype: DType) -> bool {
        matches!(dtype, DType::F16 | DType::BF16 | DType::F32)
    }

    pub(crate) fn validate_dense_metadata(
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

    pub(crate) fn flatten_to_f32(tensor: &Tensor) -> Result<Vec<f32>> {
        Ok(tensor
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?)
    }

    pub(crate) fn quantize_values(values: &[f32]) -> (f32, Vec<i8>) {
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

    pub(crate) fn quantize_rowwise(
        values: &[f32],
        row_width: usize,
    ) -> Result<(Vec<f32>, Vec<i8>)> {
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

    pub(crate) fn dequantize_values(scale: f32, bytes: &[u8]) -> Vec<f32> {
        if scale == 0.0 {
            return vec![0.0; bytes.len()];
        }

        bytes
            .iter()
            .map(|byte| (*byte as i8) as f32 * scale)
            .collect()
    }

    pub(crate) fn dequantize_rowwise(
        scales: &[f32],
        row_width: usize,
        bytes: &[u8],
    ) -> Result<Vec<f32>> {
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

    pub(crate) fn parse_rowwise_payload<'a>(
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
