use anyhow::{anyhow, bail, Result};
use candle_core::{DType, Device, Tensor};

use super::config::KvCacheMode;
use super::dense::Bf16PassthroughBackend;
use super::int8_rowwise::Int8RowwiseKvBackend;
use super::types::{
    DenseLayerKv, KvCacheBackend, KvLayerEnvelope, KvLayerPayload, TurboQuantGroupedValuePayload,
    TurboQuantKeyEncoding, TurboQuantKeyPayload, TurboQuantValuePayload,
};

pub(crate) const TURBOQUANT_VALUE_GROUP_WIDTH_CANDIDATES: &[usize] = &[32, 16, 8, 4];
pub(crate) const TURBOQUANT_K_CODEBOOK_DIM: usize = 2;
pub(crate) const TURBOQUANT_K_ROTATED_CODEBOOK: [[f32; TURBOQUANT_K_CODEBOOK_DIM]; 8] = [
    [1.0, 0.0],
    [0.0, 1.0],
    [-1.0, 0.0],
    [0.0, -1.0],
    [0.70710677, 0.70710677],
    [0.70710677, -0.70710677],
    [-0.70710677, 0.70710677],
    [-0.70710677, -0.70710677],
];

#[derive(Debug, Default, Clone, Copy)]
pub struct TurboQuantBackend;

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

    fn rotate_pair(lhs: f32, rhs: f32) -> [f32; 2] {
        const INV_SQRT_2: f32 = 0.70710677;
        [(lhs + rhs) * INV_SQRT_2, (lhs - rhs) * INV_SQRT_2]
    }

    fn rotate_row_pairs(values: &[f32]) -> Result<Vec<f32>> {
        if !values.len().is_multiple_of(2) {
            bail!(
                "turboquant rotated K path requires even row width, got {}",
                values.len()
            );
        }

        let mut rotated = Vec::with_capacity(values.len());
        for pair in values.chunks_exact(2) {
            rotated.extend_from_slice(&Self::rotate_pair(pair[0], pair[1]));
        }
        Ok(rotated)
    }

    fn flattened_rotated_codebook() -> Vec<f32> {
        TURBOQUANT_K_ROTATED_CODEBOOK
            .iter()
            .flat_map(|entry| entry.iter().copied())
            .collect()
    }

    fn nearest_rotated_codebook_index(pair: [f32; 2]) -> usize {
        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (idx, code) in TURBOQUANT_K_ROTATED_CODEBOOK.iter().enumerate() {
            let score = pair[0] * code[0] + pair[1] * code[1];
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }
        best_idx
    }

    fn key_encoding_supports_scoring(key: &TurboQuantKeyPayload) -> bool {
        matches!(
            &key.encoding,
            TurboQuantKeyEncoding::RotatedCodebookResidual { .. }
        )
    }

    pub(crate) fn key_payload_metadata_bytes(key: &TurboQuantKeyPayload) -> u64 {
        let dense_bytes = key.dense_fallback.elem_count() as u64
            * key.dense_fallback.dtype().size_in_bytes() as u64;
        let encoded_bytes = match &key.encoding {
            TurboQuantKeyEncoding::RotatedCodebookResidual {
                codebook,
                pair_scales,
                code_indices,
                residual_sketch,
                ..
            } => {
                (codebook.len() as u64 * std::mem::size_of::<f32>() as u64)
                    + (pair_scales.len() as u64 * std::mem::size_of::<f32>() as u64)
                    + code_indices.len() as u64
                    + (residual_sketch.len() as u64 * std::mem::size_of::<f32>() as u64)
            }
            TurboQuantKeyEncoding::DenseFallbackOnly { reason } => reason.len() as u64,
        };
        dense_bytes + encoded_bytes
    }

    fn validate_key_payload(
        &self,
        layer_idx: usize,
        key_shape: &[usize],
        dtype: DType,
        key: &TurboQuantKeyPayload,
    ) -> Result<()> {
        let row_width = Self::row_width(key_shape)?;
        let row_count = Self::row_count(key_shape)?;
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

        match &key.encoding {
            TurboQuantKeyEncoding::RotatedCodebookResidual {
                pair_count,
                codebook,
                pair_scales,
                code_indices,
                residual_sketch,
            } => {
                let expected_pair_count = row_width / 2;
                if row_width % 2 != 0 {
                    bail!(
                        "turboquant rotated K payload cannot validate odd row width {} for layer {}",
                        row_width,
                        layer_idx
                    );
                }
                if *pair_count != expected_pair_count {
                    bail!(
                        "turboquant rotated K pair count mismatch for layer {}: stored={} expected={}",
                        layer_idx,
                        pair_count,
                        expected_pair_count
                    );
                }
                if codebook.len() != TURBOQUANT_K_ROTATED_CODEBOOK.len() * TURBOQUANT_K_CODEBOOK_DIM
                {
                    bail!(
                        "turboquant rotated K codebook size mismatch for layer {}",
                        layer_idx
                    );
                }
                if pair_scales.len() != row_count * *pair_count {
                    bail!(
                        "turboquant rotated K scale count mismatch for layer {}: stored={} expected={}",
                        layer_idx,
                        pair_scales.len(),
                        row_count * *pair_count
                    );
                }
                if code_indices.len() != row_count * *pair_count {
                    bail!(
                        "turboquant rotated K code count mismatch for layer {}: stored={} expected={}",
                        layer_idx,
                        code_indices.len(),
                        row_count * *pair_count
                    );
                }
                if code_indices
                    .iter()
                    .any(|idx| *idx as usize >= TURBOQUANT_K_ROTATED_CODEBOOK.len())
                {
                    bail!(
                        "turboquant rotated K code index out of range for layer {}",
                        layer_idx
                    );
                }
                if residual_sketch.len() != row_count * key.sketch_dim {
                    bail!(
                        "turboquant residual sketch length mismatch for layer {}",
                        layer_idx
                    );
                }
            }
            TurboQuantKeyEncoding::DenseFallbackOnly { reason } => {
                if reason.trim().is_empty() {
                    bail!(
                        "turboquant dense-fallback-only reason must be non-empty for layer {}",
                        layer_idx
                    );
                }
            }
        }

        let dense_key_shape = key.dense_fallback.shape().dims().to_vec();
        if dense_key_shape != key_shape {
            bail!(
                "turboquant dense-fallback key shape mismatch for layer {}",
                layer_idx
            );
        }
        if key.dense_fallback.dtype() != dtype {
            bail!(
                "turboquant dense-fallback key dtype mismatch for layer {}",
                layer_idx
            );
        }

        Ok(())
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
        let encoding = if row_width % 2 != 0 {
            TurboQuantKeyEncoding::DenseFallbackOnly {
                reason: format!("rotated_codebook_k_requires_even_row_width_but_got_{row_width}"),
            }
        } else {
            let pair_count = row_width / 2;
            let mut pair_scales = Vec::with_capacity(row_count * pair_count);
            let mut code_indices = Vec::with_capacity(row_count * pair_count);
            let mut residual_sketch = Vec::with_capacity(row_count * sketch_dim);

            for row in key_values.chunks(row_width) {
                let rotated_row = Self::rotate_row_pairs(row)?;
                let mut reconstructed = Vec::with_capacity(row_width);
                for pair in rotated_row.chunks_exact(2) {
                    let scale = (pair[0] * pair[0] + pair[1] * pair[1]).sqrt();
                    pair_scales.push(scale);
                    let code_idx = if scale == 0.0 {
                        0usize
                    } else {
                        Self::nearest_rotated_codebook_index([pair[0] / scale, pair[1] / scale])
                    };
                    code_indices.push(code_idx as u8);
                    let code = TURBOQUANT_K_ROTATED_CODEBOOK[code_idx];
                    reconstructed.push(scale * code[0]);
                    reconstructed.push(scale * code[1]);
                }

                let residual = rotated_row
                    .iter()
                    .zip(reconstructed.iter())
                    .map(|(actual, approx)| actual - approx)
                    .collect::<Vec<_>>();
                residual_sketch.extend(Self::project_vector(&residual, sketch_dim));
            }

            TurboQuantKeyEncoding::RotatedCodebookResidual {
                pair_count,
                codebook: Self::flattened_rotated_codebook(),
                pair_scales,
                code_indices,
                residual_sketch,
            }
        };

        Ok(TurboQuantKeyPayload {
            row_width,
            sketch_dim,
            encoding,
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
        let grouped = Self::grouped_value_group_width(row_width)
            .map(|group_width| {
                let (group_scales, group_quants) =
                    Self::quantize_grouped_values(&value_values, row_width, group_width)?;
                Ok::<_, anyhow::Error>(TurboQuantGroupedValuePayload {
                    group_width,
                    scales: group_scales,
                    bytes: group_quants.into_iter().map(|value| value as u8).collect(),
                })
            })
            .transpose()?;
        Ok(TurboQuantValuePayload::RowwiseInt8 {
            row_width,
            scales,
            bytes: quants.into_iter().map(|value| value as u8).collect(),
            grouped,
        })
    }

    fn grouped_value_group_width(row_width: usize) -> Option<usize> {
        TURBOQUANT_VALUE_GROUP_WIDTH_CANDIDATES
            .iter()
            .copied()
            .find(|candidate| row_width >= *candidate && row_width.is_multiple_of(*candidate))
    }

    fn quantize_grouped_values(
        values: &[f32],
        row_width: usize,
        group_width: usize,
    ) -> Result<(Vec<f32>, Vec<i8>)> {
        if group_width == 0 {
            bail!("turboquant grouped V group width must be > 0");
        }
        if row_width == 0 || !row_width.is_multiple_of(group_width) {
            bail!(
                "turboquant grouped V requires row_width={} to be divisible by group_width={}",
                row_width,
                group_width
            );
        }
        if !values.len().is_multiple_of(row_width) {
            bail!(
                "turboquant grouped V requires element_count={} to be divisible by row_width={}",
                values.len(),
                row_width
            );
        }

        let groups_per_row = row_width / group_width;
        let mut scales = Vec::with_capacity((values.len() / row_width) * groups_per_row);
        let mut quants = Vec::with_capacity(values.len());
        for row in values.chunks(row_width) {
            for group in row.chunks(group_width) {
                let (scale, group_quants) = Int8RowwiseKvBackend::quantize_values(group);
                scales.push(scale);
                quants.extend(group_quants);
            }
        }
        Ok((scales, quants))
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

    pub(crate) fn weighted_value_from_grouped_payload(
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
            grouped: Some(grouped),
            ..
        } = value
        else {
            return Ok(None);
        };

        let TurboQuantGroupedValuePayload {
            group_width,
            scales: group_scales,
            bytes: group_bytes,
        } = grouped;

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

        if *group_width == 0 || *row_width % *group_width != 0 {
            bail!(
                "turboquant grouped V metadata mismatch: row_width={} group_width={}",
                row_width,
                group_width
            );
        }

        let row_count = stored_kv_heads * prefix_len;
        let groups_per_row = *row_width / *group_width;
        if group_scales.len() != row_count * groups_per_row
            || group_bytes.len() != row_count * *row_width
        {
            bail!("turboquant grouped V payload metadata mismatch");
        }

        let weights = attn_weights
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .reshape((num_heads, q_len, prefix_len))?
            .to_vec3::<f32>()?;
        let mut aggregated = vec![0.0_f32; num_heads * q_len * *row_width];
        for head_idx in 0..num_heads {
            let kv_head_idx = head_idx / num_kv_groups;
            for q_idx in 0..q_len {
                let output_offset = (head_idx * q_len + q_idx) * *row_width;
                let out_row = &mut aggregated[output_offset..output_offset + *row_width];
                for pos_idx in 0..prefix_len {
                    let weight = weights[head_idx][q_idx][pos_idx];
                    if weight == 0.0 {
                        continue;
                    }

                    let row_idx = kv_head_idx * prefix_len + pos_idx;
                    let scale_offset = row_idx * groups_per_row;
                    let byte_offset = row_idx * *row_width;
                    for group_idx in 0..groups_per_row {
                        let scale = group_scales[scale_offset + group_idx];
                        if scale == 0.0 {
                            continue;
                        }

                        let group_start = group_idx * group_width;
                        let row_group = &group_bytes
                            [byte_offset + group_start..byte_offset + group_start + group_width];
                        let out_group = &mut out_row[group_start..group_start + group_width];
                        for (dst, quantized) in out_group.iter_mut().zip(row_group.iter()) {
                            *dst += weight * ((*quantized as i8) as f32 * scale);
                        }
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
        let TurboQuantKeyEncoding::RotatedCodebookResidual {
            pair_count,
            codebook,
            pair_scales,
            code_indices,
            residual_sketch,
        } = &key.encoding
        else {
            bail!("turboquant compressed K scoring is unavailable for dense-fallback-only keys")
        };

        let row_count = key.dense_fallback.elem_count() / key.row_width;
        if codebook.len() != TURBOQUANT_K_ROTATED_CODEBOOK.len() * TURBOQUANT_K_CODEBOOK_DIM {
            bail!("turboquant rotated K codebook payload mismatch");
        }
        if pair_scales.len() != row_count * *pair_count {
            bail!(
                "turboquant rotated K scale payload mismatch: got {} values, expected {}",
                pair_scales.len(),
                row_count * *pair_count
            );
        }
        if code_indices.len() != row_count * *pair_count {
            bail!(
                "turboquant rotated K code payload mismatch: got {} values, expected {}",
                code_indices.len(),
                row_count * *pair_count
            );
        }
        if residual_sketch.len() != row_count * key.sketch_dim {
            bail!(
                "turboquant residual sketch mismatch: got {} floats, expected {}",
                residual_sketch.len(),
                row_count * key.sketch_dim
            );
        }

        let mut scores = Vec::with_capacity(query_rows.len() * row_count);
        for query_row in query_rows {
            let rotated_query = Self::rotate_row_pairs(query_row)?;
            let query_sketch = Self::project_vector(&rotated_query, key.sketch_dim);
            for row_idx in 0..row_count {
                let pair_offset = row_idx * *pair_count;
                let mut base_score = 0.0_f32;
                for pair_idx in 0..*pair_count {
                    let code_idx = code_indices[pair_offset + pair_idx] as usize;
                    let code_offset = code_idx * TURBOQUANT_K_CODEBOOK_DIM;
                    let query_offset = pair_idx * TURBOQUANT_K_CODEBOOK_DIM;
                    let scale = pair_scales[pair_offset + pair_idx];
                    base_score += scale
                        * (rotated_query[query_offset] * codebook[code_offset]
                            + rotated_query[query_offset + 1] * codebook[code_offset + 1]);
                }
                let correction = query_sketch
                    .iter()
                    .zip(
                        residual_sketch[row_idx * key.sketch_dim..(row_idx + 1) * key.sketch_dim]
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
                key: Box::new(self.encode_key_payload(&key, &key_shape)?),
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
                let value_dense_bytes = match value {
                    TurboQuantValuePayload::Dense(value) => {
                        value.elem_count() as u64 * value.dtype().size_in_bytes() as u64
                    }
                    TurboQuantValuePayload::RowwiseInt8 { scales, bytes, .. } => {
                        (scales.len() as u64 * std::mem::size_of::<f32>() as u64)
                            + bytes.len() as u64
                    }
                };
                Self::key_payload_metadata_bytes(key) + value_dense_bytes
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
                self.validate_key_payload(layer_idx, &stored.key_shape, stored.dtype, key)?;

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
                        grouped,
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
                        if let Some(grouped) = grouped {
                            if grouped.group_width == 0 || *row_width % grouped.group_width != 0 {
                                bail!(
                                    "turboquant grouped V group width mismatch for layer {}: row_width={} group_width={}",
                                    layer_idx,
                                    row_width,
                                    grouped.group_width
                                );
                            }
                            let groups_per_row = *row_width / grouped.group_width;
                            if grouped.scales.len() != expected_rows * groups_per_row {
                                bail!(
                                    "turboquant grouped V scale count mismatch for layer {}: stored={} expected={}",
                                    layer_idx,
                                    grouped.scales.len(),
                                    expected_rows * groups_per_row
                                );
                            }
                            if grouped.bytes.len() != bytes.len() {
                                bail!(
                                    "turboquant grouped V byte length mismatch for layer {}: stored={} expected={}",
                                    layer_idx,
                                    grouped.bytes.len(),
                                    bytes.len()
                                );
                            }
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

    fn supports_stored_compressed_k_scores(&self, stored: &KvLayerEnvelope) -> Result<bool> {
        self.validate_layer(stored.layer_idx, stored)?;
        let KvLayerPayload::TurboQuant { key, .. } = &stored.payload else {
            bail!(
                "KV backend '{}' expected turboquant payloads",
                self.backend_id()
            );
        };
        Ok(Self::key_encoding_supports_scoring(key))
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
        if !Self::key_encoding_supports_scoring(key) {
            return Ok(None);
        }
        let query_rows = Self::query_rows(query, key.row_width)?;
        let query_count = query_rows.len();
        let row_count = key.dense_fallback.elem_count() / key.row_width;
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
        match self.weighted_value_from_grouped_payload(
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
