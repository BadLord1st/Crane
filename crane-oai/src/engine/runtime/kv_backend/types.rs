use std::fmt;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use super::config::KvCacheMode;

pub type DenseLayerKv = Option<(Tensor, Tensor)>;
pub type LayerKvCaches = Vec<Option<KvLayerEnvelope>>;
pub type SequenceKvCaches = Vec<LayerKvCaches>;

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
    pub encoding: TurboQuantKeyEncoding,
    pub dense_fallback: Tensor,
}

#[derive(Debug, Clone)]
pub enum TurboQuantKeyEncoding {
    RotatedCodebookResidual {
        pair_count: usize,
        codebook: Vec<f32>,
        pair_scales: Vec<f32>,
        code_indices: Vec<u8>,
        residual_sketch: Vec<f32>,
    },
    DenseFallbackOnly {
        reason: String,
    },
}

#[derive(Debug, Clone)]
pub struct TurboQuantGroupedValuePayload {
    pub group_width: usize,
    pub scales: Vec<f32>,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub enum TurboQuantValuePayload {
    Dense(Tensor),
    RowwiseInt8 {
        row_width: usize,
        scales: Vec<f32>,
        bytes: Vec<u8>,
        grouped: Option<TurboQuantGroupedValuePayload>,
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

    fn supports_stored_compressed_k_scores(&self, _stored: &KvLayerEnvelope) -> Result<bool> {
        Ok(self.supports_compressed_k_scores())
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
