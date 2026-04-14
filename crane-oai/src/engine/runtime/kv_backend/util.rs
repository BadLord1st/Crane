use candle_core::Device;

use super::config::{KvBackendConfig, KvCacheMode};
use super::dense::Bf16PassthroughBackend;
use super::int8_rowwise::Int8RowwiseKvBackend;
use super::turboquant::TurboQuantBackend;
use super::types::{
    KvCacheBackend, KvLayerEnvelope, KvLayerPayload, LayerKvCaches, TurboQuantValuePayload,
};
use anyhow::Result;

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
