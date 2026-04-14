use std::str::FromStr;

use candle_core::{DType, Device, Tensor};

use super::int8_rowwise::Int8RowwiseKvBackend;
use super::turboquant::{TurboQuantBackend, TURBOQUANT_K_ROTATED_CODEBOOK};
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

fn grouped_value_rows_as_tensor(
    value: &TurboQuantValuePayload,
    num_kv_heads: usize,
    prefix_len: usize,
    row_width: usize,
) -> Tensor {
    let TurboQuantValuePayload::RowwiseInt8 {
        grouped: Some(grouped),
        ..
    } = value
    else {
        panic!("expected grouped V payload")
    };

    let groups_per_row = row_width / grouped.group_width;
    let row_count = num_kv_heads * prefix_len;
    let mut dense = Vec::with_capacity(row_count * row_width);
    for row_idx in 0..row_count {
        let scale_offset = row_idx * groups_per_row;
        let byte_offset = row_idx * row_width;
        for group_idx in 0..groups_per_row {
            let scale = grouped.scales[scale_offset + group_idx];
            let group_start = byte_offset + group_idx * grouped.group_width;
            let group_end = group_start + grouped.group_width;
            dense.extend(
                grouped.bytes[group_start..group_end]
                    .iter()
                    .map(|byte| (*byte as i8) as f32 * scale),
            );
        }
    }

    Tensor::from_vec(dense, (num_kv_heads, prefix_len, row_width), &Device::Cpu).unwrap()
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
    let TurboQuantKeyEncoding::RotatedCodebookResidual {
        pair_count,
        codebook,
        pair_scales,
        code_indices,
        residual_sketch,
    } = &key.encoding
    else {
        panic!("expected rotated-codebook K payload")
    };
    assert_eq!(*pair_count, 4);
    assert_eq!(codebook.len(), TURBOQUANT_K_ROTATED_CODEBOOK.len() * 2);
    assert_eq!(pair_scales.len(), 8);
    assert_eq!(code_indices.len(), 8);
    assert_eq!(residual_sketch.len(), 8);
    assert_eq!(key.dense_fallback.shape().dims(), &[1, 1, 2, 8]);
    assert!(matches!(value, TurboQuantValuePayload::RowwiseInt8 { .. }));
    assert_ne!(code_indices.len(), key.dense_fallback.elem_count());
}

#[test]
fn turboquant_k_payload_falls_back_to_dense_only_when_rotated_codebook_is_unsupported() {
    let backend = TurboQuantBackend;
    let key = Tensor::from_vec(
        vec![0.1_f32, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7],
        (1, 1, 1, 7),
        &Device::Cpu,
    )
    .unwrap();
    let value = Tensor::zeros((1, 1, 1, 7), DType::F32, &Device::Cpu).unwrap();

    let stored = backend
        .export_layer(0, Some((key, value)))
        .unwrap()
        .unwrap();
    let KvLayerPayload::TurboQuant { key, .. } = &stored.payload else {
        panic!("expected turboquant payload")
    };
    let TurboQuantKeyEncoding::DenseFallbackOnly { reason } = &key.encoding else {
        panic!("expected dense-fallback-only key payload")
    };
    assert!(reason.contains("even_row_width"));
    assert!(!backend
        .supports_stored_compressed_k_scores(&stored)
        .unwrap());
    let query = Tensor::zeros((1, 7), DType::F32, &Device::Cpu).unwrap();
    assert!(backend
        .score_query_against_stored_keys(&query, &stored)
        .unwrap()
        .is_none());
}

#[test]
fn turboquant_weighted_value_prefix_consumes_grouped_v_payload() {
    let backend = TurboQuantBackend;
    let key = Tensor::zeros((1, 1, 2, 8), DType::F32, &Device::Cpu).unwrap();
    let value_rows = [
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

    let expected = [
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
    let TurboQuantValuePayload::RowwiseInt8 { grouped, .. } = value else {
        panic!("expected rowwise V payload")
    };
    let grouped = grouped.as_ref().expect("expected grouped V payload");
    assert_eq!(grouped.group_width, 8);
    assert_eq!(grouped.scales.len(), 2);
    assert_eq!(grouped.bytes.len(), 16);
}

#[test]
fn turboquant_weighted_value_prefix_grouped_path_matches_dense_reference() {
    let backend = TurboQuantBackend;
    let key = Tensor::zeros((1, 2, 3, 8), DType::F32, &Device::Cpu).unwrap();
    let value_data = vec![
        0.10_f32, 0.20, 0.30, 0.40, -0.10, -0.20, -0.30, -0.40, 0.50_f32, 0.60, 0.70, 0.80, -0.50,
        -0.60, -0.70, -0.80, 0.90_f32, 1.00, 1.10, 1.20, -0.90, -1.00, -1.10, -1.20, -0.15_f32,
        -0.25, -0.35, -0.45, 0.15, 0.25, 0.35, 0.45, -0.55_f32, -0.65, -0.75, -0.85, 0.55, 0.65,
        0.75, 0.85, -0.95_f32, -1.05, -1.15, -1.25, 0.95, 1.05, 1.15, 1.25,
    ];
    let value = Tensor::from_vec(value_data.clone(), (1, 2, 3, 8), &Device::Cpu).unwrap();
    let stored = backend
        .export_layer(0, Some((key, value)))
        .unwrap()
        .unwrap();
    let attn_weights = Tensor::from_vec(
        vec![
            0.70_f32, 0.20, 0.10, 0.05_f32, 0.15, 0.80, 0.60_f32, 0.25, 0.15, 0.20_f32, 0.30, 0.50,
        ],
        (1, 4, 1, 3),
        &Device::Cpu,
    )
    .unwrap();

    let KvLayerPayload::TurboQuant { value, .. } = &stored.payload else {
        panic!("expected turboquant payload")
    };
    let hot = backend
        .weighted_value_from_grouped_payload(
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
    let dense_value = grouped_value_rows_as_tensor(value, 2, 3, 8);
    let dense_reference = attn_weights
        .reshape((2, 2, 3))
        .unwrap()
        .narrow(0, 0, 1)
        .unwrap()
        .reshape((2, 3))
        .unwrap()
        .matmul(
            &dense_value
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
            &dense_value
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
fn turboquant_weighted_value_prefix_falls_back_when_grouped_v_metadata_is_unavailable() {
    let backend = TurboQuantBackend;
    let key = Tensor::zeros((1, 1, 2, 10), DType::F32, &Device::Cpu).unwrap();
    let value = Tensor::from_vec(
        vec![
            0.10_f32, 0.20, 0.30, 0.40, 0.50, -0.10, -0.20, -0.30, -0.40, -0.50, 0.60, 0.70, 0.80,
            0.90, 1.00, -0.60, -0.70, -0.80, -0.90, -1.00,
        ],
        (1, 1, 2, 10),
        &Device::Cpu,
    )
    .unwrap();
    let stored = backend
        .export_layer(0, Some((key, value)))
        .unwrap()
        .unwrap();
    let KvLayerPayload::TurboQuant { value, .. } = &stored.payload else {
        panic!("expected turboquant payload")
    };
    let TurboQuantValuePayload::RowwiseInt8 { grouped, .. } = value else {
        panic!("expected rowwise V payload")
    };
    assert!(grouped.is_none());

    let attn_weights =
        Tensor::from_vec(vec![0.75_f32, 0.25, 0.10, 0.90], (1, 2, 1, 2), &Device::Cpu).unwrap();
    assert!(backend
        .weighted_value_from_grouped_payload(
            &attn_weights,
            &stored.value_shape,
            value,
            1,
            2,
            &Device::Cpu,
            DType::F32,
        )
        .unwrap()
        .is_none());
    assert!(backend
        .weighted_value_prefix(&attn_weights, &stored, 1, 2, &Device::Cpu, DType::F32)
        .unwrap()
        .is_some());
}

#[test]
fn turboquant_weighted_value_prefix_supports_multi_token_decode_queries() {
    let backend = TurboQuantBackend;
    let key = Tensor::zeros((1, 2, 3, 8), DType::F32, &Device::Cpu).unwrap();
    let value = Tensor::from_vec(
        vec![
            0.10_f32, 0.20, 0.30, 0.40, -0.10, -0.20, -0.30, -0.40, 0.50_f32, 0.60, 0.70, 0.80,
            -0.50, -0.60, -0.70, -0.80, 0.90_f32, 1.00, 1.10, 1.20, -0.90, -1.00, -1.10, -1.20,
            -0.15_f32, -0.25, -0.35, -0.45, 0.15, 0.25, 0.35, 0.45, -0.55_f32, -0.65, -0.75, -0.85,
            0.55, 0.65, 0.75, 0.85, -0.95_f32, -1.05, -1.15, -1.25, 0.95, 1.05, 1.15, 1.25,
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
            0.70_f32, 0.20, 0.10, 0.05_f32, 0.15, 0.80, 0.10_f32, 0.30, 0.60, 0.45_f32, 0.35, 0.20,
            0.60_f32, 0.25, 0.15, 0.20_f32, 0.30, 0.50, 0.25_f32, 0.50, 0.25, 0.55_f32, 0.15, 0.30,
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
    let dense_value = grouped_value_rows_as_tensor(value, 2, 3, 8);
    let weights = attn_weights.reshape((2, 2, 2, 3)).unwrap();
    let mut per_kv_outputs = Vec::new();
    for kv_head_idx in 0..2 {
        let dense_reference = weights
            .narrow(0, kv_head_idx, 1)
            .unwrap()
            .reshape((4, 3))
            .unwrap()
            .matmul(
                &dense_value
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
    let value_rows = [
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

    let expected = [
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
        -0.78_f32, -0.82, 0.75, 0.79, 0.11, -0.08, 0.09, -0.12, 0.84, 0.81, -0.77, -0.74, -0.09,
        0.07, -0.10, 0.08,
    ];
    let value_values = vec![
        0.1_f32, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4, 0.5, 0.6, 0.7, 0.8, -0.5, -0.6, -0.7, -0.8,
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
            let (key_scales, key_row_width, value_scales, value_row_width, key_bytes, value_bytes) =
                Int8RowwiseKvBackend::parse_rowwise_payload(bytes).unwrap();
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
