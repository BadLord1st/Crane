# Error KB

## 2026-04-13 — Gemma4 decode attention grouped-KV regression

### Symptoms
- Live chat requests failed after deploying image `sha256:a83ba4ac14a240af5a9f1a89745b089f62c80450e8bb2bf91f32d007c49f2841`.
- Runtime error:
  - `Decode forward failed: shape mismatch in matmul, lhs: [1, 8, 2, 1, 256], rhs: [1, 8, 1, 256, 22]`

### Root cause
- The decode-only Gemma4 grouped attention optimization changed the tensor shape contract for the `q_len == 1` path in `crane-core/src/models/gemma4/modeling.rs`.
- The grouped batched-matmul path was mathematically intended to avoid `repeat_kv(...).contiguous()`, but the deployed shape arrangement for live decode did not match Candle matmul expectations for the real runtime shapes.

### Fix / mitigation
- Roll back the cluster deployment to the previous working image:
  - `1stbadwolf/crane@sha256:ce2f0deaaab340eeb8e03190c8bda934c78c43c938bb06135124322f7305e49e`
- Keep the experimental attention optimization out of production until the grouped-KV shape contract is validated with a real decode-path regression test.

### Prevention
- Any future attention fast path must have an executed regression test that reproduces the live decode shape contract, not only compile checks.
- Before rollout, validate one real `/v1/chat/completions` request against the cluster model with the candidate image.
- Prefer guarded rollout for decode-path changes and keep the last known-good digest ready for immediate rollback.

### Regression references
- Candidate broken commit: `b2c0ee6` (`Optimize Gemma4 decode attention`)
- Working rollback image: `sha256:ce2f0deaaab340eeb8e03190c8bda934c78c43c938bb06135124322f7305e49e`
- Repro command:
  - `curl -sS -H 'Content-Type: application/json' -d '{"model":"gemma4","messages":[{"role":"user","content":"Кратко объясни service mesh в 2 пунктах"}],"max_tokens":48,"temperature":0.2}' http://10.14.49.42:30095/v1/chat/completions`
