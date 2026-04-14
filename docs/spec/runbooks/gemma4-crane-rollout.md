# Gemma4 Crane Rollout

Links:
- [[spec-index]]
- Primary spec: [[features/gemma4-variant-compatible-loading]]
- Tickets: T-0002

## Purpose

Roll out a Crane image containing the Gemma4 loader compatibility fix, then verify the cluster deployment becomes healthy with the target PVC-backed Gemma4 model.

## Build host

- Build on `my69` only.
- Reason: target image requires Linux `amd64` + CUDA; local macOS `arm64` is not a valid build environment.

## Build inputs

- Repo: `Crane`
- Dockerfile: `./Dockerfile`
- Required build arg: `CUDA_COMPUTE_CAP=89`
- Target runtime model path:
  - `/models/google/gemma-4-26B-A4B-it/1db3cff1840c2ae59759d8e842ff37831cf8cb63`

## Build procedure

1. Ensure the desired Git commit is pushed.
2. On `my69`, build the image with CUDA support.
3. Push the built image to the registry.
4. Record the resulting tag or digest.

## Deployment procedure

1. Update the Crane deployment image reference in `ai-crane`.
2. Scale the deployment to `1` replica if it is currently disabled.
3. Wait for pod scheduling and model startup.
4. Verify `/health` only succeeds after successful model initialization.

## Verification checklist

- deployment image matches the built image
- pod starts on a GPU-capable node
- model path points to the 26B A4B Gemma4 snapshot
- logs no longer fail on missing `layers.5.self_attn.v_proj.weight`
- `/health` responds successfully

## Manual TurboQuant real-checkpoint parity harness

Purpose: collect narrow-scope dense-vs-TurboQuant evidence on the real 26B A4B snapshot without changing production behavior.

Supported slice only:
- restored prefix
- `batch=1`
- `q_len>=1`
- restored layers may be `full_attention` or `sliding_attention`
- sliding-window TurboQuant restore/decode is bounded to layers whose effective restored window still leaves at least `16` compressed-history tokens after preserving the exact recent buffer
- shared-KV broadening is bounded to explicit owner/source mappings already encoded by the Gemma4 config:
  - only the source layer owns stored/exported KV
  - shared layers remain non-owning on extract and resolve restored-prefix lookup through their configured source layer
  - unsupported shared-source cases stay on explicit dense fallback
- hybrid history policy only:
  - compressed TurboQuant history is used only for sufficiently old tokens
  - the newest `8` restored tokens stay exact/dense in the live model KV buffer
  - TurboQuant compressed-history restore activates only when at least `16` older tokens remain after carving out that exact recent buffer
  - shorter restored histories explicitly fall back to dense restore for the whole layer

Restore boundary:
- if **every** restored layer is in the supported narrow slice, TurboQuant decode-prefix may stay enabled for those layers
- when enabled, restore is now hybrid rather than fully prefix-compressed: old history stays in the TurboQuant decode-prefix cache and the newest `8` restored tokens stay dense in the live KV cache
- sliding-window layers now keep an explicit windowed prefix budget during restore/decode: only the tail of the compressed prefix that still fits inside the current sliding window contributes scores/values, and that budget shrinks to zero as dense/live KV fills the window
- if **any** restored layer is unsupported (for example a malformed shared-KV source mapping, or a sliding-window layer whose effective restored window is too short), the adapter now performs a **semantic-preserving dense restore fallback for all layers** during restore/decode
- if restored history is too short to leave at least `16` compressed-history tokens after carving out the exact recent buffer, the adapter performs an explicit dense fallback for that restore rather than pretending TurboQuant is active
- this is intentionally **not** universal shared-KV TurboQuant support, and it is **not** TurboQuant sliding batch decode support; it is a bounded restored-prefix parity slice with explicit shared-layer ownership rules

Current adapter batch-decode broadening status:
- `turboquant` runtime batch decode is now supported only for `q_len=1` batched decode over restored KV with mixed positions and padding masks
- the supported batch path imports stored TurboQuant KV into dense batched model state for the decode step, then re-exports TurboQuant envelopes on extraction
- `bf16_dense` remains unchanged in this slice
- shared-KV and sliding-window Gemma4 layers remain explicitly out of scope for batch decode in this slice

Command:

```bash
CRANE_GEMMA4_REAL_CHECKPOINT=/models/google/gemma-4-26B-A4B-it/1db3cff1840c2ae59759d8e842ff37831cf8cb63 \
CRANE_GEMMA4_REAL_PARITY_PROMPT="Write one short sentence about cranes." \
CRANE_GEMMA4_REAL_PARITY_STEPS=4 \
cargo test -p crane-oai gemma4_real_checkpoint_turboquant_restore_parity_harness -- --ignored --nocapture
```

Notes:
- Run this only on a machine/pod where the real checkpoint path is mounted and a suitable accelerator is available.
- The harness loads the real checkpoint sequentially in `bf16_dense` and `turboquant` modes, then restores TurboQuant KV into a fresh adapter to exercise the actual restore path.
- The harness is expected to print lines beginning with `[gemma4-real-parity]` that include:
  - selected device/dtype and checkpoint path
  - prompt token count and decode-step count
  - prefill dense/turbo top-1 token IDs and max-abs logit drift
  - enabled decode-prefix layer count vs total layer count
  - open-loop dense/turbo generated token streams and the first divergence step, if any
  - per-step open-loop dense/turbo top-1 token IDs, top-5 sets, and max-abs logit drift
  - per-step dense/turbo top-1 token IDs, top-5 sets, and max-abs logit drift
- During restore/decode, structured tracing should also emit grep-friendly TurboQuant decision logs:
  - `event=gemma4_turboquant_restore_summary`
  - `event=gemma4_turboquant_restore_layer`
  - `event=gemma4_turboquant_decode_prefix_fallback`
- Supported decision categories are bounded and explicit:
  - `decision=turboquant_narrow_path` with `reason=supported_narrow_path` for `batch=1`, `q_len>=1`, all-layer-supported restored-prefix decode with hybrid history (`compressed_history_tokens>=16` and `exact_recent_tokens=8`), including bounded sliding-window layers whose active compressed-prefix tail still fits the current window and bounded shared-KV layers that resolve to an explicit source owner
  - `decision=dense_fallback` with reasons such as `history_too_short`, `mixed_layer_restore_unsupported`, `shared_kv_source_missing`, `shared_kv_unsupported`, `sliding_window_unsupported`, `backend_no_compressed_k_scores`, `batch_size_unsupported`, `query_len_unsupported`, `kv_groups_unsupported`, or `head_layout_unsupported`

Pass condition:
- the test exits successfully
- the open-loop dense/turbo generated token streams stay identical for the requested step count
- every compared decode step keeps dense/turbo top-1 agreement
- extracted restored KV payloads remain `TurboQuant` envelopes
- on the supported narrow V path, backend aggregation now consumes backend-owned grouped-int8 value metadata first; rowwise dense reconstruction remains the explicit fallback/import path

Failure interpretation:
- if the checkpoint path is inaccessible, the test fails immediately with a path/configuration message
- if `first_divergence_step` is not `None`, treat that step as the start of open-loop parity drift on the currently supported narrow path or its dense-fallback restore boundary
- if top-1 diverges on any step, treat that as real-checkpoint parity drift on the currently supported path/boundary
- if enabled decode-prefix owner count is lower than total layer count on a shared-KV model, that is expected bounded behavior: only owner layers hold prefix caches, while supported shared layers reuse those owner caches through explicit source mapping
- if hybrid restore is enabled, only the sufficiently old prefix is TurboQuant-compressed; the newest `8` restored tokens remain dense/exact by policy in this slice
- for supported sliding-window layers, the active compressed-prefix tail should monotonically shrink as decode grows the live window; once the live dense KV fully occupies the window, decode-prefix usage for that layer should naturally fall to zero without reintroducing stale prefix attention

## Rollback

1. Revert the deployment image to the prior digest.
2. Re-apply the deployment.
3. Confirm pod readiness and `/health` for the rolled-back image.
