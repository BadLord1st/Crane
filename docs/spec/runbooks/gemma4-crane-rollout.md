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
- `q_len=1`
- non-sliding / `full_attention` layers only
- no shared-KV broadening

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
  - per-step dense/turbo top-1 token IDs, top-5 sets, and max-abs logit drift

Pass condition:
- the test exits successfully
- every compared decode step keeps dense/turbo top-1 agreement
- extracted restored KV payloads remain `TurboQuant` envelopes

Failure interpretation:
- if the checkpoint path is inaccessible, the test fails immediately with a path/configuration message
- if top-1 diverges on any step, treat that as real-checkpoint parity drift on the currently supported narrow path
- if enabled decode-prefix layer count is lower than expected, inspect sliding/shared-KV gating before widening scope

## Rollback

1. Revert the deployment image to the prior digest.
2. Re-apply the deployment.
3. Confirm pod readiness and `/health` for the rolled-back image.
