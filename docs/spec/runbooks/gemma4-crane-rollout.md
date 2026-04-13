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

## Rollback

1. Revert the deployment image to the prior digest.
2. Re-apply the deployment.
3. Confirm pod readiness and `/health` for the rolled-back image.
