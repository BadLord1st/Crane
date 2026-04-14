# Project Status

Last updated: 2026-04-13T00:00:00Z

## CORE
(derived from `.pm/scopes/default/core.md`)

## Now / In Progress
- T-0001 - Make Gemma4 loading variant-compatible
- T-0002 - Build CUDA image and roll out fixed Crane

## Blocked
- (none)

## Next
- Verify pod readiness and /health for ai-crane after model load completes

## Recent pulse (append-only)
- 2026-04-14T00:00:00Z NOTE: Recorded real PVC checkpoint TurboQuant parity failure prior to mixed-restore drift fix: prefill ok, decode diverged at step 1 with enabled_layers=5/30
- 2026-04-13T00:00:00Z NOTE: Attempted grouped-KV Gemma4 decode attention optimization; live decode failed with matmul shape mismatch and deployment was rolled back to previous working digest
- 2026-04-13T00:00:00Z NOTE: Fixed Gemma4 router gather contiguity and rolled out updated image
- 2026-04-13T00:00:00Z NOTE: Fixed Gemma4 router weight dtype and rolled out updated image
- 2026-04-13T00:00:00Z NOTE: Fixed CUDA top-k launch budget and rolled out image sha256:5dab0c85481789ce64198a8c5a3e8cb21282811863cd290d17bca8b8e0302582
- 2026-04-13T00:00:00Z NOTE: Verified /health and chat completion success on final rollout without GPU top-k fallback warnings
- 2026-04-13T00:00:00Z NOTE: Confirmed target PVC model path and identified missing layers.5.self_attn.v_proj.weight in 26B Gemma4 checkpoint
- 2026-04-13T00:00:00Z STATE: T-0002 -> in_progress
- 2026-04-13T00:00:00Z NOTE: Built and pushed image 1stbadwolf/crane@sha256:1af34fa1778642e43186976c83e6409e2c8d60a3b7a05a1bb4245705db6598b1 from my69
- 2026-04-13T00:00:00Z NOTE: Updated ai-crane deployment image and scaled deployment to 1 replica
- 2026-04-13T00:00:00Z CREATED: T-0001 Make Gemma4 loading variant-compatible
- 2026-04-13T00:00:00Z CREATED: T-0002 Build CUDA image and roll out fixed Crane
- 2026-04-13T00:00:00Z STATE: T-0001 -> in_progress
- 2026-04-13T00:00:00Z NOTE: Created spec canon skeletons and handoff thread 2026-04-13-gemma4-loader-fix
- 2026-02-13T12:00:00Z INIT: Initialized PM canon for scope=default

## Recent evidence
- T-0001 tests passed for Gemma4 k==v attention loader behavior.
- T-0002 image built on my69 and pushed as sha256:1af34fa1778642e43186976c83e6409e2c8d60a3b7a05a1bb4245705db6598b1.
- ai-crane deployment updated and pod started with the new image.
- Final rollout image: sha256:5dab0c85481789ce64198a8c5a3e8cb21282811863cd290d17bca8b8e0302582.
- Final health check: 200 OK.
- Final chat completion: 200 OK, Gemma4 responded successfully on CUDA.
- Broken attention optimization was rolled back; cluster is back on working digest sha256:ce2f0deaaab340eeb8e03190c8bda934c78c43c938bb06135124322f7305e49e.
- Real PVC checkpoint TurboQuant parity was recorded as failing before the mixed-restore drift fix (prefill and first decode step ok, divergence at decode step 1).
