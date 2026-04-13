# Thread: 2026-04-13-gemma4-loader-fix

## Request Summary
Fix Crane so it can load the target smaller Gemma4 variant from the cluster PVC, then build a CUDA Docker image on my69, push it, and redeploy in the cluster.

## Normalized Request
- Outcome: Crane in the CUDA Docker image must start successfully in cluster with the target smaller Gemma4 variant from PVC, without layer-shape loader crashes.
- In scope:
  - diagnose and fix Gemma4 variant loading in Crane;
  - build amd64/CUDA Docker image on my69;
  - update cluster deployment and verify health.
- Out of scope:
  - changing PVC model contents;
  - redesigning deployment architecture;
  - broad refactors outside Gemma4 loading path.
- Constraints:
  - build host is my69 due to amd64/CUDA requirement;
  - cluster access is available via kubectl.
- Unknowns:
  - exact smaller-model path and variant structure;
  - exact failing runtime tensor/shape mismatch;
  - image tagging/pushing convention on my69.

## Project Frame
- meaning: Make Gemma4 loading variant-compatible enough for the target smaller cluster model while preserving current Gemma4 startup behavior for supported variants.
- workstreams:
  - Crane loader/spec/test fix
  - build/release path on my69
  - cluster rollout verification
- ticket strategy: multiple tickets
- primary spec strategy: create one new primary spec for Gemma4 variant-compatible loading, then update traceability and rollout notes.
- barriers:
  - Stage A: confirm spec and exact change surface
  - Stage B: code/test implementation complete
  - Stage C: verification on local tests + cluster rollout evidence
- open questions:
  - exact smaller PVC model path
  - whether live repro in cluster is needed before code change

## Ticket Set
- T-0001
- T-0002

## Artifact Index
- primary spec: docs/spec/features/gemma4-variant-compatible-loading.md
- secondary specs: docs/spec/runbooks/gemma4-crane-rollout.md
- code paths: crane-core/src/models/gemma4/model.rs; crane-core/src/models/gemma4/modeling.rs; crane-oai/src/engine/model_factory.rs; crane-oai/src/engine/adapters/gemma4_adapter.rs
- test paths: pending
- evidence: pending

## Phase Status
- Stage A: in_progress
- Stage B: pending
- Stage C: pending

## Decisions / Notes
- Current cluster deployment exists but is scaled to 0 replicas.
- Deployment currently points to /models/google/gemma-4-26B-A4B-it/1db3cff1840c2ae59759d8e842ff37831cf8cb63.
- Need spec canon creation in Crane because docs/spec is missing.
- PVC inspection confirmed two Gemma4 variants: 26B-A4B-it and 31B-it.
- The smaller target is the 26B-A4B-it snapshot currently referenced by deployment.
- Safetensors index for 26B shows `model.language_model.layers.5.self_attn.v_proj.weight` is absent while neighboring layers have it; this matches `attention_k_eq_v=true` and points to a likely loader bug in `crane-core/src/models/gemma4/modeling.rs` where `v_proj` is always required.
- After rollout, service startup succeeded and `/health` became OK, but live inference failed with `Prefill forward failed: gather only supports contiguous tensors`.
- Likely hot spot is Gemma4 MoE routing in `crane-core/src/models/gemma4/modeling.rs`, especially `gather` over tensors produced by softmax/expand in `Gemma4TextRouter::route`.
- After fixing gather contiguity and re-rolling out, live inference advanced further but now fails with `Prefill forward failed: dtype mismatch in mul, lhs: BF16, rhs: F32`.
- Likely hot spot is `Gemma4TextExperts::forward`, where routing weights are reconstructed as default F32 tensors before multiplying BF16 expert outputs.
