# Programmer Output

## Implemented scope

- Narrow fix for CUDA top-k planning in `crane-oai/src/engine/sampling.rs`.
- Planner now caps GPU top-k against the guaranteed non-opt-in dynamic shared-memory budget instead of the larger opt-in budget.
- Added focused unit coverage for top-p fallback request resolution and safe top-k cap math.

## Files changed

- `crane-oai/src/engine/sampling.rs`
- `crane-core/src/fused_ops/cuda_impl.rs`

## Tests added / updated

- `requested_gpu_topk_uses_top_p_fallback_when_top_k_disabled`
- `requested_gpu_topk_preserves_explicit_top_k`
- `max_gpu_topk_for_48k_shared_mem_budget_is_48`
- `max_gpu_topk_for_large_budget_is_still_kernel_capped`
- `build_gpu_topk_plan_caps_requested_top_k_to_safe_budget`
- `build_gpu_topk_plan_uses_cpu_when_capped_top_k_reaches_vocab`

## Commands run

- File reads only.
- No test execution performed; verification is for QA per role contract.

## Remaining risks / notes

- This fix intentionally does **not** enable CUDA opt-in shared memory for the kernel. It keeps launches in the safe ≤48 KiB budget on hardware like RTX 4070-class GPUs.
- Effective GPU top-k is now capped to 48 on the default guaranteed-safe budget, so explicit `top_k=64` requests may be reduced to 48 until kernel opt-in shared memory is implemented.
- Traceability was not updated because the current spec map only covers the Gemma4 loader contract and does not yet define a sampling-spec row for this runtime slice.

## Spec Fix Proposal (only if needed)

- None.
