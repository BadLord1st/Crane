# Programmer Output

## Implemented scope

- Fixed the narrow live Gemma4 MoE regression in `Gemma4TextExperts::forward` where router weights were rebuilt as default `F32` tensors before multiplying `BF16` expert outputs.
- Kept the earlier loader and router contiguity fixes intact.

## Files changed

- `crane-core/src/models/gemma4/modeling.rs`
  - added `Gemma4TextExperts::routing_weights_tensor(...)` to materialize router weights on the expert-output device and dtype
  - updated the expert scaling multiply path to use the expert-output dtype instead of default `F32`
  - added a focused regression test covering `BF16` expert-output scaling with router weights
- `docs/spec/traceability/traceability.md`
  - added the new BF16 router-weight regression test to the existing `T-0001` traceability row

## Tests added / updated

- Added `experts_router_weights_match_bf16_expert_output_dtype`
- Preserved and re-ran:
  - `config_deserialization_preserves_attention_k_eq_v_flag`
  - `attention_allows_missing_v_proj_when_attention_k_eq_v_is_enabled`
  - `attention_still_requires_v_proj_when_attention_k_eq_v_is_disabled`
  - `router_route_handles_non_contiguous_topk_gather_inputs`

## Commands run

- `cargo fmt --all`
  - result: passed
- `cargo test -p crane-core gemma4::modeling::tests`
  - result: passed
  - exact summary: `5 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out`
  - note: unrelated existing warnings were emitted from other model modules during test compilation; no Gemma4 test failures occurred

## Remaining risks / notes

- The fix is intentionally limited to the router-weight multiply boundary in Gemma4 MoE experts.
- CPU/F32 behavior is preserved because the helper casts weights to the current expert-output dtype rather than forcing a new dtype.
- Live CUDA inference should now avoid the previously observed `mul` BF16/F32 mismatch at this boundary.
