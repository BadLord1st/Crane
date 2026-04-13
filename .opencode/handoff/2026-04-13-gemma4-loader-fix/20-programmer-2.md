# Programmer Output

## Implemented scope

- Implemented the narrow Stage B continuation fix for the live Gemma4 inference regression in `Gemma4TextRouter::route`.
- Kept the existing startup-side `attention_k_eq_v` loader fix intact.
- Made the router's `gather` inputs contiguous at the narrowest failing boundaries:
  - `topk_idx` after `narrow(...)`
  - `scale_2d` after `unsqueeze(...).expand(...)`

## Files changed

- `crane-core/src/models/gemma4/modeling.rs`
  - added `contiguous()` before both router-path `gather` calls require it
  - added a targeted router test that exercises multi-token top-k routing over non-contiguous gather inputs
- `docs/spec/traceability/traceability.md`
  - added the new router regression test to the existing `T-0001` traceability row

## Tests added / updated

Added in `crane-core/src/models/gemma4/modeling.rs`:

- `router_route_handles_non_contiguous_topk_gather_inputs`

Existing Gemma4 loader tests remain unchanged and passing:

- `config_deserialization_preserves_attention_k_eq_v_flag`
- `attention_allows_missing_v_proj_when_attention_k_eq_v_is_enabled`
- `attention_still_requires_v_proj_when_attention_k_eq_v_is_disabled`

## Commands run

- `cargo fmt --all`
  - result: passed
- `cargo test -p crane-core gemma4::modeling::tests`
  - result: `4 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out`
- `cargo test -p crane-core gemma4::modeling::tests`
  - result: `4 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out`
  - note: repeated after formatting to confirm the final tree still passes

## Remaining risks / notes

- This fix is intentionally limited to the live inference router path and does not broaden the earlier startup-compatibility slice.
- The regression came from Candle `gather` requiring contiguous source/index layouts; the multi-token `narrow(...)` view and expanded per-expert scale tensor violated that requirement.
- No broader MoE refactor was made.
