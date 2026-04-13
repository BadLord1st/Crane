# Programmer Output

## Implemented scope

- Implemented the Stage B `T-0001` Gemma4 text-attention loader fix for `text_config.attention_k_eq_v=true`.
- Gemma4 attention now treats `v_proj` as optional only when that config flag is active and derives V from the K projection when the checkpoint omits `v_proj.weight`.
- Preserved existing behavior for checkpoints that still provide `v_proj`, and kept `v_proj` mandatory when `attention_k_eq_v` is false.

## Files changed

- `crane-core/src/models/gemma4/modeling.rs`
  - added `Config.attention_k_eq_v` with serde defaulting
  - updated `Attention` to allow missing `v_proj` only for k==v checkpoints
  - added targeted unit tests for config parsing and attention loader behavior
- `docs/spec/traceability/traceability.md`
  - added `T-0001` traceability row for the new Gemma4 tests/code path

## Tests added / updated

Added in `crane-core/src/models/gemma4/modeling.rs`:

- `config_deserialization_preserves_attention_k_eq_v_flag`
- `attention_allows_missing_v_proj_when_attention_k_eq_v_is_enabled`
- `attention_still_requires_v_proj_when_attention_k_eq_v_is_disabled`

## Commands run

- `cargo test -p crane-core gemma4::modeling::tests`
  - result: `3 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out`
- `cargo fmt --all`
- `cargo test -p crane-core gemma4::modeling::tests`
  - result: `3 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out`

## Remaining risks / notes

- This slice is intentionally minimal and unit-tested only; it does not include a real checkpoint fixture or full `Model::new` startup coverage yet.
- The fallback path derives V from the K projection output before K-specific rotary application, which matches the intended loader contract for missing-`v_proj` k==v checkpoints.
