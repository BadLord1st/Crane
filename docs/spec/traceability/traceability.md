# Traceability Map

| SPEC | AC | SC | Ticket | Tests | Code Paths |
|---|---|---|---|---|---|
| SPEC-GEMMA-001, SPEC-GEMMA-003, SPEC-GEMMA-005 | AC-1, AC-3 | SC-1, SC-5 | T-0001 | `crane-core/src/models/gemma4/modeling.rs` (`config_deserialization_preserves_attention_k_eq_v_flag`, `attention_allows_missing_v_proj_when_attention_k_eq_v_is_enabled`, `attention_still_requires_v_proj_when_attention_k_eq_v_is_disabled`, `router_route_handles_non_contiguous_topk_gather_inputs`) | `crane-core/src/models/gemma4/modeling.rs` |
