# Prompt Examples (how to invoke spec-clean-code)

These examples assume you paste a diff or point to files.

## Example 1 — PR Clean Code audit (strictly no behavior change)
Request:
- “Run spec-clean-code on this PR diff. Preserve behavior and public API. I only want refactors that are safe and reviewable.”
Expected output shape:
- P0/P1/P2 findings with file:line anchors
- refactor plan in 3–5 steps
- explicit non-goals (“did not propose architecture changes, did not change DB schema”)

## Example 2 — Naming + complexity focus
Request:
- “Audit this file for naming and complex conditionals. No refactor beyond extracting predicates/constants.”
Expected output shape:
- list of renames with rationale
- 2–6 conditional extractions with suggested predicate names

## Example 3 — DRY pass with caution
Request:
- “Identify real duplication in these two modules. Propose consolidation only when it reduces maintenance effort without adding indirection.”
Expected output shape:
- duplicates grouped by ‘identical knowledge’ vs ‘similar but domain-different’
- for each consolidation: smallest safe shared helper + call-site changes

## Example 4 — Boundary error-handling clarity
Request:
- “Check error handling in these handlers/services. Keep exceptions consistent and avoid blanket catches.”
Expected output shape:
- list of inconsistent failure modes
- recommended pattern per boundary layer (API/worker/CLI)
- minimal code-level suggestions
