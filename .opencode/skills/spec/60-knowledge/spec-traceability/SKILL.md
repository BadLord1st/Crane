---
name: spec-traceability
description: "Maintain a single traceability map: SPEC → AC → SC → tests → code paths."
metadata:
  signature: "spec-traceability :: (SpecGraph, Repo) -> TraceMap"
---

## When to use
- You want proof that requirements are tested and implemented.
- The system is drifting or regressions repeat.

## Inputs
- Spec nodes (docs/spec/**).
- Test files and code diffs.

## Outputs
- docs/spec/traceability/traceability.md updated as a single source of truth.

## Protocol
1) For each SPEC ID, locate its AC and scenarios/tests.
2) Record the mapping in a compact table or lines with file paths.
3) Keep it in one place (do not duplicate the mapping everywhere).

## Deliverables
- [ ] Traceability map updated.
- [ ] Missing links identified (spec-only or code-only).

## Anti-patterns
- Duplicating trace tables across multiple files.
- Pretending traceability exists without file paths.

## References
- [Drift Detection](references/drift-detection.md)
- [Traceability Matrix Guide](references/matrix-guide.md)
