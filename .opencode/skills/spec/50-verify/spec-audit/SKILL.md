---
name: spec-audit
description: "Skeptical audit: hunt spec drift, missing edge cases, weak tests, and misleading assumptions."
metadata:
  signature: "spec-audit :: (SpecNode, Tests, Code) -> Findings"
---

## When to use
- Before closing a non-trivial ticket.
- When the change touches critical paths.
- When regressions have happened before.

## Inputs
- SpecNode + acceptance criteria.
- Tests added/updated.
- Code diff.

## Outputs
- Findings: gaps, drift, missing tests, risky assumptions, concrete fixes.

## Protocol
1) Check: does code match spec? If not, identify drift direction.
2) Check: does each AC have test coverage?
3) Check: edge cases and failure modes are handled as specified.
4) Check: risky areas nearby for similar bugs (class of issues).

## Deliverables
- [ ] Audit report with concrete actions.
- [ ] List of missing/weak tests.
- [ ] Spec drift notes (if any).

## Anti-patterns
- Rubber-stamping without reading the diff.
- Demanding speculative improvements unrelated to the ticket.

## References
- [Regression Hunt Patterns](./references/regression-patterns.md)
- [Audit Checklists](./references/checklists.md)
