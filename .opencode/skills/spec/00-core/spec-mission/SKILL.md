---
name: spec-mission
description: "High-order workflow: choose the smallest spec-driven pipeline needed for the task (do not load everything blindly)."
metadata:
  signature: "spec-mission :: Ticket -> Pipeline"
---

## When to use
- The task is large enough to require a repeatable, end-to-end protocol.

## Inputs
- Ticket (goal, constraints, mode REVIEW/WORK).

## Outputs
- A selected pipeline: which skills to load and in what order.

## Protocol
1) Classify the task: request / bughunt / explain(whatthefuck).
2) Select the minimal skills set for that class (lazy, not maximal).
3) Execute the pipeline: Spec → Tests → Code → Verify → KB (if needed).

## Deliverables
- [ ] Pipeline selection with rationale.
- [ ] Checklist of artifacts to produce.

## Anti-patterns
- Loading every skill “because it exists”.
- Skipping tests/verification because “it’s small”.

## References

- **[Mission description example](references/phase-reference.md)**
