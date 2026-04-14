---
name: spec-gap-analysis
description: "Turn a spec/ticket into an execution plan: smallest steps first, with risks and verification points."
metadata:
  signature: "spec-gap-analysis :: (SpecNode, Snapshot) -> Plan"
---

## When to use
- The task is non-trivial (3+ steps) and needs a controlled plan.
- You suspect hidden dependencies or high regression risk.

## Inputs
- SpecNode (intended behavior).
- Snapshot (current reality).

## Outputs
- Plan: TODO list, sequencing, risks, verification commands, rollback notes (if needed).

## Protocol
1) List the minimal change surface (files/areas likely touched).
2) Produce a step-by-step plan: Spec → Tests → Code → Verify.
3) Attach a risk list: unknowns, integration points, high-churn zones.
4) Define verification points after each meaningful step.

## Deliverables
- [ ] TODO plan with ordering.
- [ ] Risks/unknowns clearly stated.
- [ ] Verification commands identified.

## Anti-patterns
- “We’ll figure it out later” planning.
- Big-bang refactors without checkpoints.

## References
- [Impact Analysis](references/impact-analysis.md)
- [Methodology](references/methodology.md)
