---
name: spec-release
description: "Produce a release/runbook plan: rollout, rollback, and verification steps."
metadata:
  signature: "spec-release :: (SpecNode, Repo) -> ReleaseRunbook"
---

## When to use
- The change is risky or needs controlled rollout.
- You need a clean deploy/rollback recipe.

## Inputs
- SpecNode + constraints.
- Current deploy/CI configuration.

## Outputs
- Runbook node: deploy steps, rollback steps, verification checklist.

## Protocol
1) Read existing CI/CD pipeline and deployment tooling.
2) Define rollout steps + smoke checks.
3) Define rollback steps (what to revert, how to validate).

## Deliverables
- [ ] Release runbook draft linked from spec-index.
- [ ] Verification checklist.

## Anti-patterns
- Deploy instructions that assume invisible tribal knowledge.
- No rollback plan for risky changes.

## References
- [Rollout & Rollback Strategies](references/rollout-rollback.md)
- [Release Guide Template](references/release-guide.md)
