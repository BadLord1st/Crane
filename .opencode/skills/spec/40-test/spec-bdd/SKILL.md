---
name: spec-bdd
description: Derive BDD scenarios from acceptance criteria with traceable IDs (SPEC/AC/SC) and clear expected outcomes.
metadata:
  signature: "spec-bdd :: SpecNode -> Scenarios"
---

## When to use
- You need executable behavior checks from the spec.
- You want stable traceability from intent to tests.

## Inputs
- SpecNode with Acceptance Criteria.
- Existing test framework conventions (if any).

## Outputs
- Scenarios mapped to AC items: SPEC-* → AC-* → SC-* → test path

## Protocol
1) For each AC, write 1..N scenarios: happy path + key failure path + essential edge case.
2) Every scenario must reference: at least one SPEC-*, one AC-*, one SC-*.
3) Prefer externally observable assertions (responses, events, stored state).
4) If the repo has no BDD harness, draft scenarios first; propose the smallest harness that matches the project tooling.

## Deliverables
- [ ] Scenario list + mapping to AC.
- [ ] Notes on fixtures/test data.
- [ ] Spec gaps found (if any).

## Anti-patterns
- Scenarios that test implementation details.
- Inventing behavior not present in the spec.
- Untraceable scenarios without IDs.

## References
- [Patterns for BDD Scenarios](references/patterns.md)
- [Tutorials for Writing BDD Scenarios](references/tutorials.md)