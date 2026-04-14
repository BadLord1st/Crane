---
name: spec-tdd
description: "Implement using a unit-boundary TDD loop: failing test → minimal fix → refactor → verify."
metadata:
  signature: "spec-tdd :: (SpecNode, Scenarios) -> Code"
---

## When to use
- You are implementing a change that can be validated at unit boundaries.
- You need high confidence with small increments.

## Inputs
- SpecNode + ACs.
- Scenario/test intent (from spec-bdd or QA plan).

## Outputs
- Code changes + unit tests that prove the behavior.

## Protocol
1) Write the smallest failing test that captures the requirement.
2) Implement the minimal change to make the test pass.
3) Refactor only if it reduces complexity (no “cleanup crusades”).
4) Re-run relevant tests after each meaningful change.

## Deliverables
- [ ] Unit tests added/updated.
- [ ] Implementation added/updated.
- [ ] Local verification evidence (basic suite).

## Anti-patterns
- Coding first, testing later.
- Refactoring the world while fixing one bug.

## References
- [TDD patterns](references/tdd-patterns.md)
- [Test Design Patterns](references/test-design.md)
