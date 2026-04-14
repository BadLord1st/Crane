---
name: spec-verify
description: "Run verification commands and produce evidence: what ran, what passed, what failed, and why."
metadata:
  signature: "spec-verify :: Repo -> Evidence"
---

## When to use
- Before claiming the task is done.
- After any meaningful change.
- When diagnosing failures.

## Inputs
- Repo with changes (or current state for diagnosis).

## Outputs
- Evidence report: commands executed, results, failing logs (trimmed), next actions.

## Protocol
1) Find the canonical commands (package scripts, Makefile, CI config).
2) Run the smallest fast checks first (unit tests/lint).
3) Run the full suite required by the ticket before completion.
4) If something fails: capture minimal logs + failing test name, not a wall of noise.

## Deliverables
- [ ] Evidence: pass/fail + commands.
- [ ] Failure summary + likely root cause hints (if failing).

## Anti-patterns
- Saying “tests pass” without stating which tests/commands ran.
- Ignoring failing checks because “probably unrelated”.

## References
- [Quality Gates for PRs](references/quality-gates.md)
- [Verification Procedures](references/verification-procedures.md)