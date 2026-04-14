---
name: spec-error-kb
description: Capture meaningful failures into docs/agent/error-kb.md with root cause, fix, and prevention + regression reference.
metadata:
  signature: "spec-error-kb :: Incident -> KBEntry"
---

## When to use
- A real bug/incident/mistake happened (not a trivial typo).
- A regression was fixed and you want to prevent repeat.

## Inputs
- Incident context: symptoms, logs, steps to reproduce, root cause, fix PR/diff.

## Outputs
- A new KB entry in docs/agent/error-kb.md that is searchable and reusable.

## Protocol
1) Write a concise entry: Symptoms → Root Cause → Fix → Prevention → Regression test link.
2) Include file paths and commands to reproduce/verify.
3) Prefer a short title that matches future search terms.

## Deliverables
- [ ] KB entry added/updated.
- [ ] Regression test reference included.

## Anti-patterns
- Writing a novel instead of an actionable entry.
- Recording “we fixed it” without the cause and prevention.

## References
- [Common Bug Patterns](references/common-patterns.md)
- [KB Entry Format](references/entry-format.md)