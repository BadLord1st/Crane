---
name: spec-git-committer
description: "Keep git history clean: concise messages, minimal diffs, and spec/test evidence in the PR narrative."
metadata:
  signature: "spec-git-committer :: Diff -> CommitPlan"
---

## When to use
- You are about to commit a non-trivial change.
- You need to split work into logical commits.

## Inputs
- Code diff.
- Ticket/spec references.

## Outputs
- Commit plan and commit message(s) linked to the ticket/spec.

## Protocol
1) Split commits by intent: spec/tests/implementation/ops changes.
2) Use descriptive messages: WHAT + WHY.
3) Reference SPEC IDs in PR description (not necessarily every commit message).

## Deliverables
- [ ] Commit plan.
- [ ] Example commit messages.

## Anti-patterns
- One giant commit for unrelated changes.
- Emoji commits in professional repos.

## References
- [Git Workflows](references/git-workflows.md)
- [Commit Conventions](references/commit-conventions.md)
