---
name: spec-env-scout
description: Extract run/test/lint/build commands and environment requirements into runbook nodes.
metadata:
  signature: "spec-env-scout :: Repo -> RunbookDraft"
---

## When to use
- You need repeatable local dev and CI instructions.
- The project is hard to run, or commands are unclear.

## Inputs
- Repo files: README, package scripts, Makefile, CI configs, docker-compose, env examples.

## Outputs
- Runbook node drafts under docs/spec/runbooks/: local-dev.md / ci.md / troubleshooting.md (as needed).

## Protocol
1) Locate the authoritative command sources (scripts, Makefile, CI).
2) Produce minimal runnable command blocks (copy/paste friendly).
3) List prerequisites and required env vars.
4) Add a “First 15 minutes” checklist (install → run → test).

## Deliverables
- [ ] Runbook draft(s) with real commands.
- [ ] List of required env vars/services.

## Anti-patterns
- Guessing commands instead of reading them from the repo.
- Mixing feature requirements into runbooks.

## References

- **[CI/CD pipeline](references/ci-cd-pipeline.md)**
- **[Environment Setup Detection](references/environment-detection.md)**
