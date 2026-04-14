---
name: spec-deps
description: "Dependency research: confirm versions, APIs, and compatibility using authoritative sources and repo constraints."
metadata:
  signature: "spec-deps :: (Repo, Question) -> Constraints"
---

## When to use
- Library behavior matters to correctness.
- You suspect version mismatch or breaking change.

## Inputs
- Repo dependency manifests (package.json, pyproject.toml, etc.).
- Specific API/behavior question.

## Outputs
- Confirmed constraints: version, API usage, compatibility notes, and spec updates if needed.

## Protocol
1) Read actual dependency constraints in the repo.
2) Prefer official docs/release notes for the relevant version range.
3) Write down the constraint in the spec (do not assume “latest”).

## Deliverables
- [ ] Version/API decision with evidence.
- [ ] Spec note capturing the constraint.

## Anti-patterns
- Assuming the latest docs match the repo’s version.
- Changing dependencies as a side quest.

# References

- **[Security Scanning Guide](references/security-scanning.md)**
- **[Upgrading Dependencies Strategy](references/upgrade-strategies.md)**
