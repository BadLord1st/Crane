---
name: spec-recon
description: "Read-only reconnaissance: map the repo into a short, evidence-based Snapshot for downstream spec/test work."
metadata:
  signature: "spec-recon :: Repo -> Snapshot"
---

## When to use
- You need to understand how the project is structured today before proposing changes.
- You need concrete run/test entrypoints, existing patterns, and risk areas.

## Inputs
- Repository filesystem (read-only).

## Outputs
- Snapshot (<=200 lines): stack, entrypoints, test/lint commands, key modules, docs/spec locations, risks.
- Suggested next skills to load (minimal set).

## Protocol
1) Identify language(s), build tool, package manager, and primary entrypoints.
2) Locate docs/specs/tests and follow existing conventions (do not invent a new layout yet).
3) Map system boundaries: services/modules, external dependencies, data stores.
4) Extract the *real* commands to run/build/test (or point to the file that defines them).
5) Emit a compact Snapshot with file paths (not vibes).

## Deliverables
- [ ] Snapshot with evidence and file paths.
- [ ] Recommended minimal next skills to load (lazy).

## Anti-patterns
- Writing a spec before you know how the repo actually works.
- Listing every file in the repo instead of the relevant surface.
- Loading a pile of skills “just in case”.

## References

- **[Architecture patterns](references/architecture-patterns.md)**
- **[Exploration techniques](references/exploration-techniques.md)**
