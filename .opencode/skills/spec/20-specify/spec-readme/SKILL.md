---
name: spec-readme
description: Create/update a project README.md. Use when the user asks "write a readme", "document this project", or wants a high-signal onboarding + ops doc.
metadata:
  signature: "spec-readme :: (Repo, Snapshot) -> README.md"
---

# README Generator

You are an expert technical writer. Produce a README.md that is:
- fast to start from zero (fresh machine),
- honest (only commands/configs that exist in the repo),
- operational (how to run/test/release/troubleshoot).

## When to use
- The user asks to create/update `README.md`.
- Onboarding is missing or out of date.

## Inputs
- Repo (read-only exploration).
- Snapshot (recommended) from `spec-recon`.
- Optional supporting artifacts (prefer linking over duplicating):
  - RunbookDraft from `spec-env-scout`.
  - ReleaseRunbook from `spec-release`.

## Output
- `README.md` in the project root.

## Protocol
1) Inventory authoritative sources (package scripts, Makefile, CI, Docker, env examples).
2) Decide the doc intent and scope:
   - onboarding (local dev),
   - architecture (only the necessary mental model),
   - operations (deploy/release/troubleshoot).
3) Write the README using the outline/template in references.
4) Prefer linking to existing spec/runbook nodes over copying them into README.
5) Ask questions only if critical (project purpose unclear; missing required secrets/URLs).

## Deliverables
- [ ] `README.md` updated with copy/paste commands that match the repo.
- [ ] Clear prerequisites and a "First 15 minutes" path.
- [ ] Environment variables documented (required vs optional).
- [ ] Troubleshooting section covers the top 3-5 likely failures.

## Anti-patterns
- Inventing commands/platforms instead of citing repo sources.
- Turning README into a mega-spec (runbooks and release plans belong in `docs/spec/runbooks/**`).
- Teaching the ecosystem instead of documenting this repo.

## References
- [Source scan checklist](references/source_scan.md)
- [README outline (template)](references/readme_outline.md)
- [Long-form example (skeleton)](references/long_form_example.md)
- [Go projects notes](references/go_projects.md)
- [Python projects notes](references/python_projects.md)
- [Tables and snippets](references/snippets.md)
- [Deployment platform detection](references/platform_detection.md)
- [Writing principles](references/writing_principles.md)
