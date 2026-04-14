---
name: clean-code
description: "Clean Code audit for a PR/diff: flag readability/maintainability issues and propose low-risk refactors (no behavior changes, no architecture rewrites)."
metadata:
  signature: "clean-code :: (CodeDiff, Context) -> Findings + RefactorPlan"
---

## When to use
- You want a consistent Clean Code pass on a PR (readability, maintainability, simplicity).
- You’re about to merge code that will be frequently modified (hot paths, core domain).
- The team wants refactor guidance that preserves semantics (safe, incremental).

## Inputs
- Code diff (preferred) or a small set of files.
- Context:
  - Language/framework conventions (e.g., Python+Litestar).
  - Team style rules (linters/formatters, naming conventions) if any.
  - Constraints: “no behavior changes”, “no public API changes”, “touch only these modules”, etc.

## Outputs
- Findings, grouped and prioritized:
  - P0: correctness risk hidden by messy code (ambiguous logic, missing error paths).
  - P1: maintainability risks (SRP violations, high coupling, duplication, unclear names).
  - P2: readability polish (formatting, minor naming, comment hygiene).
- Refactor plan with small, reviewable steps.
- Suggested tests (only if refactor safety requires it).

## Protocol
1) Confirm scope: preserve behavior; do not invent requirements; avoid “rewrite everything”.
2) Read top-down: name the intent of each changed function/class in one sentence.
3) Naming pass:
   - Identify unclear identifiers; propose intention-revealing names.
   - Fix misleading booleans (prefer positive naming).
4) Function size & responsibility pass:
   - Split multi-purpose functions; isolate I/O from pure logic.
   - Push side effects to boundaries; keep core deterministic when feasible.
5) Complexity pass:
   - Flatten deep nesting; extract complex conditionals into named predicates.
   - Remove “magic” literals into constants/enums with domain meaning.
6) Duplication pass (DRY with judgment):
   - Identify copy-paste logic; suggest a single source of truth.
   - Avoid premature abstractions that reduce clarity.
7) Comments & clarity pass:
   - Remove “what” comments that restate code; keep “why”/rationale.
   - Prefer docstrings for public surfaces and tricky invariants.
8) Error-handling pass:
   - Ensure errors are explicit and consistent (exceptions vs sentinel returns).
   - Catch narrowly; keep top-level boundaries responsible for logging.
9) Produce the report:
   - Each finding must include: location, why it matters, minimal refactor, safety notes, and (optional) test suggestion.

## Deliverables
- [ ] Clean Code report: P0/P1/P2 findings with file:line anchors.
- [ ] Refactor plan: 3–7 small steps, each “safe to review”.
- [ ] Non-goals explicitly listed (what you intentionally did NOT touch).

## Anti-patterns
- Style bikeshedding (subjective preferences) without impact on readability/maintainability.
- Refactors that change runtime behavior “accidentally” (ordering, rounding, exceptions, concurrency).
- Big-bang rewrites instead of incremental steps.
- Abstracting too early (YAGNI) or adding “flexibility” nobody asked for.
- Removing essential comments that document constraints, invariants, or rationale.

## References
- [Clean Code Principles & Checklist](./references/clean-code-checklist.md)
- [Refactor Recipes (low-risk patterns)](./references/refactor-recipes.md)
- [Prompt Examples (how to invoke this skill)](./references/examples.md)
