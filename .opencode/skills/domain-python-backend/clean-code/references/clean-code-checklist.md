# Clean Code Principles & Checklist

Source note: distilled from the project PDF section “Clean Code Principles – Writing Maintainable Code” and generalized to be language-agnostic.

## Checklist (review in this order)

1) Intent is obvious
- Names communicate purpose (variables, functions, classes).
- Public APIs have docstrings/comments describing contracts and edge cases.

2) Single responsibility
- Functions do one thing; classes represent one concept.
- I/O is separated from pure decision logic where practical.

3) No magic literals
- Replace unexplained numbers/strings with constants, enums, or domain objects.
- Include units/intent in naming (e.g., timeout_seconds).

4) Complexity is tamed
- Deep nesting is reduced; early returns are used when they clarify.
- Complex conditionals are extracted into named predicates.

5) DRY (with judgment)
- Repeated logic is consolidated into a single source of truth.
- Avoid “clever” abstractions that make simple flows harder to follow.

6) Error handling is explicit
- A function’s failure mode is consistent (exceptions vs sentinel returns).
- Exceptions are caught narrowly; boundary layers handle logging.

7) Comments are intentional
- Comments explain “why”, constraints, invariants, trade-offs.
- Comments that restate code (“what”) are avoided.

8) Consistency & standards
- Formatting, naming, and structure follow team conventions.
- Linters/formatters are treated as the baseline.

9) Tests support refactor safety
- If code is hard to test, it’s a design signal: too much responsibility or tight coupling.
- Prefer deterministic, side-effect-light units when possible.

10) KISS / YAGNI
- Solve today’s problem cleanly; don’t build speculative extension points.

## “Smells” that usually deserve a finding
- doStuff / processData style names.
- Functions with “and/then” in the name.
- Mixed concerns: business logic + DB/network + formatting in one function.
- Repeated try/except blocks with slightly different messages.
- Boolean flags with double negatives.
- “God” classes/modules with unrelated responsibilities.

## Minimal evidence for each finding
- What it impacts (readability, change risk, bug probability).
- Why now (diff touched it; upcoming changes; repeated incidents).
- Smallest safe refactor that improves it.
