# Refactor Recipes (low-risk patterns)

Goal: refactors that are easy to review and low-risk when you must preserve behavior.

## 1) Extract function for “why” clarity
Before: a 25–80 line function mixing validation, decision logic, and I/O.
After:
- extract pure helpers with intention-revealing names
- keep the outer function as orchestration

Heuristic: if you’d add a comment to explain a block, try extracting it into a helper whose name *is* the comment.

## 2) Encapsulate complex conditionals
Before: multi-clause boolean expressions in-line.
After:
- introduce intermediate booleans with names
- or extract a predicate function

Safety notes:
- preserve short-circuit behavior if it matters
- keep evaluation order if side effects exist (ideally remove side effects)

## 3) Replace magic literals
Before: `0.1`, `"OK"`, `86400`, `"US"` appear without context.
After:
- constants with domain meaning
- enums for status-like strings

Safety notes:
- ensure constant scope is correct (module vs class vs function)

## 4) Split I/O from logic
Before: DB calls inside branching logic.
After:
- compute decision first (pure)
- perform I/O second (boundary)

Safety notes:
- preserve transaction boundaries
- preserve error propagation semantics

## 5) Simplify negative conditionals
Before: `if not is_valid: return ...` nested in other negations.
After:
- rename boolean to positive meaning
- prefer early returns only if it improves readability

## 6) Introduce a small data object
Before: multiple parallel values passed around (x, y, z, flags...).
After:
- dataclass/record with named fields

Safety notes:
- avoid changing public API unless explicitly allowed

## 7) Guardrail for DRY
Duplication is not always the enemy.
- If merging duplicates increases indirection or hides domain differences, keep them.
- If duplicates are truly identical knowledge, consolidate.
