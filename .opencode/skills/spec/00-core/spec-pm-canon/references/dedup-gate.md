# Dedup gate guide

Goal: prevent duplicates by forcing an explicit UPDATE/NEW decision grounded in repo reality.

## What to scan
1) PM canon:
- `status.md`
- `.pm/scopes/default/tickets.tsv` (titles, tags, deps)
- `.pm/scopes/default/criteria.tsv` (keywords, entities)
2) Spec canon:
- `docs/spec/spec-index.md`
- `docs/spec/traceability/traceability.md`
- existing `docs/spec/**` nodes
3) Code/test surface (only if needed for confidence):
- endpoints/routes
- DB schema/migrations
- feature flags
- existing tests referencing the same behavior

## Match rubric
Strong match signals:
- same user outcome (what the user gets) AND
- same primary surface (endpoint/command/UI flow) OR same domain entity/table AND
- similar constraints or error handling

Weak match signals:
- shared entity/module, but different user outcome
- shared acceptance intent category, but different surface

## Decision rule
- If any strong match exists: UPDATE (default).
- NEW requires explicit non-overlap proof: different contract, different state machine, or different user-visible outcome.

## Output required
- Consulted artifacts list
- Potential matches list (strong/weak/no-match)
- Final decision + rationale
