# Alembic Migration Plan (Template)

## Summary
- What changes and why
- Current head(s) / DB revision state

## Generation
- Commands to create the revision (autogen vs manual)
- Expected file path and revision message

## Script Review Notes
- Risky ops (drops, type changes, constraint rebuilds)
- Data migration guards/idempotency
- Downgrade behavior

## Dry-run SQL
- Offline `--sql` commands and what to inspect

## Apply / Rollout
- Staging steps + verification
- Production steps + stop conditions

## Verification
- Minimal queries / checks proving the change is correct

## Rollback
- Commands + limitations (what cannot be undone)
