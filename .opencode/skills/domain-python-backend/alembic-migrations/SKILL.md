---
name: alembic-migrations
description: "Alembic-only migrations: generate/review safe SQLAlchemy schema+data migrations (revision --autogenerate, env.py target_metadata, offline --sql), with rollback + verification. Do NOT use for Prisma/Flyway/Liquibase."
metadata:
  signature: "spec-alembic-migrations :: (Repo, ChangeRequest) -> AlembicMigrationPlan"
---

## When to use
- Repo uses SQLAlchemy + Alembic for schema evolution.
- You need a migration plan that is safe-by-default: review-first, dry-run SQL, rollback notes.
- You are creating or validating a new Alembic revision (schema and/or data migration).

## Inputs
- ChangeRequest:
  - What changed (models / constraints / indexes / data backfill).
  - Target DB dialect(s): postgres | mysql | sqlite | other.
  - Downgrade policy: required | best-effort | unsupported (explicit).
- Repo context:
  - Alembic layout: `alembic.ini` + `env.py` + `versions/` (or `migrations/`).
  - SQLAlchemy metadata entrypoint used for autogenerate (e.g., `Base.metadata`).

## Outputs
An **AlembicMigrationPlan**:
- `summary`: 1–3 sentences of the intended DB change
- `preflight`: checks to run (revision state, heads, env.py wiring)
- `generation`: exact commands to create a revision (autogen or manual)
- `script_patch`: migration file content (or a focused diff) with `upgrade()` + `downgrade()`
- `dry_run_sql`: offline-mode `--sql` commands to inspect generated SQL
- `verification`: queries / app checks proving success
- `rollout`: staging→prod steps + rollback notes
- `risks`: locks, downtime, irreversibility, data safety concerns

## Protocol
1) Locate Alembic config and wiring:
   - Find `alembic.ini` and `env.py`.
   - Confirm `target_metadata` is correctly set to the project’s SQLAlchemy metadata (required for autogenerate).
2) Establish migration state:
   - Identify current head(s) and whether DB is up-to-date (Alembic tracks applied revision in `alembic_version`).
3) Choose strategy:
   - Simple additive changes: direct migration.
   - Potentially breaking changes: expand → backfill → switch → contract (two+ revisions).
   - SQLite / limited ALTER support: prefer batch operations.
4) Generate a candidate revision:
   - Prefer: `alembic revision --autogenerate -m "<message>"`
   - Or manual revision when autogenerate is unsafe/ambiguous.
5) Review and fix the generated script:
   - Autogenerate produces *candidate* operations; review and edit by hand.
   - Ensure constraints/index names, server defaults, nullability transitions, and data steps are explicit.
   - Ensure `downgrade()` matches project policy (required / best-effort / explicitly unsupported).
6) Produce a dry-run SQL preview (do not apply yet):
   - Use offline `--sql` output for review; use `start:end` ranges only in offline mode.
7) Apply plan (only if explicitly requested):
   - Provide the exact `alembic upgrade ...` command(s) for staging first, then production.
   - Include rollback command(s) and “stop conditions”.
8) Add verification:
   - Provide minimal verification queries and/or smoke checks.
   - Recommend CI path that migrates up (and optionally down) on a fresh DB.

## Deliverables
- [ ] One migration plan using `references/plan_template.md`
- [ ] A reviewed migration script (upgrade + downgrade policy honored)
- [ ] Dry-run SQL commands for inspection
- [ ] Verification + rollback notes

## Anti-patterns
- Blindly applying `--autogenerate` output without reading the produced `upgrade()` / `downgrade()`.
- Editing already-applied migration files; create a new corrective migration instead.
- Lumping unrelated schema changes into one revision.
- Data migrations without guards/idempotency (re-run safety), or without lock/volume consideration.

## Examples

### Example 1 — Add nullable column + backfill + make NOT NULL (safe rollout)
Input:
- ChangeRequest: "Add users.phone, backfill from profile table, then enforce NOT NULL"
Expected Output:
- Plan proposes 2–3 revisions:
  1) add nullable column
  2) backfill in batches / guarded updates
  3) set NOT NULL + add constraint/index (if needed)
- Includes dry-run `--sql` preview and rollback notes.

### Example 2 — Rename column (avoid drop/add pitfalls)
Input:
- ChangeRequest: "Rename users.fullname -> users.name"
Expected Output:
- Plan warns that naive autogen may emit drop/add; script_patch uses dialect-safe rename operation if available,
  plus data preservation considerations, plus downgrade behavior.

### Example 3 — SQLite alter table limitation (batch)
Input:
- ChangeRequest: "Change column type or drop column on SQLite"
Expected Output:
- Plan routes to batch operations context and documents move/copy behavior + verification.

## References
- [Official Alembic Tutorial](https://alembic.sqlalchemy.org/en/latest/tutorial.html)
- [Autogenerate (review-first)](https://alembic.sqlalchemy.org/en/latest/autogenerate.html)
- [Offline mode / --sql](https://alembic.sqlalchemy.org/en/latest/offline.html)
- [Batch migrations (SQLite)](https://alembic.sqlalchemy.org/en/latest/batch.html)
- [Cookbook](https://alembic.sqlalchemy.org/en/latest/cookbook.html)
- [Plan template](./references/plan_template.md)
- [Script review checklist](./references/script_review_checklist.md)
