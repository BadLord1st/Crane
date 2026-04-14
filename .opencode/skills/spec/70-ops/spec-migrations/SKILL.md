---
name: spec-migrations
description: Plan and implement schema/data migrations safely with rollback and verification.
metadata:
  signature: "spec-migrations :: (SpecNode, Repo) -> MigrationPlan"
---

## When to use
- The change affects DB schema or stored data.
- You need backward/forward compatible rollout.

## Inputs
- SpecNode with data contract changes.
- Current schema/migrations in repo.

## Outputs
- Migration plan + migration files + verification steps.

## Protocol
1) Identify current migration tool (Prisma, Alembic, Flyway, etc.).
2) Plan safe rollout: expand → backfill → switch → contract (when needed).
3) Add verification queries/checks and rollback notes.

## Deliverables
- [ ] Migration files added.
- [ ] Verification commands documented.
- [ ] Rollback plan when required.

## Anti-patterns
- Breaking production data shape without compatibility plan.
- Migrations without tests/verification.

## References
- [Zero-Downtime Deployments and Rollbacks](./references/zero-downtime.md)
- [Database Migration Patterns](./references/migration-patterns.md)
