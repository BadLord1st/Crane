# SQLAlchemy Integration (Litestar)

## What to decide
- Sync vs Async SQLAlchemy strategy.
- Session lifecycle per request (Unit of Work).
- Repository pattern usage (e.g., SQLAlchemyAsyncRepository).

## What must be in the spec
- How sessions are created, injected, and closed.
- Transaction ownership:
  - who commits on success
  - who rolls back on error
- Repository boundaries:
  - what queries are allowed
  - pagination/filtering policy
- DTO/serialization:
  - which fields are exposed
  - how internal fields are excluded

## Operational cautions
- Avoid global sessions.
- Explicitly document “N+1 risk zones” and relationship loading policy.
