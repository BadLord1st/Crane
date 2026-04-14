# Examples (Requests → Expected Output Shape)

## Example 1: New service skeleton spec
Request:
“Design a Litestar service for /v1/widgets with public read and authenticated write.”

Expected output:
- Route topology: /v1/widgets router with nested /write router guarded by auth.
- DI map: current_user, db_session, widget_repo.
- DB policy: session per request, commit on write, rollback on exception.
- Observability: /metrics, request spans, structured log fields.
- Error mapping: NotFound → 404, PermissionError → 401/403, Validation → 400.

## Example 2: Retrofit an existing repo
Request:
“Current repo has flat handlers; add structure and document DI + SQLAlchemy plugin usage.”

Expected output:
- Proposed refactor topology (routers/controllers) *as a spec*, with minimal change surface.
- DI contract with scopes and caching notes.
- Migration/verification points (routing tests, DI wiring tests).

## Example 3: Observability-first requirement
Request:
“Service is incident-prone; define telemetry requirements for critical paths.”

Expected output:
- Failure modes → signals mapping (metrics + logs + traces).
- Alert/SLO notes and correlation field contracts.
