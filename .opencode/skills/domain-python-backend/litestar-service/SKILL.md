---
name: litestar-service
description: "Define a Litestar service specification: routing topology (routers/controllers), DI contracts, database integration, and observability signals — as a spec artifact (not code)."
metadata:
  signature: "spec-litestar-service :: (SpecNode, Snapshot) -> LitestarServiceSpec"
---

## When to use
- You are building or evolving an HTTP API service specifically on **Litestar**.
- You need a crisp, testable contract for routing structure, dependency injection, and integrations (SQLAlchemy, telemetry).
- You want to prevent “framework folklore” by capturing decisions as a spec node.

## Inputs
- SpecNode (feature/system contract) describing intended behavior.
- Snapshot (current repo reality): folders, existing routers/controllers, DB stack, telemetry stack (if any).

## Outputs
- LitestarServiceSpec: a spec artifact describing:
  - Route topology (routers/controllers), versioning and grouping strategy.
  - DI map (dependency keys, scopes, lifetimes, caching policy).
  - Integration plan (SQLAlchemy plugin/repositories, DTO/serialization).
  - Observability requirements (metrics/traces/log fields) and exposure paths.
  - Error mapping rules (exception → HTTP response) and guard placement rules.

## Protocol
1) Identify the **API surface**: base paths, resources, operations, versioning, auth zones.
2) Design **routing topology**:
   - Group by feature (Router) and behavior bundles (Controller).
   - Decide guard placement level (app/router/controller/handler) and document it.
3) Define a **DI contract**:
   - List dependency keys, providers, and scopes (app/router/controller/handler).
   - State caching rules and sync/async execution assumptions.
4) Specify **data layer integration**:
   - Session / Unit-of-Work boundaries (per-request).
   - Repository usage and transaction policy (commit/rollback rules).
   - DTO/serialization policy (what is exposed vs internal).
5) Specify **observability**:
   - Minimal metrics, traces, and structured log fields (correlation IDs, tenant/user, request ID).
   - Export/endpoint conventions (e.g., /metrics) and sampling notes.
6) Define **error handling**:
   - Canonical error envelope, status codes, and exception mapping table.
   - What is safe to expose vs redact.
7) Add **verification hooks**:
   - What must be asserted in tests (routing, DI wiring, auth guards, error mapping).
   - Minimal “smoke” checks to validate budgets/telemetry endpoints.
8) Output a single “LitestarServiceSpec” deliverable with bullet-proof wording and acceptance criteria.

## Deliverables
- [ ] A LitestarServiceSpec node (one file) linked from the relevant feature/system spec.
- [ ] Route topology map (routers/controllers) with guard placement rules.
- [ ] DI map: keys → providers → scopes (+ caching / threading assumptions).
- [ ] DB integration notes: session lifecycle, repositories, transaction policy.
- [ ] Observability contract: metrics/traces/log fields + endpoint conventions.
- [ ] Error mapping rules + “do not leak” policy.
- [ ] Acceptance criteria / verification points (testable statements).

## Anti-patterns
- “Just add routes” without documenting grouping, guard scope, and DI boundaries.
- Mixing auth checks into business logic instead of using guards at the right layer.
- Injecting secrets or environment values directly into specs or logs.
- Ad-hoc DB sessions (global session, cross-request reuse) or unclear transaction ownership.
- Telemetry that collects sensitive data or lacks correlation identifiers.

## References
- [Routing + Guards patterns](./references/routing-and-guards.md)
- [Dependency Injection contract](./references/dependency-injection.md)
- [SQLAlchemy integration + repositories](./references/sqlalchemy-integration.md)
- [Observability (Prometheus + OpenTelemetry)](./references/observability.md)
- [Examples](./references/examples.md)
