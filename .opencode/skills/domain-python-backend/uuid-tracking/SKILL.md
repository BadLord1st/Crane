---
name: uuid-tracking
description: "Standardize UUID correlation IDs for every request/task: generate once, propagate across boundaries, log consistently (no secrets, never as metric labels)."
metadata:
  signature: "spec-uuid-tracking :: (Ticket, SystemContext) -> TrackingContract"
---

## When to use
- You need end-to-end tracking across services, workers, and queues.
- Incident debugging requires linking logs/traces for a single request or job.
- You’re introducing background jobs and need per-job observability.

## When not to use
- You already have a stable, enforced correlation ID contract (and propagation) — this would be redundant.
- You need ordered IDs for database locality or time sorting (consider ULID / UUIDv7 instead).
- You are designing security tokens/secrets (UUIDs are identifiers, not security controls).
- You want the assistant to edit/deploy code automatically (this skill is spec/contract-first by default).

## Inputs
- Ticket (goal + scope: HTTP/gRPC/async jobs/CLI tasks).
- Current instrumentation snapshot:
  - Logging format (JSON/key=value), field conventions (if any).
  - Tracing (OpenTelemetry or none).
  - Queue/broker (if any) and message headers/attrs.
- Existing ID fields/headers/metadata keys (if any).

### Assumptions (if inputs are missing)
- ID type: UUIDv4 (random).
- Canonical format: lowercase hex with hyphens (36 chars).
- Default names:
  - request_id: per inbound request/command
  - task_id: per background job unit
  - trace_id: tracing system’s trace id (prefer existing tracing)
  - HTTP header / gRPC metadata key: x-request-id

## Outputs
- One tracking contract doc node (spec) with:
  - ID taxonomy: request_id vs task_id vs trace_id
  - Generation rules (where/when, “generate once”)
  - Propagation map (HTTP/gRPC/messaging/internal context)
  - Log schema: mandatory fields + examples
  - Guardrails (no secrets/PII; never as metric labels)
- Integration checklist: entrypoints + boundaries + verification hooks.
- (Optional) Patch plan (files/locations) — but **no code edits by default**.

## Protocol
1) Identify “task boundaries”: request entrypoint, enqueue boundary, worker start, outbound client calls.
2) Pick the minimal ID taxonomy:
   - request_id: correlates a single inbound request/command across services.
   - task_id: correlates a single background job across producer/worker and any downstream calls.
   - trace_id: from OpenTelemetry (or equivalent). If present, log it alongside request_id/task_id.
3) Define generation rules:
   - Generate exactly once at the first boundary if missing.
   - Preserve the incoming id if provided (do not overwrite).
   - Never regenerate mid-flight; only fork child spans (tracing) if you need sub-structure.
4) Define propagation rules (contracts, not ad-hoc):
   - HTTP: x-request-id header + request-scoped context + structured log field request_id.
   - gRPC: metadata x-request-id (and forward it on outbound calls).
   - Messaging: message header/attribute request_id and/or task_id (explicitly documented).
   - Internal code: request-scoped context (ContextVar / thread-local / request state).
5) Define the logging contract:
   - Every log event MUST include request_id OR task_id (and trace_id if available).
   - Standard fields: service, operation, outcome, error_type (if any), duration_ms (when relevant).
6) Guardrails:
   - Do not log secrets or sensitive payloads; keep the ID opaque and context minimal.
   - UUIDs are not auth/session tokens and must not be treated as a security control.
   - Never put request_id/task_id as Prometheus labels (high cardinality). Keep them in logs/traces only.
7) Storage note (only if relevant):
   - Prefer native UUID / 16-byte storage where available; otherwise store canonical string.
8) Verification hooks:
   - Define a “smoke correlation”: one request/job produces logs in every component with the same id.

## Deliverables
- [ ] Tracking contract doc produced/updated (taxonomy, generation, propagation, log schema, guardrails).
- [ ] Integration checklist across entrypoints/workers/clients.
- [ ] Verification hook described (how we prove propagation works end-to-end).

## Anti-patterns
- Generating new UUIDs at every layer (“broken trace”).
- Mixing formats (uppercase, no hyphens, base64) without an explicit contract.
- Treating UUIDs as secrets or as authorization.
- Adding request_id/task_id as metric labels.
- Dumping sensitive payloads into structured logs “because it’s searchable”.

## Examples
### Example A: HTTP request correlation
Scenario: an inbound HTTP request arrives without x-request-id.

Expected contract excerpt:
- Entry: middleware generates request_id once.
- Propagation: x-request-id added to outbound HTTP/gRPC calls; request_id present in every log event.
- Sample request_id: 53407c1c-46e4-4ed0-99e5-81468d0dfcb0

### Example B: Background job tracking
Scenario: a producer enqueues a job; a worker executes it; the worker calls other services.

Expected contract excerpt:
- Producer assigns task_id at enqueue.
- Worker logs task_id and forwards it as request_id on outbound calls (or forwards both).
- Sample task_id: 49cd0135-5f63-4de7-965d-5d499183760f

### Example C: Agent/task envelope (for human tracking)
Scenario: you want every assistant “task” to be trackable in chat artifacts.

Expected envelope format (first lines of every task output):
- task_uuid: 0543dfc8-c651-42ba-be06-71f2e8e402e5
- parent_uuid: <optional>
- artifacts: <paths/links created in this task>

## References
- [Wide Events (structured logging) example](../wide-log/references/wide-log-guide.md)
- [UUID conventions (this skill)](references/uuid-conventions.md)
- [Propagation map checklist (this skill)](references/propagation-map.md)
