---
name: grpc-contract
description: "Design/update a gRPC contract (.proto + Python integration notes) with explicit deadlines, status-code error model, streaming choices, and protobuf compatibility rules. Produces a GrpcContractNode; no deployment."
metadata:
  signature: "grpc-contract :: (SpecNode, Repo) -> GrpcContractNode"
---

## When to use
- You are introducing or changing an internal service API and want a strict, typed contract.
- You need streaming RPCs (large payloads, incremental results, bidi workflows).
- You want predictable error handling and latency control (deadlines/timeouts).

## When NOT to use
- You need a public browser-facing API (prefer HTTP/JSON unless you have a controlled client fleet).
- You cannot manage client/server versioning and `.proto` governance.
- The request is “just implement the service” (this skill outputs a contract + checklist first).

## Inputs
- SpecNode: domain intent, entities, invariants, failure modes.
- Repo context: existing `.proto` layout, language(s), build tooling (protoc/buf), CI expectations.
- Runtime constraints: sync vs asyncio, expected QPS/concurrency, payload sizes, streaming needs.
- Security constraints: mTLS/TLS, auth metadata, tenancy boundaries, PII constraints.

## Outputs
- GrpcContractNode:
  - `.proto` contract skeleton (package naming + service/methods/messages).
  - Compatibility rules + versioning plan (v1/v2 coexistence).
  - Error model (status codes + canonical error details strategy if used).
  - Deadline policy (per-method defaults; cancellation semantics).
  - Python integration notes (grpc vs grpc.aio, interceptors, channel reuse, max_workers).
  - Observability notes (trace/log correlation fields; metrics candidates).

## Protocol
1) Choose API surface: bounded context, service name, method set; decide unary vs streaming per method.
2) Pick package naming and versioning: `company.product.<domain>.v1`; define migration path for v2.
3) Define messages with protobuf wire-safety:
   - Additive fields only; never reuse field numbers; reserve removed numbers/names.
   - Avoid `oneof` churn unless you understand backwards-compat pitfalls.
4) Define method contracts:
   - Idempotency expectations (read vs write), pagination/filters, streaming chunk shape.
5) Define error model:
   - Map domain failures to gRPC status codes; forbid “UNKNOWN for everything”.
   - Decide whether you use richer error details (optional) vs code+message only.
6) Define deadlines:
   - Every client call sets a deadline; define per-RPC default timeouts and max caps.
   - Define server cancellation behavior (what is safe to abandon vs must finish).
7) Define performance/concurrency:
   - Channel reuse rules; connection lifecycle; message size/streaming backpressure notes.
   - Server execution model: thread pool sizing (`max_workers`) or `grpc.aio` for asyncio.
8) Define security:
   - TLS/mTLS posture; auth metadata conventions; do-not-log fields.
9) Define operability:
   - Health checking / reflection policy (enabled by environment); basic SLO signals.

## Deliverables
- [ ] `.proto` draft (service + messages) with versioned package.
- [ ] Explicit deadline defaults per RPC.
- [ ] Error mapping (domain → StatusCode) documented.
- [ ] Compatibility rules (field numbering + reserved policy) documented.
- [ ] Python notes: grpc/grpc.aio choice, interceptors, channel reuse, server workers.
- [ ] Observability hooks (trace/log correlation + core metrics list).

## Safety & permissions
- Do not request or paste secrets (cert private keys, tokens, `.env` contents).
- If asked to “apply changes” (edit files, regenerate stubs, change server config), first produce a patch plan + dry-run commands; only apply after explicit confirmation.

## Anti-patterns
- No deadlines/timeouts (unbounded resource usage and hanging calls).
- Reusing field numbers / removing fields without `reserved`.
- Returning custom error payloads while always reporting OK/UNKNOWN.
- Creating new channels per request; mixing blocking work into `grpc.aio` handlers.
- Logging PII in metadata or “wide” request logs without redaction rules.

## Examples
- User: “Нужен gRPC для сервиса Billing: CreateInvoice, GetInvoice, ListInvoices. Ошибки: not found, validation, conflict.”
  Output: GrpcContractNode with `.proto` + StatusCode mapping + deadline defaults + versioned package.

- User: “Хотим bidi-stream для синхронизации каталогов, большие данные.”
  Output: streaming method design + chunk/message framing + backpressure notes + max message size guidance.

- User: “Переводим demo.v1 → demo.v2, нужно безопасно.”
  Output: versioning plan + coexistence strategy + `reserved`/field evolution rules.

## References
- [gRPC deadlines guide](./references/deadlines-and-cancellation.md)
- [gRPC status codes guide](./references/status-codes.md)
- [Protobuf compatibility & reserved](./references/protobuf-compat.md)
- [Python gRPC API surface (grpc/grpc.aio, interceptors, health/reflection)](./references/python-grpc-notes.md)
