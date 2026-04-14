# Tracking Contract (template)

Define, in one place:

- **ID types**: request_id (sync), task_id (async), trace_id (OTel).
- **Format**: lowercase UUID with hyphens (or binary storage internally).
- **Lifetime**:
  - request_id: per request/call.
  - task_id: per async job from enqueue to completion.
  - trace_id: per distributed trace.
- **Propagation**:
  - HTTP headers: x-request-id, x-trace-id.
  - gRPC metadata: x-request-id, x-trace-id.
  - Queue headers/attributes: task_id, parent_id.
- **Logging**: every event must include the active ID(s).

Non-goals:
- Replacing OpenTelemetry.
- Using UUIDs as a security mechanism.
