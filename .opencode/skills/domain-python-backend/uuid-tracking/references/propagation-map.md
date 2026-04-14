# Propagation map checklist

Use this to ensure the same ID survives every boundary.

## Entry points (generate once if missing)
- HTTP server middleware: read x-request-id → else generate request_id.
- gRPC server interceptor: read metadata x-request-id → else generate request_id.
- CLI/cron entry: generate request_id at process start for the command.

## Internal propagation
- Put request_id into request-scoped context (ContextVar / request.state / thread-local).
- For outbound calls, copy from context into headers/metadata.

## Messaging boundaries
- Producer:
  - generate task_id at enqueue (or accept provided task_id).
  - attach: task_id + parent request_id (if any).
- Worker:
  - read task_id from message.
  - set task_id into context; log it on every event.
  - for outbound calls, forward request_id=task_id (or forward both explicitly).

## Observability contract
- Logs: every event includes request_id or task_id (and trace_id if available).
- Traces: prefer official tracing; add request_id/task_id as span attributes if supported.
- Metrics: never include UUIDs as labels/dimensions.
