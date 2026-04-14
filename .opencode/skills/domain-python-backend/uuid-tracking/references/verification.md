# Verification checklist (UUID tracking)

## Smoke trace
Pick one request or one background job and confirm:
- The same request_id/task_id appears in:
  - entrypoint logs
  - worker logs (if async)
  - downstream client logs (or outbound request headers)
- If tracing exists, trace_id is present and consistent.

## Failure mode checks
- Missing inbound IDs are generated at the edge.
- Downstream calls always forward the ID.
- Retries do not generate a new ID (unless a new logical task is created).

## Regression hook
Add one automated check (unit/integration) that asserts:
- log records include request_id/task_id
- outbound calls include x-request-id
