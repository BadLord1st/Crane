# UUID conventions for tracking (request_id / task_id)

## Defaults
- Prefer UUIDv4 for correlation IDs (random, easy, ubiquitous).
- Use canonical string format: lowercase hex + hyphens (36 chars).
- UUIDs are **identifiers**, not secrets or authorization.

## Where UUIDs help most
- request_id: one per inbound request/command; forwarded to downstream calls.
- task_id: one per background job; preserved from enqueue → worker → downstream calls.
- trace_id: if OpenTelemetry exists, log it alongside request_id/task_id (do not replace tracing).

## Storage notes
- Prefer native UUID types / 16-byte storage where your DB/ORM supports it.
- If storing as text, store the canonical string consistently (avoid mixed case).

## Minimal Python snippet
```python
import uuid

request_id = str(uuid.uuid4())  # e.g. "f47ac10b-58cc-4372-a567-0e02b2c3d479"
```
