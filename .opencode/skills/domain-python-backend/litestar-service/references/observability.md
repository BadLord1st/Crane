# Observability (Prometheus + OpenTelemetry)

## Minimal contract
- Metrics: request count, error count, latency histogram (by endpoint/method/status).
- Traces: one span per request, propagate context to downstream calls.
- Logs: structured fields for correlation (request_id/trace_id, user/tenant, route).

## Endpoints and conventions
- Metrics endpoint: /metrics (unless overridden).
- Document cardinality rules: avoid high-cardinality labels (user_id as metric label is usually a trap).

## Sampling notes
- Development: record everything.
- Production: define sampling strategy and what must always be traced.

## Privacy
- Never emit secrets/PII into logs/spans.
- Redact error payloads and headers.
