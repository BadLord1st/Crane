# Opentelemetry Attributes Cardinality PII

## Attribute conventions
Prefer semantic conventions first:
- http.method, http.route, http.status_code
- db.system, db.statement (careful), db.operation
- messaging.system, messaging.destination

## Cardinality rules
- OK: status_code, route templates, operation names from a finite set
- NOT OK: user_id, raw URL, order_id, UUIDs as labels, unbounded strings

If you need per-entity debugging, use logs with trace_id correlation, not metric labels.

## PII / secrets rules
Forbidden in logs/attributes/labels:
- passwords, tokens, auth headers, session cookies
- raw request/response bodies
- personal identifiers unless explicitly approved + masked/redacted

Redaction policy:
- define allowlist fields
- hash or truncate where necessary
