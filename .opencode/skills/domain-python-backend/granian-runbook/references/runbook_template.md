# Granian Runbook Spec

## 0) Scope
Service name:
Environment (dev/stage/prod):
Owner/on-call:

This runbook defines how to serve an ASGI application with Granian. It is a *spec* (no changes applied).

## 1) App entrypoint
- ASGI import path: `__REPLACE_WITH_module:app__`
- App type: ASGI 3 (async) / mixed
- Blocking operations: none known / list hotspots:

## 2) Deployment topology
- Runtime: systemd / Docker / Kubernetes
- Reverse proxy / ingress: nginx / traefik / ALB / none
- TLS termination: proxy / ingress / app (avoid in-app unless required)
- Listening address/port:

## 3) Baseline Granian configuration (hypothesis)
Goal: a *starting point* to benchmark.

- Workers: __N__
- Threads per worker: __M__
- Rationale:
  - CPU cores available:
  - Workload profile (CPU vs I/O):
  - Expected concurrency:

Example (Python API):

```python
from granian import Granian

server = Granian("module:app", workers=4, threads=4, http_port=8000)
server.serve()
```

## 4) Operational behaviors
- Startup time budget:
- Graceful shutdown budget (drain + stop):
- Max request timeout (upstream + app):
- Keep-alives / connection reuse expectations:

## 5) Health, readiness, and observability
Health endpoints:
- Liveness:
- Readiness:

Logging:
- Format: JSON / key=value
- Required fields: `service`, `env`, `request_id` (or `trace_id`), `route`, `status`, `latency_ms`
- Sensitive fields to never log:

Metrics/tracing (optional):
- Prometheus / OpenTelemetry / other:

## 6) Run instructions
### systemd
- Unit file: see `references/systemd_unit_template.md`
- Start/stop:
- Logs location:

### Docker
- Compose snippet: see `references/docker_compose_snippet.md`
- Resource limits:

### Kubernetes (if applicable)
- Deployment resources (requests/limits):
- Probes:

## 7) Tuning plan
We will finalize workers/threads only after running the benchmark plan.

- Candidate configs to test:
- Expected bottleneck:
- Decision rule (what metric wins):

## 8) Risks and rollback
- Main risks (CPU saturation, tail latency, connection limits, log volume):
- Rollback plan:

## 9) Acceptance criteria
- p50/p95/p99 latency targets:
- Error rate target:
- Throughput target:
- Resource ceiling (CPU/mem):
