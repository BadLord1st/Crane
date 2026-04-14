---
name: granian-runbook
description: Produce a deployment/runbook spec for serving an ASGI app with Granian (workers/threads, logging, health, graceful shutdown), including a benchmark plan; no live changes, no secret handling.
metadata:
  signature: "spec-granian-runbook :: (AppTarget, EnvConstraints) -> GranianRunbookSpec"
---

## When to use
- You want a clear, reproducible runbook for running an ASGI app on **Granian**.
- You need guidance on **workers/threads tuning** and what to benchmark.
- You’re migrating from Uvicorn/Gunicorn and want an operational checklist.

## Inputs
- App target (one of):
  - Python import path (e.g., `module:app`), or
  - A short description of how the app is created (factory, DI, etc.).
- Deployment context: bare metal / VM / Docker / Kubernetes / systemd.
- Constraints: CPU cores, memory limits, expected RPS/latency goals, I/O profile (CPU-bound vs I/O-bound), upstream proxy (nginx/traefik), log/metrics stack.

If the user doesn’t provide constraints, assume:
- 2–4 CPU cores, moderate I/O, Docker or systemd, reverse proxy terminates TLS.

## Outputs
A **Granian Runbook Spec** containing:
- Minimal “how to run” commands/config (systemd/Docker/K8s).
- Worker/thread tuning guidance (with benchmark-first stance).
- Logging/observability notes (what to log, what to avoid).
- Health checks and graceful shutdown expectations.
- A benchmark plan and acceptance criteria.

## Protocol
1) Confirm app entrypoint format (`module:app`) and whether the app is fully ASGI-compliant.
2) Identify deployment wrapper (systemd/Docker/K8s) and where TLS termination happens.
3) Propose an initial worker/thread configuration **as a hypothesis**, then require benchmarking to finalize.
4) Define logging expectations (structured logs preferred), and what fields must be present for correlation.
5) Define health/readiness checks and startup/shutdown behaviors (timeouts, drain).
6) Provide a minimal benchmark plan: endpoints, load shape, metrics to record, and how to compare configurations.
7) Produce a final runbook spec using the templates in `references/`.

## Deliverables
- [ ] One filled **Runbook Spec** using `references/runbook_template.md`
- [ ] One **Benchmark Plan** using `references/bench_template.md`
- [ ] Optional: a **systemd unit** and/or **Docker compose snippet** from templates

## Anti-patterns
- Claiming a “best” worker/thread count without measuring in the target environment.
- Suggesting production deployments that embed secrets in commands or logs.
- Mixing runbook/spec creation with actually applying changes (deploy, restart, edit files).
- Recommending direct internet exposure without TLS / reverse proxy considerations.

## References
- [Runbook Spec Template](references/runbook_template.md)
- [Benchmark Plan Template](references/bench_template.md)
- [systemd Unit Template](references/systemd_unit_template.md)
- [Docker Compose Snippet](references/docker_compose_snippet.md)
- [TLS & Proxy Checklist](references/tls_proxy_checklist.md)
