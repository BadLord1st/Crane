---
name: opentelemetry
description: Define an OpenTelemetry instrumentation spec (traces/metrics/log correlation, context propagation, OTLP export, sampling) as a spec artifact; do NOT modify code or deploy infra.
metadata:
  signature: "opentelemetry :: SpecNode -> OTelSpec"
---

## When to use
- You need distributed tracing across services (HTTP/gRPC/queues) and want vendor-neutral telemetry.
- You want a concrete OTel plan (what to instrument, how to propagate context, where to export).
- You need to align teams on span taxonomy, attributes, sampling, and PII rules before implementation.

## Inputs
- SpecNode (service list, main request flows, key dependencies: DB, cache, queues, external APIs).
- Runtime context: language(s), frameworks (e.g., Litestar/FastAPI, gRPC libs), job runner (Celery/TaskIQ), deployment (k8s/VM).
- Telemetry backend constraints: Jaeger/Tempo/Zipkin, Prometheus, vendor APM, existing logging stack.
- Non-functional constraints: overhead budget, data retention, PII/security constraints, SLOs.

## Outputs
- OTelSpec: an implementation-ready plan (instrumentation map + config intent + validation checks).

## Protocol
1) Inventory “critical paths” (top user journeys + async continuations) and define trace boundaries.
2) Decide propagation strategy end-to-end (HTTP headers / gRPC metadata / message headers) and name the carrier fields.
3) Choose instrumentation mode per component:
   - auto-instrument where safe/available (framework, HTTP client, DB),
   - manual spans where you need domain semantics (use-case spans, queue handoff, custom IO).:contentReference[oaicite:4]{index=4}
4) Define export topology:
   - default: OTLP -> OpenTelemetry Collector -> backend(s),
   - specify endpoints/env intent (no secrets), batching/retries responsibility boundary.:contentReference[oaicite:5]{index=5}
5) Define sampling policy (dev vs prod, head/tail, always-sample “important transactions”).:contentReference[oaicite:6]{index=6}
6) Define span taxonomy + attribute conventions:
   - semantic conventions first (http.*, db.*, messaging.*),
   - guardrails: avoid high-cardinality dimensions (e.g., user_id on metrics; raw URLs).:contentReference[oaicite:7]{index=7}:contentReference[oaicite:8]{index=8}
7) Log correlation stance:
   - include trace_id/request_id in structured logs,
   - explicitly forbid sensitive data in logs/attributes; define redaction/masking rules.:contentReference[oaicite:9]{index=9}
8) Define “smoke-level” verification: one request flow + one async handoff + one DB call showing linked spans.

## Deliverables
- [ ] Service/flow instrumentation map (what emits spans/metrics/log correlation).
- [ ] Context propagation plan (carriers, boundaries, fallbacks).
- [ ] Export & Collector topology (OTLP endpoints as intent, not secrets).
- [ ] Sampling policy (dev/prod, special-case rules).
- [ ] Attribute conventions + cardinality/PII guardrails.
- [ ] Minimal validation checklist (how to prove it works).

## Anti-patterns
- “We added OTel” without a span taxonomy or propagation plan (broken traces).
- Over-instrumenting hot low-level loops; ignoring overhead budgets.:contentReference[oaicite:10]{index=10}
- High-cardinality labels/attributes that explode costs (user_id, raw URL, unbounded IDs).:contentReference[oaicite:11]{index=11}
- Leaking secrets/PII into logs, span attributes, or metric labels.:contentReference[oaicite:12]{index=12}

## References
- [OTel Instrumentation Map Template](references/otel-instrumentation-map.md)
- [Collector / Export / Sampling Notes](references/otel-collector-export-sampling.md)
- [Attributes, Cardinality, PII Rules](references/otel-attributes-cardinality-pii.md)
