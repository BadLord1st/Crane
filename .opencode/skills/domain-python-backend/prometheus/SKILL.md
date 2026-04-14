---
name: prometheus
description: Define a Prometheus monitoring contract (metrics/labels, scrape job, alerts, dashboards) for a change; outputs are doc/snippets only (no infra/code changes).
metadata:
  signature: "spec-prometheus :: SpecNode -> PrometheusPlan"
---

## When to use
- You monitor the service with Prometheus/Grafana and need actionable SLIs/SLOs/alerts.
- You add/modify endpoints, workers, queues, or DB paths and want a metrics contract before implementation.
- Incidents are hard to detect early (no clear error/latency/saturation signals).

## When not to use
- You need logs/traces design first (use spec-observability for broader telemetry contracts).
- You use a non-Prometheus metrics backend without PromQL semantics (adapt another skill).
- You are asking to deploy/configure a real Prometheus server now (this skill outputs specs/snippets only).

## Inputs
- SpecNode (feature/system change) + critical operations + known failure modes.
- Runtime/deploy context: where scraping happens (K8s/VM), typical scrape interval, instance topology.
- Existing metric conventions (prefix/subsystem, label standards), if any.
- SLO intent (if known): e.g., “p95 < 300ms”, “5xx < 0.5%”.

## Outputs
- PrometheusPlan:
  - Metric inventory (name, type, unit, HELP text, labels, cardinality policy, expected volume).
  - Alert rules (PromQL + severity + paging policy + runbook link placeholders).
  - Dashboard outline (panel list + key queries).
  - Scrape job snippet (job_name, metrics_path, discovery/targets notes).

## Protocol
1) Identify the minimal “golden signals” per critical operation:
   - Throughput, latency, errors, saturation.
2) Define metric primitives:
   - Counters for totals (requests/jobs/errors), gauges for “now”, histograms for latency distributions.
3) Naming rules:
   - snake_case, subsystem grouping, unit/type suffixes (_total, _seconds, etc.).
4) Label policy (strict allowlist):
   - Labels must be bounded sets (status_code, method, route_id, operation).
   - Explicitly ban high-cardinality labels (user_id, raw URL, unbounded IDs).
5) Latency design:
   - Prefer histograms for percentiles and aggregation across instances (avoid “per-instance-only” percentile traps).
6) Alert design:
   - Start with a small set: error-rate spike, latency SLO burn, saturation/backlog growth.
   - Each alert must name: symptom, suspected causes, first 3 checks, mitigation steps (runbook stub).
7) Scrape & exposure constraints:
   - Define /metrics endpoint expectations, auth/network notes (without secrets).
8) Verification hooks:
   - Define “acceptance checks”: which metric names must exist; which labels must be present; sample queries.

## Deliverables
- [ ] Metric inventory with label/cardinality policy and naming conventions.
- [ ] 3–7 alert rules with severity + runbook placeholders.
- [ ] Dashboard outline (panels + PromQL queries).
- [ ] Scrape job snippet + assumptions (discovery mode, metrics_path, port).

## Anti-patterns
- “Add more metrics” without mapping to SLIs/SLOs.
- High-cardinality labels or unbounded label values (raw URL, IDs).
- Client-side percentiles you can’t aggregate across instances when you actually need fleet-level views.
- Exporting everything “just in case” (unbounded time-series growth).
- Putting secrets/PII into labels.

## Edge cases
- Short-lived jobs: define how metrics are exposed (long-lived exporter vs gateway pattern) and what you lose.
- Multi-tenant systems: tenant label only if tenant set is bounded; otherwise model via logs/traces.
- Very high QPS: keep label sets tiny; avoid per-route explosion; prefer route templates or operation IDs.
- Batch/queue workloads: model backlog and processing latency explicitly (queue depth gauge + job duration histogram).

## Examples
### Example 1: New HTTP endpoint
Request: “Добавляем POST /v1/payments; нужен мониторинг и алерты.”
Expected: metrics for request rate, error rate, duration histogram (with bounded labels), p95 query, 2–3 alerts + dashboard panels.

### Example 2: Worker + queue
Request: “Добавляем фонового воркера; очередь может расти.”
Expected: queue_depth gauge, job_started_total/job_failed_total counters, job_duration_seconds histogram, alert on sustained backlog growth + saturation signals.

### Example 3: DB hotspot
Request: “Появился критичный SQL путь; хотим видеть деградации.”
Expected: db_query_duration_seconds histogram by operation (bounded), timeout/error counters, alert on p95/p99 burn + error spikes, dashboard drill-down panels.

## References
- [Metrics Contract](./references/metrics-contract.md)
- [Alerting & Dashboards](./references/alerts-and-dashboards.md)
- [PromQL Snippets](./references/promql-snippets.md)
