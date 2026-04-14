# Alerting & dashboards (minimal, useful)

## Alert rules principles
- Start small: 3–7 alerts beats 50 noisy ones.
- Every alert includes: symptom, severity, owner, runbook link (even stub), “first checks”.
- Prefer rate-based alerts over raw counters.

## Suggested baseline alerts
1) Error ratio spike (HTTP 5xx or job failures):
   - Trigger when error_ratio over N minutes exceeds threshold.
2) Latency SLO burn:
   - Trigger on p95/p99 crossing threshold or burn-rate style windows (fast + slow).
3) Saturation / backlog:
   - Queue depth increasing for sustained window, or utilization pinned high.

## Dashboard outline (Grafana)
- Overview:
  - RPS / throughput, error ratio, p95 latency, saturation gauges.
- Drill-down:
  - By operation/route template (bounded).
  - Queue panels: depth, processing rate, duration percentiles.
- “What changed?” helper panels:
  - Deploy markers (if you have them), instance count, restarts.

## Runbook stub template (per alert)
- What it means (user impact).
- Most likely causes (top 3).
- First 3 checks (queries/log links).
- Mitigations (safe steps).
- Escalation.
