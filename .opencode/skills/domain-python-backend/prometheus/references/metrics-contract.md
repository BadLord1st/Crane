# Prometheus metrics contract (practical rules)

## Exposure
- Expose an HTTP metrics endpoint (commonly `/metrics`) that Prometheus scrapes on an interval.
- Keep it fast and stable; treat it as part of the service contract.

## Metric types (choose correctly)
- Counter: monotonic totals (requests_total, jobs_failed_total).
- Gauge: instantaneous values (queue_depth, in_flight, open_fds).
- Histogram: distributions you want to aggregate (request/job/db duration in seconds, payload sizes).

## Naming conventions
- snake_case.
- Suffixes:
  - `_total` for counters that represent totals.
  - `_seconds` for durations.
- Group by subsystem:
  - `http_requests_total`, `http_request_duration_seconds`, `db_query_duration_seconds`.

## Labels (dimensions)
- Labels must come from bounded sets (status_code, method, operation, route_template).
- Avoid high cardinality:
  - user_id, request_id, raw url, arbitrary exception text, unbounded IDs.

## Cardinality budget (suggested contract)
- Per metric: <= 1k series per instance; prefer <= 200.
- Max label count: 0–3, unless you can prove boundedness.

## “Golden signals” starter set
- Throughput: requests/jobs per second.
- Errors: 5xx / failed jobs per second, plus error ratio.
- Latency: histogram-based percentiles (p50/p95/p99).
- Saturation: queue depth, thread pool usage, DB pool usage, memory, file descriptors.
