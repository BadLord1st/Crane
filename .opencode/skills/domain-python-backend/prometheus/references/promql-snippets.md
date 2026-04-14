# PromQL snippets (copy/paste)

## HTTP request rate (RPS)
sum(rate(http_requests_total[5m]))

## Error ratio (5xx / all)
sum(rate(http_requests_total{status=~"5.."}[5m]))
/
sum(rate(http_requests_total[5m]))

## p95 latency from histogram
histogram_quantile(
  0.95,
  sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
)

## p95 latency by operation (bounded label)
histogram_quantile(
  0.95,
  sum(rate(http_request_duration_seconds_bucket[5m])) by (le, operation)
)

## Queue backlog growth (sustained increase)
deriv(queue_depth[10m]) > 0

## Job processing throughput
sum(rate(jobs_processed_total[5m]))

## Worker error ratio
sum(rate(jobs_failed_total[5m])) / sum(rate(jobs_started_total[5m]))
