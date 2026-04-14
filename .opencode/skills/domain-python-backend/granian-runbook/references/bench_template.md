# Granian Benchmark Plan

## 1) Objectives
- What are we optimizing for? (throughput / latency / cost)
- Primary SLOs:
  - p50/p95/p99 latency:
  - error rate:
  - throughput (RPS):

## 2) Test environment
- Where will this run? (same node class as prod?)
- CPU cores / memory limits:
- Reverse proxy in path? (yes/no)
- TLS enabled? (yes/no)

## 3) Workload definition
Endpoints to test:
- `GET /__` (cheap)
- `POST /__` (representative)

Load shape:
- Concurrency levels: __
- Duration: __
- Warmup: __

Payload sizes:
- request size:
- response size:

## 4) Candidate configurations
Test matrix (each row is one run):
- workers=__ threads=__
- workers=__ threads=__
- workers=__ threads=__

## 5) Measurement
Record for each run:
- RPS, p50/p95/p99 latency
- Error breakdown (timeouts, 5xx)
- CPU utilization, memory RSS, context switches
- Connection counts / keep-alive hit ratio (if available)

## 6) Tools
- Load generator (wrk/k6/hey/locust):
- System metrics (pidstat, sar, docker stats, cAdvisor):
- App metrics/tracing (if any):

## 7) Decision rule
Choose the config that:
- Meets SLOs at target RPS
- Minimizes tail latency (p99)
- Stays within CPU/memory ceilings

## 8) Report output
Deliver as:
- One paragraph summary
- Best config + runner-up
- Raw results (attach or link)
- Notes on anomalies
