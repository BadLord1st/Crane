# Opentelemetry Collector Export Sampling

## Export topology (default)
App -> OTLP -> OpenTelemetry Collector -> backends (traces/metrics/logs)

Why Collector:
- centralizes batching/retries and routing to Jaeger/Tempo/Zipkin/Prometheus-compatible endpoints
- decouples app config from backend changes

## OTLP endpoints (intent only)
- OTLP/gRPC: <host:4317>
- OTLP/HTTP: <host:4318>

Do NOT embed credentials/tokens here; keep auth out-of-band.

## Sampling policy
Dev:
- AlwaysOn (record everything), short retention

Prod:
- Ratio-based head sampling as baseline (e.g., 1–10%)
- “Important transactions” may be force-sampled
- Tail sampling (collector-side) if you need “keep errors / slow traces”

## Verification (smoke)
- One HTTP request: verify trace spans include inbound + DB + outbound HTTP
- One gRPC call: verify cross-service trace continuity
- One async job: verify linkage to originating request trace
