# Observability Instrumentation Patterns

## The Three Pillars

| Pillar | Purpose | When to Use |
|--------|---------|-------------|
| Metrics | Aggregated, time-series data | Dashboards, alerts, trends |
| Logs | Discrete events with context | Debugging, audit trail |
| Traces | Request flow across services | Latency analysis, debugging |

## Metrics Instrumentation

### Metric Types

| Type | Use Case | Example |
|------|----------|---------|
| Counter | Cumulative count | Requests, errors, items processed |
| Gauge | Current value | Queue size, active connections |
| Histogram | Distribution | Request latency, response size |
| Summary | Client-side percentiles | Similar to histogram |

### Naming Conventions
```
# Format: <namespace>_<subsystem>_<name>_<unit>

# Good
http_requests_total
http_request_duration_seconds
db_connections_active
queue_messages_pending

# Bad
requests          # No context
httpRequestTime   # camelCase
latency          # No unit
```

### Python (Prometheus)
```python
from prometheus_client import Counter, Histogram, Gauge

# Counter for requests
http_requests = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

# Histogram for latency
request_latency = Histogram(
    'http_request_duration_seconds',
    'Request latency',
    ['method', 'endpoint'],
    buckets=[.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10]
)

# Gauge for connections
active_connections = Gauge(
    'db_connections_active',
    'Active database connections'
)

# Usage
http_requests.labels(method='GET', endpoint='/api/users', status='200').inc()
request_latency.labels(method='GET', endpoint='/api/users').observe(0.125)
active_connections.set(pool.size())
```

### Node.js (prom-client)
```javascript
const { Counter, Histogram, Gauge, register } = require('prom-client');

const httpRequests = new Counter({
  name: 'http_requests_total',
  help: 'Total HTTP requests',
  labelNames: ['method', 'endpoint', 'status']
});

const requestLatency = new Histogram({
  name: 'http_request_duration_seconds',
  help: 'Request latency',
  labelNames: ['method', 'endpoint'],
  buckets: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
});

// Usage in middleware
app.use((req, res, next) => {
  const start = process.hrtime.bigint();
  res.on('finish', () => {
    const duration = Number(process.hrtime.bigint() - start) / 1e9;
    httpRequests.labels(req.method, req.path, res.statusCode).inc();
    requestLatency.labels(req.method, req.path).observe(duration);
  });
  next();
});
```

## Logging Best Practices

### Structured Logging
```python
import structlog

logger = structlog.get_logger()

# Always structured, not string interpolation
logger.info(
    "user_login",
    user_id=user.id,
    email=user.email,
    method="password",
    ip_address=request.client.host,
    user_agent=request.headers.get("user-agent")
)
```

### Log Levels
| Level | When to Use |
|-------|-------------|
| DEBUG | Detailed diagnostic info |
| INFO | Normal operation events |
| WARNING | Something unexpected, but handled |
| ERROR | Operation failed, needs attention |
| CRITICAL | System is unusable |

### Context Enrichment
```python
# Add context to all logs in request
logger = logger.bind(
    request_id=request.id,
    user_id=current_user.id,
    trace_id=get_trace_id()
)

# All subsequent logs include this context
logger.info("processing_order", order_id=order.id)
logger.info("payment_initiated", amount=order.total)
```

### What to Log
```
✅ DO LOG:
- Request/response metadata (not bodies)
- User actions (login, logout, permission change)
- Business events (order placed, payment processed)
- Errors with context
- Performance markers (slow queries, timeouts)
- Security events (failed login, permission denied)

❌ DON'T LOG:
- Passwords, tokens, secrets
- Full request/response bodies
- Personal data (unless required)
- High-frequency events without sampling
```

## Distributed Tracing

### Trace Anatomy
```
Trace: abc123
├── Span: api-request (root)
│   ├── Span: auth-middleware
│   ├── Span: db-query
│   │   └── Span: connection-acquire
│   └── Span: response-serialize
```

### OpenTelemetry Python
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Setup
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="otel-collector:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)

# Manual instrumentation
@tracer.start_as_current_span("process_order")
def process_order(order_id: str):
    span = trace.get_current_span()
    span.set_attribute("order.id", order_id)
    
    with tracer.start_as_current_span("validate_order"):
        validate(order_id)
    
    with tracer.start_as_current_span("charge_payment"):
        charge(order_id)
```

### Context Propagation
```python
# Extract context from incoming request
from opentelemetry.propagate import extract

@app.middleware("http")
async def tracing_middleware(request, call_next):
    context = extract(request.headers)
    with tracer.start_as_current_span("handle_request", context=context):
        response = await call_next(request)
    return response

# Inject context into outgoing request
from opentelemetry.propagate import inject

def call_external_service(url):
    headers = {}
    inject(headers)  # Adds trace context headers
    return requests.get(url, headers=headers)
```

## SLOs and Alerting

### Define SLOs
```yaml
# Example SLO definition
slos:
  - name: api_availability
    target: 99.9
    window: 30d
    indicator:
      ratio:
        good: http_requests_total{status!~"5.."}
        total: http_requests_total

  - name: api_latency
    target: 95
    window: 30d
    indicator:
      ratio:
        good: http_request_duration_seconds_bucket{le="0.5"}
        total: http_request_duration_seconds_count
```

### Alert Rules (Prometheus)
```yaml
groups:
  - name: slo-alerts
    rules:
      - alert: HighErrorRate
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[5m]))
            /
            sum(rate(http_requests_total[5m]))
          ) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Error rate exceeds 1%"

      - alert: HighLatency
        expr: |
          histogram_quantile(0.95, 
            sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
          ) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "p95 latency exceeds 500ms"
```

## Dashboard Design

### Essential Panels

1. **Rate**: Requests per second
2. **Errors**: Error rate %
3. **Duration**: p50, p95, p99 latency
4. **Saturation**: Resource usage (CPU, memory, connections)

### Grafana Query Examples
```promql
# Request rate
sum(rate(http_requests_total[5m]))

# Error rate percentage
sum(rate(http_requests_total{status=~"5.."}[5m])) 
/ 
sum(rate(http_requests_total[5m])) 
* 100

# Latency percentiles
histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))
histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))

# Active connections
db_connections_active
```
