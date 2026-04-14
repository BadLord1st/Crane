# Wide Events (Canonical Log Lines)

## What is a Wide Event?

One comprehensive, structured log event per unit of work (request, job, message) containing all context needed for debugging.

**Instead of**:
```
INFO  Received request /api/orders
DEBUG Authenticating user
INFO  User authenticated: user_123
DEBUG Querying database
INFO  Found 5 orders
DEBUG Serializing response
INFO  Request completed in 234ms
```

**Use**:
```json
{
  "event_type": "wide_event",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_abc123",
  "trace_id": "trace_xyz789",
  "http": {
    "method": "GET",
    "path": "/api/orders",
    "status_code": 200
  },
  "user": {
    "id": "user_123",
    "tier": "premium"
  },
  "duration_ms": 234,
  "db": {
    "query_count": 2,
    "slowest_ms": 45
  },
  "outcome": "success"
}
```

## Schema Version 1

### Required Fields
| Field | Type | Description |
|-------|------|-------------|
| `event_type` | string | Always "wide_event" for filtering |
| `schema_version` | string | "1" |
| `timestamp` | ISO 8601 | Event time |
| `service.name` | string | Service identifier |
| `request.id` | string | Unique request ID |
| `duration_ms` | number | Total request time |
| `outcome` | enum | success \| error |

### HTTP Context
```json
{
  "http.method": "POST",
  "http.path": "/api/orders",
  "http.route": "/api/orders",
  "http.status_code": 201,
  "http.user_agent": "Mozilla/5.0..."
}
```

### User Context
```json
{
  "user.id": "usr_12345",
  "user.tier": "premium",
  "user.org_id": "org_789",
  "user.is_internal": false
}
```

### Error Context (when outcome=error)
```json
{
  "error.type": "ValidationError",
  "error.code": "INVALID_EMAIL",
  "error.message": "Email format invalid",
  "error.retriable": true
}
```

### Dependency Aggregation
```json
{
  "db.query_count": 5,
  "db.total_ms": 120,
  "db.slowest_ms": 45,
  "db.error_count": 0,
  
  "ext.call_count": 2,
  "ext.total_ms": 200,
  "ext.targets": ["payment-service", "email-service"]
}
```

## Implementation Pattern

### Python (FastAPI)
```python
from contextvars import ContextVar
from dataclasses import dataclass, field
import time
import structlog

wide_event_ctx: ContextVar["WideEvent"] = ContextVar("wide_event")
logger = structlog.get_logger()

@dataclass
class WideEvent:
    event_type: str = "wide_event"
    schema_version: str = "1"
    timestamp: str = ""
    service_name: str = "my-api"
    request_id: str = ""
    trace_id: str = ""
    
    http_method: str = ""
    http_path: str = ""
    http_status_code: int = 0
    
    user_id: str = ""
    user_tier: str = ""
    
    duration_ms: float = 0
    outcome: str = "success"
    
    error_type: str = ""
    error_message: str = ""
    
    db_query_count: int = 0
    db_slowest_ms: float = 0
    
    _extra: dict = field(default_factory=dict)
    
    def add(self, **kwargs):
        self._extra.update(kwargs)
    
    def to_dict(self):
        d = {k: v for k, v in self.__dict__.items() 
             if not k.startswith("_") and v}
        d.update(self._extra)
        return d


@app.middleware("http")
async def wide_event_middleware(request, call_next):
    event = WideEvent(
        timestamp=datetime.utcnow().isoformat() + "Z",
        request_id=request.headers.get("x-request-id", str(uuid4())),
        trace_id=request.headers.get("x-trace-id", ""),
        http_method=request.method,
        http_path=request.url.path,
    )
    wide_event_ctx.set(event)
    
    start = time.perf_counter()
    try:
        response = await call_next(request)
        event.http_status_code = response.status_code
        if response.status_code >= 500:
            event.outcome = "error"
        return response
    except Exception as e:
        event.outcome = "error"
        event.error_type = type(e).__name__
        event.error_message = str(e)[:500]  # Truncate
        raise
    finally:
        event.duration_ms = (time.perf_counter() - start) * 1000
        
        if should_sample(event):
            logger.info("wide_event", **event.to_dict())


def enrich_event(**kwargs):
    """Call from anywhere to add context"""
    event = wide_event_ctx.get(None)
    if event:
        event.add(**kwargs)


# In handlers
@app.get("/api/orders")
async def get_orders():
    enrich_event(user_id=current_user.id, user_tier=current_user.tier)
    orders = await db.get_orders()
    enrich_event(order_count=len(orders))
    return orders
```

## Tail Sampling

Decide what to log **after** the request completes:

```python
def should_sample(event: WideEvent) -> bool:
    # Always log errors
    if event.outcome == "error":
        return True
    
    # Always log slow requests
    if event.duration_ms > SLOW_THRESHOLD_MS:  # e.g., 2000
        return True
    
    # Always log VIP users
    if event.user_tier in ("enterprise", "vip"):
        return True
    
    # Always log internal users (for debugging)
    if event._extra.get("user_is_internal"):
        return True
    
    # Sample regular successful requests
    import random
    return random.random() < SAMPLE_RATE  # e.g., 0.01 = 1%
```

### Configuration
```python
# Environment-based config with defaults
SLOW_THRESHOLD_MS = int(os.getenv("WIDE_EVENT_SLOW_MS", "2000"))
SAMPLE_RATE = float(os.getenv("WIDE_EVENT_SAMPLE_RATE", "0.01"))
VIP_TIERS = os.getenv("WIDE_EVENT_VIP_TIERS", "enterprise,vip").split(",")
```

## Sanitization

```python
BLOCKED_FIELDS = {"authorization", "cookie", "password", "secret", "token"}
PII_FIELDS = {"email", "phone", "ssn", "address"}

def sanitize_event(event_dict: dict) -> dict:
    result = {}
    for key, value in event_dict.items():
        key_lower = key.lower()
        
        # Block sensitive fields entirely
        if any(blocked in key_lower for blocked in BLOCKED_FIELDS):
            continue
        
        # Hash PII fields
        if any(pii in key_lower for pii in PII_FIELDS):
            if isinstance(value, str):
                result[f"{key}_hash"] = hashlib.sha256(value.encode()).hexdigest()[:16]
                continue
        
        # Truncate long strings
        if isinstance(value, str) and len(value) > 2000:
            value = value[:2000] + "...[truncated]"
        
        result[key] = value
    
    return result
```

## Querying Wide Events

### Find slow requests
```sql
SELECT * FROM logs
WHERE event_type = 'wide_event'
  AND duration_ms > 1000
ORDER BY duration_ms DESC
LIMIT 100
```

### Error analysis by user tier
```sql
SELECT user_tier, 
       COUNT(*) as error_count,
       AVG(duration_ms) as avg_duration
FROM logs
WHERE event_type = 'wide_event'
  AND outcome = 'error'
GROUP BY user_tier
ORDER BY error_count DESC
```

### Request patterns by endpoint
```sql
SELECT http_path,
       COUNT(*) as requests,
       AVG(duration_ms) as avg_ms,
       PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95_ms
FROM logs
WHERE event_type = 'wide_event'
  AND timestamp > NOW() - INTERVAL '1 hour'
GROUP BY http_path
ORDER BY requests DESC
```

## Benefits vs Traditional Logging

| Aspect | Traditional | Wide Events |
|--------|-------------|-------------|
| Lines per request | 5-20 | 1 |
| Context correlation | Manual grep | Single record |
| Storage cost | Higher (many lines) | Lower (sampled) |
| Query speed | Slow (text search) | Fast (structured) |
| Debug experience | Piece together | Complete picture |
