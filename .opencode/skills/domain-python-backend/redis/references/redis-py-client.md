# Client Config (redis-py)

Goal: stable latency and predictable failure modes.

## Connection management
- Reuse a singleton client/pool per process.
- Set bounded `max_connections` (avoid connection storms).
- Prefer separate pools for distinct workloads if they have very different latency profiles.

## Timeouts (principles)
- Set `socket_connect_timeout` (fail fast on network issues).
- Set `socket_timeout` (bound read latency).
- Avoid infinite waits.

## Retries (principles)
- Retries should be limited and aligned with idempotency.
- Do not blindly retry writes that are not safe to repeat.
- If retrying, use small backoff + jitter.

## Minimal sync example (template)
```python
from redis import Redis
from redis.connection import ConnectionPool

pool = ConnectionPool.from_url(
    "redis://localhost:6379/0",
    max_connections=50,
    decode_responses=False,
    health_check_interval=30,
)

r = Redis(
    connection_pool=pool,
    socket_connect_timeout=0.2,
    socket_timeout=0.5,
)
```

## Async example (template)
```python
from redis.asyncio import Redis

r = Redis.from_url(
    "redis://localhost:6379/0",
    max_connections=50,
    socket_connect_timeout=0.2,
    socket_timeout=0.5,
    health_check_interval=30,
)
```

## Pipelines
- Use pipelines to reduce RTT when you have multiple independent ops.
- Keep pipeline size bounded to avoid long blocking.

## Serialization
- Choose a single serialization format per domain.
- Include a schema/version field inside values if evolution is expected.
