# Keyspace & TTL

## Naming convention
Prefer: `{service}:{env}:{domain}:v{N}:{entity}:{id}:{sub}`

Rules:
- Always include `service` and `env` (or an explicit single-tenant assumption).
- Include a version segment (`v1`, `v2`) for schema/key migration safety.
- Keep keys short but readable; avoid PII in keys.
- Document the *owner* of each domain prefix.

Examples
- `api:prod:user:v1:profile:{user_id}`
- `api:prod:auth:v1:session:{session_id}`
- `api:prod:rl:v1:login:{ip}:{yyyymmddhhmm}`
- `api:prod:idemp:v1:charge:{request_id}`

## TTL policy
Default stance: *everything* has TTL unless explicitly justified.

A simple TTL matrix (example)
- User profile cache: 5–15 min (+ jitter)
- Feature flags cache: 15–60 sec (+ jitter)
- Session: 1–30 days (business-dependent)
- Rate limit windows: 1–10 min
- Idempotency keys: 24–72 hours
- Locks: 5–60 sec (must be bounded)

## Stampede protection (thundering herd)
Pick one (document the choice):
- TTL jitter: randomize expiry (e.g., `ttl = base ± 10%`).
- Single-flight per key (in-process) to collapse concurrent refresh.
- Soft TTL (serve stale + refresh in background) if staleness is acceptable.
- Leases/locks for refresh (only if truly necessary; keep TTL short).

## Value format
- Prefer compact JSON or msgpack for application caches.
- Prefer bytes and explicit schema/versioning when performance matters.
- Avoid untrusted pickle.

## Guardrails
- Explicit max value size (e.g., “<= 16KB” for hot cache values).
- Explicit max cardinality per key pattern (to prevent memory bloat).
- If a key can become “hot”: include sharding strategy (`{hash(id)%N}`) or redesign.

## Common failure modes
- Keys without TTL accumulate → eviction storms / OOM.
- Too-low TTL → churn and increased load on the backing DB.
- Hot key → single-threaded saturation on one shard/instance.
