---
name: redis
description: "Specify Redis usage in a service: keyspace/TTL/eviction/client config/atomicity/security/observability as a spec artifact (not implementation)."
metadata:
  signature: "redis :: (SpecNode, Repo) -> RedisUsageSpec"
---

## When to use
- You are adding Redis for cache/session/rate-limit/coordination and want rules, not folklore.
- You need explicit TTL/invalidation, and a plan for stampede/hot keys.
- You need eviction + OOM/degradation behavior documented.
- You need safe atomic patterns (counters, idempotency keys, locks).

## Inputs
- SpecNode + constraints (SLO, acceptable staleness, what may be lost).
- Workload sketch: RPS, key cardinality, value size bounds, read/write ratio.
- Redis topology (managed/standalone/sentinel/cluster) — if unknown, mark ASSUMPTION.
- Client/runtime constraints (language, sync/async).

## Outputs
- A `RedisUsageSpec` node (Markdown) that includes:
  - Intent: what Redis is used for (cache vs coordination vs messaging).
  - Keyspace: naming, versioning, examples, size/TTL constraints.
  - TTL & invalidation: default TTL, exceptions, stampede protection.
  - Capacity & eviction: maxmemory assumptions, policy, OOM behavior.
  - Atomicity: which operations must be atomic, recommended primitives/Lua.
  - Client config: pooling, timeouts, retries/backoff (no secrets).
  - Observability: metrics/alerts and what “good” looks like.
  - Security baseline: network isolation, ACL/TLS, dangerous commands.
  - Risks: top risks + mitigations.

## Protocol
1) Classify the Redis roles (cache/session/rate-limit/locks/streams/pubsub).
2) Specify data semantics: TTL, staleness, idempotency, acceptable loss.
3) Design keyspace: prefixes (service/env/version), max value sizes, TTL matrix.
4) Define cache semantics: cache-aside vs write-through, invalidation, stampede controls.
5) Define capacity & eviction: what must have TTL, what may be evicted, how the app degrades.
6) Define atomic patterns: counters, rate-limit, idempotency keys, locks (if needed).
7) Define client config: pooling, connect/read timeouts, retry policy, backpressure.
8) Define observability & security: metrics + baseline hardening.

## Deliverables
- [ ] `RedisUsageSpec` drafted and linked from spec-index.
- [ ] Keyspace examples + TTL matrix.
- [ ] Eviction/OOM/degradation plan.
- [ ] Atomicity/locking guidance (or an explicit “no locks needed”).
- [ ] Minimal metrics/alerts list.

## Anti-patterns
- Redis as a primary database for critical data without an explicit durability/failover spec.
- Keys without TTL by default; uncontrolled cardinality.
- Large blobs (multi-MB) or unbounded lists/streams without retention.
- Non-atomic read-modify-write where atomic primitives exist.
- New connection per request (no pooling / no reuse).
- Unsafe serialization of untrusted data (e.g., pickle).
- Redis exposed to the internet; shared credentials; prod has dangerous commands enabled.

## References
- [Keyspace & TTL](references/keyspace-ttl.md)
- [Eviction & Capacity](references/eviction-capacity.md)
- [Client Config (redis-py)](references/redis-py-client.md)
- [Atomicity Patterns (Commands/Lua)](references/atomicity.md)
- [Security Baseline (Network/ACL/TLS)](references/security.md)
