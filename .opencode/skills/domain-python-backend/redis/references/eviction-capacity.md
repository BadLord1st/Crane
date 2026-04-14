# Eviction & Capacity

## Baseline approach
- Treat Redis memory as a *budgeted* resource.
- Decide which data is *evictable* (best-effort) vs *must be retained*.

## maxmemory & policies (rule of thumb)
- Pure cache workloads: consider `allkeys-lru` or `allkeys-lfu`.
- Mixed workloads where only some keys have TTL: `volatile-*` policies can surprise you (non-TTL keys become non-evictable).
- `noeviction` is safe for correctness but can shift failures to clients (writes fail) — document the degrade path.

## Required spec items
- `maxmemory` assumption (or managed default) + headroom.
- Eviction policy and the rationale.
- “What happens when memory is full?”
  - Which operations fail?
  - Does the app degrade to DB, return 429, or shed load?

## Signals to watch
- `evicted_keys` increasing steadily → memory pressure / TTL misdesign.
- High `used_memory_rss` vs `used_memory` gap → fragmentation.
- Rising latency + low hit rate → cache churn or hot keys.
- Increasing `rejected_connections` / client timeouts → pool saturation or instance overload.

## Mitigations
- Ensure TTL on cache keys; cap cardinality.
- Reduce value sizes; store references not blobs.
- Add sharding for hot keys or redesign key patterns.
- Separate workloads (e.g., cache vs locks/queues) into different Redis instances.
- Adjust TTLs and refresh strategy to reduce churn.

## Anti-patterns
- “We’ll just increase memory” without key/TTL discipline.
- Unbounded lists/streams without retention trimming.
