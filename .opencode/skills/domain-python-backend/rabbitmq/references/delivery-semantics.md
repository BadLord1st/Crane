# Delivery semantics, retries, and idempotency

This note is an implementation-oriented checklist for making messaging reliable.

## Shared baseline (RabbitMQ and Kafka)
- Assume at-least-once delivery unless you can prove otherwise.
- Design consumers to be idempotent (dedupe by event_id/task_id, or by deterministic side-effect).
- Prefer small, versioned message schemas; set explicit payload size limits.
- Always carry correlation identifiers (trace_id/request_id, tenant/user IDs where allowed).

## RabbitMQ specifics

Durability and ack:
- Durable queues + persistent messages for critical tasks.
- Manual acks: only ack after the side-effect is committed.
- Use QoS prefetch (often prefetch_count=1 for fairness in worker pools).

Routing:
- Direct exchange: point-to-point.
- Fanout exchange: broadcast.
- Topic exchange: pattern routing (e.g., user.*).

Retries and DLQ:
- Prefer a bounded retry strategy: immediate retries (small count) + delayed retry queue (TTL) + DLQ.
- For poison messages, route to DLQ with context (error class, last_attempt_at, attempts).

Ordering and duplicates:
- Ordering is per queue; duplicates can occur (consumer crash after processing before ack).

## Kafka specifics

Topics, partitions, and keys:
- Ordering is per partition; choose a key that preserves the ordering you actually need (e.g., user_id).
- Scale is via partitions + consumer groups; consumers in the same group share partitions.

Offsets and delivery:
- Messages are retained; consumption advances offsets.
- Commit offsets only after processing is done (or use transactional patterns if available).

Durability:
- Use replication factor >= 2/3 for production topics.
- Producers should use acks=all for stronger durability.

Schema evolution:
- If you publish structured events (JSON/Avro/Protobuf), consider schema registry.
- Adopt compatibility rules (backward/forward) and explicit versioning.

Retries and DLQ:
- Prefer explicit retry topics with backoff (e.g., topic.retry.1m, topic.retry.10m) and a dead-letter topic.
- Avoid infinite retry loops; cap attempts and escalate.

Exactly-once caveat:
- Exactly-once is possible in narrow cases (Streams/transactions), but the safe default is at-least-once + idempotent consumers.
