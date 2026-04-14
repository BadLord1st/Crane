---
name: rabbitmq
description: "Define messaging requirements for RabbitMQ/Kafka: broker choice, topology, delivery semantics, and operability."
metadata:
  signature: "spec-messaging :: (SpecNode, Repo) -> MessagingNode"
---

## When to use
- The change introduces or modifies async workflows, background jobs, or event streams.
- You need explicit guarantees around ordering, retries, durability, or replay.
- You need a clear RabbitMQ vs Kafka choice (or a split) that downstream teams can implement.

## Inputs
- SpecNode (domain events / commands) + non-functional requirements (latency, throughput, retention, ordering).
- Current broker(s), client libraries, and topology (if already present).

## Outputs
- Messaging node: broker selection, topology (topics/queues/exchanges), message contracts, delivery/ordering semantics, retry/DLQ strategy, and observability hooks.

## Protocol
1) Classify message intent: command vs event vs task; fanout vs competing consumers; sync boundary vs async boundary.
2) Choose broker strategy (RabbitMQ, Kafka, or split) and document the rationale + constraints.
3) Define topology:
   - RabbitMQ: exchanges (direct/topic/fanout), routing keys, queues, bindings, DLX/DLQ.
   - Kafka: topics, partitions, keys, replication factor, retention.
4) Define producer guarantees: durability settings, publish confirms / acks, batching/flush behavior.
5) Define consumer guarantees: ack/commit mode, idempotency keys, retry policy, poison-message handling, backpressure.
6) Define schema + evolution rules: versioning, compatibility, optional schema registry.
7) Define operability + observability: key metrics, structured log fields, trace context propagation.
8) Safety: do not propose destructive broker operations by default (purge queues, delete topics). If needed, require explicit confirmation and provide a dry-run plan.
9) Emit the Messaging node and link it from the spec-index; list open questions and risk hotspots.

## Deliverables
- [ ] Broker choice + rationale (RabbitMQ vs Kafka vs both).
- [ ] Topology diagram (topics/queues/exchanges) + naming conventions.
- [ ] Message contract(s): fields, versioning, size limits, PII policy.
- [ ] Delivery semantics: at-least-once assumptions + idempotency plan.
- [ ] Retry/DLQ strategy + operational alerts.

## Anti-patterns
- Treating Kafka like a per-message work queue (or RabbitMQ like an event log) without acknowledging the semantics differences.
- “Fire-and-forget” producers (no durability/acks) for critical flows.
- Consumers without idempotency handling (duplicates happen).
- Embedding secrets/PII in payloads or logs.
- Recommending purge/delete operations without confirmation + safe rollback.

## References
- [RabbitMQ vs Kafka selection guide](./references/rabbitmq-vs-kafka.md)
- [Delivery semantics, retries, and idempotency](./references/delivery-semantics.md)
- [Examples: Messaging node outputs](./references/examples.md)
