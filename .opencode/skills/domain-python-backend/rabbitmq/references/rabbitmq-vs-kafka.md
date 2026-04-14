# RabbitMQ vs Kafka selection guide

RabbitMQ and Kafka overlap, but they optimize for different constraints.

## Use RabbitMQ when
- You need traditional messaging / task queues: one consumer processes a message and it is removed.
- You need low per-message latency and explicit per-message acknowledgements (confirm “handled”).
- You need flexible routing (direct/topic/fanout exchanges), multiple queues, and per-message handling.

Typical patterns: background jobs, work queues, RPC-style request/reply, job fanout to worker pools.

## Use Kafka when
- You need high-throughput event streaming with retained history (replay) and multiple independent consumers.
- Messages represent an append-only log of events, not “tasks to do right now”.
- You need horizontal scalability via partitions + consumer groups.

Typical patterns: domain event streams, event sourcing, analytics pipelines, CDC, streaming joins/aggregations.

## Use both when
- You have a stream of domain/user events (Kafka) but still need immediate offloading of slow tasks (RabbitMQ work queue).

A useful mental model:
- RabbitMQ: “deliver this task to exactly one worker (but duplicates can still happen).”
- Kafka: “publish an ordered log per partition; consumers track offsets; replay is a feature.”

## First questions to answer (decision checklist)
1) Do we need replay/history (hours/days) and multiple consumer groups? If yes, Kafka is a strong default.
2) Is per-message confirmation / complex routing / work-queue semantics central? If yes, RabbitMQ is a strong default.
3) What are the ordering requirements?
   - Strict ordering for a key/entity: Kafka (per partition) or RabbitMQ (per queue) can work.
   - Strict global ordering: expensive; usually implies one partition/queue and lower throughput.
4) What is the expected throughput and payload size? (Kafka is generally the safer choice at very high throughput.)
5) How will we handle retries, poison messages, and idempotency? (Required in both.)
