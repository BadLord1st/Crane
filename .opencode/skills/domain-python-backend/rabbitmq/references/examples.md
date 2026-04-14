# Examples: Messaging node outputs

These are intentionally compact “what good looks like” examples.

## Example 1 — Work queue for slow tasks (RabbitMQ)

Request:
- “We need to offload PDF generation + email sending from the request path. Reliability matters.”

Messaging node (sketch):
- Broker: RabbitMQ
- Exchange: jobs (direct)
- Queues:
  - pdf_tasks (durable, DLX=jobs.dlx)
  - email_tasks (durable, DLX=jobs.dlx)
  - jobs.dlq (durable)
- Routing keys: pdf.generate, email.send
- Producer:
  - persistent messages
  - publisher confirms enabled (if client supports)
- Consumer:
  - manual ack after side-effect commit
  - prefetch_count=1..N (start with 1 for fairness)
  - idempotency by task_id (UUID)
- Retry:
  - max_attempts=5
  - delayed retry queue (TTL/backoff), then DLQ
- Observability:
  - metrics: queue_depth, consumer_utilization, dlq_rate, processing_latency
  - logs: task_id, trace_id, routing_key, attempt

## Example 2 — Domain event stream with multiple consumers (Kafka)

Request:
- “We need a durable stream of user actions. Several services consume it independently, and replay is required.”

Messaging node (sketch):
- Broker: Kafka
- Topic: user-events
- Partitions: 24 (tune to throughput); key=user_id
- Replication factor: 3
- Retention: 14d (time-based) + size cap
- Producer:
  - acks=all
  - idempotent producer on (if supported)
- Consumers:
  - consumer groups: analytics, notifications, fraud
  - commit offsets after processing
  - idempotency by event_id
- Schema:
  - Protobuf/Avro + schema registry
  - compatibility: backward
- Retry/DLQ:
  - user-events.retry.1m / user-events.retry.10m
  - user-events.dlq with error metadata
- Observability:
  - metrics: consumer_lag, produce_errors, retry_rate, dlq_rate
  - logs: event_id, user_id, partition, offset, trace_id

## Example 3 — Split: Kafka for events, RabbitMQ for tasks

Request:
- “We want event-driven integration (replayable), but also need a worker pool for CPU-heavy jobs.”

Messaging node (sketch):
- Kafka topic: domain-events (business events, retained)
- RabbitMQ exchange/queues: jobs/* (derived tasks, ephemeral)
- Bridge rule:
  - consumer group job-dispatcher reads domain-events
  - emits tasks to RabbitMQ with task_id and trace context
- Safety:
  - backpressure: if RabbitMQ queue depth > threshold, slow/stop dispatch
  - poison handling: tasks go to DLQ; events remain in Kafka for audit/replay
