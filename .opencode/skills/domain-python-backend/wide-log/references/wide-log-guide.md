# Wide Log Guide

## Принцип
Wide log = один структурированный “канонический” event на unit-of-work (запрос/вызов/джоб/сообщение) на сервис.
Цель: все ключевые ответы для дебага — в одном событии, по которому удобно фильтровать.

## Event naming
- event_name: стабильное имя действия (например, `checkout.request`, `orders.create`, `job.weekly_digest`).
- Не включать динамику (id/uuid/email) в имя события.

## Envelope (обязательное)
Рекомендуемый минимум:
- event_type: `wide_event`
- schema_version: `"1"`
- timestamp (ISO8601)
- level
- service: name, version, env, region (если есть)
- correlation: request_id, trace_id, span_id (что доступно)
- unit: ровно один из блоков:
  - http: method, route/path, status_code
  - rpc: system (grpc), service, method, status_code
  - job: job_name, job_id, attempt, queue
  - msg: topic, partition, offset, attempt
- outcome: `success|error|timeout|canceled`
- duration_ms

## Payload (доменные блоки)
Собирайте только то, что реально нужно для “вопросов к логам”:
- actor: user_id/org_id/tenant_id, plan/tier (без PII)
- feature_flags/experiment
- domain: order_id/payment_id/cart_id и т.п. (ID обычно ок)
- dependencies (агрегаты, не “спам”):
  - db: query_count, total_ms, slowest_ms, error_count
  - cache: hit/miss counts, total_ms
  - ext: call_count, total_ms, targets (список сервисов/провайдеров)

## Error block
Только если outcome != success:
- error.type (класс/тип)
- error.code (стабильный код)
- error.message (обрезать длину)
- error.retriable (если известно)
Важно: никаких секретов, токенов, дампа payload.

## Tail sampling
Решение “логировать или нет” — в конце unit-of-work:
- Always: outcome=error
- Always: duration_ms > SLOW_THRESHOLD
- Always: VIP tiers / internal users (если принято)
- Else: random sample rate (например 1%)

## Sanitization / Safety
- Полный запрет ключей по подстроке: authorization, cookie, password, secret, token, api_key, private_key.
- PII (email/phone/address/ip): либо не логировать, либо хранить маску/хеш.
- Truncation: любые строки > N символов обрезать.
- Запрет “логировать целиком”: request.headers, request.body, response.body.

## Граничные случаи (про которые надо договориться)
1) Процесс упал до финализации: нужен хотя бы минимальный “start” event или гарантия flush на выходе (иначе wide event не появится).
2) Слишком “толстый” event: фиксируйте контракт и версионируйте schema_version, не расширяйте хаотично.
3) Высокий объём: sampling обязателен, иначе wide log станет DDoS на лог-хранилище.
4) PII/секреты: по умолчанию deny-list, allow-list для чувствительных доменных зон.
