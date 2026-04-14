---
name: wide-log
description: "Define a Wide Log (canonical wide event) contract: 1 structured event per unit-of-work with correlation IDs, outcome, latency, domain context, sampling and redaction rules."
metadata:
  signature: "spec-wide-log :: (SpecNode, FailureModes) -> WideLogSpec"
---

## When to use
- Вы хотите перейти от “много строк на запрос” к “1 wide event per unit-of-work” (HTTP/gRPC/job/message).
- Инциденты требуют склейки контекста по request_id/trace_id, и это занимает время.
- Нужно согласовать единый контракт полей (корреляция, outcome, latency, actor/tenant, доменные ID, агрегаты зависимостей).
- Нужно safe-by-default логирование: redaction/PII правила и запрет на дамп секретов.

## Inputs
- Тип unit-of-work: HTTP request, gRPC call, background job, consumer message (можно несколько).
- Список failure modes / “вопросов к логам” (пример: “почему оплаты падают у premium”).
- Какие корреляционные ID уже есть: request_id, trace_id/span_id (или ничего).
- Доменные сущности и “разрезы”: user/org/tenant, plan/tier, order/payment, feature flags.
- Политика данных: что считать PII/секретами, что можно логировать только хешом/маской.

## Outputs
- WideLogSpec: схема (schema_version), обязательный envelope, доменные payload-блоки, правила именования событий.
- Emission strategy: где собирать и где эмитить 1 wide event, что делать при exception, формат (JSON, single-line).
- Sampling strategy (tail sampling): что логировать всегда, что семплировать.
- Redaction/sanitization policy: deny/allow, truncation, hashing, защита от log-injection.
- Набор примеров wide events + набор “query recipes”.

## Protocol
1) Зафиксировать unit-of-work и правило: “1 event per unit-of-work per service hop”.
2) Определить `event_name` (стабильное) и `schema_version` (версионирование контракта).
3) Спроектировать envelope (минимальный обязательный набор):
   - timestamp, level
   - service.name/service.version/env/region (если есть)
   - correlation: request_id и/или trace_id + span_id
   - unit: http.* или rpc.* или job.* или msg.*
   - outcome + status_code + duration_ms
4) Спроектировать доменные payload-блоки (вложенные объекты) и агрегаты:
   - actor: user_id/org_id/tenant_id (без PII)
   - feature_flags/experiment
   - db/cache/external_calls как агрегаты (counts, totals, slowest, targets)
   - error-блок только при outcome=error (type/code/message, без секретов)
5) Определить emission point:
   - сбор контекста в течение работы (context store),
   - эмиссия в конце (finally/after_response), чтобы знать outcome и duration.
6) Tail sampling:
   - всегда логировать ошибки,
   - всегда логировать “slow” выше порога,
   - всегда логировать VIP/интернал (если нужно),
   - остальное — rate-based.
7) Sanitization:
   - запрет на секреты (authorization/cookie/token/password/key),
   - PII — только маска/хеш по политике,
   - truncation длинных строк,
   - запрет “логировать целиком request/response/body/headers”.
8) Сгенерировать примеры (JSONL) и query recipes, плюс rollout checklist (MVP → расширение → контроль объёма).

## Deliverables
- [ ] SKILL-уровневый контракт wide event (schema_version + envelope + payload-блоки).
- [ ] Политика redaction/PII и правила truncation/hashing.
- [ ] Tail sampling правила (errors/slow/VIP + rate).
- [ ] 3–6 примеров wide events в JSONL.
- [ ] 5–8 query recipes (по outcome, duration, endpoint/method, user_tier, error_code, dependency targets).

## Anti-patterns
- “Добавим ещё логов” без контракта полей и без единичного wide event на unit-of-work.
- Динамические `event_name` (с user_id/uuid в имени).
- Мультилайн и “размазанные” stacktrace-строки вместо структурированного error-блока.
- Логирование секретов/PII или дамп целых объектов (request/headers/body).

## References
- [Wide Log Guide](./references/wide-log-guide.md)
- [Wide Log Examples (JSONL + queries)](./references/wide-log-examples.md)
