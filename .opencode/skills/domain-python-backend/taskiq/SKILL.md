---
name: taskiq
description: "Спроектировать интеграцию TaskIQ: выбор broker/result backend, воркерная топология, ретраи/таймауты, расписание и наблюдаемость (без внесения правок в код по умолчанию)."
metadata:
  signature: "spec-taskiq :: (SpecNode, InfraHints, Repo?) -> TaskIQPlan"
---

## When to use
- Нужны фоновые задачи вне request/response (email, обработка файлов, синхронизации, ETL, batch-операции).
- Нужны ретраи/отложенные запуски/периодические задания.
- Нужна эксплуатационная ясность: как запускать воркеры, как мониторить, как безопасно выкатывать изменения задач.

## Inputs
- SpecNode: что именно уносится в background, какие входы/выходы, SLO/временные ограничения.
- InfraHints: что доступно (RabbitMQ/Redis/NATS), окружение (docker/k8s/systemd), требования по надёжности.
- (Опционально) Repo: текущие фреймворки/DI, логирование, метрики, конфиги.

## Outputs
- TaskIQPlan (узел/док): 
  - broker + result backend выбор и аргументы
  - контракты задач (сигнатуры, idempotency key, дедупликация, side-effects)
  - worker topology (пулы, разделение CPU/I/O, приоритеты, изоляция опасных задач)
  - retry/timeout policy + failure taxonomy
  - scheduling (periodic/delayed) + требования к отдельному scheduler-процессу (если применимо)
  - observability: логи/метрики/трейсы + correlation IDs
  - runbook: как стартовать/останавливать/скейлить/дебажить

## Protocol
1) Классифицировать задачи: I/O-bound vs CPU-bound, latency-critical vs batch, допустимость повторного выполнения.
2) Выбрать broker и result backend под требования (durability, throughput, ops):
   - зафиксировать why: надёжность, сложность эксплуатации, задержки, стоимость.
3) Определить lifecycle интеграции:
   - где и когда вызываются startup/shutdown брокера;
   - как задачи регистрируются (import-time) и где живёт “broker instance”.
4) Описать контракты задач:
   - идемпотентность (ключ/дедуп), границы транзакций, exactly-once vs at-least-once ожидания;
   - политика повторов, backoff, максимальные попытки; что считать retryable.
5) Таймауты и отмена:
   - per-task timeout (и/или asyncio.wait_for внутри задач), поведение при истечении.
6) Scheduling / delayed:
   - какие задачи периодические, какие отложенные; требования по таймзонам/UTC.
7) Observability:
   - “wide events” лог-поля: task_id, correlation_id, user/tenant, attempt, duration, error_class;
   - метрики: task_started/finished/failed, retries, duration histogram, queue lag (если доступно).
8) Runbook:
   - команды/параметры воркера, параллелизм, health checks, рестарты, алерты, канареечные проверки.
9) Явно перечислить риски/неясности и что нужно уточнить у команды/инфры.

## Deliverables
- [ ] TaskIQPlan (одним документом) с зафиксированными решениями broker/backend + rationale.
- [ ] Матрица задач (тип, критичность, ретраи, таймауты, идемпотентность, наблюдаемость).
- [ ] Runbook: запуск/скейл/рестарт/дебаг воркеров и scheduler (если применимо).
- [ ] Минимальный “definition of done” для production: метрики/логи/алерты, лимиты ретраев, тест-кейсы.

## Examples
1) Запрос: “Нужно отправлять письма и генерировать PDF; письма можно ретраить, PDF CPU-bound”.
   Ожидаемый результат: раздельные worker pools (I/O vs CPU), разные ретраи/таймауты, идемпотентность на уровне business-key.

2) Запрос: “Каждый час чистить просроченные сессии; важно не заддосить БД”.
   Ожидаемый результат: periodic scheduling + rate limiting/батчирование, отдельные метрики длительности/ошибок, защитные лимиты.

3) Запрос: “Отложенная задача через 5 минут после signup”.
   Ожидаемый результат: delayed execution pattern (broker capability), требования к надёжности (что если брокер недоступен), поля корреляции.

## Anti-patterns
- Использовать InMemory broker для production.
- Не описать startup/shutdown и получить “задачи не отправляются/не принимаются”.
- Неидемпотентные задачи с побочными эффектами без дедуп-ключа.
- Безлимитные ретраи, отсутствие backoff/таймаутов.
- Блокирующие CPU/IO операции внутри async-задач без изоляции (thread/process separation).
- Логи/метрики без task_id/correlation_id; протечки секретов в логах.

## References
- TaskIQ: getting started, scheduling, middlewares, broker lifecycle.
- [Design Patterns](./references/taskiq-design-patterns.md)
- [Snippets](./references/taskiq-snippets.md)
