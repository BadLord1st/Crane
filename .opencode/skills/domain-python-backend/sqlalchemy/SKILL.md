---
name: sqlalchemy
description: "Зафиксировать правила использования SQLAlchemy (engine/session/transactions/loading/testing) как проверяемую спецификацию и конвенции; без запроса секретов и без выполнения миграций/DDL."
metadata:
  signature: "spec-sqlalchemy :: (SpecNode, Repo) -> DataAccessSpec"
---

## When to use
- Фича/фикс читает или пишет в БД через SQLAlchemy (ORM или Core).
- Нужно формализовать границы транзакций и жизненный цикл Session/AsyncSession.
- Есть риски перформанса: N+1, лишние lazy-load в сериализации, проблемы с пулом.
- Нужен единый стиль репозиториев/Unit of Work и тестовых фикстур.

## Inputs
- SpecNode (или тикет) с операциями чтения/записи и требованиями к консистентности.
- Снимок репо: где создаётся engine, как создаётся Session/AsyncSession, есть ли Alembic.
- Режим: sync или async (если неизвестно — явно зафиксировать допущение).
- Диалект БД (Postgres/MySQL/SQLite) и ожидаемая нагрузка (хотя бы “низкая/средняя/высокая”).

## Outputs
- DataAccessSpec: краткая спецификация конвенций (Session/UoW, транзакции, репозитории, loading-стратегия, тестирование).
- Минимальные кодовые шаблоны (snippets), а не “массовые правки файлов”.
- Verification steps: как поймать N+1/утечки сессии и проверить миграции “smoke-level”.

## Protocol
1) Определить контекст: SQLAlchemy 2.x стиль (`select()`/`session.execute()`), sync vs async, где живёт UoW (web-request/job/CLI).
2) Зафиксировать “Session per Unit of Work”: где создаём Session/AsyncSession, где commit/rollback, где закрываем.
3) Описать транзакционную модель: одна транзакция на кейс? нужны ли savepoints? какие инварианты.
4) Описать репозитории: интерфейсы в домене, реализации в infra; запрет на “ORM повсюду”.
5) Описать loading-стратегию: где допускается lazy-load, где обязателен eager-load; как предотвращаем N+1.
6) Описать правила сериализации/DTO: нельзя “случайно” триггерить lazy-load вне сессии.
7) Тестирование: фикстуры с rollback, стратегия БД для тестов, проверка соответствия схемы (Alembic smoke).
8) Safety-by-default: никаких DSN/паролей; любые DDL/миграции/опасные команды — только как dry-run план + явное подтверждение.

## Deliverables
- [ ] DataAccessSpec оформлен (конвенции + rationale + минимальные примеры).
- [ ] Шаблоны для Session/UoW и загрузки связей добавлены в references.
- [ ] Проверки: “как воспроизвести и поймать” N+1 и проблемы с lifecycle.
- [ ] Для изменений модели: указан путь “model -> migration -> verify” (без выполнения).

## Anti-patterns
- Глобальная/долго живущая Session (reuse между запросами/джобами).
- Возврат ORM-объектов наружу без гарантированной загруженности данных (detached + lazy-load).
- Сериализация ORM-объектов, которая провоцирует скрытые запросы (N+1).
- Commit в цикле без явной модели транзакций.
- “Починим перформанс” без измерений и воспроизводимого сценария.

## Examples
1) Запрос: “Добавляем Invoice + операции create/pay/cancel. Нужны транзакционные границы и репозитории.”
   Ожидаемо: DataAccessSpec с UoW на use-case, интерфейсами репозиториев, правилами commit/rollback, примерами запросов.
2) Запрос: “Есть N+1 при выдаче Users с Posts. Где правильно eager-load и как поймать регресс?”
   Ожидаемо: loading-стратегия (selectinload/joinedload), запреты на lazy-load в DTO, минимальный тест/профайлинг.
3) Запрос: “Переходим на AsyncSession.”
   Ожидаемо: async UoW-шаблон, правила `async with session.begin()`, ограничения на sync IO, тестовые фикстуры.

## References
- [Session + Unit of Work Patterns](./references/session_uow.md)
- [Loading Strategy + N+1 Guardrails](./references/loading_strategy.md)
- [Testing + Migration Alignment (smoke)](./references/testing_and_migrations.md)
