# Loading Strategy + N+1 Guardrails

По умолчанию связи часто lazy-loaded; для предсказуемой производительности нужно явно выбирать eager-loading там, где данные точно понадобятся, чтобы избежать N+1.

## Базовая политика (пример)
- В handler/endpoint слое запрещаем неявные lazy-load (DTO/serializer не должен триггерить запросы).
- В repo/query-функциях явно указываем loader options.
- Для коллекций чаще подходит selectinload; joinedload полезен для “1:1 / many-to-one”, но может раздувать строки.

## SQLAlchemy 2.x examples
```python
from sqlalchemy import select
from sqlalchemy.orm import selectinload, joinedload

# Users + Posts (коллекция) — обычно selectinload
stmt = select(User).options(selectinload(User.posts)).where(User.is_active == True)
users = (session.execute(stmt).scalars().all())

# Order + Customer (many-to-one) — часто joinedload
stmt = select(Order).options(joinedload(Order.customer)).where(Order.id == order_id)
order = session.execute(stmt).scalar_one()
```

## “Поймать” случайные lazy-load

Подходы (фиксируем в спеках как guardrails):

* Настроить отношения с `lazy="raise"` (или эквивалент) для критичных моделей.
* В тестах включать логирование SQL и/или считать количество запросов на use-case.
* DTO строить из заранее загруженных данных (не из “живых” ORM объектов с ленивыми связями).

## Граничные случаи

* “Большие коллекции”: eager-load может взорвать память — тогда делаем pagination/батчинг, а не “загрузить всё”.
* “Смешанные графы”: разные use-case требуют разного loading; это нормально — фиксируем per-query.
