# Testing + Migration Alignment (smoke)

## Тесты: rollback-per-test (идея)
Цель — быстрые тесты без утечек данных и без “грязного” состояния между кейсами.

### Sync sketch
```python
import pytest
from sqlalchemy.orm import Session

@pytest.fixture()
def session(SessionLocal) -> Session:
    with SessionLocal() as s:
        trans = s.begin()  # открываем транзакцию
        try:
            yield s
        finally:
            trans.rollback()  # откатываем всё, что сделал тест
```

### Async sketch

```python
import pytest

@pytest.fixture()
async def session(async_sessionmaker):
    async with async_sessionmaker() as s:
        async with s.begin():
            yield s
        # rollback/commit управляется begin() (зависит от выбранной стратегии)
```

## Migration alignment (smoke-level)

Если меняются модели:

* Спека должна описывать путь “model -> alembic revision -> verify”.
* Проверка “smoke”: в CI прогоняем upgrade до head на чистой БД (или хотя бы валидируем, что миграции компилируются).
  Важно: этот skill не запускает миграции сам — только описывает шаги и критерии готовности.

## Типовые ошибки, которые фиксируем в спеках

* Тесты проходят на SQLite, а миграции/типизация ломаются на Postgres.
* Модели меняются, но миграции забыты.
* В тестах Session живёт дольше одного кейса (утечки).
