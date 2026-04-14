# Session + Unit of Work (UoW) Patterns

Ключевое правило: Session/AsyncSession — короткоживущая, привязана к одному “юниту работы” (обычно web-request или одна job). Не делаем глобальную Session и не шарим её между независимыми операциями.

## Sync (SQLAlchemy 2.x) — базовый шаблон

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

engine = create_engine(
    DATABASE_URL,  # без секретов в коде/логах
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=True,  # см. примечание ниже
)

def uow(fn, *args, **kwargs):
    """Обёртка UoW: один вход -> одна транзакция."""
    with SessionLocal() as session:  # гарантирует close()
        try:
            result = fn(session, *args, **kwargs)
            session.commit()
            return result
        except Exception:
            session.rollback()
            raise
```

## Async — базовый шаблон

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

engine = create_async_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,  # часто удобнее в async-сценариях
)

async def uow_async(fn, *args, **kwargs):
    async with AsyncSessionLocal() as session:
        async with session.begin():  # begin() = commit/rollback автоматически
            return await fn(session, *args, **kwargs)
```

## expire_on_commit: важная развилка

* True: после commit объекты “протухают” и при обращении могут рефетчиться (требуется активная Session).
* False: удобнее для DTO/response, но повышает риск работать со “старыми” данными.
  Решение фиксируем в DataAccessSpec по проекту, не “как получится”.

## Мини-правило для сервисного слоя

Сервисные функции принимают Session/AsyncSession параметром (явная зависимость), а не создают её внутри “где-то в глубине”.
