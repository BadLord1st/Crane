# Alembic Script Review Checklist

- Does autogen propose any DROP that should be a RENAME or ALTER instead?
- Are NULLability transitions safe (backfill before NOT NULL)?
- Are server defaults explicit where needed (avoid implicit runtime defaults)?
- Are constraint/index names stable and intentional?
- For data migrations:
  - Is it guarded / idempotent?
  - Will it lock large tables? Is batching needed?
- Does `downgrade()` match the project policy (required / best-effort / explicitly unsupported)?
- Is there a minimal verification query for each critical change?
- If SQLite:
  - Are batch operations used for ALTER patterns that SQLite can’t do directly?
