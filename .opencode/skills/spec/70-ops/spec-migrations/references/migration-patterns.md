# Database Migration Patterns

## Migration File Structure

### Naming Convention
```
YYYYMMDDHHMMSS_description.sql
# or
001_create_users_table.sql
002_add_email_index.sql
```

### Standard Template
```sql
-- Migration: 20240115120000_create_users
-- Description: Create users table with basic auth fields
-- Author: dev@example.com

-- ============ UP ============

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);

-- ============ DOWN ============

DROP INDEX IF EXISTS idx_users_email;
DROP TABLE IF EXISTS users;
```

## Common Patterns

### 1. Add Column (Safe)
```sql
-- UP: Add nullable column first
ALTER TABLE users ADD COLUMN phone VARCHAR(20);

-- Later migration: Add constraint after backfill
ALTER TABLE users ALTER COLUMN phone SET NOT NULL;
```

**Anti-pattern**: Adding NOT NULL without default blocks writes
```sql
-- ❌ DANGEROUS: Locks table, fails if data exists
ALTER TABLE users ADD COLUMN phone VARCHAR(20) NOT NULL;

-- ✅ SAFE: Nullable first, then backfill, then constraint
ALTER TABLE users ADD COLUMN phone VARCHAR(20);
UPDATE users SET phone = 'unknown' WHERE phone IS NULL;
ALTER TABLE users ALTER COLUMN phone SET NOT NULL;
```

### 2. Rename Column (Zero-Downtime)
```sql
-- Step 1: Add new column
ALTER TABLE users ADD COLUMN full_name VARCHAR(255);

-- Step 2: Backfill data
UPDATE users SET full_name = name;

-- Step 3: Deploy app reading both columns
-- Step 4: Migrate app to write both, read new
-- Step 5: Deploy app reading/writing only new

-- Step 6: Drop old column (separate migration)
ALTER TABLE users DROP COLUMN name;
```

### 3. Add Index (Concurrent)
```sql
-- ❌ DANGEROUS: Locks table for writes
CREATE INDEX idx_users_email ON users(email);

-- ✅ SAFE: Concurrent (PostgreSQL)
CREATE INDEX CONCURRENTLY idx_users_email ON users(email);
```

### 4. Change Column Type
```sql
-- ❌ DANGEROUS: May lock table, lose data
ALTER TABLE orders ALTER COLUMN amount TYPE DECIMAL(10,2);

-- ✅ SAFE: New column approach
ALTER TABLE orders ADD COLUMN amount_decimal DECIMAL(10,2);
UPDATE orders SET amount_decimal = amount::DECIMAL(10,2);
-- Deploy app to use new column
ALTER TABLE orders DROP COLUMN amount;
ALTER TABLE orders RENAME COLUMN amount_decimal TO amount;
```

### 5. Add Foreign Key
```sql
-- ❌ DANGEROUS: Validates all rows, locks table
ALTER TABLE orders 
ADD CONSTRAINT fk_orders_user 
FOREIGN KEY (user_id) REFERENCES users(id);

-- ✅ SAFE: Add as NOT VALID, validate later
ALTER TABLE orders 
ADD CONSTRAINT fk_orders_user 
FOREIGN KEY (user_id) REFERENCES users(id) NOT VALID;

-- Later, in separate migration (doesn't lock)
ALTER TABLE orders VALIDATE CONSTRAINT fk_orders_user;
```

## ORM Migration Tools

### Alembic (Python/SQLAlchemy)
```bash
# Generate migration
alembic revision --autogenerate -m "add users table"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

```python
# alembic/versions/001_add_users.py
def upgrade():
    op.create_table(
        'users',
        sa.Column('id', sa.UUID(), primary_key=True),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
    )

def downgrade():
    op.drop_table('users')
```

### Prisma (Node.js)
```bash
# Generate migration
npx prisma migrate dev --name add_users

# Apply in production
npx prisma migrate deploy
```

### Goose (Go)
```bash
# Create migration
goose create add_users sql

# Apply
goose up

# Rollback
goose down
```

### Flyway (Java)
```bash
# Apply migrations
flyway migrate

# Info
flyway info

# Repair (after failed migration)
flyway repair
```

## Pre-Migration Checklist

```
- [ ] Backup database (or have restore plan)
- [ ] Test migration on staging
- [ ] Measure migration duration on staging
- [ ] Check for table locks
- [ ] Schedule maintenance window if needed
- [ ] Prepare rollback script
- [ ] Notify team/stakeholders
- [ ] Monitor during migration
```

## Rollback Strategies

### 1. Down Migration
```sql
-- If up creates table, down drops it
-- Risk: Data loss
```

### 2. Forward Fix
```sql
-- Don't rollback, fix forward
-- Create new migration that fixes the issue
-- Safer for production
```

### 3. Reversible Operations Only
Keep migrations reversible:
- Add column ↔ Drop column
- Create index ↔ Drop index
- Rename ↔ Rename back

Non-reversible (avoid or plan carefully):
- Drop table (data loss)
- Change column type (data conversion)
- Delete rows (data loss)

## Large Table Migrations

### Batched Updates
```sql
-- Don't update all at once
-- ❌ UPDATE users SET status = 'active';

-- ✅ Batch by ID ranges
DO $$
DECLARE
    batch_size INT := 1000;
    last_id UUID := '00000000-0000-0000-0000-000000000000';
BEGIN
    LOOP
        UPDATE users 
        SET status = 'active' 
        WHERE id > last_id 
        AND id <= (
            SELECT id FROM users 
            WHERE id > last_id 
            ORDER BY id LIMIT 1 OFFSET batch_size
        );
        
        IF NOT FOUND THEN EXIT; END IF;
        
        SELECT MAX(id) INTO last_id FROM users WHERE status = 'active';
        COMMIT;
        PERFORM pg_sleep(0.1); -- Pause between batches
    END LOOP;
END $$;
```

### Parallel Processing (Application Layer)
```python
# Process in chunks from application
def migrate_users_in_batches(batch_size=1000):
    last_id = None
    while True:
        query = User.query.order_by(User.id)
        if last_id:
            query = query.filter(User.id > last_id)
        batch = query.limit(batch_size).all()
        
        if not batch:
            break
        
        for user in batch:
            user.status = 'active'
        
        db.session.commit()
        last_id = batch[-1].id
        time.sleep(0.1)
```

## Testing Migrations

```python
# pytest fixture for migration testing
@pytest.fixture
def db_with_migrations(database):
    # Apply all migrations
    alembic.command.upgrade(config, "head")
    yield database
    # Rollback all migrations
    alembic.command.downgrade(config, "base")

def test_migration_up_down():
    # Test upgrade
    alembic.command.upgrade(config, "+1")
    assert table_exists('users')
    
    # Test downgrade
    alembic.command.downgrade(config, "-1")
    assert not table_exists('users')

def test_migration_with_data():
    # Insert test data
    db.execute("INSERT INTO users (email) VALUES ('test@example.com')")
    
    # Apply migration that modifies data
    alembic.command.upgrade(config, "+1")
    
    # Verify data migrated correctly
    result = db.execute("SELECT full_name FROM users").fetchone()
    assert result[0] == 'test@example.com'
```
