# Regression Patterns & Detection

## Common Regression Categories

### 1. Silent Behavior Changes

**Pattern**: Code change modifies behavior without test failure.

**Detection**:
```python
# Before: returns empty list
def get_items(filter=None):
    if not filter:
        return []
    return db.query(filter)

# After: returns all items (REGRESSION!)
def get_items(filter=None):
    if not filter:
        return db.query_all()  # Silent behavior change
    return db.query(filter)
```

**Hunt Strategy**:
- Review all conditional branches modified
- Check default parameter behaviors
- Trace callers to verify expectations

### 2. Error Handling Regressions

**Pattern**: Error paths broken or swallowed.

**Detection**:
```python
# Before: proper error propagation
try:
    result = risky_operation()
except OperationError as e:
    logger.error(f"Operation failed: {e}")
    raise

# After: error swallowed (REGRESSION!)
try:
    result = risky_operation()
except Exception:
    result = None  # Caller never knows about failure
```

**Hunt Strategy**:
- Review all try/except blocks touched
- Verify error logging still works
- Check that callers handle None/error cases

### 3. Performance Regressions

**Pattern**: N+1 queries, missing indexes, unbounded loops.

**Detection**:
```python
# Before: single query
users = db.query(User).filter(active=True).all()

# After: N+1 queries (REGRESSION!)
users = db.query(User).filter(active=True).all()
for user in users:
    user.permissions = db.query(Permission).filter(user_id=user.id).all()
```

**Hunt Strategy**:
- Check query count before/after
- Review loops that touch database/network
- Verify pagination on list endpoints

### 4. Concurrency Regressions

**Pattern**: Race conditions, deadlocks, lost updates.

**Detection**:
```python
# Before: atomic operation
counter.increment()

# After: race condition (REGRESSION!)
value = counter.get()
counter.set(value + 1)  # Lost update if concurrent
```

**Hunt Strategy**:
- Review shared state modifications
- Check lock acquisition order
- Verify transaction boundaries

### 5. Configuration Regressions

**Pattern**: Config changes break environments.

**Detection**:
```yaml
# Before: optional with default
database:
  pool_size: ${DB_POOL_SIZE:-10}

# After: required, breaks if not set (REGRESSION!)
database:
  pool_size: ${DB_POOL_SIZE}  # Fails if env var missing
```

**Hunt Strategy**:
- Compare config changes across environments
- Verify defaults still work
- Check CI/CD environment variables

## Regression Hunt Protocol

### Step 1: Identify Impact Surface
```
1. List all files modified
2. Find all callers/importers of modified code
3. Identify integration points (API, DB, events)
4. Map to user-facing features
```

### Step 2: Trace Critical Paths
```
For each feature in impact surface:
1. Start from entry point (API, UI, job)
2. Follow execution through modified code
3. Verify exit points (response, side effects)
4. Check error paths explicitly
```

### Step 3: Verify Adjacent Workflows
```
Pick 2-3 workflows that:
- Share code with modified paths
- Use same database tables
- Depend on same services
- Handle similar data types
```

### Step 4: Document Findings
```markdown
## Regression Hunt: [Feature/PR Name]

### Verified Paths
- [ ] Path A: OK
- [ ] Path B: OK
- [ ] Path C: Found issue → fixed in commit X

### Adjacent Workflows Tested
- Workflow X: No regression
- Workflow Y: No regression

### Deferred Checks
- Performance testing (needs load test)
- Browser compatibility (needs manual QA)
```

## Red Flags During Audit

| Red Flag | Action |
|----------|--------|
| Test deleted without replacement | Investigate why, restore or replace |
| Catch-all exception handler added | Verify errors aren't swallowed |
| Default parameter changed | Verify all callers expect new default |
| Public API signature changed | Check for breaking changes |
| Database migration without backfill | Verify existing data handled |
| Config key renamed | Check all environments updated |
| Logging removed | Verify observability maintained |
