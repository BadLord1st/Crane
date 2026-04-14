# Common Error Patterns

## Pattern Library

### 1. The Silent Failure

**Description**: Operation fails but no error is raised or logged.

**Example**:
```python
def save_user(user):
    try:
        db.save(user)
    except Exception:
        pass  # Silent failure!
```

**Detection**:
- Look for empty `except` blocks
- Check for missing error logging
- Review functions with no return value verification

**Fix Pattern**:
```python
def save_user(user):
    try:
        db.save(user)
        return True
    except DatabaseError as e:
        logger.error(f"Failed to save user {user.id}: {e}")
        raise UserSaveError(f"Could not save user: {e}") from e
```

---

### 2. The Missing Null Check

**Description**: Code assumes value exists when it might be null/undefined.

**Example**:
```javascript
function getUserEmail(userId) {
    const user = getUser(userId);
    return user.email;  // Crash if user is null!
}
```

**Detection**:
- Review all `.property` access after function calls
- Check database query results before access
- Audit optional parameters

**Fix Pattern**:
```javascript
function getUserEmail(userId) {
    const user = getUser(userId);
    if (!user) {
        throw new NotFoundError(`User ${userId} not found`);
    }
    return user.email;
}
```

---

### 3. The Race Condition

**Description**: Behavior depends on timing of concurrent operations.

**Example**:
```python
if not cache.exists(key):
    value = expensive_computation()
    cache.set(key, value)  # Another thread might set between check and set
```

**Detection**:
- Look for check-then-act patterns
- Review shared state modifications
- Audit background job interactions

**Fix Pattern**:
```python
# Use atomic operation
value = cache.get_or_set(key, lambda: expensive_computation())

# Or use locks
with cache.lock(key):
    if not cache.exists(key):
        value = expensive_computation()
        cache.set(key, value)
```

---

### 4. The Resource Leak

**Description**: Resources (connections, files, handles) not properly released.

**Example**:
```python
def read_config():
    f = open('config.json')
    data = json.load(f)
    return data  # File never closed!
```

**Detection**:
- Look for `open()` without context manager
- Check database connection handling
- Review HTTP client usage

**Fix Pattern**:
```python
def read_config():
    with open('config.json') as f:
        return json.load(f)
```

---

### 5. The Timeout Trap

**Description**: External call without timeout leads to infinite hang.

**Example**:
```python
response = requests.get(external_api_url)  # No timeout!
```

**Detection**:
- Audit all HTTP client calls
- Check database query configurations
- Review message queue consumers

**Fix Pattern**:
```python
response = requests.get(
    external_api_url,
    timeout=(3.05, 27)  # (connect timeout, read timeout)
)
```

---

### 6. The Unbounded Query

**Description**: Query returns unlimited results, causing memory issues.

**Example**:
```python
def get_all_users():
    return db.query(User).all()  # Millions of rows!
```

**Detection**:
- Look for `.all()` or `SELECT *` without LIMIT
- Check list endpoints without pagination
- Review batch processing loops

**Fix Pattern**:
```python
def get_users(page: int = 1, per_page: int = 100):
    return db.query(User).limit(per_page).offset((page - 1) * per_page).all()
```

---

### 7. The Hardcoded Secret

**Description**: Credentials embedded in source code.

**Example**:
```python
API_KEY = "sk-1234567890abcdef"  # Committed to git!
```

**Detection**:
- Run secret scanners (gitleaks, trufflehog)
- Search for `password`, `secret`, `key`, `token`
- Check config files for hardcoded values

**Fix Pattern**:
```python
import os
API_KEY = os.environ["API_KEY"]

# Or with defaults for development
API_KEY = os.environ.get("API_KEY", "dev-only-key")
```

---

### 8. The Missing Validation

**Description**: User input used without validation.

**Example**:
```python
def delete_file(filename):
    os.remove(f"/uploads/{filename}")  # Path traversal!
```

**Detection**:
- Trace user input through code
- Check file path construction
- Review SQL query building

**Fix Pattern**:
```python
def delete_file(filename):
    # Validate: no path traversal
    if '..' in filename or '/' in filename:
        raise ValueError("Invalid filename")
    
    safe_path = os.path.join("/uploads", os.path.basename(filename))
    if not safe_path.startswith("/uploads/"):
        raise ValueError("Invalid path")
    
    os.remove(safe_path)
```

---

### 9. The Floating Point Trap

**Description**: Using floating point for money/precise calculations.

**Example**:
```python
total = 0.1 + 0.2  # = 0.30000000000000004
```

**Detection**:
- Look for `float` in financial code
- Check price/amount calculations
- Review percentage calculations

**Fix Pattern**:
```python
from decimal import Decimal

total = Decimal("0.1") + Decimal("0.2")  # = 0.3

# Or use integer cents
total_cents = 10 + 20  # = 30 cents
```

---

### 10. The Missing Transaction

**Description**: Multi-step database operations without transaction.

**Example**:
```python
def transfer_money(from_id, to_id, amount):
    debit_account(from_id, amount)
    # If this fails, money is lost!
    credit_account(to_id, amount)
```

**Detection**:
- Look for multiple DB writes in one function
- Check for error handling between writes
- Review batch update operations

**Fix Pattern**:
```python
def transfer_money(from_id, to_id, amount):
    with db.transaction():
        debit_account(from_id, amount)
        credit_account(to_id, amount)
        # Both succeed or both fail
```

## Quick Reference Table

| Pattern | Symptom | Detection | Risk |
|---------|---------|-----------|------|
| Silent Failure | Data loss, inconsistency | Empty except blocks | High |
| Missing Null | NullPointerException | Property access after call | Medium |
| Race Condition | Intermittent failures | Check-then-act | High |
| Resource Leak | Memory/connection exhaustion | Missing context managers | High |
| Timeout Trap | Hanging requests | HTTP calls without timeout | Critical |
| Unbounded Query | OOM, slow responses | .all() without limit | High |
| Hardcoded Secret | Security breach | Secret scanner | Critical |
| Missing Validation | Injection attacks | User input trace | Critical |
| Floating Point | Wrong calculations | Float in money code | Medium |
| Missing Transaction | Data inconsistency | Multiple DB writes | High |
