# Error Knowledge Base Structure

## Entry Format

Each KB entry follows this structure:

```markdown
## [ERROR_CODE] Short Description

**Category**: [runtime|build|test|config|integration|security]
**Severity**: [critical|high|medium|low]
**First Seen**: YYYY-MM-DD
**Last Seen**: YYYY-MM-DD
**Occurrences**: N

### Symptoms
What the developer sees when this error occurs.

### Root Cause
Why this error happens (the underlying issue).

### Solution
Step-by-step fix.

### Prevention
How to prevent this in the future.

### Related
Links to related errors, docs, or commits.
```

## Example Entries

### [NPM-001] Module Not Found After Clean Install

**Category**: build
**Severity**: medium
**First Seen**: 2024-01-15
**Last Seen**: 2024-03-20
**Occurrences**: 5

#### Symptoms
```
Error: Cannot find module '@company/internal-lib'
```
After `npm ci` or on CI, but works locally.

#### Root Cause
Private package requires authentication. Local environment has cached credentials, CI does not.

#### Solution
1. Create `.npmrc` with registry config:
   ```
   @company:registry=https://npm.company.com/
   //npm.company.com/:_authToken=${NPM_TOKEN}
   ```
2. Add `NPM_TOKEN` to CI secrets

#### Prevention
- Document private registry setup in CONTRIBUTING.md
- Add CI check that validates `.npmrc` presence
- Use `npm whoami` step before install

#### Related
- [NPM-003] Auth token expired
- docs/setup.md#private-packages

---

### [PG-001] Connection Refused to PostgreSQL

**Category**: integration
**Severity**: high
**First Seen**: 2024-02-01
**Last Seen**: 2024-02-28
**Occurrences**: 12

#### Symptoms
```
psycopg2.OperationalError: could not connect to server: Connection refused
    Is the server running on host "localhost" and accepting TCP/IP connections on port 5432?
```

#### Root Cause
Multiple possible causes:
1. PostgreSQL not running
2. Wrong port in connection string
3. Docker container not ready
4. Firewall blocking connection

#### Solution
```bash
# 1. Check if PostgreSQL is running
pg_isready -h localhost -p 5432

# 2. If using Docker, verify container
docker ps | grep postgres
docker logs postgres-container

# 3. Wait for readiness in scripts
until pg_isready -h localhost -p 5432; do
  sleep 1
done
```

#### Prevention
- Add health checks to docker-compose
- Use connection retry with backoff
- Add startup check to test setup

#### Related
- docker-compose.yml health check config
- [PG-002] Authentication failed

---

### [TEST-001] Flaky Test Due to Timing

**Category**: test
**Severity**: medium
**First Seen**: 2024-01-10
**Last Seen**: 2024-04-01
**Occurrences**: 23

#### Symptoms
Test passes sometimes, fails randomly. Often in CI but not locally.

```
AssertionError: Expected 'completed' but got 'pending'
```

#### Root Cause
Test checks state before async operation completes. Race condition between test assertion and background process.

#### Solution
```python
# Bad: immediate assertion
result = start_async_job()
assert result.status == 'completed'

# Good: wait with timeout
result = start_async_job()
assert wait_for_status(result, 'completed', timeout=10)

# Or use polling
from tenacity import retry, stop_after_delay
@retry(stop=stop_after_delay(10))
def check_completed():
    assert get_status() == 'completed'
```

#### Prevention
- Never assert on async state immediately
- Use explicit waits with timeouts
- Add retry decorators for flaky assertions
- Run tests multiple times in CI

#### Related
- [TEST-002] Flaky test due to date/time
- Testing best practices doc

## Categories

### Runtime Errors
Errors that occur during application execution:
- Null/undefined access
- Type errors
- Resource exhaustion (memory, connections)
- Timeout errors

### Build Errors
Errors during build/compile phase:
- Missing dependencies
- Type check failures
- Syntax errors
- Configuration errors

### Test Errors
Errors specific to test execution:
- Flaky tests
- Fixture problems
- Mock configuration
- Assertion failures

### Config Errors
Configuration-related issues:
- Missing environment variables
- Invalid config format
- Wrong environment (dev/prod mismatch)
- Secret management

### Integration Errors
External service issues:
- Database connection
- API failures
- Authentication errors
- Network timeouts

### Security Errors
Security-related issues:
- Vulnerability discovered
- Secret exposed
- Authentication bypass
- Input validation failure

## KB Maintenance

### Adding New Entry
1. Check if similar entry exists
2. Use template above
3. Add to appropriate category file
4. Cross-reference related entries

### Updating Entry
1. Increment occurrence count
2. Update "Last Seen" date
3. Add new symptoms/solutions if discovered
4. Link to fix commit

### Archiving Entry
If error is permanently fixed (code removed, dependency removed):
1. Move to `archive/` folder
2. Add "Archived" badge
3. Note why it's no longer relevant
