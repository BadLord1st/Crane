# Drift Detection & Prevention

## What is Drift?

Drift occurs when the actual state diverges from the expected state:

- **Spec Drift**: Code doesn't match specification
- **Test Drift**: Tests don't cover current behavior
- **Config Drift**: Running config differs from documented
- **Schema Drift**: Database differs from migrations

## Types of Drift

### 1. Specification Drift

**Symptoms**:
- Code does things not in the spec
- Spec describes features that don't exist
- Edge cases in code not documented

**Detection**:
```bash
# Compare spec acceptance criteria vs test coverage
grep -h "should\|must\|will" specs/*.md | wc -l  # Expected behaviors
grep -h "def test_" tests/*.py | wc -l           # Actual tests

# Find undocumented endpoints
# Compare OpenAPI spec vs actual routes
diff <(yq '.paths | keys' openapi.yaml | sort) \
     <(grep -oh '@app\.[a-z]*("[^"]*"' src/*.py | sed 's/.*"\([^"]*\)"/\1/' | sort)
```

**Prevention**:
```markdown
- Write spec before code
- Review spec changes in PRs
- Link tests to spec sections
- Periodic spec-code audits
```

### 2. Test Drift

**Symptoms**:
- Tests pass but bugs exist
- New features have no tests
- Refactored code, tests not updated

**Detection**:
```bash
# Code coverage delta
coverage run -m pytest
coverage report --show-missing

# Find recent changes without test changes
git log --oneline --name-only -- src/ | head -20
git log --oneline --name-only -- tests/ | head -20

# Mutation testing (find tests that don't really test)
mutmut run --paths-to-mutate src/
```

**Prevention**:
```markdown
- Require tests in PR checklist
- Minimum coverage thresholds
- Run mutation testing monthly
- TDD (test-first) approach
```

### 3. Configuration Drift

**Symptoms**:
- Works in production, fails locally
- "It works on my machine"
- Undocumented environment variables

**Detection**:
```bash
# Compare environment variables across environments
diff <(sort .env.example) <(sort .env.production)

# Find env var usage in code
grep -roh 'os\.environ\[['"'"'"]([A-Z_]*)['"'"'"]\]' src/ | \
  sort -u > code_env_vars.txt

# Compare with documented
diff code_env_vars.txt docs/configuration.md
```

**Prevention**:
```markdown
- Infrastructure as Code (Terraform, Pulumi)
- .env.example always updated
- Config validation on startup
- Document all config in one place
```

### 4. Schema Drift

**Symptoms**:
- Migration files don't match actual DB
- ORM models don't match schema
- Missing indexes, wrong constraints

**Detection**:
```bash
# Compare migrations vs actual schema (PostgreSQL)
pg_dump --schema-only production_db > actual_schema.sql
alembic upgrade head  # Apply all migrations
pg_dump --schema-only test_db > expected_schema.sql
diff actual_schema.sql expected_schema.sql

# SQLAlchemy model vs database
python -c "
from app import models, db
from sqlalchemy import inspect
inspector = inspect(db.engine)
for table in inspector.get_table_names():
    print(f'{table}: {inspector.get_columns(table)}')
"
```

**Prevention**:
```markdown
- Never modify production DB directly
- Verify migrations in CI (up and down)
- Schema snapshots in version control
- Regular drift checks
```

## Drift Detection Automation

### Pre-commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit

# Check for spec drift
if git diff --cached --name-only | grep -q "^src/"; then
    if ! git diff --cached --name-only | grep -q "^specs/\|^tests/"; then
        echo "WARNING: Code changes without spec/test updates"
        read -p "Continue? [y/N] " -n 1 -r
        [[ $REPLY =~ ^[Yy]$ ]] || exit 1
    fi
fi
```

### CI Check
```yaml
name: Drift Detection

on: [push, pull_request]

jobs:
  check-drift:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Check spec-test alignment
        run: |
          ./scripts/check_traceability.sh
          
      - name: Check schema drift
        run: |
          docker-compose up -d db
          alembic upgrade head
          ./scripts/compare_schema.sh
          
      - name: Check config drift
        run: |
          ./scripts/validate_config.sh
```

### Scheduled Audit
```yaml
name: Weekly Drift Audit

on:
  schedule:
    - cron: '0 9 * * 1'  # Monday 9am

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Generate drift report
        run: ./scripts/drift_report.sh > drift_report.md
        
      - name: Create issue if drift found
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: 'Weekly Drift Audit: Issues Found',
              body: require('fs').readFileSync('drift_report.md', 'utf8'),
              labels: ['drift', 'technical-debt']
            })
```

## Drift Remediation

### Quick Fixes
```bash
# Sync ORM models to database (DANGEROUS)
# Only in development!
alembic revision --autogenerate -m "sync_schema"
alembic upgrade head

# Regenerate OpenAPI from code
python -c "from app import app; import json; print(json.dumps(app.openapi()))" > openapi.json

# Update .env.example from code
grep -roh "os\.environ\[.*\]" src/ | \
  sed 's/os\.environ\[['"'"'"]\([^'"'"'"]*\)['"'"'"]\]/\1=/' | \
  sort -u > .env.example
```

### Remediation Process
```markdown
## Drift Remediation Ticket

**Type**: [Spec | Test | Config | Schema]
**Severity**: [High | Medium | Low]
**Discovered**: 2024-01-15

### Current State
[What actually exists]

### Expected State
[What should exist per spec/docs]

### Remediation Plan
1. [ ] Update [spec/code/tests/config]
2. [ ] Add prevention measure
3. [ ] Update documentation
4. [ ] Verify in all environments

### Root Cause
[Why did this drift occur?]

### Prevention
[What process/automation prevents recurrence?]
```

## Drift Metrics Dashboard

```markdown
## Drift Health Dashboard

| Area | Status | Last Check | Trend |
|------|--------|------------|-------|
| Spec Coverage | 92% | 2024-01-15 | ↑ |
| Test Coverage | 85% | 2024-01-15 | → |
| Config Alignment | 100% | 2024-01-15 | ↑ |
| Schema Alignment | 100% | 2024-01-15 | → |

### Open Drift Issues
- DRIFT-001: Missing tests for payment retry (Medium)
- DRIFT-002: Undocumented FEATURE_X env var (Low)

### Recent Drift Fixes
- DRIFT-000: Schema migration gap (Fixed 2024-01-10)
```
