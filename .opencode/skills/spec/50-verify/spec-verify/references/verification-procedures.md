# Verification Procedures

## Verification Levels

| Level | What | When | Who |
|-------|------|------|-----|
| Unit | Individual functions/methods | Every commit | Developer |
| Integration | Component interactions | Every PR | CI |
| System | Full application | Pre-deploy | CI/QA |
| Acceptance | User requirements | Pre-release | Product/QA |

## Pre-Commit Verification

### Quick Checks (< 30 seconds)
```bash
# Format check
black --check src/
prettier --check "src/**/*.ts"

# Lint
ruff check src/
eslint src/

# Type check
mypy src/
tsc --noEmit

# Fast unit tests
pytest tests/unit -x -q
```

### Git Hook Setup
```bash
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: format
        name: Format
        entry: black
        language: python
        types: [python]
      
      - id: lint
        name: Lint
        entry: ruff check
        language: python
        types: [python]
      
      - id: typecheck
        name: Type Check
        entry: mypy
        language: python
        types: [python]
```

## PR Verification

### CI Pipeline
```yaml
name: PR Verification

on: pull_request

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      # Quality gates
      - name: Format
        run: black --check src/
      
      - name: Lint
        run: ruff check src/
      
      - name: Type check
        run: mypy src/
      
      # Tests
      - name: Unit tests
        run: pytest tests/unit --cov
      
      - name: Integration tests
        run: pytest tests/integration
      
      # Security
      - name: Dependency audit
        run: pip-audit
      
      - name: SAST
        run: bandit -r src/
```

### Verification Checklist
```markdown
## PR Verification Checklist

### Automated (CI)
- [ ] All tests pass
- [ ] Coverage >= 80%
- [ ] No lint errors
- [ ] No type errors
- [ ] No security vulnerabilities

### Manual Review
- [ ] Code matches spec
- [ ] Edge cases handled
- [ ] Error handling appropriate
- [ ] No debug code
- [ ] Documentation updated
```

## Test Execution Verification

### Test Run Report
```bash
# Generate detailed report
pytest --tb=short --junitxml=report.xml --cov --cov-report=html

# Verify critical tests ran
pytest --collect-only | grep -c "test session starts"
```

### Coverage Verification
```python
# Enforce minimum coverage
# pytest.ini or pyproject.toml
[tool.pytest.ini_options]
addopts = "--cov --cov-fail-under=80"

# Per-module coverage
[tool.coverage.run]
branch = true

[tool.coverage.report]
fail_under = 80
exclude_lines = [
    "pragma: no cover",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
]
```

### Test Quality Verification
```bash
# Mutation testing - verify tests actually test
mutmut run --paths-to-mutate src/
mutmut results

# Good: Mutation Score > 70%
# All mutations detected by tests
```

## Build Verification

### Build Smoke Test
```bash
#!/bin/bash
# verify_build.sh

set -e

echo "Building application..."
docker build -t app:test .

echo "Starting container..."
docker run -d --name app-test -p 8080:8080 app:test

echo "Waiting for health..."
timeout 30 bash -c 'until curl -s http://localhost:8080/health; do sleep 1; done'

echo "Running smoke tests..."
curl -f http://localhost:8080/health
curl -f http://localhost:8080/api/version

echo "Cleanup..."
docker stop app-test
docker rm app-test

echo "✓ Build verification passed"
```

### Artifact Verification
```bash
# Verify Docker image
docker run --rm app:test --version
docker run --rm app:test python -c "import app; print(app.__version__)"

# Verify package
pip install dist/*.whl
python -c "import mypackage; print(mypackage.__version__)"
```

## Environment Verification

### Startup Verification
```python
def verify_environment():
    """Run at application startup."""
    errors = []
    
    # Required environment variables
    required = ['DATABASE_URL', 'SECRET_KEY', 'API_KEY']
    for var in required:
        if not os.environ.get(var):
            errors.append(f"Missing required env var: {var}")
    
    # Database connection
    try:
        db.execute("SELECT 1")
    except Exception as e:
        errors.append(f"Database connection failed: {e}")
    
    # External services
    try:
        requests.get(PAYMENT_SERVICE_URL + "/health", timeout=5)
    except Exception as e:
        errors.append(f"Payment service unreachable: {e}")
    
    if errors:
        raise EnvironmentError("\n".join(errors))
    
    logger.info("Environment verification passed")
```

### Health Check Endpoint
```python
@app.get("/health")
def health_check():
    checks = {
        "database": check_database(),
        "cache": check_redis(),
        "external_api": check_external_api(),
    }
    
    all_healthy = all(c["status"] == "healthy" for c in checks.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "checks": checks,
        "version": app.version,
        "timestamp": datetime.utcnow().isoformat()
    }

def check_database():
    try:
        db.execute("SELECT 1")
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

## Deployment Verification

### Post-Deploy Smoke Tests
```bash
#!/bin/bash
# post_deploy_verify.sh

BASE_URL=${1:-https://api.example.com}

echo "Verifying deployment to $BASE_URL"

# Health check
curl -f "$BASE_URL/health" || exit 1

# Version check
DEPLOYED_VERSION=$(curl -s "$BASE_URL/api/version" | jq -r '.version')
EXPECTED_VERSION=$(cat VERSION)
if [ "$DEPLOYED_VERSION" != "$EXPECTED_VERSION" ]; then
    echo "Version mismatch: expected $EXPECTED_VERSION, got $DEPLOYED_VERSION"
    exit 1
fi

# Critical endpoint checks
curl -f "$BASE_URL/api/users" -H "Authorization: Bearer $TEST_TOKEN" || exit 1

# Performance check
RESPONSE_TIME=$(curl -o /dev/null -s -w '%{time_total}' "$BASE_URL/health")
if (( $(echo "$RESPONSE_TIME > 1.0" | bc -l) )); then
    echo "Health check too slow: ${RESPONSE_TIME}s"
    exit 1
fi

echo "✓ Deployment verification passed"
```

### Canary Verification
```yaml
# Argo Rollouts analysis
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: success-rate
spec:
  metrics:
    - name: success-rate
      interval: 1m
      successCondition: result[0] >= 0.99
      provider:
        prometheus:
          query: |
            sum(rate(http_requests_total{status!~"5..",version="canary"}[5m]))
            /
            sum(rate(http_requests_total{version="canary"}[5m]))
```

## Verification Evidence

### Audit Trail
```markdown
## Verification Record

**Deployment**: v1.2.3
**Environment**: production
**Date**: 2024-01-15 14:30 UTC
**Verified By**: CI (run #4567)

### Pre-Deploy Checks
- [x] All tests passed (4567 passed, 0 failed)
- [x] Coverage: 87% (threshold: 80%)
- [x] Security scan: 0 high, 2 medium
- [x] Dependency audit: clean

### Post-Deploy Checks
- [x] Health check: healthy
- [x] Version verified: 1.2.3
- [x] Smoke tests: 5/5 passed
- [x] Latency p95: 180ms (budget: 200ms)

### Evidence Links
- CI Run: https://ci.example.com/runs/4567
- Coverage Report: https://coverage.example.com/1.2.3
- Deployment Logs: https://logs.example.com/deploy/1.2.3
```
