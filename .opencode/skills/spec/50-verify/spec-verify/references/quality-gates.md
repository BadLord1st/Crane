# Quality Gates Reference

## What are Quality Gates?

Quality gates are checkpoints that code must pass before proceeding to the next stage. They enforce minimum quality standards automatically.

## Gate Stages

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Pre-Commit │───▶│     PR      │───▶│   Pre-Deploy│───▶│ Post-Deploy │
│    Gate     │    │    Gate     │    │    Gate     │    │    Gate     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                  │                  │                  │
   Format             Tests             Security           Health
   Lint              Coverage          Approval          Metrics
   Types             Review            Sign-off          Alerts
```

## Pre-Commit Gate

### Criteria
| Check | Threshold | Blocking |
|-------|-----------|----------|
| Format | 100% | Yes |
| Lint errors | 0 | Yes |
| Type errors | 0 | Yes |
| Fast tests | 100% pass | Optional |

### Configuration (example)
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.1.0
    hooks:
      - id: black
        
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.14
    hooks:
      - id: ruff
        args: [--fix]
        
  - repo: local
    hooks:
      - id: mypy
        name: mypy
        entry: mypy
        language: system
        types: [python]
        args: [--strict]
```

## PR Gate

### Criteria
| Check | Threshold | Blocking |
|-------|-----------|----------|
| All tests pass | 100% | Yes |
| Code coverage | >= 80% | Yes |
| Coverage delta | >= 0% | Yes (no decrease) |
| Security scan | 0 high/critical | Yes |
| Dependency audit | 0 vulnerabilities | Yes (high+) |
| Peer review | 1+ approvals | Yes |
| PR size | < 400 lines | Warning |

### GitHub Actions Implementation
```yaml
name: PR Gate

on: pull_request

jobs:
  quality-gate:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt
        
      # Quality checks
      - name: Format check
        run: black --check src/
        
      - name: Lint
        run: ruff check src/
        
      - name: Type check
        run: mypy src/
        
      # Tests with coverage
      - name: Run tests
        run: |
          pytest --cov=src --cov-report=xml --cov-fail-under=80
          
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          fail_ci_if_error: true
          
      # Security
      - name: Dependency audit
        run: pip-audit --strict
        
      - name: SAST
        run: bandit -r src/ -f json -o bandit-report.json
        
      # Size check (warning only)
      - name: Check PR size
        run: |
          LINES=$(git diff --stat origin/main | tail -1 | grep -o '[0-9]* insertion' | grep -o '[0-9]*')
          if [ "$LINES" -gt 400 ]; then
            echo "::warning::Large PR: $LINES lines. Consider splitting."
          fi
```

### Branch Protection Rules
```json
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "quality-gate",
      "security-scan"
    ]
  },
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false
  },
  "enforce_admins": true,
  "restrictions": null
}
```

## Pre-Deploy Gate

### Criteria
| Check | Threshold | Blocking |
|-------|-----------|----------|
| All CI checks | Pass | Yes |
| E2E tests | 100% pass | Yes |
| Performance baseline | Within 10% | Yes |
| Security sign-off | Approved | Yes (for major) |
| Changelog | Updated | Yes |
| Version bumped | Yes | Yes |

### Implementation
```yaml
name: Pre-Deploy Gate

on:
  push:
    branches: [main]

jobs:
  pre-deploy:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      # Build
      - name: Build
        run: docker build -t app:${{ github.sha }} .
        
      # E2E tests
      - name: E2E tests
        run: |
          docker-compose -f docker-compose.test.yml up -d
          npm run test:e2e
          
      # Performance regression
      - name: Performance check
        run: |
          k6 run load-tests/baseline.js
          # Fails if p95 latency > baseline + 10%
          
      # Security
      - name: Container scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: app:${{ github.sha }}
          exit-code: 1
          severity: HIGH,CRITICAL
          
      # Verification
      - name: Verify changelog
        run: |
          if ! grep -q "## \[${{ github.ref_name }}\]" CHANGELOG.md; then
            echo "Changelog not updated for this version"
            exit 1
          fi
```

## Post-Deploy Gate

### Criteria
| Check | Threshold | Blocking Rollback |
|-------|-----------|-------------------|
| Health check | Healthy | Yes |
| Error rate | < 1% | Yes |
| Latency p95 | < baseline + 20% | Yes |
| Smoke tests | Pass | Yes |
| Critical alerts | 0 | Yes |

### Automated Verification
```yaml
name: Post-Deploy Gate

on:
  workflow_dispatch:
    inputs:
      environment:
        required: true
        type: choice
        options: [staging, production]

jobs:
  verify:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Health check
        run: |
          for i in {1..30}; do
            curl -sf ${{ inputs.environment }}.example.com/health && exit 0
            sleep 2
          done
          exit 1
          
      - name: Smoke tests
        run: npm run smoke:${{ inputs.environment }}
        
      - name: Check error rate
        run: |
          ERROR_RATE=$(curl -s "$PROMETHEUS/api/v1/query?query=..." | jq '.data.result[0].value[1]')
          if (( $(echo "$ERROR_RATE > 0.01" | bc -l) )); then
            echo "Error rate too high: $ERROR_RATE"
            exit 1
          fi
          
      - name: Trigger rollback on failure
        if: failure()
        run: |
          gh workflow run rollback.yml \
            -f environment=${{ inputs.environment }} \
            -f reason="Post-deploy gate failed"
```

## Gate Metrics Dashboard

```markdown
## Quality Gate Health

### Pass Rates (Last 30 Days)

| Gate | Pass Rate | Avg Duration | Trend |
|------|-----------|--------------|-------|
| Pre-Commit | 95% | 12s | → |
| PR Gate | 87% | 4m 30s | ↑ |
| Pre-Deploy | 98% | 8m | → |
| Post-Deploy | 99.5% | 2m | ↑ |

### Top Failure Reasons

| Gate | Reason | Count |
|------|--------|-------|
| PR | Coverage below threshold | 23 |
| PR | Lint errors | 18 |
| PR | Type errors | 12 |
| Pre-Deploy | E2E flaky test | 5 |
| Post-Deploy | Latency spike | 2 |
```

## Gate Override Process

### Emergency Override
```markdown
## Gate Override Request

**Gate**: PR Gate
**Reason**: Security hotfix - CVE-2024-1234
**Requested By**: @developer
**Approved By**: @security-lead

### Justification
Critical security vulnerability requires immediate fix.
Full test suite takes 20 minutes, patch is one-line.

### Conditions
- [ ] Security lead approval
- [ ] Change is minimal and isolated
- [ ] Follow-up PR for full testing within 24h
- [ ] Incident ticket created

### Override Commands
```bash
gh pr merge --admin --squash
```
```

### Override Audit Log
```json
{
  "timestamp": "2024-01-15T14:30:00Z",
  "gate": "pr-gate",
  "pr": 1234,
  "override_by": "security-lead",
  "reason": "Security hotfix CVE-2024-1234",
  "approvers": ["security-lead", "cto"],
  "follow_up_ticket": "SEC-789"
}
```

## Customizing Gates

### Per-Team Overrides
```yaml
# .github/quality-gates.yml
defaults:
  coverage_threshold: 80
  max_pr_size: 400
  required_approvers: 1

overrides:
  # Higher standards for core modules
  "src/core/**":
    coverage_threshold: 95
    required_approvers: 2
    
  # Relaxed for docs
  "docs/**":
    coverage_threshold: 0
    required_approvers: 1
    
  # Stricter for security
  "src/auth/**":
    coverage_threshold: 95
    required_approvers: 2
    require_security_review: true
```

### Dynamic Thresholds
```python
def calculate_coverage_threshold(file_path: str) -> int:
    """Dynamic coverage based on file criticality."""
    critical_paths = ['auth/', 'payment/', 'core/']
    new_paths = ['experimental/', 'beta/']
    
    if any(p in file_path for p in critical_paths):
        return 95
    elif any(p in file_path for p in new_paths):
        return 70
    else:
        return 80
```
