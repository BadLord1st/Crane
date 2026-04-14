# Gap Analysis Methodology

## What is Gap Analysis?

Gap analysis identifies the delta between:
- **Current State**: What the system does now
- **Desired State**: What the spec says it should do

Output: A prioritized list of changes needed to close the gap.

## Analysis Framework

### Step 1: Baseline Current State

```markdown
## Current State Assessment

### Functionality
| Feature | Status | Notes |
|---------|--------|-------|
| User login | ✅ Implemented | Basic auth only |
| Password reset | ❌ Missing | Not yet built |
| 2FA | ⚠️ Partial | TOTP only, no SMS |

### Codebase Health
- Test coverage: 67%
- Known bugs: 3 open
- Tech debt items: 5

### Infrastructure
- Deployment: Manual
- Monitoring: Basic (uptime only)
- Scaling: Single instance
```

### Step 2: Define Desired State

```markdown
## Desired State (from Spec)

### Required Functionality
| Requirement | Priority | Acceptance Criteria |
|-------------|----------|---------------------|
| REQ-001 | Must | Users can login with email/password |
| REQ-002 | Must | Users can reset password via email |
| REQ-003 | Should | Users can enable 2FA (TOTP or SMS) |
| REQ-004 | Could | Social login (Google, GitHub) |

### Quality Attributes
- Response time: < 200ms p95
- Availability: 99.9%
- Test coverage: > 80%

### Infrastructure
- Deployment: CI/CD automated
- Monitoring: Full observability stack
- Scaling: Auto-scaling enabled
```

### Step 3: Identify Gaps

```markdown
## Gap Identification

### Functional Gaps
| ID | Current | Desired | Gap | Effort |
|----|---------|---------|-----|--------|
| G-001 | No password reset | Password reset via email | Implement email flow | 3 days |
| G-002 | TOTP only | TOTP + SMS | Add SMS provider | 2 days |
| G-003 | No social login | Google + GitHub | OAuth integration | 5 days |

### Non-Functional Gaps
| ID | Current | Desired | Gap | Effort |
|----|---------|---------|-----|--------|
| G-010 | Manual deploy | CI/CD | GitHub Actions setup | 2 days |
| G-011 | 67% coverage | 80% coverage | Add tests | 3 days |
| G-012 | No APM | Full traces | Add OpenTelemetry | 4 days |

### Technical Debt
| ID | Issue | Impact | Effort |
|----|-------|--------|--------|
| D-001 | No input validation | Security risk | 2 days |
| D-002 | SQL queries not parameterized | SQL injection | 1 day |
```

### Step 4: Prioritize & Plan

```markdown
## Prioritized Gap Closure Plan

### Phase 1: Critical (Week 1-2)
- [D-002] Fix SQL injection vulnerability
- [D-001] Add input validation
- [G-001] Implement password reset

### Phase 2: Required (Week 3-4)
- [G-010] Set up CI/CD pipeline
- [G-011] Increase test coverage

### Phase 3: Enhancement (Week 5-6)
- [G-002] Add SMS 2FA
- [G-012] Implement observability

### Phase 4: Nice-to-have (Backlog)
- [G-003] Social login
```

## Gap Types

### 1. Functional Gaps
Missing or incomplete features:
- Feature not implemented
- Feature partially working
- Feature works but doesn't match spec

### 2. Quality Gaps
Non-functional requirements not met:
- Performance below target
- Availability SLA not met
- Security controls missing

### 3. Coverage Gaps
Insufficient test coverage:
- Untested code paths
- Missing edge case tests
- No integration tests

### 4. Documentation Gaps
Missing or outdated docs:
- API not documented
- Setup instructions missing
- Architecture diagrams outdated

### 5. Compliance Gaps
Regulatory/policy violations:
- GDPR requirements not met
- Accessibility standards violated
- Security certifications missing

## Tools & Techniques

### Code Coverage Analysis
```bash
# Python
pytest --cov --cov-report=html

# JavaScript
npm test -- --coverage

# Go
go test -cover ./...
```

### Spec-to-Code Tracing
```markdown
| Spec Section | Code Location | Test | Status |
|--------------|---------------|------|--------|
| 2.1 Login | auth/login.py | test_login.py | ✅ |
| 2.2 Logout | auth/logout.py | - | ❌ No test |
| 2.3 Register | - | - | ❌ Not implemented |
```

### Dependency Analysis
```bash
# Find outdated
npm outdated
pip list --outdated

# Find security issues
npm audit
pip-audit
```

## Gap Analysis Checklist

```
- [ ] Current state documented
- [ ] Desired state defined (from spec)
- [ ] All features compared
- [ ] Non-functional requirements checked
- [ ] Test coverage analyzed
- [ ] Security gaps identified
- [ ] Tech debt catalogued
- [ ] Gaps prioritized by risk/value
- [ ] Effort estimated for each gap
- [ ] Closure plan created
- [ ] Dependencies identified
- [ ] Risks documented
```

## Output Template

```markdown
# Gap Analysis Report

**Project**: [Name]
**Date**: YYYY-MM-DD
**Analyst**: [Name]

## Executive Summary
[2-3 sentences on key findings]

## Current State Overview
[Summary of what exists]

## Gap Summary
- Critical gaps: N
- High priority: N
- Medium priority: N
- Low priority: N
- Total estimated effort: X weeks

## Detailed Gaps
[Table of all gaps with priority, effort, risk]

## Recommended Plan
[Phased approach to close gaps]

## Risks
[What could go wrong during gap closure]

## Dependencies
[External factors affecting the plan]
```
