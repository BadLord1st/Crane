# Traceability Matrix Guide

## What is Traceability?

Traceability links requirements to their implementations and tests, ensuring:
- Every requirement is implemented
- Every implementation has a test
- Changes can be traced through the system

```
Requirement → Design → Code → Test
     ↑          ↑        ↑       ↑
     └──────────┴────────┴───────┘
           Bidirectional links
```

## Traceability Matrix Format

### Simple Matrix

| Requirement | Spec Section | Code Location | Test | Status |
|-------------|--------------|---------------|------|--------|
| REQ-001 | spec/auth.md#login | auth/login.py | test_login.py | ✅ |
| REQ-002 | spec/auth.md#logout | auth/logout.py | test_logout.py | ✅ |
| REQ-003 | spec/auth.md#mfa | - | - | ⏳ Not started |
| REQ-004 | spec/orders.md | orders/create.py | test_orders.py | ⚠️ No test |

### Detailed Matrix

```markdown
## REQ-001: User Login

**Description**: Users can login with email and password

**Spec**: [spec/auth.md#login](spec/auth.md#login)

**Implementation**:
- `auth/login.py:LoginHandler` (main logic)
- `auth/validators.py:validate_credentials` (validation)
- `auth/tokens.py:create_session_token` (token generation)

**Tests**:
- `tests/test_auth.py::test_login_with_valid_credentials`
- `tests/test_auth.py::test_login_with_invalid_password`
- `tests/test_auth.py::test_login_with_nonexistent_user`

**BDD Scenarios**:
- `features/auth.feature:Scenario: Successful login`
- `features/auth.feature:Scenario: Failed login with wrong password`

**Status**: ✅ Complete
**Last Verified**: 2024-01-15
```

## Generating Traceability Matrix

### From Code Comments
```python
# REQ-001: User login functionality
def login(email: str, password: str) -> LoginResult:
    """
    Implements user login.
    
    Requirements:
        - REQ-001: Basic login
        - REQ-005: Rate limiting
    
    Tests:
        - test_auth.py::test_login_*
    """
    pass
```

### From Test Names
```python
def test_REQ001_user_can_login_with_valid_credentials():
    """Covers REQ-001: User login"""
    pass

def test_REQ001_user_cannot_login_with_wrong_password():
    """Covers REQ-001: User login (error case)"""
    pass
```

### Automated Extraction
```bash
# Extract all REQ references from code
grep -rn "REQ-[0-9]*" --include="*.py" src/

# Extract all REQ references from tests
grep -rn "REQ-[0-9]*" --include="*.py" tests/

# Find coverage by requirement
for req in $(grep -oh "REQ-[0-9]*" specs/*.md | sort -u); do
    echo "=== $req ==="
    echo "Code:"
    grep -l "$req" src/**/*.py 2>/dev/null
    echo "Tests:"
    grep -l "$req" tests/**/*.py 2>/dev/null
done
```

## Detecting Coverage Gaps

### Requirements Without Implementation
```bash
# List all requirements from spec
spec_reqs=$(grep -oh "REQ-[0-9]*" specs/*.md | sort -u)

# List all implemented requirements
code_reqs=$(grep -oh "REQ-[0-9]*" src/**/*.py | sort -u)

# Find missing
comm -23 <(echo "$spec_reqs") <(echo "$code_reqs")
```

### Implementation Without Tests
```python
# Find functions/classes without test references
import ast
import os

def find_untested(src_dir, test_dir):
    # Parse source files for defined functions
    defined = set()
    for file in os.listdir(src_dir):
        if file.endswith('.py'):
            with open(f"{src_dir}/{file}") as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    defined.add(node.name)
    
    # Parse test files for tested functions
    tested = set()
    for file in os.listdir(test_dir):
        if file.startswith('test_') and file.endswith('.py'):
            with open(f"{test_dir}/{file}") as f:
                content = f.read()
            for name in defined:
                if name in content:
                    tested.add(name)
    
    return defined - tested
```

### BDD Scenarios Without Implementation
```bash
# Find scenario steps without step definitions
grep -h "Given\|When\|Then" features/*.feature | \
  sed 's/.*\(Given\|When\|Then\)//' | \
  sort -u | \
while read step; do
    if ! grep -q "$step" features/steps/*.py; then
        echo "Missing step: $step"
    fi
done
```

## Maintaining Traceability

### On Requirement Change
```markdown
1. Update spec document
2. Find all REQ-XXX references in code
3. Update implementation if needed
4. Update tests to match new behavior
5. Update traceability matrix
6. Mark for review
```

### On Code Change
```markdown
1. Identify which requirements are affected
2. Verify tests still cover the requirement
3. Update traceability matrix if structure changed
4. Run affected tests
```

### Periodic Audit
```markdown
Weekly:
- [ ] Run automated gap detection
- [ ] Review new requirements for coverage

Sprint End:
- [ ] Full traceability matrix review
- [ ] Update status of all requirements
- [ ] Identify tech debt (missing tests)

Release:
- [ ] Verify all requirements for release are covered
- [ ] Generate release traceability report
- [ ] Sign-off on coverage
```

## Traceability Tooling

### Test Markers (pytest)
```python
import pytest

@pytest.mark.requirement("REQ-001")
def test_user_login():
    pass

# Generate coverage report
# pytest --requirement-coverage
```

### GitHub Issues Integration
```markdown
Issue Title: Implement user login
Labels: requirement

## Requirement
REQ-001: Users can login with email and password

## Implementation
- [ ] Create login endpoint
- [ ] Add validation
- [ ] Add tests

## Links
- Spec: link-to-spec
- PR: link-to-pr
- Tests: link-to-tests
```

### VS Code Task
```json
{
  "label": "Generate Traceability Matrix",
  "type": "shell",
  "command": "python scripts/generate_traceability.py > TRACEABILITY.md",
  "problemMatcher": []
}
```

## Traceability Report Template

```markdown
# Traceability Report

**Generated**: 2024-01-15
**Version**: 1.2.0

## Summary

| Metric | Value |
|--------|-------|
| Total Requirements | 45 |
| Implemented | 42 (93%) |
| Tested | 40 (89%) |
| Gaps | 5 |

## Coverage by Category

| Category | Requirements | Implemented | Tested |
|----------|--------------|-------------|--------|
| Authentication | 10 | 10 | 10 |
| Orders | 15 | 14 | 13 |
| Payments | 12 | 12 | 11 |
| Admin | 8 | 6 | 6 |

## Gaps

### Requirements Without Implementation
- REQ-035: Export orders to PDF
- REQ-036: Bulk order import
- REQ-045: Admin dashboard

### Implementation Without Tests
- REQ-022: Order status webhook (orders/webhooks.py)
- REQ-028: Payment retry logic (payments/retry.py)

## Full Matrix

[See TRACEABILITY_MATRIX.md for complete matrix]
```
