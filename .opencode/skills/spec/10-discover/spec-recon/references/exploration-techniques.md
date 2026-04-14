# Codebase Exploration Techniques

## Quick Project Assessment

### 30-Second Overview
```bash
# What language/framework?
ls *.json *.toml *.yaml Makefile Dockerfile 2>/dev/null

# Project structure
find . -maxdepth 2 -type d | head -20

# Entry points
ls main.* app.* index.* src/main.* 2>/dev/null

# Config files
ls *.config.* .env* 2>/dev/null
```

### 2-Minute Deep Dive
```bash
# README for context
head -100 README.md

# Package dependencies
cat package.json | jq '.dependencies, .devDependencies' 2>/dev/null
cat pyproject.toml | grep -A 50 '\[tool.poetry.dependencies\]' 2>/dev/null
cat go.mod 2>/dev/null

# Test setup
ls -la tests/ test/ __tests__/ spec/ 2>/dev/null
grep -r "test" package.json pyproject.toml 2>/dev/null | head -5

# CI/CD
ls -la .github/workflows/ .gitlab-ci.yml .circleci/ 2>/dev/null
```

## Understanding Code Flow

### Entry Point Discovery
```bash
# Web frameworks
grep -r "app = " --include="*.py" | head -5
grep -r "createApp\|express()" --include="*.ts" --include="*.js" | head -5

# Main functions
grep -rn "def main\|func main\|void main" --include="*.py" --include="*.go" --include="*.java"

# Route definitions
grep -rn "@app.route\|@router\|app.get\|app.post" --include="*.py" | head -20
grep -rn "router\.\|app\.get\|app\.post" --include="*.ts" --include="*.js" | head -20
```

### Dependency Mapping
```bash
# Python imports
grep -rh "^from\|^import" --include="*.py" | sort | uniq -c | sort -rn | head -20

# JavaScript/TypeScript imports
grep -rh "^import.*from\|require(" --include="*.ts" --include="*.js" | sort | uniq -c | sort -rn | head -20

# Internal vs external
grep -rh "from \.\|from src\|from app" --include="*.py" | head -20
```

### Call Chain Tracing
```bash
# Find function definition
grep -rn "def process_order\|func processOrder" .

# Find all callers
grep -rn "process_order\|processOrder" . | grep -v "def \|func "

# Build call chain manually
# 1. handler → 2. service → 3. repository → 4. database
```

## Pattern Recognition

### Common Architectural Patterns

**Layered Architecture**
```
handlers/     → HTTP handlers, route definitions
services/     → Business logic
repositories/ → Data access
models/       → Domain entities
```

**Domain-Driven Design**
```
domain/
  user/
    entity.py
    repository.py
    service.py
  order/
    entity.py
    repository.py
    service.py
```

**Hexagonal (Ports & Adapters)**
```
core/         → Business logic, no dependencies
ports/        → Interfaces (what we need)
adapters/     → Implementations (how we get it)
  http/
  database/
  messaging/
```

### Detecting Patterns
```bash
# Repository pattern
grep -rn "class.*Repository\|Repository:" --include="*.py" --include="*.ts"

# Service layer
grep -rn "class.*Service\|Service:" --include="*.py" --include="*.ts"

# Factory pattern
grep -rn "class.*Factory\|Factory\." --include="*.py" --include="*.ts"

# Dependency injection
grep -rn "@inject\|@Inject\|Depends(" --include="*.py" --include="*.ts"
```

## Configuration Discovery

### Environment Variables
```bash
# Find all env var references
grep -rh "os.environ\|process.env\|os.Getenv" --include="*.py" --include="*.ts" --include="*.go"

# Find .env files
find . -name ".env*" -o -name "*.env"

# Extract env var names
grep -roh 'os\.environ\[["'"'"']\([A-Z_]*\)' --include="*.py" | sed 's/.*\[\(['"'"'"]\)\([^'"'"'"]*\).*/\2/' | sort -u
```

### Configuration Files
```bash
# Find config files
find . -name "config.*" -o -name "settings.*" -o -name "*.config.*"

# Common locations
cat config/default.json settings.py config.yaml 2>/dev/null | head -50
```

## Database Schema Discovery

### ORM Models
```python
# SQLAlchemy models
grep -rn "class.*Base\)\|Column(" --include="*.py"

# Django models
grep -rn "class.*models.Model\|models.CharField" --include="*.py"

# Prisma schema
cat prisma/schema.prisma
```

### Migrations
```bash
# Find migration files
find . -path "*/migrations/*" -name "*.py" -o -path "*/alembic/*" -name "*.py"
find . -path "*/migrations/*" -name "*.sql"

# Latest migrations
ls -lt alembic/versions/*.py 2>/dev/null | head -5
ls -lt migrations/*.sql 2>/dev/null | head -5
```

## API Contract Discovery

### OpenAPI/Swagger
```bash
# Find spec files
find . -name "openapi.*" -o -name "swagger.*" -o -name "api-spec.*"

# Extract endpoints from spec
cat openapi.yaml | grep -E "^  /|^    (get|post|put|delete|patch):"
```

### Route Extraction
```bash
# FastAPI
grep -rn "@app\.\(get\|post\|put\|delete\|patch\)" --include="*.py" | \
  sed 's/.*@app\.\([a-z]*\)("\([^"]*\)".*/\1 \2/'

# Express
grep -rn "app\.\(get\|post\|put\|delete\)" --include="*.js" --include="*.ts" | \
  sed 's/.*app\.\([a-z]*\)(['"'"'"]\([^'"'"'"]*\).*/\1 \2/'
```

## Testing Structure Discovery

### Test Organization
```bash
# Find test files
find . -name "*_test.py" -o -name "test_*.py" -o -name "*.test.ts" -o -name "*.spec.ts"

# Test to source mapping
# test_user.py → user.py
# user.test.ts → user.ts
# user_test.go → user.go
```

### Test Coverage Gaps
```bash
# Find source files without tests
for f in src/*.py; do
  base=$(basename "$f" .py)
  if [ ! -f "tests/test_${base}.py" ]; then
    echo "Missing test for: $f"
  fi
done
```

## Documentation Discovery

### Inline Documentation
```bash
# Python docstrings
grep -rn '"""' --include="*.py" | head -20

# JSDoc
grep -rn '/\*\*' --include="*.ts" --include="*.js" | head -20

# Go doc comments
grep -rn "^// [A-Z]" --include="*.go" | head -20
```

### External Docs
```bash
# Find docs
find . -name "*.md" -path "*/docs/*" -o -name "*.rst"
ls docs/ doc/ documentation/ 2>/dev/null
```

## Output Template

```markdown
# Codebase Recon Report

## Project Overview
- **Language**: Python 3.11
- **Framework**: FastAPI
- **Package Manager**: Poetry

## Structure
```
src/
├── api/          # HTTP handlers
├── services/     # Business logic
├── repositories/ # Data access
├── models/       # Domain entities
└── utils/        # Helpers
```

## Entry Points
- `src/main.py` → FastAPI app creation
- `src/api/routes.py` → Route registration

## Key Patterns
- Repository pattern for data access
- Dependency injection via FastAPI Depends
- Pydantic models for validation

## Test Infrastructure
- Framework: pytest
- Location: tests/
- Coverage: 72%

## Configuration
- Environment: `.env` + `config.py`
- Key vars: DATABASE_URL, SECRET_KEY, DEBUG

## Dependencies
- Critical: SQLAlchemy, Pydantic, httpx
- Dev: pytest, mypy, black
```
