# Environment Detection Patterns

## Project Type Detection

### By File Presence

| Files | Project Type | Build System |
|-------|--------------|--------------|
| `package.json` | Node.js | npm/yarn/pnpm |
| `pyproject.toml` | Python | poetry/hatch/pdm |
| `requirements.txt` | Python | pip |
| `go.mod` | Go | go modules |
| `Cargo.toml` | Rust | cargo |
| `pom.xml` | Java | Maven |
| `build.gradle` | Java/Kotlin | Gradle |
| `mix.exs` | Elixir | Mix |
| `Gemfile` | Ruby | Bundler |
| `composer.json` | PHP | Composer |

### Framework Detection

```bash
# Node.js frameworks
grep -l "next" package.json        # Next.js
grep -l "nuxt" package.json        # Nuxt
grep -l "react" package.json       # React
grep -l "vue" package.json         # Vue
grep -l "angular" package.json     # Angular
grep -l "express" package.json     # Express
grep -l "fastify" package.json     # Fastify
grep -l "nestjs" package.json      # NestJS

# Python frameworks
grep -l "django" pyproject.toml    # Django
grep -l "fastapi" pyproject.toml   # FastAPI
grep -l "flask" pyproject.toml     # Flask
grep -l "pytorch" pyproject.toml   # PyTorch
```

## Test Framework Detection

### Node.js
```bash
# Check package.json scripts
jq '.scripts.test' package.json

# Common test runners
grep -E "jest|vitest|mocha|ava|tape" package.json
```

| Framework | Config File | Run Command |
|-----------|-------------|-------------|
| Jest | jest.config.js | `npm test` / `jest` |
| Vitest | vitest.config.ts | `vitest` |
| Mocha | .mocharc.json | `mocha` |
| Playwright | playwright.config.ts | `playwright test` |
| Cypress | cypress.config.js | `cypress run` |

### Python
```bash
# Check pyproject.toml
grep -E "pytest|unittest|nose" pyproject.toml
```

| Framework | Config | Run Command |
|-----------|--------|-------------|
| pytest | pytest.ini / pyproject.toml | `pytest` |
| unittest | - | `python -m unittest` |
| nose2 | nose2.cfg | `nose2` |

### Go
```bash
# Go uses built-in testing
go test ./...

# With coverage
go test -cover ./...
```

### Rust
```bash
# Built-in test framework
cargo test

# With nextest
cargo nextest run
```

## CI/CD Detection

### GitHub Actions
```bash
ls -la .github/workflows/
# Common files: ci.yml, test.yml, build.yml, deploy.yml
```

### GitLab CI
```bash
cat .gitlab-ci.yml
```

### Other CI Systems
| File | CI System |
|------|-----------|
| `.travis.yml` | Travis CI |
| `Jenkinsfile` | Jenkins |
| `.circleci/config.yml` | CircleCI |
| `azure-pipelines.yml` | Azure DevOps |
| `bitbucket-pipelines.yml` | Bitbucket |
| `.drone.yml` | Drone |

## Environment Variables

### Detection Script
```bash
# Common env file patterns
ls -la .env* 2>/dev/null
ls -la *.env 2>/dev/null

# Check for env templates
cat .env.example 2>/dev/null
cat .env.template 2>/dev/null
```

### Required vs Optional
```bash
# Find env var references
grep -rh '\$\{.*\}' --include='*.yaml' --include='*.yml' .
grep -rh 'process\.env\.' --include='*.ts' --include='*.js' .
grep -rh 'os\.environ' --include='*.py' .
grep -rh 'os\.Getenv' --include='*.go' .
```

## Database Detection

### By Connection Strings
```bash
# PostgreSQL
grep -ri "postgres\|psql\|pg_" . --include='*.env*'

# MySQL
grep -ri "mysql\|mariadb" . --include='*.env*'

# MongoDB
grep -ri "mongodb\|mongo" . --include='*.env*'

# Redis
grep -ri "redis" . --include='*.env*'

# SQLite
find . -name "*.db" -o -name "*.sqlite*"
```

### By ORM/Driver
| Package | Database |
|---------|----------|
| `pg` / `psycopg2` | PostgreSQL |
| `mysql2` / `mysqlclient` | MySQL |
| `mongodb` / `pymongo` | MongoDB |
| `redis` / `redis-py` | Redis |
| `prisma` | Multiple (check schema.prisma) |
| `sqlalchemy` | Multiple (check connection string) |

## Container Detection

### Docker
```bash
# Dockerfile presence
ls Dockerfile* 2>/dev/null
ls docker-compose*.yml 2>/dev/null
ls .dockerignore 2>/dev/null

# Parse compose for services
yq '.services | keys' docker-compose.yml
```

### Kubernetes
```bash
# K8s manifests
find . -name "*.yaml" -exec grep -l "apiVersion:" {} \;
ls -la k8s/ kubernetes/ manifests/ 2>/dev/null

# Helm
ls Chart.yaml 2>/dev/null
ls -la charts/ 2>/dev/null
```

## Environment Report Template

```markdown
## Environment Scout Report

### Project Type
- **Language**: Python 3.11
- **Framework**: FastAPI
- **Package Manager**: poetry

### Test Setup
- **Framework**: pytest
- **Config**: pyproject.toml [tool.pytest.ini_options]
- **Command**: `poetry run pytest`
- **Coverage**: pytest-cov configured

### CI/CD
- **Platform**: GitHub Actions
- **Workflows**: 
  - ci.yml (test on push)
  - deploy.yml (deploy on tag)

### Database
- **Primary**: PostgreSQL 15
- **ORM**: SQLAlchemy 2.0
- **Migrations**: Alembic

### Environment Variables
- **Required**: DATABASE_URL, SECRET_KEY, API_KEY
- **Optional**: DEBUG, LOG_LEVEL
- **Template**: .env.example exists

### Containers
- **Docker**: Yes (Dockerfile + docker-compose.yml)
- **Services**: app, postgres, redis

### Recommendations
1. Add .env.example to track required vars
2. Consider adding pre-commit hooks
3. Missing: health check endpoint
```
