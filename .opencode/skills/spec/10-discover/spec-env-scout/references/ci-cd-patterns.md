# CI/CD Configuration Patterns

## GitHub Actions

### Basic Test Workflow
```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        node-version: [18, 20, 22]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node.js ${{ matrix.node-version }}
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Run tests
        run: npm test
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        if: matrix.node-version == 20
```

### Python with Poetry
```yaml
name: Python CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: test_db
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Install Poetry
        uses: snok/install-poetry@v1
        with:
          version: 1.7.0
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'poetry'
      
      - name: Install dependencies
        run: poetry install
      
      - name: Run tests
        env:
          DATABASE_URL: postgresql://postgres:test@localhost:5432/test_db
        run: poetry run pytest --cov
```

### Go
```yaml
name: Go CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Go
        uses: actions/setup-go@v5
        with:
          go-version: '1.22'
      
      - name: Download dependencies
        run: go mod download
      
      - name: Run tests
        run: go test -v -race -coverprofile=coverage.out ./...
      
      - name: Run linter
        uses: golangci/golangci-lint-action@v4
```

## GitLab CI

### Multi-stage Pipeline
```yaml
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_DRIVER: overlay2

.test-template: &test-template
  stage: test
  image: python:3.11
  before_script:
    - pip install poetry
    - poetry install
  cache:
    paths:
      - .venv/

unit-tests:
  <<: *test-template
  script:
    - poetry run pytest tests/unit

integration-tests:
  <<: *test-template
  services:
    - postgres:15
  variables:
    DATABASE_URL: postgresql://postgres:postgres@postgres:5432/test
  script:
    - poetry run pytest tests/integration

build:
  stage: build
  image: docker:24
  services:
    - docker:24-dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main

deploy-staging:
  stage: deploy
  script:
    - ./deploy.sh staging
  environment:
    name: staging
  only:
    - main
```

## Common Patterns

### Caching Strategies

| Language | Cache Path | Key |
|----------|------------|-----|
| Node.js | `~/.npm` or `node_modules` | `package-lock.json` hash |
| Python | `~/.cache/pip` or `.venv` | `poetry.lock` hash |
| Go | `~/go/pkg/mod` | `go.sum` hash |
| Rust | `~/.cargo` + `target` | `Cargo.lock` hash |

### Service Containers
```yaml
# PostgreSQL
services:
  postgres:
    image: postgres:15
    env:
      POSTGRES_PASSWORD: test
    ports: ['5432:5432']

# Redis
services:
  redis:
    image: redis:7
    ports: ['6379:6379']

# MySQL
services:
  mysql:
    image: mysql:8
    env:
      MYSQL_ROOT_PASSWORD: test
    ports: ['3306:3306']
```

### Matrix Builds
```yaml
strategy:
  fail-fast: false
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    python-version: ['3.10', '3.11', '3.12']
    exclude:
      - os: windows-latest
        python-version: '3.10'
```

### Conditional Jobs
```yaml
# Only on main branch
if: github.ref == 'refs/heads/main'

# Only on tags
if: startsWith(github.ref, 'refs/tags/')

# Skip if [skip ci] in commit
if: "!contains(github.event.head_commit.message, '[skip ci]')"

# Only when specific files changed
paths:
  - 'src/**'
  - 'tests/**'
  - 'package.json'
```

### Secrets & Environment
```yaml
# GitHub Actions
env:
  API_KEY: ${{ secrets.API_KEY }}
  DATABASE_URL: ${{ secrets.DATABASE_URL }}

# With environments
jobs:
  deploy:
    environment: production
    steps:
      - run: echo ${{ secrets.PROD_API_KEY }}
```

## Troubleshooting CI

### Common Failures

| Issue | Cause | Fix |
|-------|-------|-----|
| Tests pass locally, fail in CI | Missing env vars | Add to CI secrets |
| Flaky tests | Race conditions | Add retries, fix test |
| Timeout | Slow tests/network | Increase timeout, parallelize |
| Cache miss | Key changed | Check lock file in commit |
| Permission denied | File permissions | Use `chmod` in script |
| Service not ready | Container startup | Add health checks, wait |

### Debugging Commands
```yaml
# Print environment
- run: env | sort

# Print file structure
- run: find . -type f -name "*.py" | head -20

# Interactive debugging (GitHub)
- uses: mxschmitt/action-tmate@v3
  if: failure()
```
