# Source Scan Checklist

The README must be grounded in the repo. Before writing, locate the authoritative sources for:

## Project identity
- Existing `README.md` / `docs/**` for intent and terminology.
- Product name, repo name, and any public URLs.

## How to run (local dev)
- `package.json` scripts / `pnpm-lock.yaml` / `yarn.lock`
- `Makefile`
- `Dockerfile`, `docker-compose.yml`, `compose.yaml`
- `bin/` scripts (common in Rails and other ecosystems)
- `Procfile` / `Procfile.dev`

## Configuration / environment variables
- `.env.example`, `.env.sample`, `.env.template`, `.envrc`
- `config/**` (framework-specific configuration)
- Any documented secret management (`.github/`, `k8s/`, `helm/`, `terraform/`)

## Tests, lint, typecheck, build
- Test runner configs: `pytest.ini`, `jest.config.*`, `vitest.config.*`, `go test` conventions, etc.
- Lint configs: `.eslintrc*`, `.ruff.toml`, `pyproject.toml`, `.golangci.yml`, etc.
- Build commands: CI definitions or package scripts.
- CI files: `.github/workflows/**`, `.gitlab-ci.yml`, `azure-pipelines.yml`, `circleci/**`

## Data stores
- Database configs: `prisma/schema.prisma`, `db/schema.rb`, migration folders, etc.
- Local dev dependencies: Postgres/Redis/MinIO/etc.

## Deployment and operations
- Deployment descriptors (see `platform_detection.md`).
- Observability (logs/metrics/traces) if present.

## If something is unclear
Prefer asking for:
- A 2-3 sentence description of what the project does.
- Any required external services (and whether substitutes exist for local dev).
- Where secrets are expected to come from in dev and in prod.

Avoid asking for:
- Things already present in CI configs, scripts, or runbooks.
