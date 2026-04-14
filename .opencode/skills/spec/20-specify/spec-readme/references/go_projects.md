# Go Projects: README Notes

Use this when the repo is primarily Go.

## What to scan

- `go.mod` / `go.sum` (module name, Go version, key deps)
- `cmd/**` (main binaries / entrypoints)
- `internal/**` (application code that should not be imported externally)
- `pkg/**` (libraries intended for reuse; not always present)
- `configs/**` / `config/**` (config loaders, defaults)
- `deploy/**` / `k8s/**` / `helm/**` (deployment)
- `Makefile` (canonical commands)
- `.github/workflows/**` (CI truth)
- `Dockerfile` / `compose.yaml`

## Commands section: preferred patterns

- Build:
  - `go build ./...` (libraries) or `go build ./cmd/<app>`
- Test:
  - `go test ./...`
  - If race matters: `go test -race ./...`
- Lint (if present):
  - `golangci-lint run` (prefer `Makefile` wrapper if it exists)
- Format:
  - `gofmt -w .` (again: prefer repo wrapper)

## Configuration patterns

- Explicitly document how config is loaded:
  - env vars (common)
  - config file flags (e.g. `-config`)
  - default config locations
- If the repo uses `cobra` / `urfave/cli`, document the primary binary and flags.

## Common README pitfalls

- Don’t say “run `go run main.go`” unless that’s actually how it’s structured.
- If there are multiple binaries under `cmd/`, list them and the intended usage.
