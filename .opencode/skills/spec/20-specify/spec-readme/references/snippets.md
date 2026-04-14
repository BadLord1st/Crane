# Tables and Snippets

## Environment variables table

```md
## Environment Variables

### Required

| Variable | Description | Example |
|---|---|---|
| `DATABASE_URL` | Postgres connection string | `postgres://...` |

### Optional

| Variable | Description | Default |
|---|---|---|
| `LOG_LEVEL` | Logging verbosity | `info` |
```

Guidelines:
- If a secret must not be committed, say so explicitly.
- If an env var is used only in production, label it.

## Commands table

```md
## Common Commands

| Command | Description |
|---|---|
| `<cmd>` | Run the app |
| `<cmd>` | Run tests |
| `<cmd>` | Run lint |
```

Guidelines:
- Prefer the repo's canonical commands (scripts/Makefile) over ad-hoc invocations.
- If commands differ between OSes, call it out.

## Troubleshooting pattern

```md
## Troubleshooting

### <Symptom>

**What you see:** `<error message>`

**Likely cause:** <1 sentence>

**Fix:**
```bash
<commands>
```
```
