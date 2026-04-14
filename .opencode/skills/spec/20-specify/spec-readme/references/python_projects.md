# Python Projects: README Notes

Use this when the repo is primarily Python.

## What to scan

- `pyproject.toml` (tooling, dependencies, scripts/entrypoints)
- `poetry.lock` / `uv.lock` / `requirements*.txt` / `Pipfile.lock`
- `src/**` layout vs flat layout
- `tests/**` (pytest/unittest)
- `Makefile` / `noxfile.py` / `tox.ini`
- `.python-version` / `runtime.txt` (version pinning)
- `.env.example` / `dotenv` usage
- `.github/workflows/**` (CI truth)
- `Dockerfile` / `compose.yaml`

## Commands section: preferred patterns

Pick the package manager that exists in the repo:

- Poetry:
  - install: `poetry install`
  - run: `poetry run <cmd>`
  - tests: `poetry run pytest`

- uv:
  - install/sync: `uv sync`
  - run: `uv run <cmd>`
  - tests: `uv run pytest`

- pip:
  - recommend venv creation (copy/paste commands)
  - install: `pip install -r requirements.txt`

## Configuration patterns

- If env vars are used, provide a table and a `cp .env.example .env` step.
- If settings are file-based (YAML/TOML), document default path + override mechanism.

## Common README pitfalls

- Don’t invent “python -m app” if the package/module name differs.
- Don’t assume Conda unless the repo uses it.
- If there’s a CLI entrypoint in `pyproject.toml` (console_scripts), document that command.
