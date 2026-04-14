---
description: "REVIEW: explain the project and produce a runnable, Obsidian-friendly mental model"
agent: orchestrator
model: openai/gpt-5.4
temperature: 0.2
---

# What The Fuck Is This?

Explain this project based on specs, code, and configuration.

## Protocol

### Phase 0) Mode

This command is **always REVIEW**.

- No repo mutations.
- No speculative refactors.
- Output must be actionable and runnable.

### Phase 1) Skills (lazy-load)

Load only what you need.

Start with recon:

```bash
skill "discover_spec_recon"
```

If you need to extract run/test/deploy commands or env details:

```bash
skill "discover_spec_env_scout"
```

### Phase 2) Gather context (read, don't assume)

- README.md, docs/, specs/
- package.json / pyproject.toml / Cargo.toml / etc.
- Configuration files (.env.example, docker-compose, etc.)
- Entry points (main.py, index.ts, cmd/, etc.)
- CI/CD configuration (.github/workflows/, etc.)

### Phase 3) Answer these questions

### What is this?

- Purpose and main functionality
- Target users/audience
- Key features

### How is it built?

- Tech stack (language, framework, key dependencies)
- Architecture overview
- Key modules/components and their responsibilities

### How to run it?

#### Prerequisites

- Required tools (Node, Python, Docker, etc.)
- Required services (databases, APIs, etc.)
- Environment variables needed

#### Development

```bash
# Commands to set up and run locally
```

#### Tests

```bash
# Commands to run tests
```

### How to configure?

- Environment variables and their purpose
- Configuration files
- Feature flags (if any)

### How to deploy?

- Deployment targets (cloud, docker, serverless)
- CI/CD pipeline overview
- Required secrets/credentials

### Gotchas & Tips

- Common issues and solutions
- Performance considerations
- Security notes

## Output Format

Provide a clear, structured explanation that a new developer could follow to:

1. Understand what this project does
2. Set it up locally
3. Run and test it
4. Make their first contribution

Be specific — include actual commands, file paths, and configuration values.

## Deliverables

1) A newcomer-friendly explanation:
   - what it does
   - how it's built
   - how to run/test/configure/deploy

2) A "First 15 minutes" checklist:
   - install
   - run
   - test
   - fix one small thing

3) An Obsidian-friendly map draft (do not write files in REVIEW):
   - suggested nodes like:
     - `[[spec-index]]`
     - `[[runbook-local-dev]]`
     - `[[runbook-ci]]`
     - `[[architecture]]`
     - `[[test-plan]]`
   - and what each should contain
