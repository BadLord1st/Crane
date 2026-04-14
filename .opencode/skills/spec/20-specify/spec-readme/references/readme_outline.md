# README Outline (Template)

Use this as the default structure. Keep it copy/paste friendly.

```md
# <Project Name>

<1-3 sentences. What it is, who it is for, what problem it solves.>

## Key Features
- ...

## Tech Stack
- Language/runtime:
- Frameworks:
- Data stores:
- Tooling (tests/lint/build):

## Prerequisites
- ... (pin major versions if the repo requires it)

## First 15 Minutes
1. Clone
2. Install deps
3. Configure env
4. Start services (db/cache)
5. Run app
6. Run tests

## Getting Started (Local Development)
### Clone

### Install Dependencies

### Configure Environment
Include:
- how to copy example env file
- required vs optional env vars
- where secrets come from

### Start Dependencies
DB/cache/queues; include Docker commands if the repo supports it.

### Run
The single best command, plus alternatives.

## Common Commands
Table or list:
- test
- lint
- typecheck
- build
- format
- db migrate

## Architecture (Mental Model)
Explain only the minimum needed to work on the repo:
- directory layout (high-level)
- request/data flow (if applicable)
- key modules and responsibilities

## Configuration Reference
Environment variables table and any config files that matter.

## Deployment
If a runbook exists, link it.
Otherwise: describe the detected deployment target and the repo's deploy entrypoints.

## Troubleshooting
3-5 likely failures:
- symptoms
- how to confirm
- fix

## Contributing (optional)

## License (optional)
```
