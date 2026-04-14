# Ledger schema (PM canon)

Default scope: `.pm/scopes/default/`

Files:
- `meta.env` — contains `NEXT_TICKET_ID=T-0001` (or next available)
- `tickets.tsv` — authoritative ticket list
- `criteria.tsv` — acceptance criteria items per ticket
- `evidence.tsv` — proof references per ticket
- `pulse.log` — append-only event stream
- `core.md` — durable context

## tickets.tsv columns
`id, state, title, owner, scope, spec_path, created_at, updated_at, deps, tags`

- `id`: `T-0001` format
- `state`: `NEW | IN_PROGRESS | BLOCKED | DONE | DROPPED`
- `spec_path`: MUST include the ticket id in file name
- `deps`: comma-separated ticket IDs
- `tags`: comma-separated tags

## criteria.tsv columns
`ticket_id, ac_id, checked, text, spec_ids`

- `ac_id`: `AC-1`, `AC-2`, …
- `checked`: `[ ]` or `[x]` (or `0/1`, but be consistent)
- `text`: externally observable behavior
- `spec_ids`: comma-separated `SPEC-*` IDs (can be blank until spec exists)

## evidence.tsv columns
`ticket_id, date, kind, ref, note`

- `kind`: `tests | logs | screenshot | pr | note`
- `ref`: file path or link

## pulse.log columns (append-only)
`timestamp, event, message`

- `event`: `CREATED | STATE | DONE | BLOCKED | NOTE`
