---
name: spec-pm-canon
description: "Maintain PM canon (tasks-as-code): ledger + status snapshot + dedup gate. Prevent duplicate work by enforcing UPDATE>NEW."
metadata:
  signature: "spec-pm-canon :: (Repo, Goal, Mode) -> (TicketID, PMCanonDelta)"
---

## When to use
- You are about to create/propose work and must avoid duplicates.
- You need a canonical task state that survives chat context.
- You need proof: criteria + evidence + traceability before calling something DONE.

## Inputs
- Goal statement (user request).
- Mode: REVIEW or WORK.
- PM canon files:
  - `./opencode/status.md`
  - `.pm/scopes/<scope>/*` (default: `default`)
- Spec canon files:
  - `docs/spec/spec-index.md`
  - `docs/spec/traceability/traceability.md`

## Outputs
- Canon Load + Dedup Gate results (consulted artifacts + matches).
- Ticket decision: UPDATE existing `T-XXXX` or NEW `T-XXXX`.
- PM canon deltas (WORK): updated ledger rows + updated `status.md`.

## Protocol

### 1) Canon Load (read-first)
1) Read `status.md` (snapshot).
2) Read ledger files under `.pm/scopes/<scope>/`:
   - `meta.env`, `tickets.tsv`, `criteria.tsv`, `evidence.tsv`, `pulse.log` (+ `core.md` if present).
3) Read spec canon:
   - `docs/spec/spec-index.md`
   - `docs/spec/traceability/traceability.md`
4) If any canonical file is missing:
   - REVIEW: report + propose creation (no writes).
   - WORK: create missing skeletons, then continue.

### 2) Dedup Gate (UPDATE by default)
1) Search for overlap using:
   - ticket titles, tags, owners
   - key nouns from the Goal (endpoints, entities, modules, flags, tables)
   - spec titles and SPEC IDs in traceability
2) Classify matches:
   - Strong match → treat as duplicate unless you prove non-overlap.
   - Weak match → treat as overlap; propose merge strategy.
3) Decision rule:
   - If any strong match exists → UPDATE.
   - NEW is allowed only after you explicitly rule out overlap (scope/outcome/contract).

### 3) Ticket ID allocation
- ID format: `T-0001`, `T-0002`, …
- WORK: allocate from `meta.env` (`NEXT_TICKET_ID=...`). If absent/corrupt, compute from max in `tickets.tsv`, then repair `meta.env`.
- REVIEW: use provisional `T-????` when allocation is unsafe.

### 4) Ledger updates (WORK)
Update only the minimum necessary, but keep invariants.

`tickets.tsv`:
- create or update the row for `T-XXXX`
- keep `state`, `owner`, `spec_path`, `updated_at` coherent

`criteria.tsv`:
- store acceptance criteria as externally observable checkboxes
- link each AC row to relevant SPEC IDs (when known)

`evidence.tsv`:
- add references to proof (test command output, logs, PRs, screenshots, etc.)

`pulse.log` (append-only):
- append events: `CREATED`, `STATE`, `DONE`, `BLOCKED`, `NOTE`

### 5) status.md snapshot (WORK)
Update `status.md` after any ledger change.
It must be consistent with the ledger and include:
- last updated timestamp
- CORE context (from `.pm/.../core.md` if present)
- tickets grouped by state
- active tickets with AC checklist summaries
- recent pulse + evidence (last ~10)

## Deliverables
- [ ] Consulted artifacts list (paths + 1-line takeaways).
- [ ] Match classification + UPDATE/NEW decision with rationale.
- [ ] In WORK: ledger updated + status snapshot updated.

## Invariants
- Prefer UPDATE over NEW when overlap exists.
- A ticket is DONE only when:
  - all AC are checked
  - evidence exists
  - traceability links exist

## Anti-patterns
- Creating new tickets/specs without reading the canon.
- Using titles/slugs as identity instead of stable IDs.
- Rewriting `pulse.log` history.
- Marking DONE without checked AC + evidence + traceability.

## References
- [Ledger schema](references/ledger-schema.md)
- [status.md template](references/status-template.md)
- [Dedup gate guide](references/dedup-gate.md)
- [State transitions](references/state-transitions.md)
