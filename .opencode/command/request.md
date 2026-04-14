---
description: "Request a feature/change: Canon (PM+Spec) → Project Frame → Ticket Set → Primary Spec → BDD AC → Plan → Verify (REVIEW by default)"
agent: orchestrator
model: openai/gpt-5.4
temperature: 0.2
---

# Request

## Input (verbatim)
$ARGUMENTS

## Phase 0.5) Intent Normalization Gate (MANDATORY)

Before Phase 1, you MUST produce a **Normalized Request** block and then use ONLY it downstream.

Output:
- Outcome (1 sentence; system-level; externally observable)
- In-scope (≤3 bullets)
- Out-of-scope / Non-goals (≤3 bullets)
- Constraints (optional; ≤2 bullets)
- Unknowns (≤3; only blockers)

## Protocol

### Phase 0) Mode

Unless the user explicitly switched to WORK, treat `/request` as **REVIEW**.

- REVIEW = **no repo mutations**. Output deliverables only (proposed diffs allowed).
- WORK = mutate repo. Create/update PM canon + spec canon + tests + code + evidence.

REVIEW delegation rules:
- Any repo exploration/search/reading SHOULD be delegated to `@explore` (read-only).
- `@general` MAY be used for synthesis/challenge/decomposition, but not as the primary source of repo facts.
- Role agents (`Architect`, `Designer`, `Programmer`, `QA`, `Security`) MAY be used in read-only advisory mode when their specialty is relevant.
- In REVIEW, all subagent output is advisory. Final decisions remain with Orchestrator.
- If no repo reads happened, say so explicitly and mark: `Repo visibility: none`.

### Phase 1) Canon Load + Dedup Gate (MANDATORY)

You are not allowed to create new tickets/specs until this phase is complete.

#### 1.1) Load PM canon (tasks-as-code)

Follow the `spec-pm-canon` skill.

PM canon has a single entrypoint and a ledger:

- `./status.md` (human-readable canonical snapshot; treated as **derived**)
- `./.pm/scopes/<scope>/` (ledger; treated as **source of truth**)

Default scope: `default`.

Minimum ledger files:
- `.pm/scopes/default/meta.env` (NEXT_TICKET_ID=…)
- `.pm/scopes/default/tickets.tsv`
- `.pm/scopes/default/criteria.tsv`
- `.pm/scopes/default/evidence.tsv`
- `.pm/scopes/default/pulse.log` (append-only)

In REVIEW:
- If any are missing, report that explicitly and include a **creation plan** (but do not create them).

In WORK:
- If missing, create the canon files from templates in this repo (see `docs/pm/`), then continue.

#### 1.2) Load spec canon

Spec canon lives under `docs/spec/`:

- `docs/spec/spec-index.md` (index; MUST exist)
- `docs/spec/traceability/traceability.md` (single trace map; MUST exist)
- relevant spec nodes under:
  - `docs/spec/features/`
  - `docs/spec/contracts/`
  - `docs/spec/system/`
  - `docs/spec/runbooks/`
  - `docs/spec/testing/`

In REVIEW:
- If missing, report and propose creation.
In WORK:
- Create missing index/traceability skeletons, then continue.

#### 1.3) Repo dedup search (broad, then narrow)

Perform a dedup search using:
- goal keywords + domain nouns (endpoints, entities, modules, feature flags, DB tables)
- existing ticket titles
- spec node titles
- SPEC-* IDs (traceability map)

Output a short, explicit block:

- **Consulted artifacts** (≤12): list file paths you actually read (PM canon + spec canon + any key code/tests).
- **Potential matches** (≤6 total):
  - Strong match: likely duplicate → prefer UPDATE
  - Weak match: possible overlap → propose merge strategy
  - No match: NEW allowed

Hard rule:
- If any strong match exists, default to **UPDATE** unless you prove non-overlap in scope/outcome/contract.

### Phase 2) Project Framing + Fan-out Gate (MANDATORY)

Before creating tickets, produce a **Project Frame** using the Normalized Request and dedup results.

Output:
- Project meaning (1–2 sentences; what behavior/capability changes as a whole)
- Workstreams (1–5; grouped by meaningful slices, not by repo folders alone)
- Ticket strategy:
  - single ticket
  - multiple tickets
  - update existing + add new
- Primary spec strategy:
  - update existing primary spec
  - create one new primary spec
  - update primary spec + targeted existing nodes
- Parallel lanes (which can run concurrently without conflicting)

Hard rules:
- Do NOT decompose into multiple tickets unless the workstreams have distinct scope, ownership, risk, or verification surfaces.
- Do NOT keep one ticket if multiple independently verifiable workstreams clearly exist.
- Spec is organized by **project meaning / behavioral contract**, not by ticket count.

### Phase 3) Ticket Set (NEW / UPDATE / MIXED)

Use the **Outcome** and **Non-goals** from the Normalized Request.
Do not pull wording from Input verbatim.

#### 3.1) Choose scope + ID strategy

- Scope: `default` unless a specific subsystem warrants a new scope.
- ID format: `T-0001`, `T-0002`, …

In WORK:
- Allocate IDs from `.pm/scopes/<scope>/meta.env` (or compute from `tickets.tsv` if meta is missing, then repair meta).

In REVIEW:
- Use provisional IDs (`T-????`) when safe allocation is not possible.

#### 3.2) Ticket Set output (copy/paste friendly)

Produce:
1. A short **Ticket Set Summary**
2. Then 1..N tickets in the following format

```md
# Ticket: <T-XXXX> - <short outcome>

## Goal
<what this slice must achieve>

## Non-Goals
- <what this slice does NOT do>

## Acceptance Intent (high-level)
- [ ] <externally observable behavior>

## Depends on
- <ticket ids or "none">

## Constraints
- runtime:
- versions:
- performance:
- security:

## Artifacts
- primary spec: <semantic spec path>
- trace: docs/spec/traceability/traceability.md
- tests: <path>
- code: <path>
- pm: .pm/scopes/<scope>/*

## Plan / TODO
1) …

## Risks / Unknowns
- …
````

In WORK:

* Write/update the ledger:

  * append/update row(s) in `tickets.tsv`
  * add AC rows to `criteria.tsv` (initially unchecked)
  * update `status.md` snapshot (see Phase 7)

In REVIEW:

* Provide a concise proposed diff (what rows would be added/changed).

### Phase 4) Delegation Strategy (parallel-safe; staged)

Use concurrency only for independent, read-safe work.

Allowed early parallel lanes:

* `Architect` → project-level behavioral contract / primary spec
* `Designer` → UI/VAC/design contract when UI or visuals are in scope
* `@explore` → repo-grounded fact finding
* `@general` → synthesis/challenge/decomposition
* `Security` → early advisory only if attack surface or compliance risk is obvious

Barrier A:

* Do NOT start implementation until the Project Frame and primary spec direction are stable.

Allowed mid-stage parallel lanes:

* `Programmer` → implementation by ticket slice
* `Security` → targeted threat/observability/perf review if relevant

Barrier B:

* Do NOT declare done until integrated verification is complete.

Final lane:

* `QA` → slice verification + integrated verification with evidence

Hard rules:

* Subagent outputs are advisory until Orchestrator consolidates them.
* Only Orchestrator may finalize Ticket/spec/canon decisions.
* Parallelism must never create competing final specs or conflicting canon writes.

### Phase 5) Skill discovery (lazy-load; enforce the right defaults)

Discover broadly, load narrowly.

Always load for any non-trivial work:

* `spec-pm-canon`
* `spec-recon`
* `spec-traceability`
* `spec-specification`
* `spec-bdd`
* `spec-gap-analysis`

Conditionally:

* Regression risk high → `spec-audit`
* Tests needed → `spec-tdd`, `spec-verify`
* Env unclear → `spec-env-scout`
* Dependencies/versions matter → `spec-deps`
* DB changes → `spec-migrations`
* Security surface expands → `spec-threat-model`
* Observability/release → `spec-observability`, `spec-release`
* Meaningful failure occurred → `spec-error-kb`

### Phase 6) Specification (Architect)

Produce or update exactly **one primary spec node** for the project meaning.

You MAY also update existing secondary spec nodes if required for correctness, but do NOT create parallel duplicates.

Requirements:

* testable
* implementation-agnostic
* explicit about edge cases & failure modes
* stable IDs: `SPEC-<AREA>-NNN`
* acceptance criteria: `AC-1`, `AC-2`, … and each AC references ≥1 SPEC-*
* include at least one Mermaid `stateDiagram-v2` if behavior changes
* prefer semantic filenames; do NOT key spec filenames to ticket IDs

Minimum structure:

```md
# <Project / Capability / Change>

Links:
- [[spec-index]]
- Tickets: T-XXXX, T-YYYY

## Overview

## Scope / Non-goals

## User-visible behavior

## Inputs / Outputs

## Edge cases

## Errors & failure modes

## Compatibility / migration notes

## Acceptance Criteria (BDD-ready)
```

In WORK:

* Update `docs/spec/spec-index.md` to link the primary node.
* Update `docs/spec/traceability/traceability.md` with SPEC → AC mapping.

### Phase 7) BDD-ready Acceptance Criteria + QA plan

From the primary spec:

* Produce scenarios: `SC-1`, `SC-2`, … for each AC (happy + failure + essential edge)
* For multiple tickets, indicate which scenarios are:

  * slice-local
  * cross-slice / integrated
* Add a short verification plan:

  * layers to test
  * regression zones
  * minimal commands to verify

In WORK:

* Implement the smallest correct changes + tests.
* Update traceability with:

  * SPEC-* → AC-* → SC-* → ticket(s) → test path(s) → code path(s)

### Phase 8) PM canon maintenance (WORK only)

Maintain PM canon using `spec-pm-canon` and the reference formats in `docs/pm/`.

Hard requirements:

* Keep the ledger authoritative (`tickets.tsv`, `criteria.tsv`, `evidence.tsv`, `pulse.log`).
* Keep `status.md` consistent with the ledger.
* Never mark DONE without checked AC + evidence + traceability.
* For multi-ticket work, mark tickets independently; do not collapse partial progress into fake DONE.

### Phase 9) Non-Negotiable Deliverables

Always:

* [ ] Canon Load + Dedup Gate results
* [ ] Project Frame (meaning + workstreams + ticket strategy + spec strategy)
* [ ] Ticket Set with explicit scope + acceptance intent
* [ ] Primary spec (or spec update) aligned with the project meaning
* [ ] BDD-ready acceptance criteria + scenario list
* [ ] Verification plan (commands + regression zones)

Formatting / size budget:

* Project Frame: ≤40 lines
* Each Ticket: ≤50 lines
* Primary spec node: ≤180 lines
* BDD scenarios: ≤24 total
* Verification plan: ≤15 lines

In WORK:

* [ ] PM canon updated (ledger + status snapshot)
* [ ] Tests + implementation + evidence
* [ ] Traceability map updated
* [ ] If a meaningful mistake occurred → update `docs/agent/error-kb.md`

## Mindset

* Prefer UPDATE over NEW when overlap exists
* Prefer one primary spec per project meaning over many ticket-shaped specs
* Parallelize research; serialize final decisions
* If you can't test it, you don't understand it
* PM canon is the single source of task truth
* Spec canon is the single source of behavioral truth
