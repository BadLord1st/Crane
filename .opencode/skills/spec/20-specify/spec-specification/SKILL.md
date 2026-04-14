---
name: spec-specification
description: Write/update a spec node in docs/spec using a graph-friendly structure (Obsidian links), stable IDs, and testable acceptance criteria.
metadata:
  signature: "spec-specification :: (Snapshot, Ticket) -> SpecNode"
---

## When to use
- A feature/change/bugfix needs a crisp contract for intended behavior.
- You want docs that are linkable, small, and test-driven.

## Inputs
- Snapshot (from spec-recon).
- Task Ticket: goal, constraints, acceptance intent.

## Outputs
- A single SpecNode file under docs/spec/** (feature/system/contract/runbook/testing).
- A link from docs/spec/spec-index.md to that node (or create index if missing).

## Protocol
1) Pick exactly **one** node to write/update (one change = one node).
2) Place the node under the correct folder: features/, system/, contracts/, runbooks/, testing/, traceability/.
3) Use stable requirement IDs: `SPEC-<AREA>-NNN` and keep them stable once published.
4) Put acceptance criteria in the node (AC-1, AC-2, …). Each AC must reference at least one SPEC-* ID.
5) Add a `Links:` block at the top (Obsidian graph): always include [[spec-index]].
6) If the topic grows >300 lines, split into additional nodes and link them.

## Diagram requirements (Mermaid)

### Mandatory per change
Every spec node that represents a behavioral change must include at least one **state machine** diagram (`stateDiagram-v2`).
- If the change touches multiple bounded contexts, include one state machine per context (or link to dedicated files under `docs/spec/system/state-machines/`).

### Strongly recommended
- **Sequence diagram** (`sequenceDiagram`) for any change that spans multiple actors/components (UI ↔ API ↔ worker ↔ DB).
- **Flowchart** (`flowchart`) when the change is primarily algorithmic / branching logic (acts as an “activity diagram” substitute).
- **Class diagram** (`classDiagram`) for domain model / type surfaces (public DTOs, core aggregates).
- **ER diagram** (`erDiagram`) for relational persistence shape (tables/entities + relationships).
- **Requirement diagram** (`requirementDiagram`) for traceability when requirements/ACs are non-trivial (IDs + relationships).

### Mermaid robustness rules (escaping / “don’t break rendering”)
Use these defaults to keep diagrams resilient across renderers and avoid parse surprises:
1) Use **ASCII identifiers** for technical IDs (`[A-Za-z0-9_]+`). Put human text in labels.
2) Quote any label that contains spaces or punctuation. Prefer explicit aliasing:
   - State: `state "Human label with spaces" as ST_FOO`
   - Flowchart node: `A["Human label"]`
3) Never use reserved-ish tokens (e.g., `end`) as raw node IDs. If unavoidable, keep it in a **label**, not an ID.
4) Prefer `stateDiagram-v2` over v1 for state machines.
5) Keep one diagram per concern; link out instead of building a single unreadable mega-diagram.

## Mermaid diagram catalog (supported diagram types)

This skill only uses Mermaid diagram types that are documented by Mermaid itself:
`flowchart`, `sequenceDiagram`, `classDiagram`, `stateDiagram-v2`, `erDiagram`, `journey`, `gantt`, `pie`,
`quadrantChart`, `requirementDiagram`, `gitGraph`, `C4Context`, `C4Container`, `C4Component`, `C4Dynamic`, `C4Deployment`, `mindmap`, `timeline`, `zenuml`, `sankey`,
`xychart`, `block`, `packet`, `kanban`, `architecture-beta`, `radar-beta`, `treemap-beta`.

For copy/paste-ready examples, see:
- [Mermaid Catalog / Gallery](references/mermaid_catalog.md)
- [Mermaid Safety / Escaping Guide](references/mermaid_safety.md)


## Deliverables
- [ ] Spec node created/updated (one node only).
- [ ] Spec index updated with a link to the node.
- [ ] Acceptance criteria suitable for BDD/tests.

## Anti-patterns
- Creating a mega-spec that contains architecture + runbooks + testplan + everything.
- Vague requirements without acceptance criteria.
- Changing semantics in code without updating the spec node.

## References
- [Recommended File Structure](references/file_structure.md)
- [Spec Node Template](references/spec_node_template.md)
- [Spec Graph Structure](references/spec_graph.md)
- [Spec Writing Tutorial: Acceptance Criteria](references/tutorials.md)
