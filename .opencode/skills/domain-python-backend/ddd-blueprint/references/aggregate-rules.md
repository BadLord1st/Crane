# Aggregate design rules (quick reference)

Aggregate = consistency boundary. An aggregate root protects invariants.

Rules of thumb:
- Keep aggregates small; prefer references (IDs) over deep object graphs.
- Only one aggregate root per transaction.
- Invariants must be enforceable inside the aggregate boundary.
- Cross-aggregate consistency: prefer eventual consistency via domain events.

Deliverable format per aggregate:
- Name + Aggregate Root
- Entities / Value Objects
- Invariants
- Commands (methods) and validation
- Domain events emitted
- Repository interface boundaries
