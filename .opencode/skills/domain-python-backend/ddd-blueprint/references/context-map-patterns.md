# Context map patterns (quick reference)

A context map makes boundaries and relationships explicit.

Common relationship axes:
- Upstream/downstream (who changes first).
- Conformist vs customer/supplier.
- Translation need: Anti-Corruption Layer (ACL) vs shared kernel.

Integration styles to annotate on the map:
- Synchronous API
- Events (pub/sub)
- Data replication (careful: can blur boundaries)

Deliverable format:
- For each edge: direction, integration style, contract, ownership, failure modes.
