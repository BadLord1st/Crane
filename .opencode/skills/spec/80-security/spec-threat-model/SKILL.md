---
name: spec-threat-model
description: "Threat modeling as a spec node: assets → threats → mitigations → security acceptance criteria."
metadata:
  signature: "spec-threat-model :: (SpecNode, Snapshot) -> ThreatModelNode"
---

## When to use
- Attack surface expands (new endpoint, auth flow, secrets, permissions).
- Handling sensitive data.

## Inputs
- SpecNode describing the change.
- Snapshot of current architecture/boundaries.

## Outputs
- Threat model node under docs/spec/system/ or docs/spec/contracts/ with explicit mitigations and ACs.

## Protocol
1) List assets and trust boundaries.
2) Enumerate threats (STRIDE-style is fine) that are relevant to the change.
3) Define mitigations that are implementable and testable.
4) Add security acceptance criteria to be turned into tests.

## Deliverables
- [ ] Threat model node with mitigations and ACs.
- [ ] Links back to the feature/contract nodes.

## Anti-patterns
- Boiling the ocean with irrelevant threats.
- Hand-wavy mitigations that can’t be verified.

## References
- [API Security Checklist](references/security-checklists.md)
- [Dependency Security Checklist](references/security-checklists.md)