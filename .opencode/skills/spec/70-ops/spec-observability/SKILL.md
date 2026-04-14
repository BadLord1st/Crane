---
name: spec-observability
description: "Define observability requirements as a node: logs/metrics/traces that prove the system behavior."
metadata:
  signature: "spec-observability :: SpecNode -> ObservabilityNode"
---

## When to use
- The change is operationally risky or needs SLO visibility.
- Incidents are hard to debug without signals.

## Inputs
- SpecNode + failure modes.
- Current logging/metrics setup (if any).

## Outputs
- Observability node: required events, fields, metrics, alerts/slo notes.

## Protocol
1) For each failure mode, define the signal needed to detect it.
2) Specify log fields (correlation IDs, user/tenant, request IDs) as contracts.
3) Define minimal metrics (latency, error rate, queue depth, etc.).

## Deliverables
- [ ] Observability node linked to the feature/system spec.
- [ ] Concrete log/metric definitions.

## Anti-patterns
- “Add more logs” without defining what and why.
- Collecting sensitive data in logs.

## References
- [Wide Events Logging](./references/wide-events.md)
- [Instrumentation Examples](./references/instrumentation.md)
