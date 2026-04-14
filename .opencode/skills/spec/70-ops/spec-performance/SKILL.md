---
name: spec-performance
description: Define performance budgets and verification checks as spec artifacts (not folklore).
metadata:
  signature: "spec-performance :: SpecNode -> PerfBudget"
---

## When to use
- The ticket has latency/throughput constraints.
- You suspect performance regressions.

## Inputs
- SpecNode + constraints.
- Current perf characteristics (if known).

## Outputs
- Perf budget + validation commands/checks (smoke-level).

## Protocol
1) Define measurable budgets (p95 latency, memory ceiling, etc.).
2) Identify the simplest repeatable check (benchmark, load test, timing harness).
3) Capture the command and the pass/fail threshold.

## Deliverables
- [ ] Performance budget documented.
- [ ] A reproducible validation check.

## Anti-patterns
- Vague “should be fast”.
- Optimizing without measurement.

## References
- [Optimization Patterns](references/optimization-patterns.md)
- [Testing Strategies](references/testing-strategies.md)
