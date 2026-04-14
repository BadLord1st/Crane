---
name: ddd-blueprint
description: "Turn a ticket/spec into a Domain-Driven Design blueprint: bounded contexts, ubiquitous language, aggregates/invariants, and integration map. No code changes; produce a structured model pack."
metadata:
  signature: "ddd-blueprint :: Ticket -> DDDBlueprint"
---

## When to use
- You need DDD strategic/tactical design artifacts before implementing (or refactoring) a domain.
- The domain is large enough that you expect multiple models/teams (bounded contexts).
- You want explicit consistency boundaries (aggregates) and domain events.

## Inputs
- Ticket/spec (goal, constraints, success criteria).
- Domain language samples (terms, policies, sample scenarios) — can be raw notes.
- Current snapshot (optional): existing services/modules, DB schema, APIs.

## Outputs
- DDD Blueprint (structured):
  1) Ubiquitous Language glossary.
  2) Candidate bounded contexts + responsibilities.
  3) Context map (relationships + integration style).
  4) Aggregate catalog per context (aggregate root, entities, value objects, invariants).
  5) Domain events + commands (what triggers what).
  6) Layer boundaries (domain/application/infrastructure) + suggested folder/module split.
  7) Risks/unknowns + questions for domain experts.

## Protocol
1) Extract ubiquitous language: terms, verbs, business rules, “same word — different meaning” cases.
2) Identify bounded contexts: split by purpose, policy differences, and team ownership.
3) Sketch a context map: upstream/downstream, translation needs, shared kernel vs separate models.
4) For each context, define aggregates: consistency boundary, invariants, transactions, aggregate root.
5) Derive domain events and commands from key state changes and business milestones.
6) Map to code boundaries: domain model vs application orchestration vs infrastructure adapters.
7) Produce a minimal “implementation slice” plan (thin vertical slice through 1 context + 1 aggregate).

## Safety / Permissions
- Default: DESIGN-ONLY. Do not modify repo files, run migrations, or propose destructive commands.
- Treat secrets as toxic (do not request or output .env, tokens, keys).
- If the user asks to implement, first deliver the blueprint + a dry-run plan, then ask for explicit confirmation.

## Deliverables
- [ ] Glossary (ubiquitous language).
- [ ] Bounded contexts list with responsibilities.
- [ ] Context map with integration notes.
- [ ] Aggregates + invariants per context.
- [ ] Domain events + commands.
- [ ] Proposed module/folder split and boundary rules.
- [ ] Open questions + risks.

## Anti-patterns
- Starting from DB tables/CRUD endpoints instead of business rules.
- One “god context” / “one model to rule them all”.
- Aggregates that span many business capabilities or require cross-aggregate transactions.
- Mixing domain logic into controllers/handlers.

## Examples
1) Request: “We’re building subscriptions + invoicing. Define bounded contexts, aggregates, and events.”
   Expected output: 2–3 contexts (e.g., Billing, Subscription, Customer), context map, aggregate catalog (Subscription, Invoice), events (SubscriptionActivated, InvoiceIssued), invariants.

2) Request: “Refactor an existing monolith into modules. Use DDD boundaries; no code yet.”
   Expected output: candidate contexts + mapping to current modules, anti-corruption layer candidates, migration slice plan.

3) Request: “We keep breaking invariants around orders and stock. Propose aggregate boundaries.”
   Expected output: aggregate options (Order vs Order+Items; Inventory as separate context), explicit invariants, consistency strategy, event-driven integration notes.

## References
- [Bounded contexts & ubiquitous language](references/bounded-contexts.md)
- [Context mapping patterns](references/context-map-patterns.md)
- [Aggregate design rules](references/aggregate-rules.md)
- [Event storming notes](references/event-storming-notes.md)
- [Output template](references/output-template.md)
