# Bounded contexts & ubiquitous language (quick reference)

Bounded Context is the unit of strategic design: each context owns a model and the same term can legitimately mean different things across contexts.

Heuristics to split contexts:
- Policy differences: rules/approvals/SLAs differ.
- Pace of change: high-churn vs stable areas.
- Different stakeholders and language.
- Data ownership: which context is the source of truth.

Ubiquitous language checklist:
- Terms (nouns): business concepts.
- Verbs: behaviors and commands.
- Rules: “must/never/only if”.
- Ambiguities: same word used differently.

Deliverable format:
- Glossary entry: Term, definition, synonyms, context, examples, forbidden uses.
