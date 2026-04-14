# Writing Principles

## Be thorough where it matters

The README exists to prevent wasted time. Prioritize:
- setup steps that fail often,
- non-obvious prerequisites,
- the real commands to run/test/build,
- where configuration and secrets come from.

## Be honest

- If a piece is unknown or not represented in the repo, say so.
- Prefer: "This repo expects X (found in `path/to/file`)" over "Typically you should...".

## Keep the doc maintainable

- If content belongs in runbooks/spec nodes, link to them.
- Avoid duplicating long operational procedures in README.
- Add a table of contents only when it helps navigation.

## Ask only when critical

Good questions are blockers:
- "What does this project do?" (if it cannot be inferred)
- "Which external services are required in dev?"
- "Where do prod secrets come from?"

Avoid preference questions unless the user asked for it.
