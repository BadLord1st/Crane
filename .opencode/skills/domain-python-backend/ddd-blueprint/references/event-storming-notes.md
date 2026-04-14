# Event storming notes (quick reference)

Event storming is a collaborative technique to discover domain events, commands, and hotspots.

Minimal sequence:
1) List domain events (past tense) on the timeline.
2) Attach commands (imperative) that trigger events.
3) Identify aggregates / policies that decide.
4) Mark external systems and integration points.

Remote-friendly artifact:
- Timeline of events
- For each event: meaning, data, producer, consumers, invariants involved
