---
description: "Run technical quality checks across accessibility, performance, theming, responsive design, and anti-patterns. Generates a scored report with P0-P3 severity ratings and actionable plan. Use when the user wants an accessibility check, performance audit, or technical quality review."
agent: orchestrator
model: openai/gpt-5.4
temperature: 0.2
---

# Audit

**Target**: $ARGUMENTS

## Protocol

1) Load skills:

   ```bash
   skill "frontend-design"
   skill "audit"
   ```

2) Follow the skill's instructions on $ARGUMENTS.
