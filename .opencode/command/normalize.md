---
description: "Audits and realigns UI to match design system standards, spacing, tokens, and patterns. Use when the user mentions consistency, design drift, mismatched styles, tokens, or wants to bring a feature back in line with the system."
agent: orchestrator
model: openai/gpt-5.4
temperature: 0.2
---

# Normalize

**Target**: $ARGUMENTS

## Protocol

1) Load skills:

   ```bash
   skill "frontend-design"
   skill "normalize"
   ```

2) Follow the skill's instructions on $ARGUMENTS.
