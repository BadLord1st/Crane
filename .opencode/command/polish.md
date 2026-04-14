---
description: "Performs a final quality pass fixing alignment, spacing, consistency, and micro-detail issues before shipping. Use when the user mentions polish, finishing touches, pre-launch review, something looks off, or wants to go from good to great."
agent: orchestrator
model: openai/gpt-5.4
temperature: 0.2
---

# Polish

**Target**: $ARGUMENTS

## Protocol

1) Load skills:

   ```bash
   skill "frontend-design"
   skill "polish"
   ```

2) Follow the skill's instructions on $ARGUMENTS.
