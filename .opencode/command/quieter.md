---
description: "Tones down visually aggressive or overstimulating designs, reducing intensity while preserving quality. Use when the user mentions too bold, too loud, overwhelming, aggressive, garish, or wants a calmer, more refined aesthetic."
agent: orchestrator
model: openai/gpt-5.4
temperature: 0.2
---

# Quieter

**Target**: $ARGUMENTS

## Protocol

1) Load skills:

   ```bash
   skill "frontend-design"
   skill "quieter"
   ```

2) Follow the skill's instructions on $ARGUMENTS.
