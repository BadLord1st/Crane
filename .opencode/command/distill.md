---
description: "Strip designs to their essence by removing unnecessary complexity. Great design is simple, powerful, and clean. Use when the user asks to simplify, declutter, reduce noise, remove elements, or make a UI cleaner and more focused."
agent: orchestrator
model: openai/gpt-5.4
temperature: 0.2
---

# Distill

**Target**: $ARGUMENTS

## Protocol

1) Load skills:

   ```bash
   skill "frontend-design"
   skill "distill"
   ```

2) Follow the skill's instructions on $ARGUMENTS.
