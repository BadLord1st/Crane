---
description: "Evaluate design from a UX perspective, assessing visual hierarchy, information architecture, emotional resonance, cognitive load, and overall quality with quantitative scoring, persona-based testing, and actionable feedback. Use when the user asks to review, critique, evaluate, or give feedback on a design or component."
agent: orchestrator
model: openai/gpt-5.4
temperature: 0.2
---

# Critique

**Target**: $ARGUMENTS

## Protocol

1) Load skills:

   ```bash
   skill "frontend-design"
   skill "critique"
   ```

2) Follow the skill's instructions on $ARGUMENTS.
