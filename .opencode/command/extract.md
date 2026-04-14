---
description: "Extract and consolidate reusable components, design tokens, and patterns into your design system. Identifies opportunities for systematic reuse and enriches your component library. Use when the user asks to create components, refactor repeated UI patterns, build a design system, or extract tokens."
agent: orchestrator
model: openai/gpt-5.4
temperature: 0.2
---

# Extract

**Target**: $ARGUMENTS

## Protocol

1) Load skills:

   ```bash
   skill "frontend-design"
   skill "extract"
   ```

2) Follow the skill's instructions on $ARGUMENTS.
