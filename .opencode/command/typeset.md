---
description: "Improves typography by fixing font choices, hierarchy, sizing, weight, and readability so text feels intentional. Use when the user mentions fonts, type, readability, text hierarchy, sizing looks off, or wants more polished, intentional typography."
agent: orchestrator
model: openai/gpt-5.4
temperature: 0.2
---

# Typeset

**Target**: $ARGUMENTS

## Protocol

1) Load skills:

   ```bash
   skill "frontend-design"
   skill "typeset"
   ```

2) Follow the skill's instructions on $ARGUMENTS.
