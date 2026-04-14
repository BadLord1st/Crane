---
description: "Improve unclear UX copy, error messages, microcopy, labels, and instructions to make interfaces easier to understand. Use when the user mentions confusing text, unclear labels, bad error messages, hard-to-follow instructions, or wanting better UX writing."
agent: orchestrator
model: openai/gpt-5.4
temperature: 0.2
---

# Clarify

**Target**: $ARGUMENTS

## Protocol

1) Load skills:

   ```bash
   skill "frontend-design"
   skill "clarify"
   ```

2) Follow the skill's instructions on $ARGUMENTS.
