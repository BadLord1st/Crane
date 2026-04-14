---
description: "Designs and improves onboarding flows, empty states, and first-run experiences to help users reach value quickly. Use when the user mentions onboarding, first-time users, empty states, activation, getting started, or new user flows."
agent: orchestrator
model: openai/gpt-5.4
temperature: 0.2
---

# Onboard

**Target**: $ARGUMENTS

## Protocol

1) Load skills:

   ```bash
   skill "frontend-design"
   skill "onboard"
   ```

2) Follow the skill's instructions on $ARGUMENTS.
