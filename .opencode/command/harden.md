---
description: "Improve interface resilience through better error handling, i18n support, text overflow handling, and edge case management. Makes interfaces robust and production-ready. Use when the user asks to harden, make production-ready, handle edge cases, add error states, or fix overflow and i18n issues."
agent: orchestrator
model: openai/gpt-5.4
temperature: 0.2
---

# Harden

**Target**: $ARGUMENTS

## Protocol

1) Load skills:

   ```bash
   skill "frontend-design"
   skill "harden"
   ```

2) Follow the skill's instructions on $ARGUMENTS.
