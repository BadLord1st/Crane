---
description: "Add strategic color to features that are too monochromatic or lack visual interest, making interfaces more engaging and expressive. Use when the user mentions the design looking gray, dull, lacking warmth, needing more color, or wanting a more vibrant or expressive palette."
agent: orchestrator
model: openai/gpt-5.4
temperature: 0.2
---

# Colorize

**Target**: $ARGUMENTS

## Protocol

1) Load skills:

   ```bash
   skill "frontend-design"
   skill "colorize"
   ```

2) Follow the skill's instructions on $ARGUMENTS.
