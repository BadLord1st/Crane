---
name: spec-vision-inspect
description: Inspect an image/screenshot/diagram and produce a structured observation report (what is visible, what looks wrong, and what to check next).
metadata:
  signature: "spec-vision-inspect :: Image -> VisualObservationReport"
---

## When to use
- A user provides a screenshot instead of a textual report.
- You need a grounded read of a UI state, terminal output, diagram, dashboard, or document screenshot.
- You want to turn a visual artifact into actionable next steps and hypotheses.

## Inputs
- One image (screenshot, photo, diagram, dashboard, or document capture).
- Optional context (1–3 sentences): expected vs actual, where the image came from.

## Outputs
A **Visual Observation Report** with:
- Summary: what this image appears to be
- Observations: key visible elements (with rough locations)
- Anomalies: what looks wrong or suspicious
- Hypotheses: 3–7 ranked likely causes (with evidence)
- Next checks: cheapest validations first
- Confidence and unknowns

## Protocol
1) Classify the image type: UI / terminal / diagram / dashboard / document.
2) Describe what is visible in 1–2 sentences (no theories yet).
3) List key elements and labels, using location anchors (top-left, center, right panel, footer).
4) Identify anomalies or mismatches versus the provided context.
5) Produce 3–7 hypotheses ranked by likelihood. Each hypothesis must cite at least one visual clue.
6) Provide next checks in order of cheapest signal first (log line, config, route, API call, CSS class, feature flag).
7) If this should become a spec requirement, propose **Visual Acceptance Criteria (VAC)** candidates.

## Deliverables
- [ ] One Visual Observation Report using the template in `references/report_template.md`

## Anti-patterns
- Inventing UI elements or text you cannot see.
- Claiming certainty without evidence.
- Jumping straight to code changes without a validation step.
- Suggesting expensive investigations before cheap checks.

## References

- [Visual Acceptance Criteria (VAC) Node Template](references/vac_template.md)
- [Visual Observation Report Template](references/report_template.md)
