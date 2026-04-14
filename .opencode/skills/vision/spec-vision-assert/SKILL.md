---
name: spec-vision-assert
description: Validate an image against Visual Acceptance Criteria (VAC). Produces pass/fail evidence per criterion.
metadata:
  signature: "spec-vision-assert :: (Image, VAC) -> VisualCheckReport"
---

## When to use
- You have explicit visual expectations ("this screen must show X").
- You want screenshot-based verification for UI or docs.
- You want a lightweight alternative to pixel-perfect diff.

## Inputs
- Image (current state screenshot).
- Visual Acceptance Criteria (VAC) list.

## Outputs
A **Visual Check Report**:
- VAC results: PASS/FAIL/UNKNOWN per item
- Evidence: what part of the image supports the judgment
- Notes: ambiguities or missing context
- Suggested follow-ups if FAIL/UNKNOWN

## Protocol
1) Parse the VAC list into atomic checks.
2) For each VAC item, decide:
   - PASS: clearly satisfied
   - FAIL: clearly contradicted
   - UNKNOWN: not determinable from the image
3) For each decision, include a brief evidence note (what you see and where).
4) Summarize overall status and list the top blockers.
5) If VAC items are vague, rewrite them into deterministic versions and flag the rewrite as a suggestion.

## Deliverables
- [ ] Table-like report with one row per VAC item
- [ ] Evidence notes for each row

## Anti-patterns
- Passing checks based on "probably".
- Failing without pointing to what contradicts it.
- Treating subjective aesthetics as requirements.

## References
- [Visual Check Report Template](references/report_template.md)
