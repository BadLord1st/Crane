---
name: spec-vision-diff
description: Compare two images (before/after) and report meaningful semantic differences for UI and diagrams.
metadata:
  signature: "spec-vision-diff :: (ImageBefore, ImageAfter) -> VisualDiffReport"
---

## When to use
- You have "before" and "after" screenshots and want a semantic diff.
- You want to summarize changes without pixel-level tooling.
- You want regression detection (missing element, changed state, new error).

## Inputs
- Image A (before)
- Image B (after)
- Optional focus ("layout", "text", "errors", "component presence")

## Outputs
A **Visual Diff Report**:
- Added / Removed elements
- Text changes
- State changes (enabled/disabled, loading, selected)
- Error/alert changes
- Likely cause hints (optional, clearly marked as hypotheses)

## Protocol
1) Describe Image A and Image B briefly.
2) List differences grouped by type: structure, text, state, errors.
3) Highlight high-impact changes first (missing button, new error, blank screen).
4) If differences imply a regression, point to likely modules to inspect.
5) Keep it semantic. Do not pretend this is pixel-perfect.

## Deliverables
- [ ] Structured diff with grouped changes
- [ ] One-paragraph summary of the most important regressions

## Anti-patterns
- Over-indexing on tiny visual noise.
- Making causal claims without evidence.

## References
- [Visual Regression Testing Best Practices](https://example.com/visual-regression-best-practices)
