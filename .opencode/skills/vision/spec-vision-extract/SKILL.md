---
name: spec-vision-extract
description: Extract structured information from an image (error text, UI fields, tables, chart values) and return it in a usable schema.
metadata:
  signature: "spec-vision-extract :: Image -> ExtractedArtifacts"
---

## When to use
- You have a screenshot of an error message, stack trace, UI form, table, or dashboard.
- You need clean structured data to drive debugging or spec writing.

## Inputs
- One image.
- Optional extraction goal (examples):
  - "extract error message and context"
  - "extract table as rows"
  - "extract visible form fields and their values"

## Outputs
An **ExtractedArtifacts** object:
- `kind`: ui | terminal | table | chart | diagram | document
- `text_blocks`: list of visible text (preserve line breaks where relevant)
- `entities`: key-value facts (status code, route, user id, version, timestamp)
- `ui_elements`: list of visible controls (button, input, checkbox, toast)
- `tables`: list of rows/columns when present
- `notes`: ambiguity and missing pieces

## Protocol
1) Identify the primary `kind`.
2) Extract all clearly visible text blocks (do not hallucinate missing lines).
3) Normalize obvious entities (HTTP status, error codes, file names, timestamps).
4) If a table is present, extract headers and rows.
5) If a chart is present, extract axis labels and visible values (approximate only if explicit).
6) Return both the structured output and a short human summary.

## Deliverables
- [ ] Structured extraction in the schema described above
- [ ] Short summary of what matters for debugging/spec

## Anti-patterns
- Making up hidden text.
- Converting blurry text into confident claims.
- Mixing interpretation into extraction (keep interpretation for inspect/assert skills).

## References
- [Vision Outputs Schema](references/schema.md)
