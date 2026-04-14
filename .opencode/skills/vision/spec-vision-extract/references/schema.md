# ExtractedArtifacts Schema

```json
{
  "kind": "ui | terminal | table | chart | diagram | document",
  "text_blocks": ["string"],
  "entities": {"key": "value"},
  "ui_elements": [
    {"type": "button|input|select|checkbox|toast|modal|link|badge", "label": "...", "state": "enabled|disabled|unknown", "location": "..."}
  ],
  "tables": [
    {
      "title": "optional",
      "columns": ["..."],
      "rows": [["..."]]
    }
  ],
  "notes": ["..." ]
}
```
