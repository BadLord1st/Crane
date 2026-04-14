# Protobuf Compat

Wire-safe defaults:
- Adding new fields is safe; old clients ignore unknown fields.
- Removing fields is safe ONLY if you never reuse the field number again.
- Always `reserved` removed field numbers (and often names) to prevent accidental reuse.
- Prefer additive evolution; if you need breaking changes, bump package version (v1 → v2) and run them side-by-side.
Notes:
- Be cautious with `oneof` evolution; tag reuse can lead to ambiguity across versions.
