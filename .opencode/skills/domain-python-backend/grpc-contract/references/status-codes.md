# references/status-codes.md

- Use standard gRPC status codes; reserve application-specific detail for message/details (optional).
- Maintain a stable mapping: domain failure → StatusCode (+ message).
- Typical mapping examples:
  - Validation → INVALID_ARGUMENT
  - Missing entity → NOT_FOUND
  - Conflict on create → ALREADY_EXISTS
  - Authn/authz → UNAUTHENTICATED / PERMISSION_DENIED
  - Rate/size limits → RESOURCE_EXHAUSTED
  - Dependency outage → UNAVAILABLE
- Ban “UNKNOWN for everything”; it destroys client behavior predictability.
