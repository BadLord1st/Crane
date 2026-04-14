# Dependency Injection Contract (Litestar)

## DI invariants (write them into the spec)
- Dependency keys must match kwarg names on handlers.
- Providers must be callables and declared using Provide.
- Dependencies are scoped to where they are declared (app/router/controller/handler).

## What to specify
For each dependency:
- key (string) and type (informational)
- provider function/class
- scope (app/router/controller/handler)
- caching (enabled/disabled) and lifetime expectations
- sync vs async and whether it is safe to run in-thread

## Testing posture
- Handlers should remain callable with injected fakes (pure-ish functions).
- Unit tests validate provider wiring and boundary behavior.
