# Routing + Guards (Litestar)

## Goals
- Route grouping by feature.
- Consistent application of cross-cutting concerns (auth, DI) via hierarchy.

## Recommended topology
- Use Router for feature-level grouping and shared dependencies.
- Use Controller for cohesive handler bundles under a path prefix.
- Prefer nested Routers to isolate “public vs write/admin” surfaces.

## Guard placement rule
Pick the *highest* layer that matches scope:
- handler: single endpoint rule
- controller: all endpoints in the controller
- router: all endpoints in a feature group
- app: global policy

Document *why* the chosen layer is correct (blast radius, reuse, clarity).

## Notes
- Guards are lists and can be composed.
- Don’t leak auth into business logic; keep handlers narrow and testable.
