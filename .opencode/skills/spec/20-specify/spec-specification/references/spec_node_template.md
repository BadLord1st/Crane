# Spec Node Template

Use this as a starting point for any node in `docs/spec/**`.

```md
# <Title>

Links:
- [[spec-index]]
- [[runbooks/local-dev]]
- [[testing/test-plan]]

Spec Namespace: SPEC-<AREA>
Status: Draft | Proposed | Accepted | Deprecated
Version: 1.0
Owners:
Last Updated: YYYY-MM-DD

## Overview

## User-visible behavior

## Inputs / Outputs

## Diagrams (Mermaid)

### State machine (REQUIRED per change)
```mermaid
stateDiagram-v2
  direction LR
  [*] --> ST_IDLE

  state "Idle" as ST_IDLE
  state "Pending approval" as ST_PENDING
  state "Active" as ST_ACTIVE
  state "Failed" as ST_FAILED

  ST_IDLE --> ST_PENDING: "submit()"
  ST_PENDING --> ST_ACTIVE: "approve()"
  ST_PENDING --> ST_FAILED: "reject()"
  ST_ACTIVE --> ST_IDLE: "reset()"
  ST_FAILED --> ST_IDLE: "reset()"
```

### Sequence diagram (recommended for cross-component behavior)
```mermaid
sequenceDiagram
  autonumber
  participant U as "User"
  participant UI as "Client/UI"
  participant API as "API"
  participant SVC as "Service"
  participant DB as "DB"

  U->>UI: "Click 'Submit'"
  UI->>API: "POST /v1/items"
  API->>SVC: "CreateItem(cmd)"
  SVC->>DB: "INSERT item"
  DB-->>SVC: "ok"
  SVC-->>API: "ItemCreated(id)"
  API-->>UI: "201 Created (id)"
  UI-->>U: "Shows success"
```

### Flowchart (activity-style branching / algorithm)
```mermaid
flowchart TD
  A["Start"] --> B{"Valid input?"}
  B -- "no" --> E["Return error"]
  B -- "yes" --> C["Compute result"]
  C --> D["Persist / emit event"]
  D --> F["Done"]
```

### Class diagram (types / domain model)
```mermaid
classDiagram
  class Item {
    +string id
    +string status
    +datetime createdAt
  }

  class CreateItemCommand {
    +string requestId
    +string payload
  }

  CreateItemCommand --> Item : "creates"
```

### ER diagram (persistence model)
```mermaid
erDiagram
  ITEM ||--o{ ITEM_EVENT : "emits"
  ITEM {
    string id
    string status
    datetime created_at
  }
  ITEM_EVENT {
    string id
    string item_id
    string type
    datetime created_at
  }
```

### Requirement diagram (traceability for SPEC/AC IDs)
```mermaid
requirementDiagram
  requirement SPEC_AREA_001 {
    id: "SPEC-<AREA>-001"
    text: "System shall validate inputs before processing."
    risk: "medium"
    verifymethod: "test"
  }

  requirement SPEC_AREA_002 {
    id: "SPEC-<AREA>-002"
    text: "System shall persist a durable record for each accepted request."
    risk: "high"
    verifymethod: "test"
  }

  SPEC_AREA_001 - derives -> SPEC_AREA_002
```
### ID conventions

- Requirements: `SPEC-<AREA>-NNN`
- Acceptance criteria inside a node: `AC-1`, `AC-2`, ...
- Scenario IDs (when BDD exists): `SC-1`, `SC-2`, ...

### Splitting

If a node grows too large:
- move contracts to `docs/spec/contracts/...`
- move run commands to `docs/spec/runbooks/...`
- link them from `Links:`
