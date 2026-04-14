# Mermaid Catalog (all diagram types used by this spec system)

This catalog is intentionally “copy/paste first”: every example uses ASCII IDs and quoted labels to reduce renderer breakage.

> Note: Some diagram types are marked as experimental/beta in Mermaid docs (e.g., `architecture-beta`, `radar-beta`, `treemap-beta`). Ensure your renderer supports Mermaid v11+ if you use them.

## Flowchart

```mermaid
flowchart LR
  A["Start"] --> B{"Decision?"}
  B -- "path A" --> C["Do A"]
  B -- "path B" --> D["Do B"]
  C --> E["End"]
  D --> E
```

## Sequence

```mermaid
sequenceDiagram
  autonumber
  participant A as Actor A
  participant B as Actor B
  A->>B: Request
  B-->>A: Response
```

## Class

```mermaid
classDiagram
  class User {
    +string id
    +string email
  }
  class Session {
    +string id
    +datetime expiresAt
  }
  User "1" --> "0..*" Session : has
```

## State machine

```mermaid
stateDiagram-v2
  direction LR
  [*] --> ST_INIT
  state "Init" as ST_INIT
  state "Running" as ST_RUN
  state "Stopped" as ST_STOP
  ST_INIT --> ST_RUN: start()
  ST_RUN --> ST_STOP: stop()
  ST_STOP --> ST_RUN: start()
```

## ER

```mermaid
erDiagram
  USER ||--o{ SESSION : "has"
  USER {
    string id
    string email
  }
  SESSION {
    string id
    string user_id
    datetime expires_at
  }
```

## User Journey

```mermaid
journey
  title Checkout flow
  section Browse
    Find item: 5: User
    Open item page: 4: User
  section Pay
    Enter details: 3: User
    Confirm: 4: User
```

## Gantt

```mermaid
gantt
  title Release plan
  dateFormat  DD-MM-YYYY
  section Spec
    Write spec: a1, 01-02-2026, 2d
  section Build
    Implement: a2, after a1, 5d
  section Ship
    Release: a3, after a2, 1d
```

## Pie

```mermaid
pie
  title Work split
  "Spec": 20
  "Implementation": 60
  "Tests": 20
```

## Quadrant chart

```mermaid
quadrantChart
  title "Impact vs Effort"
  x-axis "Low effort" --> "High effort"
  y-axis "Low impact" --> "High impact"
  quadrant-1 "Quick wins"
  quadrant-2 "Big bets"
  quadrant-3 "Low value"
  quadrant-4 "Traps"
  "Change A": [0.2, 0.8]
  "Change B": [0.7, 0.7]
```

## Requirement diagram

```mermaid
requirementDiagram
  requirement REQ_001 {
    id: "SPEC-EXAMPLE-001"
    text: "System shall accept requests with valid payload."
    risk: medium
    verifymethod: test
  }

  requirement REQ_002 {
    id: "SPEC-EXAMPLE-002"
    text: "System shall reject invalid payloads with a structured error."
    risk: high
    verifymethod: test
  }

  REQ_001 - refines -> REQ_002
```

## GitGraph

```mermaid
gitGraph
  commit id:"c1"
  branch feature
  checkout feature
  commit id:"c2"
  checkout main
  merge feature
  commit id:"c3"
```

## C4 (all supported entrypoints)

> C4 support in Mermaid is marked experimental and is PlantUML-compatible. Keep diagrams minimal and version-check your renderer.

### C4Context

```mermaid
C4Context
  title System Context
  Person(user, "User", "Primary user")
  System(app, "App", "Delivers the feature")
  System_Ext(idp, "Identity Provider", "External auth")
  Rel(user, app, "Uses")
  Rel(app, idp, "Authenticates with")
```

### C4Container

```mermaid
C4Container
  title Container diagram
  Person(user, "User", "Primary user")
  System_Boundary(app, "App") {
    Container(web, "Web", "Browser", "UI")
    Container(api, "API", "HTTP", "Backend API")
    ContainerDb(db, "DB", "SQL", "Relational store")
  }
  Rel(user, web, "Uses")
  Rel(web, api, "Calls")
  Rel(api, db, "Reads/Writes")
```

### C4Component

```mermaid
C4Component
  title Component diagram
  Container_Boundary(api, "API") {
    Component(ctrl, "Controller", "HTTP", "Routes requests")
    Component(svc, "Service", "Code", "Business logic")
    ComponentDb(repo, "Repository", "SQL", "Persistence adapter")
  }
  Rel(ctrl, svc, "Calls")
  Rel(svc, repo, "Uses")
```

### C4Dynamic

```mermaid
C4Dynamic
  title Dynamic diagram for Internet Banking System - API Application

  ContainerDb(c4, "Database", "Relational Database Schema", "Stores user registration information, hashed authentication credentials, access logs, etc.")
  Container(c1, "Single-Page Application", "JavaScript and Angular", "Provides all of the Internet banking functionality to customers via their web browser.")
  Container_Boundary(b, "API Application") {
    Component(c3, "Security Component", "Spring Bean", "Provides functionality Related to signing in, changing passwords, etc.")
    Component(c2, "Sign In Controller", "Spring MVC Rest Controller", "Allows users to sign in to the Internet Banking System.")
  }
  Rel(c1, c2, "Submits credentials to", "JSON/HTTPS")
  Rel(c2, c3, "Calls isAuthenticated() on")
  Rel(c3, c4, "select * from users where username = ?", "JDBC")

  UpdateRelStyle(c1, c2, $textColor="red", $offsetY="-40")
  UpdateRelStyle(c2, c3, $textColor="red", $offsetX="-40", $offsetY="60")
  UpdateRelStyle(c3, c4, $textColor="red", $offsetY="-40", $offsetX="10")


```

### C4Deployment

```mermaid
C4Deployment
  title Deployment diagram
  Deployment_Node(prod, "Prod") {
    Node(api_node, "API Node") {
      Container(api, "API", "HTTP", "Backend API")
    }
    Node(db_node, "DB Node") {
      ContainerDb(db, "DB", "SQL", "Relational store")
    }
  }
  Rel(api, db, "Connects to")
```

## Mindmap

```mermaid
mindmap
  root(("Spec"))
    behavior
      states
      interactions
    data
      contracts
      persistence
    operations
      runbooks
      monitoring
```

## Timeline

```mermaid
timeline
  title Delivery milestones
  2026-02-03 : "Spec updated"
  2026-02-10 : "Implementation complete"
  2026-02-12 : "Release"
```

## ZenUML (sequence-style)

```mermaid
zenuml
  title "ZenUML minimal example"
  Client->Service: "request()"
  Service->Client: "response()"
```

## Sankey

```mermaid
sankey

Ingress,Service,80
Ingress,Reject,20
Service,DB,60
Service,Cache,20
```

## XY Chart

```mermaid
xychart
  title "Requests per minute"
  x-axis "t" ["t1", "t2", "t3", "t4"]
  y-axis "rpm" 0 --> 100
  line [10, 30, 80, 60]
  bar [5, 20, 40, 30]
```

## Block diagram

```mermaid
block
  columns 2
  A["A"] B["B"]
  A --> B
```

## Packet

```mermaid
packet
  0-3: "Version"
  4-7: "IHL"
  8-15: "DSCP/ECN"
  16-31: "Total Length"
```

## Kanban

```mermaid
kanban
  todo[Todo]
    t1[Write spec]@{ assigned: "dev" }
  doing[In progress]
    t2[Implement]@{ assigned: "dev" }
  done[Done]
    t3[Ship]@{ assigned: "release" }
```

## Architecture

```mermaid
architecture-beta
  group app(cloud)[Application]
  service api(server)[API] in app
  service worker(server)[Worker] in app

  group data(cloud)[Data]
  service db(database)[DB] in data
  service mq(cloud)[Queue] in data

  api:R --> L:db
  api:B --> T:mq
  worker:B --> T:mq
  worker:R --> L:db
```

## Radar

```mermaid
radar-beta
  title Quality profile
  axis a["Correctness"], b["Maintainability"], c["Observability"], d["Performance"], e["Security"]
  curve v1["v1"]{60, 70, 40, 50, 65}
  curve v2["v2"]{75, 80, 65, 60, 78}
```

## Treemap

```mermaid
treemap-beta
  "system"
    "features": 50
    "contracts": 20
    "runbooks": 10
    "testing": 20
```
