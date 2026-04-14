# Spec Graph Structure (Obsidian-friendly)

This workspace uses **small spec nodes linked together** instead of one giant spec file.

## Root

- `docs/spec/spec-index.md` is the entry point.
- Every spec node must link back to `[[spec-index]]`.

## Node categories

```
docs/spec/
  spec-index.md
  glossary.md

  system/
    architecture.md
    data-flow.md
    error-model.md
    state-machines/
      <machine>.md

  contracts/
    api/
      <service>.md
    data/
      <entity>.md

  features/
    <feature>.md

  runbooks/
    local-dev.md
    ci.md
    deploy.md
    troubleshooting.md

  testing/
    test-plan.md
    bdd-index.md

  traceability/
    traceability.md
```

## Rules

1) **One change = one node**. If the node grows, split into smaller nodes and link them.
2) **Links first**. The `Links:` block at the top is mandatory.
3) `SPEC-<AREA>-NNN` IDs are stable once published.
4) Keep operational how-to in `runbooks/` and behavior intent in `features/`.

## Why this works

- Specs become navigable and composable.
- Obsidian graph becomes a real map, not a hairball.
- Each agent/skill can “own” one node type without stepping on others.


## Visual map (Mermaid)

### High-level folder topology

```mermaid
flowchart TB
  IDX["docs/spec/spec-index.md"]:::root

  subgraph SYS["system/"]
    ARCH["architecture.md"]
    DF["data-flow.md"]
    ERR["error-model.md"]
    subgraph SM["state-machines/"]
      SM1["<machine>.md"]
    end
  end

  subgraph CTR["contracts/"]
    subgraph API["api/"]
      SVC["<service>.md"]
    end
    subgraph DATA["data/"]
      ENT["<entity>.md"]
    end
  end

  subgraph FEAT["features/"]
    F1["<feature>.md"]
  end

  subgraph RUN["runbooks/"]
    R1["local-dev.md"]
    R2["ci.md"]
    R3["deploy.md"]
    R4["troubleshooting.md"]
  end

  subgraph TST["testing/"]
    TP["test-plan.md"]
    BDD["bdd-index.md"]
  end

  subgraph TR["traceability/"]
    TRC["traceability.md"]
  end

  IDX --> SYS
  IDX --> CTR
  IDX --> FEAT
  IDX --> RUN
  IDX --> TST
  IDX --> TR

  classDef root stroke-width:2px;
```

### Concept map (what links to what)

```mermaid
mindmap
  root(("spec-index"))
    system
      architecture
      data_flow
      error_model
      state_machines
    contracts
      api_contracts
      data_contracts
    features
      feature_nodes
    runbooks
      local_dev
      ci
      deploy
      troubleshooting
    testing
      test_plan
      bdd_index
    traceability
      mapping
```
