# Mission Phase Reference

## Quick Reference: When to Use Each Phase

| Phase | Trigger | Skip When |
|-------|---------|-----------|
| 0. Recon | Always first | Never skip |
| 1. Spec | New feature, behavior change | Pure refactor with no behavior change |
| 2. Gap Analysis | After spec, before coding | Trivial change |
| 3. BDD | Feature with user-facing behavior | Internal utility only |
| 4. TDD | All implementation | Spike/prototype (delete after) |
| 5. Verify | Before any merge/commit | Never skip |
| 6. Audit | Before merge to main | Quick fix in non-critical area |
| 7. Error KB | Bug found during work | No errors encountered |
| 8. Git Commit | Always | Never skip |
| 9. Release | Production deployment | Development branch only |

## Phase Quick Reference

### Phase 0: Recon
```
skill "spec-recon"
```

**Output Checklist**:
- [ ] Project structure mapped
- [ ] Existing patterns documented
- [ ] Test infrastructure identified
- [ ] Build/lint commands verified
- [ ] Dependencies assessed

**Time Budget**: 15-30 minutes

### Phase 1: Specification
```
skill "spec-specification"
```

**Output Checklist**:
- [ ] Acceptance criteria defined (testable)
- [ ] State transitions documented
- [ ] Error cases enumerated
- [ ] API contracts drafted
- [ ] Performance requirements stated

**Time Budget**: 30-60 minutes

### Phase 2: Gap Analysis
```
skill "spec-gap-analysis"
```

**Output Checklist**:
- [ ] Current vs desired state compared
- [ ] Impact surface identified
- [ ] Risk assessment complete
- [ ] Step plan created
- [ ] Effort estimated

**Time Budget**: 15-30 minutes

### Phase 3: BDD
```
skill "spec-bdd"
```

**Output Checklist**:
- [ ] Feature files created
- [ ] Scenarios cover acceptance criteria
- [ ] Happy path tested
- [ ] Edge cases covered
- [ ] Error cases tested

**Time Budget**: 30-60 minutes

### Phase 4: TDD
```
skill "spec-tdd"
```

**Output Checklist**:
- [ ] Unit test written first
- [ ] Test fails for right reason
- [ ] Minimal code to pass
- [ ] Refactor complete
- [ ] All tests green

**Time Budget**: Varies with scope

### Phase 5: Verify
```
skill "spec-verify"
```

**Output Checklist**:
- [ ] All tests pass
- [ ] Lint/type checks pass
- [ ] Build succeeds
- [ ] No regressions detected
- [ ] Evidence captured

**Time Budget**: 10-20 minutes

### Phase 6: Audit
```
skill "spec-audit"
```

**Output Checklist**:
- [ ] Spec-code consistency verified
- [ ] No debug code left
- [ ] No accidental changes
- [ ] Traceability matrix updated
- [ ] Security considerations reviewed

**Time Budget**: 15-30 minutes

### Phase 7: Error KB
```
skill "spec-error-kb"
```

**Output Checklist**:
- [ ] Error categorized
- [ ] Root cause identified
- [ ] Solution documented
- [ ] Prevention added
- [ ] Related entries linked

**Time Budget**: 10-15 minutes

### Phase 8: Git Commit
```
skill "spec-git-committer"
```

**Output Checklist**:
- [ ] Conventional commit format
- [ ] Clear, descriptive message
- [ ] Atomic change (one logical unit)
- [ ] References issue/ticket
- [ ] Signed if required

**Time Budget**: 5 minutes

### Phase 9: Release
```
skill "spec-release"
```

**Output Checklist**:
- [ ] Version bumped correctly
- [ ] Changelog updated
- [ ] Release notes drafted
- [ ] Rollout strategy defined
- [ ] Rollback plan documented

**Time Budget**: 15-30 minutes

## Domain Loading Decision Tree

```
Is this an ML/AI project?
├─ Yes → skill "theta-ml-research"
└─ No → Continue

Is this a web frontend project?
├─ Yes → skill "theta-web" (when available)
└─ No → Continue

Is this security-sensitive?
├─ Yes → skill "theta-security" (when available)
└─ No → Continue with core skills only
```

## Adapting for Task Size

### Trivial Fix (< 30 min)
```
Recon (5 min) → TDD → Verify → Commit
```

### Small Feature (1-4 hours)
```
Recon → Spec (light) → BDD → TDD → Verify → Audit → Commit
```

### Medium Feature (1-3 days)
```
Full pipeline, all phases
```

### Large Feature (> 3 days)
```
Full pipeline + break into multiple PRs
Each PR follows: BDD → TDD → Verify → Audit → Commit
```

## Phase Dependencies

```
Phase 0 (Recon)
    │
    ▼
Phase 1 (Spec) ◄─── Domain Loading
    │
    ▼
Phase 2 (Gap Analysis)
    │
    ├────────────────┐
    ▼                ▼
Phase 3 (BDD)    Phase 1b (Threat Model, Observability, Performance)
    │                │
    └────────────────┘
              │
              ▼
         Phase 4 (TDD)
              │
              ▼
         Phase 5 (Verify)
              │
              ▼
         Phase 6 (Audit)
              │
    ┌─────────┴─────────┐
    ▼                   ▼
Phase 7 (Error KB)  Phase 8 (Commit)
                        │
                        ▼
                   Phase 9 (Release) [optional]
```
