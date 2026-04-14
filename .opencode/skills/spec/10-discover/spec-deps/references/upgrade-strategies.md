# Dependency Upgrade Strategies

## Risk Assessment Matrix

| Change Type | Risk Level | Strategy |
|-------------|------------|----------|
| Patch (1.2.3 → 1.2.4) | Low | Auto-merge with tests |
| Minor (1.2.3 → 1.3.0) | Medium | Review changelog, test |
| Major (1.2.3 → 2.0.0) | High | Full migration plan |
| Security fix | Critical | Immediate, any version |

## Pre-Upgrade Checklist

```
- [ ] Current version documented
- [ ] Target version identified
- [ ] Changelog reviewed
- [ ] Breaking changes listed
- [ ] Migration guide found (if major)
- [ ] Dependency tree checked (transitive deps)
- [ ] Lock file backed up
```

## Upgrade Workflows

### 1. Safe Patch Upgrade
```bash
# 1. Update lock file only
npm update package-name  # or pip install -U, go get -u

# 2. Run full test suite
npm test

# 3. Smoke test critical paths
npm run e2e

# 4. Commit with clear message
git commit -m "chore(deps): bump package-name 1.2.3 → 1.2.4"
```

### 2. Minor Version Upgrade
```bash
# 1. Read changelog for new features/deprecations
# 2. Check for deprecation warnings in current code
grep -r "DeprecatedAPI" src/

# 3. Update and test
npm install package-name@1.3.0
npm test

# 4. Address any deprecation warnings
# 5. Commit with changelog reference
git commit -m "chore(deps): upgrade package-name to 1.3.0

- New feature X available
- Deprecated Y, migrated to Z
- See: https://github.com/org/package/releases/tag/v1.3.0"
```

### 3. Major Version Migration
```markdown
## Migration Plan: package-name 1.x → 2.x

### Phase 1: Assessment
- [ ] Read migration guide
- [ ] List all breaking changes
- [ ] Identify affected code paths
- [ ] Estimate effort (hours/days)

### Phase 2: Preparation
- [ ] Create feature branch
- [ ] Add compatibility shims if available
- [ ] Update types/interfaces

### Phase 3: Migration
- [ ] Update import paths
- [ ] Fix API call signatures
- [ ] Update configuration format
- [ ] Fix type errors

### Phase 4: Verification
- [ ] All tests passing
- [ ] Manual smoke test
- [ ] Performance comparison
- [ ] No deprecation warnings

### Phase 5: Rollout
- [ ] Deploy to staging
- [ ] Monitor for 24h
- [ ] Deploy to production
- [ ] Keep rollback ready
```

## Language-Specific Strategies

### Python (pip/poetry)
```bash
# Check outdated
pip list --outdated
poetry show --outdated

# Check security
pip-audit
safety check

# Upgrade with constraints
poetry update package-name --dry-run
poetry update package-name
```

### Node.js (npm/yarn/pnpm)
```bash
# Check outdated
npm outdated
yarn outdated

# Check security
npm audit
yarn audit

# Interactive upgrade
npx npm-check-updates -i
```

### Go
```bash
# Check for updates
go list -m -u all

# Upgrade specific
go get package@v2.0.0

# Tidy dependencies
go mod tidy

# Verify
go mod verify
```

### Rust (Cargo)
```bash
# Check outdated
cargo outdated

# Check security
cargo audit

# Update
cargo update -p package-name
```

## Handling Breaking Changes

### API Signature Changes
```python
# Before (v1)
client.send(message, timeout=30)

# After (v2) - options object
client.send(message, SendOptions(timeout=30))

# Migration shim
def send_compat(message, timeout=30):
    return client.send(message, SendOptions(timeout=timeout))
```

### Configuration Format Changes
```yaml
# Before (v1)
database:
  host: localhost
  port: 5432

# After (v2) - connection string
database:
  url: "postgresql://localhost:5432/db"

# Migration: support both during transition
database:
  url: ${DATABASE_URL:-postgresql://${DB_HOST}:${DB_PORT}/${DB_NAME}}
```

### Import Path Changes
```python
# Create compatibility layer
# compat.py
try:
    from new_package import NewAPI as API
except ImportError:
    from old_package import OldAPI as API

# Usage stays stable
from myapp.compat import API
```

## Rollback Procedures

### Immediate Rollback
```bash
# Restore lock file
git checkout HEAD~1 -- package-lock.json
npm ci

# Or revert entire commit
git revert HEAD
```

### Gradual Rollback (feature flags)
```python
if feature_flags.get("use_new_client_v2"):
    from new_client import Client
else:
    from old_client import Client
```

## Dependency Freeze Periods

When NOT to upgrade:
- 48h before major release
- During incident response
- Before holiday/vacation
- Without test coverage for affected code

Document freeze periods in team calendar.
