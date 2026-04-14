# Impact Analysis Techniques

## What is Impact Analysis?

Impact analysis determines:
1. What code/components will be affected by a change
2. What could break if we modify X
3. What's the risk surface of a change

## Analysis Levels

### 1. Static Analysis (Code)

#### Direct Dependencies
```bash
# Find all imports of a module
grep -r "from module_x import" src/
grep -r "import module_x" src/

# Find all callers of a function
grep -r "function_name(" src/
```

#### Call Graph Analysis
```python
# Python: use pycallgraph or pyan
pyan3 src/**/*.py --grouped --dot | dot -Tpng -o callgraph.png

# JavaScript: use madge
npx madge --image graph.png src/
```

#### Type/Interface Usage
```bash
# Find all implementations of interface
grep -r "implements InterfaceName" src/
grep -r "class.*\(InterfaceName\)" src/
```

### 2. Dynamic Analysis (Runtime)

#### Trace Execution Paths
```python
# Python: trace module
python -m trace --count -C coverage_dir script.py

# Or use coverage
coverage run script.py
coverage report -m
```

#### Profile Hot Paths
```bash
# Python
python -m cProfile -o profile.stats script.py
python -c "import pstats; p=pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
```

### 3. Data Flow Analysis

#### Input Tracing
```markdown
User Input → Validation → Processing → Storage
     ↓           ↓            ↓          ↓
  [HTTP]     [Schema]    [Business]   [Database]

Change at: Processing
Impact: Storage layer might receive different format
```

#### Output Tracing
```markdown
Storage → Processing → Serialization → Response
   ↓          ↓             ↓            ↓
 [DB]    [Business]     [JSON]       [HTTP]

Change at: Processing
Impact: Response format may change (breaking API)
```

## Impact Categories

### 1. Direct Impact
Components that directly use the changed code:
- Callers of modified function
- Subclasses of modified class
- Importers of modified module

### 2. Indirect Impact
Components affected through chain:
- Callers of callers
- Data consumers downstream
- Tests that exercise the path

### 3. External Impact
Effects outside the codebase:
- API contract changes
- Database schema changes
- Configuration changes
- Third-party integrations

## Risk Assessment Matrix

| Change Type | Direct Impact | Indirect Impact | External Impact | Risk |
|-------------|---------------|-----------------|-----------------|------|
| Rename variable | Low | None | None | Low |
| Change function signature | Medium | Medium | Possible | Medium |
| Modify algorithm | Low | High | Possible | High |
| Change database schema | Medium | High | High | Critical |
| Modify API contract | Medium | Medium | Critical | Critical |

## Impact Analysis Checklist

```
- [ ] Identify changed components
- [ ] Find all direct callers/users
- [ ] Trace indirect dependencies
- [ ] Check for reflection/dynamic usage
- [ ] Review configuration dependencies
- [ ] Check database interactions
- [ ] Verify API contract
- [ ] List affected tests
- [ ] Assess external integrations
- [ ] Document breaking changes
```

## Tools by Language

### Python
```bash
# Find usages
grep -r "function_name" --include="*.py"

# Call graph
pyan3 --grouped --dot src/**/*.py

# Import analysis
python -c "import modulefinder; mf=modulefinder.ModuleFinder(); mf.run_script('script.py'); print('\n'.join(mf.modules.keys()))"
```

### JavaScript/TypeScript
```bash
# Find usages
grep -r "functionName" --include="*.ts" --include="*.js"

# Dependency graph
npx madge --circular --extensions ts src/

# Type references
npx ts-prune  # Find unused exports
```

### Go
```bash
# Find usages
grep -r "FunctionName" --include="*.go"

# Call graph
go install golang.org/x/tools/cmd/callgraph@latest
callgraph -algo=pta ./...

# Dependency analysis
go mod graph
```

### Java
```bash
# Find usages
grep -r "methodName" --include="*.java"

# Or use IDE's "Find Usages"

# Dependency analysis
mvn dependency:tree
gradle dependencies
```

## Impact Report Template

```markdown
# Impact Analysis: [Change Description]

## Summary
[What is being changed and why]

## Direct Impact

### Modified Files
- `src/module/file.py`: Changed function X

### Direct Callers
| File | Function | Impact |
|------|----------|--------|
| api/routes.py | get_user() | Signature changed |
| services/auth.py | validate() | Return type changed |

## Indirect Impact

### Affected Workflows
1. User login flow
   - Uses validate() → affected
2. Token refresh flow
   - Uses validate() → affected

### Affected Tests
- tests/test_auth.py::test_login
- tests/test_routes.py::test_get_user

## External Impact

### API Changes
- Endpoint `/api/users/{id}` response format changed
- Breaking change: `email` field now required

### Database Changes
- None

### Configuration Changes
- New env var required: `AUTH_TIMEOUT`

## Risk Assessment
- Overall Risk: **Medium**
- Mitigation: Feature flag, gradual rollout

## Recommendations
1. Update API documentation
2. Notify API consumers
3. Add migration guide
4. Deploy with feature flag
```

## Common Pitfalls

### 1. Missing Reflection Usage
```python
# Not caught by static analysis
method_name = "process_" + type
getattr(handler, method_name)()
```
**Mitigation**: Search for `getattr`, `eval`, `exec`, dynamic imports

### 2. Missing Config Dependencies
```python
# Change in config affects behavior
if config.get("feature_flag"):
    new_behavior()
```
**Mitigation**: Search for config key usage

### 3. Missing Event/Message Consumers
```python
# Publishers don't know about subscribers
event_bus.publish("user_created", user)
```
**Mitigation**: Document event contracts, search for event name

### 4. Missing Database Dependencies
```sql
-- Views, triggers, stored procedures
CREATE VIEW active_users AS SELECT * FROM users WHERE status = 'active';
```
**Mitigation**: Review DB schema, run migration dry-run
