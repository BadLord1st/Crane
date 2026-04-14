# Release Management Guide

## Semantic Versioning

### Format
```
MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]

1.0.0        # Initial release
1.0.1        # Patch: bug fix
1.1.0        # Minor: new feature (backward compatible)
2.0.0        # Major: breaking change
2.0.0-alpha  # Pre-release
2.0.0-rc.1   # Release candidate
2.0.0+build.123  # Build metadata
```

### When to Increment

| Change Type | Version Bump | Example |
|-------------|--------------|---------|
| Breaking API change | MAJOR | Remove endpoint, change response format |
| New feature | MINOR | Add endpoint, add optional field |
| Bug fix | PATCH | Fix incorrect behavior |
| Security fix | PATCH (usually) | Vulnerability fix |
| Dependency update (breaking) | MAJOR | If it changes your API |
| Dependency update (safe) | PATCH | Security or bug fixes |

### Breaking Change Examples
```
# Breaking: response format changed
# Before: { "user": "John" }
# After:  { "data": { "user": "John" } }

# Breaking: required field added
# Before: POST /users { "name": "John" }
# After:  POST /users { "name": "John", "email": "required" }

# NOT breaking: optional field added
# Before: { "name": "John" }
# After:  { "name": "John", "age": 30 }  # age is optional
```

## Changelog Format

### Keep a Changelog Standard
```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- New payment processing endpoint

### Changed
- Improved error messages for validation failures

### Deprecated
- Old `/v1/users` endpoint (use `/v2/users`)

### Removed
- Support for Node.js 16

### Fixed
- Race condition in order processing

### Security
- Updated dependencies to fix CVE-2024-1234

## [1.2.0] - 2024-01-15

### Added
- User profile photo upload
- Export orders to CSV

### Fixed
- Incorrect tax calculation for EU orders

## [1.1.0] - 2024-01-01
...
```

### Automated Changelog (Conventional Commits)
```bash
# Generate changelog from commits
npx conventional-changelog -p angular -i CHANGELOG.md -s

# Or use release-please (GitHub Action)
# Creates PR with version bump and changelog
```

## Release Process

### Pre-Release Checklist
```
- [ ] All tests passing
- [ ] No critical/high vulnerabilities
- [ ] Performance benchmarks acceptable
- [ ] Documentation updated
- [ ] Breaking changes documented
- [ ] Migration guide written (if major)
- [ ] Changelog updated
- [ ] Version bumped in code
- [ ] Release branch created (if using GitFlow)
```

### Release Steps
```bash
# 1. Create release branch (optional)
git checkout -b release/1.2.0

# 2. Bump version
npm version minor  # or poetry version minor, etc.

# 3. Update changelog
vim CHANGELOG.md

# 4. Commit
git commit -am "chore: prepare release 1.2.0"

# 5. Create tag
git tag -a v1.2.0 -m "Release 1.2.0: Add user profiles"

# 6. Push
git push origin release/1.2.0 --tags

# 7. Merge to main
git checkout main
git merge --no-ff release/1.2.0

# 8. Create GitHub release
gh release create v1.2.0 --notes-from-tag
```

### Post-Release
```
- [ ] Monitor error rates
- [ ] Watch performance metrics
- [ ] Check customer feedback channels
- [ ] Merge release branch back to develop
- [ ] Announce release (if public)
```

## Release Notes Template

```markdown
# Release Notes: v1.2.0

**Release Date**: 2024-01-15

## Highlights

This release introduces user profile photos and order export functionality.

## New Features

### User Profile Photos
Users can now upload and manage their profile photos. Supports JPEG and PNG formats up to 5MB.

```bash
# Upload photo
curl -X POST /api/users/me/photo -F "file=@photo.jpg"
```

### Order Export
Export your orders to CSV format for accounting purposes.

```bash
# Export last 30 days
curl /api/orders/export?days=30 > orders.csv
```

## Bug Fixes

- Fixed incorrect tax calculation for EU orders (#234)
- Fixed timeout on large order lists (#256)

## Breaking Changes

None in this release.

## Upgrade Guide

No special upgrade steps required. Simply deploy the new version.

## Known Issues

- Profile photo upload may timeout on slow connections (#278)

## Deprecations

The `/v1/users` endpoint is deprecated and will be removed in v2.0.0. Please migrate to `/v2/users`.

## Contributors

Thanks to @alice, @bob, and @charlie for their contributions!
```

## Rollout Strategies

### Staged Rollout
```yaml
# Stage 1: Internal
- environment: internal
  percentage: 100
  duration: 1 day
  
# Stage 2: Canary
- environment: production
  percentage: 1
  duration: 1 hour
  metrics_check: true
  
# Stage 3: Expand
- environment: production
  percentage: 10
  duration: 4 hours
  
# Stage 4: Full
- environment: production
  percentage: 100
```

### Feature Flags
```python
# Gradual rollout by user segment
if feature_flags.is_enabled("new_checkout", 
    user_id=user.id,
    percentage=10,  # 10% of users
    segments=["beta_testers"]  # + all beta testers
):
    return new_checkout()
else:
    return old_checkout()
```

### Blue-Green
```bash
# Deploy to green
kubectl apply -f deployment-green.yaml

# Run smoke tests
./run-smoke-tests.sh green

# Switch traffic
kubectl patch service myapp -p '{"spec":{"selector":{"version":"green"}}}'

# Monitor
sleep 300 && ./check-metrics.sh

# Rollback if needed
kubectl patch service myapp -p '{"spec":{"selector":{"version":"blue"}}}'
```

## Rollback Procedures

### Immediate Rollback
```bash
# Git-based
git revert HEAD
git push

# Container-based
kubectl rollout undo deployment/myapp

# Tag-based
git checkout v1.1.0
./deploy.sh
```

### Database Rollback Considerations
```markdown
1. Check if migrations are reversible
2. If data was migrated, check if it can be restored
3. Consider: is forward-fix faster than rollback?
4. Have backup restore procedure ready
```

### Rollback Checklist
```
- [ ] Identify the issue
- [ ] Assess rollback vs forward-fix time
- [ ] Notify stakeholders
- [ ] Execute rollback
- [ ] Verify rollback successful
- [ ] Monitor for issues
- [ ] Post-mortem scheduled
```

## Hotfix Process

```bash
# 1. Branch from production tag
git checkout v1.2.0
git checkout -b hotfix/1.2.1

# 2. Apply fix
vim src/bug.py
git commit -m "fix: critical bug in payment processing"

# 3. Test
./run-tests.sh

# 4. Bump patch version
npm version patch

# 5. Tag and deploy
git tag v1.2.1
./deploy.sh production

# 6. Merge back
git checkout main
git merge hotfix/1.2.1
git checkout develop
git merge hotfix/1.2.1
```
