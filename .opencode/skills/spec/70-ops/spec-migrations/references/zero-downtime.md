# Zero-Downtime Deployment Strategies

## Core Principles

1. **Backward Compatibility**: New code works with old data
2. **Forward Compatibility**: Old code works with new data
3. **Incremental Changes**: Small, reversible steps
4. **Feature Flags**: Control feature visibility

## Deployment Patterns

### 1. Rolling Deployment
```
Before: [v1] [v1] [v1] [v1]
Step 1: [v2] [v1] [v1] [v1]
Step 2: [v2] [v2] [v1] [v1]
Step 3: [v2] [v2] [v2] [v1]
After:  [v2] [v2] [v2] [v2]
```

Requirements:
- v1 and v2 compatible with same database schema
- Health checks to verify new instances
- Load balancer draining connections

### 2. Blue-Green Deployment
```
Blue (current):  [v1] [v1] [v1]  ← traffic
Green (new):     [v2] [v2] [v2]  ← idle

Switch:
Blue:   [v1] [v1] [v1]  ← idle
Green:  [v2] [v2] [v2]  ← traffic
```

Requirements:
- Double infrastructure
- Shared database compatible with both versions
- Fast DNS/load balancer switch
- Easy rollback (switch back)

### 3. Canary Deployment
```
Production: [v1] [v1] [v1] [v1] [v1]  ← 100% traffic

Canary:
[v2]           ← 1% traffic
[v1] [v1] [v1] [v1] [v1]  ← 99% traffic

Gradual:
[v2] [v2]      ← 20% traffic
[v1] [v1] [v1] ← 80% traffic

Full:
[v2] [v2] [v2] [v2] [v2]  ← 100% traffic
```

Requirements:
- Traffic splitting capability
- Monitoring to detect issues
- Automatic rollback on errors

## Database Migration Strategies

### Expand and Contract

**Phase 1: Expand** (backward compatible)
```sql
-- Add new column, keep old
ALTER TABLE users ADD COLUMN full_name VARCHAR(255);
```

**Phase 2: Migrate** (dual write)
```python
# App writes to both columns
user.name = value
user.full_name = value
```

**Phase 3: Backfill** (catch up old data)
```sql
UPDATE users SET full_name = name WHERE full_name IS NULL;
```

**Phase 4: Switch** (read from new)
```python
# App reads from new column only
display_name = user.full_name
```

**Phase 5: Contract** (remove old)
```sql
ALTER TABLE users DROP COLUMN name;
```

### Read-Modify-Write Pattern
```python
# Safe concurrent updates
def update_user_email(user_id, new_email):
    while True:
        user = get_user(user_id)
        old_version = user.version
        
        user.email = new_email
        user.version += 1
        
        updated = db.update(
            User,
            where=(User.id == user_id) & (User.version == old_version),
            values=user
        )
        
        if updated:
            break  # Success
        # Retry on conflict
```

## Feature Flags

### Simple Toggle
```python
if feature_flags.is_enabled("new_checkout"):
    return new_checkout_flow()
else:
    return old_checkout_flow()
```

### Gradual Rollout
```python
# Roll out to percentage of users
if feature_flags.is_enabled("new_checkout", user_id=user.id, percentage=10):
    return new_checkout_flow()
```

### User Segment Targeting
```python
# Enable for specific segments
if feature_flags.is_enabled("new_checkout", 
    user_id=user.id,
    segments=["beta_testers", "employees"]):
    return new_checkout_flow()
```

### Kill Switch
```python
# Emergency disable
if feature_flags.is_killed("payment_processing"):
    return maintenance_page()
```

## API Versioning for Zero-Downtime

### URL Versioning
```
/api/v1/users  ← old clients
/api/v2/users  ← new clients
```

### Header Versioning
```
GET /api/users
Accept: application/vnd.api+json; version=2
```

### Backward Compatible Changes
```json
// v1 response
{ "name": "John" }

// v2 response (additive, backward compatible)
{ "name": "John", "full_name": "John Doe" }
```

### Deprecation Process
```
1. Announce deprecation (docs, headers)
2. Add sunset header: Sunset: Sat, 1 Jan 2025 00:00:00 GMT
3. Monitor v1 usage
4. Notify remaining users
5. Return 410 Gone after sunset
```

## Rollback Procedures

### Application Rollback
```bash
# Kubernetes
kubectl rollout undo deployment/app

# Docker Compose
docker-compose up -d --scale app=0
docker-compose up -d app  # with previous image tag

# AWS ECS
aws ecs update-service --cluster prod --service app --task-definition app:previous
```

### Database Rollback
```bash
# If using reversible migrations
alembic downgrade -1

# If destructive migration applied
# Restore from backup (last resort)
pg_restore -d mydb backup.dump
```

### Feature Flag Rollback
```bash
# Instant rollback via flag
curl -X POST https://flags.internal/api/flags/new_checkout/disable
```

## Checklist: Zero-Downtime Deploy

```
Pre-deploy:
- [ ] Database migrations are backward compatible
- [ ] New code handles old data format
- [ ] Old code handles new data format (if rollback needed)
- [ ] Feature flags in place for risky changes
- [ ] Rollback plan documented
- [ ] Monitoring dashboards ready

During deploy:
- [ ] Watch error rates
- [ ] Watch latency p99
- [ ] Watch resource usage
- [ ] Verify health checks passing
- [ ] Test critical paths manually

Post-deploy:
- [ ] Verify all instances healthy
- [ ] Check for error spikes
- [ ] Confirm feature working as expected
- [ ] Update runbook if issues found
- [ ] Schedule cleanup of old code/flags
```

## Common Pitfalls

| Pitfall | Impact | Prevention |
|---------|--------|------------|
| NOT NULL without default | Migration fails | Add column nullable first |
| Removing column still in use | App errors | Remove from code first |
| Incompatible API change | Client errors | Version or feature flag |
| Long-running migration lock | Downtime | Use CONCURRENTLY, batches |
| No rollback plan | Extended outage | Test rollback in staging |
