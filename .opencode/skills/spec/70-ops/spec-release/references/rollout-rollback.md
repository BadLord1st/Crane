# Rollout & Rollback Strategies

## Rollout Patterns

### 1. Big Bang (All at Once)

```
Before: [v1] [v1] [v1] [v1]
After:  [v2] [v2] [v2] [v2]
```

**Pros**: Simple, fast
**Cons**: High risk, no gradual validation
**When**: Low-risk changes, development environments

### 2. Rolling Deployment

```
Step 1: [v2] [v1] [v1] [v1]
Step 2: [v2] [v2] [v1] [v1]
Step 3: [v2] [v2] [v2] [v1]
Step 4: [v2] [v2] [v2] [v2]
```

**Pros**: Zero downtime, gradual
**Cons**: Mixed versions serving traffic
**When**: Stateless services, backward-compatible changes

```yaml
# Kubernetes rolling update
spec:
  replicas: 4
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1        # Extra pods during update
      maxUnavailable: 0  # Never reduce capacity
```

### 3. Blue-Green Deployment

```
Blue (active):  [v1] [v1] → [v1] [v1] (standby)
Green (standby):[v2] [v2] → [v2] [v2] (active)
                    ↑
              Traffic switch
```

**Pros**: Instant rollback, full testing before switch
**Cons**: Double infrastructure cost
**When**: Critical services, major changes

```bash
# Switch traffic (nginx)
upstream backend {
    server green.internal:8080;  # Point to green
    # server blue.internal:8080;  # Comment out blue
}
```

### 4. Canary Deployment

```
         1%           10%           50%          100%
[v2]─────────────────────────────────────────────────►
[v1][v1][v1][v1] → [v1][v1][v1] → [v1][v1] → [v1] → ∅
```

**Pros**: Real traffic validation, gradual risk
**Cons**: Complex traffic splitting, monitoring required
**When**: User-facing changes, risk mitigation needed

```yaml
# Istio traffic split
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
spec:
  http:
    - route:
        - destination:
            host: myapp
            subset: v2
          weight: 10  # 10% to canary
        - destination:
            host: myapp
            subset: v1
          weight: 90
```

### 5. Feature Flags (Dark Launch)

```python
# Deploy code, activate later
if feature_flags.is_enabled("new_search", user_id=user.id):
    return new_search_algorithm()
else:
    return old_search_algorithm()
```

**Pros**: Decouple deploy from release, instant toggle
**Cons**: Code complexity, flag cleanup needed
**When**: A/B testing, gradual features, risky changes

## Canary Metrics & Promotion

### Key Metrics to Watch
```yaml
canary_metrics:
  - name: error_rate
    threshold: "< 1%"
    comparison: "vs baseline"
    
  - name: latency_p99
    threshold: "< 500ms"
    comparison: "< 110% of baseline"
    
  - name: success_rate
    threshold: "> 99%"
    comparison: "vs baseline"
```

### Automated Canary Analysis
```yaml
# Argo Rollouts
apiVersion: argoproj.io/v1alpha1
kind: Rollout
spec:
  strategy:
    canary:
      steps:
        - setWeight: 5
        - pause: { duration: 10m }
        - analysis:
            templates:
              - templateName: success-rate
        - setWeight: 20
        - pause: { duration: 10m }
        - analysis:
            templates:
              - templateName: success-rate
        - setWeight: 50
        - pause: { duration: 10m }
        - setWeight: 100
```

### Manual Promotion Decision
```markdown
## Canary Promotion Checklist

### At 1% (after 15 minutes)
- [ ] Error rate within 10% of baseline
- [ ] No new error types in logs
- [ ] Latency p99 within 20% of baseline
- [ ] No customer complaints

→ If OK: Promote to 10%
→ If NOT: Rollback immediately

### At 10% (after 1 hour)
- [ ] All above metrics still healthy
- [ ] Conversion rate stable
- [ ] No increase in support tickets

→ If OK: Promote to 50%
→ If NOT: Rollback immediately

### At 50% (after 4 hours)
- [ ] All metrics healthy for sustained period
- [ ] No degradation over time

→ If OK: Promote to 100%
→ If NOT: Investigate before proceeding
```

## Rollback Procedures

### Automated Rollback Triggers
```yaml
# Prometheus alerting rule
- alert: CanaryHighErrorRate
  expr: |
    sum(rate(http_requests_total{version="canary",status=~"5.."}[5m]))
    /
    sum(rate(http_requests_total{version="canary"}[5m]))
    > 0.05
  for: 2m
  annotations:
    action: "Trigger automatic rollback"
```

### Manual Rollback Commands

**Kubernetes**
```bash
# Rollback to previous revision
kubectl rollout undo deployment/myapp

# Rollback to specific revision
kubectl rollout undo deployment/myapp --to-revision=2

# Check rollout history
kubectl rollout history deployment/myapp
```

**Docker Compose**
```bash
# Pull previous image
docker-compose pull --ignore-pull-failures
docker-compose up -d myapp

# Or specify previous tag
docker-compose up -d myapp:v1.2.0
```

**AWS ECS**
```bash
# Update to previous task definition
aws ecs update-service \
  --cluster production \
  --service myapp \
  --task-definition myapp:42  # Previous version
```

**Feature Flag Rollback**
```bash
# Instant disable via API
curl -X POST https://flags.internal/api/flags/new_feature/disable

# Or via CLI
ff disable new_feature --environment production
```

### Rollback Decision Tree
```
Is the issue critical? (data loss, security, widespread errors)
├── YES → Immediate rollback
│         1. Execute rollback command
│         2. Notify stakeholders
│         3. Investigate root cause
│
└── NO → Assess options
          ├── Can we forward-fix in < 30 minutes?
          │   └── YES → Forward-fix with monitoring
          │   └── NO  → Rollback, then fix
          │
          └── Is the issue isolated to canary?
              └── YES → Just remove canary traffic
              └── NO  → Full rollback
```

## Rollback Communication

### Status Page Update
```markdown
**Investigating** - 14:30 UTC
We are investigating elevated error rates on the checkout service.

**Identified** - 14:35 UTC
The issue has been identified as related to today's deployment.

**Monitoring** - 14:40 UTC
We have rolled back the deployment and are monitoring the recovery.

**Resolved** - 14:50 UTC
The rollback is complete. All services are operating normally.
Error rates have returned to baseline levels.
```

### Internal Communication
```markdown
## Rollback Notification

**Service**: checkout-service
**Time**: 2024-01-15 14:40 UTC
**Action**: Rollback from v2.3.0 to v2.2.1

### Impact
- ~5% of checkout attempts failed for 10 minutes
- No data loss
- No security impact

### Root Cause (preliminary)
Database connection pool exhaustion under load

### Next Steps
1. Post-mortem scheduled for tomorrow 10:00 UTC
2. Fix will be developed and tested
3. Next deployment will include additional monitoring
```

## Post-Rollback Checklist

```
- [ ] Verify rollback successful (metrics normal)
- [ ] Notify stakeholders
- [ ] Update status page
- [ ] Preserve logs and metrics from incident
- [ ] Schedule post-mortem
- [ ] Create ticket for root cause investigation
- [ ] Document what went wrong
- [ ] Plan for re-deployment with fixes
```
