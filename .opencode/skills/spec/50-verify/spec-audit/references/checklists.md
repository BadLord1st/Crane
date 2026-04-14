# Audit Checklists

## Pre-Audit Preparation

```
- [ ] All tests passing locally
- [ ] Git status clean (no uncommitted changes)
- [ ] Dependencies up to date
- [ ] Environment matches CI
```

## Code Quality Audit

### 1. Accidental Changes
```
- [ ] No debug statements (console.log, print, debugger)
- [ ] No commented-out code blocks
- [ ] No TODO/FIXME left unaddressed
- [ ] No hardcoded secrets or credentials
- [ ] No test-only code in production paths
```

### 2. Consistency Check
```
- [ ] Naming conventions followed
- [ ] Error handling consistent with codebase patterns
- [ ] Logging follows established format
- [ ] Configuration follows existing patterns
```

### 3. Spec Alignment
```
- [ ] All acceptance criteria have corresponding tests
- [ ] No implemented behavior missing from spec
- [ ] No spec requirements left unimplemented
- [ ] Edge cases from spec are covered
```

## Regression Hunt Checklist

### Adjacent Workflows
```
- [ ] List 3-5 workflows that touch modified code
- [ ] Manually trace each workflow path
- [ ] Verify error handling in each path
- [ ] Check boundary conditions
```

### Integration Points
```
- [ ] API contracts unchanged (or documented)
- [ ] Database schema compatible
- [ ] External service calls still valid
- [ ] Event/message formats unchanged
```

## Domain-Specific Checklists

### Security Audit (when domain-security loaded)
```
- [ ] No new secrets in code
- [ ] Input validation on all entry points
- [ ] Authentication checks in place
- [ ] Authorization properly scoped
- [ ] Sensitive data not logged
- [ ] Dependencies scanned for vulnerabilities
```

### Web Audit (when domain-web loaded)
```
- [ ] Accessibility (a11y) not regressed
- [ ] Performance budget maintained
- [ ] Mobile responsiveness preserved
- [ ] Browser compatibility verified
- [ ] SEO meta tags intact
```

### ML Audit (when domain-ml loaded)
```
- [ ] Model versioning in place
- [ ] Training data lineage documented
- [ ] Inference latency within SLA
- [ ] Bias/fairness metrics captured
- [ ] Reproducibility verified (seeds, versions)
```

## Post-Audit Actions

### If Issues Found
1. Document each issue with severity
2. Link to relevant spec/test
3. Propose fix or mitigation
4. Update error KB if pattern detected

### If Clean
1. Record what was verified
2. Note any deferred checks
3. Update traceability matrix
4. Approve for merge/commit
