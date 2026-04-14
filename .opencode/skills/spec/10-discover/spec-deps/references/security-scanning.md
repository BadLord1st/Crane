# Dependency Security Scanning

## Vulnerability Databases

| Database | Coverage | URL |
|----------|----------|-----|
| NVD (NIST) | All languages | nvd.nist.gov |
| GitHub Advisory | All languages | github.com/advisories |
| OSV | Open source | osv.dev |
| Snyk DB | All languages | snyk.io/vuln |
| npm audit | Node.js | npmjs.com |
| PyPI Advisory | Python | pypi.org/security |
| RustSec | Rust | rustsec.org |

## Scanning Tools by Language

### Python
```bash
# pip-audit (recommended)
pip install pip-audit
pip-audit

# safety
pip install safety
safety check

# With requirements file
pip-audit -r requirements.txt
safety check -r requirements.txt

# Output formats
pip-audit --format=json > audit.json
```

### Node.js
```bash
# Built-in npm audit
npm audit
npm audit --json > audit.json

# Fix automatically (careful!)
npm audit fix
npm audit fix --force  # breaking changes allowed

# yarn
yarn audit
yarn audit --json > audit.json

# pnpm
pnpm audit
```

### Go
```bash
# govulncheck (official)
go install golang.org/x/vuln/cmd/govulncheck@latest
govulncheck ./...

# nancy (Sonatype)
go list -json -deps ./... | nancy sleuth
```

### Rust
```bash
# cargo-audit
cargo install cargo-audit
cargo audit

# With fix suggestions
cargo audit fix --dry-run
cargo audit fix
```

### Java/Kotlin
```bash
# OWASP Dependency-Check
mvn org.owasp:dependency-check-maven:check
gradle dependencyCheckAnalyze

# Snyk
snyk test --all-projects
```

## CI/CD Integration

### GitHub Actions
```yaml
name: Security Scan

on:
  push:
    branches: [main]
  pull_request:
  schedule:
    - cron: '0 0 * * *'  # Daily

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run npm audit
        run: npm audit --audit-level=high
        
      - name: Run Snyk
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
```

### GitLab CI
```yaml
security_scan:
  stage: test
  script:
    - npm audit --audit-level=high
  allow_failure: false
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
```

## Severity Levels & Response

| Severity | CVSS Score | Response Time | Action |
|----------|------------|---------------|--------|
| Critical | 9.0-10.0 | Immediate | Stop work, patch now |
| High | 7.0-8.9 | 24 hours | Prioritize fix |
| Medium | 4.0-6.9 | 1 week | Schedule fix |
| Low | 0.1-3.9 | 1 month | Track for next upgrade |

## Vulnerability Triage

### Questions to Ask
```
1. Is the vulnerable code path reachable in our app?
2. Is the vulnerability exploitable in our deployment?
3. Do we pass untrusted input to the vulnerable function?
4. Is there a workaround without upgrading?
5. What's the upgrade risk vs vulnerability risk?
```

### Triage Template
```markdown
## Vulnerability: CVE-XXXX-XXXXX

**Package**: example-lib@1.2.3
**Severity**: High (CVSS 7.5)
**Description**: Remote code execution via crafted input

### Reachability Analysis
- [ ] We use the affected function: `parseUntrusted()`
- [ ] We pass user input to this function
- [ ] The function is exposed to network

### Decision
- [ ] Upgrade immediately
- [ ] Apply workaround
- [ ] Accept risk (document why)
- [ ] Not applicable (explain)

### Notes
[Why this decision was made]
```

## Handling Transitive Vulnerabilities

### Identify the Chain
```bash
# npm
npm ls vulnerable-package

# pip
pipdeptree -p vulnerable-package

# go
go mod graph | grep vulnerable-package
```

### Resolution Options

1. **Upgrade direct dependency** (preferred)
   ```bash
   npm update parent-package
   ```

2. **Override/force version** (temporary)
   ```json
   // package.json
   "overrides": {
     "vulnerable-package": "2.0.0"
   }
   ```

3. **Replace dependency** (if unmaintained)
   ```bash
   npm uninstall parent-package
   npm install alternative-package
   ```

4. **Fork and patch** (last resort)
   ```bash
   git clone parent-package
   # Apply security patch
   # Publish to private registry
   ```

## False Positive Handling

### Document False Positives
```yaml
# .snyk or audit-ci config
ignore:
  - id: SNYK-JS-EXAMPLE-12345
    reason: "Not exploitable - we don't use the affected API"
    expires: 2025-06-01
```

### npm audit exceptions
```json
// package.json
"auditConfig": {
  "ignore": ["CVE-2024-12345"]
}
```

## Regular Maintenance

### Weekly
- Run security scan
- Review new advisories
- Update patch versions

### Monthly
- Review minor version updates
- Check for deprecated packages
- Update security tooling

### Quarterly
- Full dependency review
- License compliance check
- Remove unused dependencies
