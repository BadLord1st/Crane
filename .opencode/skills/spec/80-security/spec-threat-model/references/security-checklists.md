# Security Checklists

## Authentication Checklist

### Password Security
```
- [ ] Minimum password length: 12 characters
- [ ] Password complexity requirements
- [ ] Password history (prevent reuse of last N passwords)
- [ ] Secure password hashing (bcrypt, Argon2, scrypt)
- [ ] Salt per password (not global)
- [ ] Password reset via secure token (time-limited)
- [ ] No password hints or security questions
```

### Session Management
```
- [ ] Session ID regeneration on login
- [ ] Secure session cookie flags (HttpOnly, Secure, SameSite)
- [ ] Session timeout (idle and absolute)
- [ ] Session invalidation on logout
- [ ] Single session or limited concurrent sessions
- [ ] Session binding (IP, user agent - carefully)
```

### Multi-Factor Authentication
```
- [ ] MFA option available for all users
- [ ] MFA enforced for privileged accounts
- [ ] Backup codes for account recovery
- [ ] Rate limiting on MFA attempts
- [ ] MFA bypass logging and alerting
```

### Account Protection
```
- [ ] Account lockout after failed attempts
- [ ] Login attempt logging
- [ ] Suspicious login detection (location, device)
- [ ] Notification on new device login
- [ ] Account recovery process is secure
```

## Authorization Checklist

### Access Control
```
- [ ] Principle of least privilege applied
- [ ] Role-based access control (RBAC) implemented
- [ ] Authorization checks on every request
- [ ] Server-side authorization (not just UI hiding)
- [ ] Deny by default, allow explicitly
```

### API Security
```
- [ ] Authentication required on all non-public endpoints
- [ ] Authorization checked after authentication
- [ ] Rate limiting per user/API key
- [ ] API versioning strategy
- [ ] Deprecation policy for old versions
```

### Resource Access
```
- [ ] Direct object reference protection (IDOR)
- [ ] User can only access their own resources
- [ ] Admin actions require admin role check
- [ ] Sensitive operations require re-authentication
```

## Input Validation Checklist

### General
```
- [ ] Validate on server-side (not just client)
- [ ] Whitelist validation preferred over blacklist
- [ ] Validate data type, length, format, range
- [ ] Reject unexpected input (fail closed)
- [ ] Canonicalize before validation
```

### Injection Prevention
```
- [ ] Parameterized queries for SQL
- [ ] ORM used correctly (no raw queries with user input)
- [ ] NoSQL injection prevention
- [ ] Command injection prevention
- [ ] LDAP injection prevention
- [ ] XPath injection prevention
```

### Output Encoding
```
- [ ] HTML encoding for HTML context
- [ ] JavaScript encoding for JS context
- [ ] URL encoding for URL parameters
- [ ] CSS encoding for CSS context
- [ ] JSON encoding for JSON output
```

## Cryptography Checklist

### Encryption
```
- [ ] TLS 1.2+ for all connections
- [ ] Strong cipher suites only
- [ ] Certificate validation enabled
- [ ] Sensitive data encrypted at rest
- [ ] Encryption keys properly managed
```

### Hashing
```
- [ ] Secure hash algorithms (SHA-256+)
- [ ] Password-specific hashing (bcrypt, Argon2)
- [ ] HMAC for integrity verification
- [ ] No MD5 or SHA-1 for security purposes
```

### Key Management
```
- [ ] Keys not hardcoded in source
- [ ] Keys stored in secure vault
- [ ] Key rotation procedure documented
- [ ] Separate keys per environment
- [ ] Encryption key different from signing key
```

## Data Protection Checklist

### Sensitive Data
```
- [ ] Sensitive data identified and classified
- [ ] Minimal data collection (data minimization)
- [ ] Data retention policy defined
- [ ] Data deletion/anonymization process
- [ ] Sensitive data not in URLs or logs
```

### PII Handling
```
- [ ] PII inventory maintained
- [ ] Consent obtained for PII collection
- [ ] Access to PII limited and logged
- [ ] PII encrypted in transit and at rest
- [ ] Right to deletion supported
```

### Logging and Monitoring
```
- [ ] Security events logged
- [ ] Logs protected from tampering
- [ ] No sensitive data in logs
- [ ] Log retention defined
- [ ] Alerting on suspicious activity
```

## API Security Checklist

### Request Handling
```
- [ ] Rate limiting implemented
- [ ] Request size limits
- [ ] Content-Type validation
- [ ] Request timeout configured
- [ ] Pagination on list endpoints
```

### Response Handling
```
- [ ] Security headers set (see below)
- [ ] No sensitive data in error messages
- [ ] Consistent error format
- [ ] CORS properly configured
- [ ] No unnecessary information exposure
```

### Security Headers
```
- [ ] Content-Security-Policy
- [ ] X-Content-Type-Options: nosniff
- [ ] X-Frame-Options: DENY or SAMEORIGIN
- [ ] Strict-Transport-Security (HSTS)
- [ ] X-XSS-Protection: 0 (use CSP instead)
- [ ] Referrer-Policy
- [ ] Permissions-Policy
```

## Dependency Security Checklist

```
- [ ] Dependencies from trusted sources
- [ ] Dependency versions pinned
- [ ] Regular vulnerability scanning
- [ ] Automated security updates for patches
- [ ] License compliance checked
- [ ] Transitive dependencies reviewed
```

## Pre-Deployment Security Checklist

```
- [ ] SAST scan completed (no high/critical)
- [ ] DAST scan completed (no high/critical)
- [ ] Dependency scan completed
- [ ] Secret scanning completed
- [ ] Threat model reviewed/updated
- [ ] Security test cases passed
- [ ] Penetration test (if required)
- [ ] Security sign-off obtained
```

## Incident Response Preparation

```
- [ ] Incident response plan documented
- [ ] Security contacts defined
- [ ] Escalation path documented
- [ ] Runbooks for common incidents
- [ ] Breach notification process defined
- [ ] Regular incident response drills
```
