# STRIDE Threat Modeling

## What is STRIDE?

STRIDE is a mnemonic for six categories of security threats:

| Letter | Threat | Property Violated | Question to Ask |
|--------|--------|-------------------|-----------------|
| **S** | Spoofing | Authentication | Can someone pretend to be another user/system? |
| **T** | Tampering | Integrity | Can someone modify data they shouldn't? |
| **R** | Repudiation | Non-repudiation | Can someone deny they did something? |
| **I** | Information Disclosure | Confidentiality | Can someone see data they shouldn't? |
| **D** | Denial of Service | Availability | Can someone make the system unavailable? |
| **E** | Elevation of Privilege | Authorization | Can someone do something they shouldn't be allowed to? |

## STRIDE Process

### Step 1: Create Data Flow Diagram

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Browser   │─────▶│  API Server │─────▶│  Database   │
│   (User)    │◀─────│  [Process]  │◀─────│ [Data Store]│
└─────────────┘      └─────────────┘      └─────────────┘
      │                    │
      │    Trust Boundary  │
      └────────────────────┘
      
Legend:
─────▶  Data Flow
[    ]  Process
(    )  External Entity
〈    〉  Data Store
━━━━━   Trust Boundary
```

### Step 2: Enumerate Assets

```markdown
## Assets

### Data Assets
- User credentials (passwords, tokens)
- Personal information (email, address)
- Payment data (card numbers, bank accounts)
- Business data (orders, inventory)

### System Assets
- API servers
- Database servers
- Authentication service
- Admin interfaces

### Access Assets
- API keys
- Service account credentials
- Admin accounts
```

### Step 3: Apply STRIDE to Each Element

#### External Entity (Browser/User)
| Threat | Applicable? | Scenario |
|--------|-------------|----------|
| Spoofing | ✅ | Attacker impersonates legitimate user |
| Repudiation | ✅ | User denies placing an order |

#### Process (API Server)
| Threat | Applicable? | Scenario |
|--------|-------------|----------|
| Spoofing | ✅ | Attacker pretends to be the API server |
| Tampering | ✅ | Attacker modifies request in transit |
| Repudiation | ✅ | No audit logs of actions |
| Info Disclosure | ✅ | Error messages leak internal details |
| DoS | ✅ | API overwhelmed with requests |
| EoP | ✅ | Regular user accesses admin endpoints |

#### Data Store (Database)
| Threat | Applicable? | Scenario |
|--------|-------------|----------|
| Tampering | ✅ | Direct database modification |
| Info Disclosure | ✅ | SQL injection exposes data |
| DoS | ✅ | Expensive queries lock database |

#### Data Flow (Network)
| Threat | Applicable? | Scenario |
|--------|-------------|----------|
| Tampering | ✅ | Man-in-the-middle attack |
| Info Disclosure | ✅ | Traffic interception |

### Step 4: Document Threats and Mitigations

```markdown
## Threat: User Credential Spoofing

**Category**: Spoofing
**Asset**: User authentication
**Attack Vector**: Attacker obtains user credentials through phishing or data breach

### Risk Assessment
- Likelihood: Medium
- Impact: High
- Risk: High

### Mitigations
1. Implement multi-factor authentication (MFA)
2. Enforce strong password policy
3. Detect and block suspicious login attempts
4. Implement account lockout after failed attempts

### Acceptance Criteria
- [ ] MFA available for all users
- [ ] Password minimum 12 characters with complexity
- [ ] Account locks after 5 failed login attempts
- [ ] Suspicious login alerts sent to user
```

## STRIDE Per Element Table

| Element | S | T | R | I | D | E |
|---------|---|---|---|---|---|---|
| User | ✅ | | ✅ | | | |
| API Server | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Database | | ✅ | | ✅ | ✅ | |
| Network (Browser→API) | | ✅ | | ✅ | | |
| Network (API→DB) | | ✅ | | ✅ | | |

## Common Mitigations by Threat

### Spoofing Mitigations
```
- [ ] Strong authentication (passwords, MFA)
- [ ] Mutual TLS for service-to-service
- [ ] API key validation
- [ ] Session management with secure tokens
- [ ] Certificate pinning
```

### Tampering Mitigations
```
- [ ] Input validation on all boundaries
- [ ] Parameterized queries (SQL injection)
- [ ] CSRF tokens
- [ ] Integrity checks (HMAC, signatures)
- [ ] Immutable audit logs
```

### Repudiation Mitigations
```
- [ ] Comprehensive audit logging
- [ ] Secure log storage (tamper-evident)
- [ ] User action confirmation emails
- [ ] Digital signatures for critical actions
- [ ] Timestamp all events
```

### Information Disclosure Mitigations
```
- [ ] Encrypt data at rest
- [ ] Encrypt data in transit (TLS)
- [ ] Minimize data exposure (field-level)
- [ ] Secure error handling (no stack traces)
- [ ] Access control on all queries
```

### Denial of Service Mitigations
```
- [ ] Rate limiting
- [ ] Request size limits
- [ ] Timeout configurations
- [ ] Query complexity limits
- [ ] Load balancing and scaling
- [ ] CDN for static assets
```

### Elevation of Privilege Mitigations
```
- [ ] Principle of least privilege
- [ ] Role-based access control (RBAC)
- [ ] Input validation to prevent injection
- [ ] Secure defaults
- [ ] Sandboxing/isolation
```

## Threat Model Document Template

```markdown
# Threat Model: [Feature/System Name]

**Version**: 1.0
**Date**: 2024-01-15
**Author**: Security Team
**Status**: Draft | In Review | Approved

## 1. System Description
[Brief description of the system/feature being modeled]

## 2. Data Flow Diagram
[Insert DFD here]

## 3. Assets
[List of valuable assets]

## 4. Trust Boundaries
[Identify where trust levels change]

## 5. Threat Enumeration

### 5.1 [Threat Name]
- **Category**: [STRIDE category]
- **Description**: [How the attack works]
- **Asset Affected**: [Which asset]
- **Likelihood**: Low/Medium/High
- **Impact**: Low/Medium/High
- **Risk**: Low/Medium/High/Critical

**Mitigations**:
1. [Mitigation 1]
2. [Mitigation 2]

**Acceptance Criteria**:
- [ ] [Testable criterion]

[Repeat for each threat]

## 6. Residual Risks
[Risks that remain after mitigations]

## 7. Review History
| Date | Reviewer | Notes |
|------|----------|-------|
| 2024-01-15 | @security-lead | Initial review |
```

## Security Acceptance Criteria Examples

```gherkin
Feature: Authentication Security

  Scenario: Account lockout after failed attempts
    Given a user with valid credentials
    When the user fails to login 5 times
    Then the account should be locked for 15 minutes
    And an email notification should be sent

  Scenario: Session expiry
    Given a logged-in user
    When the user is inactive for 30 minutes
    Then the session should expire
    And the user should be redirected to login

  Scenario: MFA enforcement for admin
    Given a user with admin role
    When the user attempts to login
    Then MFA should be required
    And login should fail without valid MFA token
```
