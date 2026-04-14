# BDD Writing Tutorials

Step-by-step guides for deriving BDD scenarios from specifications with proper ID traceability.

## Contents

- [Tutorial 1: From Spec to Scenarios](#tutorial-1-from-spec-to-scenarios)
- [Tutorial 2: Gherkin Best Practices](#tutorial-2-gherkin-best-practices)
- [Tutorial 3: Test Data and Fixtures](#tutorial-3-test-data-and-fixtures)
- [Tutorial 4: Edge Cases and Error Paths](#tutorial-4-edge-cases-and-error-paths)
- [Tutorial 5: Framework Integration](#tutorial-5-framework-integration)

---

## Tutorial 1: From Spec to Scenarios

### Step 1: Read the Acceptance Criteria

From your spec (`docs/spec/auth.md`):

```markdown
| AC ID | Description | SPEC Refs | SC Placeholders |
|-------|-------------|-----------|-----------------|
| AC-1 | User can login with valid credentials | SPEC-AUTH-001 | SC-1, SC-2 |
| AC-2 | Login fails with invalid password | SPEC-AUTH-001 | SC-3 |
```

### Step 2: Create scenario file

```bash
# Match structure to spec
mkdir -p tests/features
touch tests/features/auth.feature
```

### Step 3: Write scenarios with IDs

```gherkin
Feature: Authentication
  As a user
  I want to login to the application
  So that I can access protected resources

  @SPEC-AUTH-001 @AC-1 @SC-1
  Scenario: Successful login with valid credentials
    Given a registered user with email "test@example.com"
    And the user's password is "SecurePass123"
    When the user submits login with email "test@example.com" and password "SecurePass123"
    Then the response status should be 200
    And the response should contain "accessToken"
    And the response should contain "refreshToken"

  @SPEC-AUTH-001 @AC-1 @SC-2
  Scenario: Concurrent login invalidates previous session
    Given a user is logged in on device A
    When the user logs in on device B
    Then device A's session should be invalidated
    And device B should have a valid session

  @SPEC-AUTH-001 @AC-2 @SC-3
  Scenario: Login fails with incorrect password
    Given a registered user with email "test@example.com"
    When the user submits login with email "test@example.com" and password "WrongPass"
    Then the response status should be 400
    And the error code should be "INVALID_CREDENTIALS"
```

### Step 4: Verify traceability

Every scenario MUST have:
- `@SPEC-*` tag (links to specification)
- `@AC-*` tag (links to acceptance criterion)
- `@SC-*` tag (unique scenario ID)

```bash
# Check all scenarios have required tags
grep -E "@SC-[0-9]+" tests/features/*.feature | wc -l
grep -E "Scenario:" tests/features/*.feature | wc -l
# Numbers should match
```

---

## Tutorial 2: Gherkin Best Practices

### Rule 1: Observable behavior only

**Bad** (implementation detail):
```gherkin
Then the database should contain a new session record
And the Redis cache should be updated
```

**Good** (observable behavior):
```gherkin
Then the user should be logged in
And the session should be valid for 7 days
```

### Rule 2: Declarative over imperative

**Bad** (imperative):
```gherkin
Given I navigate to the login page
And I click on the email field
And I type "test@example.com"
And I click on the password field
And I type "password123"
And I click the login button
```

**Good** (declarative):
```gherkin
Given I am on the login page
When I submit login with email "test@example.com" and password "password123"
```

### Rule 3: One behavior per scenario

**Bad** (multiple behaviors):
```gherkin
Scenario: User login and dashboard access
  When the user logs in
  Then they should be authenticated
  When they access the dashboard
  Then they should see their data
```

**Good** (single behavior each):
```gherkin
Scenario: Successful login
  When the user logs in
  Then they should be authenticated

Scenario: Authenticated user accesses dashboard
  Given the user is logged in
  When they access the dashboard
  Then they should see their data
```

### Rule 4: Use consistent language

Define a ubiquitous language:

```markdown
## Glossary (from spec)

- "logged in" = has valid access token
- "session expired" = access token invalid, refresh token may be valid
- "authenticated user" = user with valid session
```

Use these exact terms in all scenarios.

---

## Tutorial 3: Test Data and Fixtures

### Step 1: Define data requirements

From scenario:
```gherkin
Given a registered user with email "test@example.com"
```

**Fixture needed**: User with specific email, known password.

### Step 2: Create fixture definitions

```javascript
// fixtures/users.js
export const testUsers = {
  standard: {
    email: "test@example.com",
    password: "SecurePass123",
    name: "Test User"
  },
  admin: {
    email: "admin@example.com",
    password: "AdminPass123",
    role: "admin"
  }
};
```

### Step 3: Implement step definitions

```javascript
// steps/auth.steps.js
import { Given, When, Then } from '@cucumber/cucumber';

Given('a registered user with email {string}', async function(email) {
  // Create or ensure user exists
  this.testUser = await createTestUser({ email });
});

When('the user submits login with email {string} and password {string}', 
  async function(email, password) {
    this.response = await api.post('/auth/login', { email, password });
  }
);

Then('the response status should be {int}', function(status) {
  expect(this.response.status).to.equal(status);
});
```

### Step 4: Data cleanup strategy

```javascript
// hooks.js
import { Before, After, AfterAll } from '@cucumber/cucumber';

Before(async function() {
  // Reset database to known state
  await db.reset();
});

After(async function(scenario) {
  // Capture debug info on failure
  if (scenario.result.status === 'failed') {
    await this.captureDebugInfo();
  }
});

AfterAll(async function() {
  // Cleanup test data
  await db.cleanup();
});
```

---

## Tutorial 4: Edge Cases and Error Paths

### Step 1: Identify edge cases from spec

From spec error model:
```markdown
| Code | Message |
|------|---------|
| AUTH_RATE_LIMITED | Too many attempts |
| AUTH_SESSION_EXPIRED | Session expired |
```

### Step 2: Write error scenarios

```gherkin
@SPEC-AUTH-003 @AC-4 @SC-6
Scenario: Rate limiting after 5 failed attempts
  Given a registered user with email "test@example.com"
  When the user fails login 5 times
  And the user attempts login again
  Then the response status should be 429
  And the error code should be "AUTH_RATE_LIMITED"
  And the error should include retry-after time

@SPEC-AUTH-002 @AC-3 @SC-5
Scenario: Expired refresh token forces logout
  Given a user with an expired refresh token
  When the user attempts to refresh their session
  Then the response status should be 401
  And the error code should be "AUTH_SESSION_EXPIRED"
  And the user should be logged out
```

### Step 3: Write boundary scenarios

```gherkin
@SPEC-AUTH-001 @AC-1 @SC-7
Scenario Outline: Password validation boundaries
  Given a registered user
  When the user attempts login with password "<password>"
  Then the result should be "<result>"

  Examples:
    | password   | result  | notes                    |
    | 1234567    | invalid | 7 chars, below min (8)   |
    | 12345678   | valid   | 8 chars, at min          |
    | <128chars> | valid   | 128 chars, at max        |
    | <129chars> | invalid | 129 chars, above max     |
```

### Step 4: Write concurrency scenarios

```gherkin
@SPEC-AUTH-002 @AC-1 @SC-8
Scenario: Concurrent requests don't create race condition
  Given a user is authenticated
  When 10 concurrent requests are made
  Then all requests should succeed or fail consistently
  And no duplicate sessions should be created
```

---

## Tutorial 5: Framework Integration

### Cucumber.js Setup

```javascript
// cucumber.js config
module.exports = {
  default: {
    require: ['tests/steps/**/*.js', 'tests/support/**/*.js'],
    format: ['progress', 'json:reports/cucumber.json'],
    publishQuiet: true,
  }
};
```

### Playwright BDD Setup

```typescript
// playwright.config.ts
import { defineConfig } from '@playwright/test';
import { defineBddConfig } from 'playwright-bdd';

export default defineConfig({
  ...defineBddConfig({
    features: 'tests/features',
    steps: 'tests/steps',
  }),
});
```

### pytest-bdd Setup

```python
# pytest.ini
[pytest]
bdd_features_base_dir = tests/features/

# tests/step_defs/test_auth.py
from pytest_bdd import scenario, given, when, then

@scenario('auth.feature', 'Successful login with valid credentials')
def test_successful_login():
    pass

@given('a registered user with email "test@example.com"')
def registered_user():
    return create_test_user(email="test@example.com")

@when('the user submits login')
def submit_login(registered_user, api_client):
    return api_client.post('/auth/login', json={...})

@then('the response status should be 200')
def check_status(response):
    assert response.status_code == 200
```

---

## Quick Reference: BDD Checklist

Before merging scenarios:

- [ ] Every scenario has @SPEC-*, @AC-*, @SC-* tags
- [ ] Scenarios test observable behavior (not implementation)
- [ ] Language matches spec glossary
- [ ] One behavior per scenario
- [ ] Happy path covered
- [ ] Key error paths covered
- [ ] Edge cases/boundaries covered
- [ ] Step definitions implemented
- [ ] Fixtures/test data defined
- [ ] Cleanup hooks in place
- [ ] All scenarios pass locally

---

## Mapping Template

Create this mapping in your spec:

```markdown
## BDD Traceability

| SPEC ID | AC ID | SC ID | Feature File | Status |
|---------|-------|-------|--------------|--------|
| SPEC-AUTH-001 | AC-1 | SC-1 | auth.feature:L10 | ✅ |
| SPEC-AUTH-001 | AC-1 | SC-2 | auth.feature:L18 | ✅ |
| SPEC-AUTH-001 | AC-2 | SC-3 | auth.feature:L26 | ✅ |
| SPEC-AUTH-002 | AC-3 | SC-4 | auth.feature:L34 | 🔄 |
| SPEC-AUTH-002 | AC-3 | SC-5 | auth.feature:L42 | ⏳ |
```

Legend: ✅ Implemented | 🔄 In Progress | ⏳ Planned
