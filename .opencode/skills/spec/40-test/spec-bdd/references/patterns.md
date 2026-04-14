# BDD Patterns and Anti-Patterns

Common patterns for effective BDD scenarios and anti-patterns to avoid.

## Contents

- [Patterns](#patterns)
- [Anti-Patterns](#anti-patterns)
- [Framework-Specific Patterns](#framework-specific-patterns)

---

## Patterns

### Pattern 1: Background for Common Setup

When multiple scenarios share setup:

```gherkin
Feature: Shopping Cart

  Background:
    Given a user is logged in
    And the user has an empty cart

  @SC-1
  Scenario: Add item to cart
    When the user adds product "Widget" to cart
    Then the cart should contain 1 item

  @SC-2
  Scenario: Remove item from cart
    Given the cart contains product "Widget"
    When the user removes "Widget" from cart
    Then the cart should be empty
```

### Pattern 2: Scenario Outline for Data Variations

When testing same behavior with different data:

```gherkin
@SPEC-PAYMENT-001 @AC-1
Scenario Outline: Payment validation
  Given a cart with total <amount>
  When the user pays with <method>
  Then the payment should be <result>

  @SC-10
  Examples: Valid payments
    | amount | method | result   |
    | 100    | card   | accepted |
    | 50     | paypal | accepted |

  @SC-11
  Examples: Invalid payments
    | amount | method | result   |
    | 0      | card   | rejected |
    | -10    | card   | rejected |
```

### Pattern 3: Tags for Organization

Use tags for filtering and organization:

```gherkin
@auth @critical
Feature: Authentication

  @happy-path @smoke
  Scenario: Successful login
    ...

  @error-path
  Scenario: Failed login
    ...

  @slow @nightly
  Scenario: Session timeout after 24 hours
    ...
```

Run subsets:
```bash
cucumber --tags "@smoke"
cucumber --tags "@critical and not @slow"
```

### Pattern 4: Doc Strings for Complex Data

For multi-line or structured data:

```gherkin
Scenario: Create user with profile
  When I create a user with:
    """json
    {
      "name": "John Doe",
      "email": "john@example.com",
      "preferences": {
        "theme": "dark",
        "notifications": true
      }
    }
    """
  Then the user should be created
```

### Pattern 5: Data Tables for Structured Input

For tabular data:

```gherkin
Scenario: Bulk user import
  When I import users:
    | name     | email              | role  |
    | Alice    | alice@example.com  | admin |
    | Bob      | bob@example.com    | user  |
    | Charlie  | charlie@example.com| user  |
  Then 3 users should be created
```

### Pattern 6: Hooks for Setup/Teardown

```gherkin
# In step definitions, not feature file:

Before('@database') do
  DatabaseCleaner.start
end

After('@database') do
  DatabaseCleaner.clean
end

Before('@slow') do
  increase_timeout(60)
end
```

---

## Anti-Patterns

### Anti-Pattern 1: UI Details in Scenarios

**Bad**:
```gherkin
Scenario: Login
  Given I am on the login page
  When I click the "email" input field
  And I type "user@example.com"
  And I click the "password" input field
  And I type "password123"
  And I click the "Login" button
  And I wait for 2 seconds
  Then I should see the dashboard
```

**Good**:
```gherkin
Scenario: Successful login
  Given I am on the login page
  When I login with email "user@example.com" and password "password123"
  Then I should see the dashboard
```

### Anti-Pattern 2: Implementation Details

**Bad**:
```gherkin
Scenario: Create order
  When I create an order
  Then a row should be inserted into the orders table
  And a message should be published to the order-created queue
  And the inventory service should be called via gRPC
```

**Good**:
```gherkin
Scenario: Create order
  When I create an order
  Then the order should be created successfully
  And I should receive an order confirmation
  And the inventory should be reserved
```

### Anti-Pattern 3: Incidental Details

**Bad**:
```gherkin
Scenario: User registration
  Given it is Monday, January 15, 2024 at 10:30 AM
  And the server is running on port 3000
  And the database connection pool size is 10
  When John Smith with ID 12345 registers with email "john@gmail.com"
  Then user ID 12345 should exist in the users table
```

**Good**:
```gherkin
Scenario: User registration
  When a new user registers with valid details
  Then the user should be able to login
```

### Anti-Pattern 4: Conditional Logic

**Bad**:
```gherkin
Scenario: Checkout
  When I checkout
  Then if I am a premium member the discount should be 20%
  But if I am a regular member the discount should be 10%
  And if I have a coupon it should be applied
```

**Good** (separate scenarios):
```gherkin
Scenario: Premium member checkout
  Given I am a premium member
  When I checkout
  Then I should receive a 20% discount

Scenario: Regular member checkout
  Given I am a regular member
  When I checkout
  Then I should receive a 10% discount

Scenario: Checkout with coupon
  Given I have a valid coupon
  When I checkout
  Then the coupon discount should be applied
```

### Anti-Pattern 5: Long Scenarios

**Bad**: Scenarios with 15+ steps testing multiple behaviors.

**Good**: Split into focused scenarios, use Background for shared setup.

### Anti-Pattern 6: Testing Internal State

**Bad**:
```gherkin
Then the session object should have isAuthenticated = true
And the token should be stored in localStorage
And the Redux store should contain the user data
```

**Good**:
```gherkin
Then I should be logged in
And I should see my profile information
```

---

## Framework-Specific Patterns

### Cucumber.js: Async Steps

```javascript
When('I fetch user data', async function() {
  this.response = await api.get('/users/me');
});

Then('I should see my profile', async function() {
  await expect(this.response.data).to.have.property('email');
});
```

### Playwright BDD: Page Objects

```typescript
// pages/login.page.ts
export class LoginPage {
  constructor(private page: Page) {}
  
  async login(email: string, password: string) {
    await this.page.fill('[data-testid="email"]', email);
    await this.page.fill('[data-testid="password"]', password);
    await this.page.click('[data-testid="submit"]');
  }
}

// steps/login.steps.ts
When('I login with email {string} and password {string}', 
  async ({ page }, email, password) => {
    const loginPage = new LoginPage(page);
    await loginPage.login(email, password);
  }
);
```

### pytest-bdd: Fixtures as Context

```python
import pytest
from pytest_bdd import given, when, then, parsers

@pytest.fixture
def api_client():
    return APIClient(base_url="http://localhost:3000")

@given(parsers.parse('a user with email "{email}"'))
def user(email):
    return create_user(email=email)

@when('the user logs in')
def login(user, api_client):
    return api_client.login(user.email, user.password)
```

### Behave (Python): Context Object

```python
from behave import given, when, then

@given('a registered user')
def step_impl(context):
    context.user = create_test_user()

@when('the user logs in')
def step_impl(context):
    context.response = login(context.user)

@then('the login should succeed')
def step_impl(context):
    assert context.response.status_code == 200
```

---

## Checklist: Is This Scenario Good?

- [ ] Single behavior tested
- [ ] No implementation details
- [ ] Uses domain language from spec
- [ ] Has SPEC/AC/SC tags
- [ ] Would a business person understand it?
- [ ] Does it describe WHAT, not HOW?
- [ ] Is it independent from other scenarios?
- [ ] Can it run in any order?
- [ ] Is it deterministic (no flakiness)?
