# TDD Patterns & Practices

## The TDD Cycle

```
    ┌─────────────────────────┐
    │                         │
    ▼                         │
┌───────┐   ┌───────┐   ┌─────┴─┐
│  RED  │ → │ GREEN │ → │REFACTOR│
└───────┘   └───────┘   └───────┘
 Write a     Make it     Improve
 failing     pass        design
 test
```

### RED: Write a Failing Test
```python
def test_user_can_login_with_valid_credentials():
    user = create_user(email="test@example.com", password="secret")
    
    result = login(email="test@example.com", password="secret")
    
    assert result.success is True
    assert result.user.email == "test@example.com"
```

**Key Points**:
- Test name describes the behavior
- Test fails for the right reason (not syntax error)
- One assertion per test (or related assertions)

### GREEN: Make It Pass
```python
def login(email: str, password: str) -> LoginResult:
    user = db.query(User).filter_by(email=email).first()
    if user and user.check_password(password):
        return LoginResult(success=True, user=user)
    return LoginResult(success=False, user=None)
```

**Key Points**:
- Write the simplest code that passes
- Don't over-engineer
- It's okay to hardcode (you'll generalize later)

### REFACTOR: Improve Design
```python
def login(email: str, password: str) -> LoginResult:
    user = find_user_by_email(email)
    if user is None:
        return LoginResult.failure("User not found")
    
    if not user.verify_password(password):
        return LoginResult.failure("Invalid password")
    
    return LoginResult.success(user)
```

**Key Points**:
- Tests still pass (safety net)
- Improve naming, structure, duplication
- Extract methods, classes as needed

## Test Naming Conventions

### Behavior-Focused Names
```python
# Pattern: test_<action>_<condition>_<expected_result>

def test_login_with_valid_credentials_succeeds():
    ...

def test_login_with_wrong_password_fails():
    ...

def test_login_with_nonexistent_user_fails():
    ...

def test_login_after_three_failures_locks_account():
    ...
```

### Given-When-Then in Name
```python
def test_given_valid_user_when_login_then_returns_token():
    ...

def test_given_locked_account_when_login_then_returns_locked_error():
    ...
```

## Test Structure: Arrange-Act-Assert

```python
def test_order_total_includes_tax():
    # Arrange: Set up test data
    order = Order()
    order.add_item(Product(name="Widget", price=100))
    tax_rate = 0.1
    
    # Act: Execute the behavior
    total = order.calculate_total(tax_rate=tax_rate)
    
    # Assert: Verify the result
    assert total == 110  # 100 + 10% tax
```

## Common TDD Patterns

### 1. Triangulation
Start with the simplest case, add more tests to force generalization.

```python
# Test 1: Simplest case
def test_fibonacci_of_0_is_0():
    assert fibonacci(0) == 0

# Implementation (can be hardcoded)
def fibonacci(n):
    return 0

# Test 2: Force generalization
def test_fibonacci_of_1_is_1():
    assert fibonacci(1) == 1

# Updated implementation
def fibonacci(n):
    if n <= 1:
        return n
    ...

# Test 3: More triangulation
def test_fibonacci_of_5_is_5():
    assert fibonacci(5) == 5

# Final implementation
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

### 2. One to Many
Start with a single item, then generalize to collections.

```python
# Test 1: Single item
def test_cart_total_with_one_item():
    cart = Cart()
    cart.add(Product(price=100))
    assert cart.total == 100

# Test 2: Multiple items
def test_cart_total_with_multiple_items():
    cart = Cart()
    cart.add(Product(price=100))
    cart.add(Product(price=50))
    assert cart.total == 150
```

### 3. Fake It Till You Make It
Return a constant first, then generalize.

```python
# Test
def test_add():
    assert add(2, 3) == 5

# Fake implementation
def add(a, b):
    return 5  # Fake it!

# Add another test to force real implementation
def test_add_different_numbers():
    assert add(1, 1) == 2

# Real implementation
def add(a, b):
    return a + b
```

### 4. Obvious Implementation
When the solution is obvious, just write it.

```python
def test_user_full_name():
    user = User(first_name="John", last_name="Doe")
    assert user.full_name == "John Doe"

# Obvious, just implement it
@property
def full_name(self):
    return f"{self.first_name} {self.last_name}"
```

## Test Doubles

### Mocks (Verify Behavior)
```python
def test_order_sends_confirmation_email():
    email_service = Mock()
    order = Order(email_service=email_service)
    
    order.complete()
    
    email_service.send.assert_called_once_with(
        to=order.customer.email,
        template="order_confirmation",
        data={"order_id": order.id}
    )
```

### Stubs (Provide Canned Answers)
```python
def test_order_total_with_discount():
    discount_service = Mock()
    discount_service.get_discount.return_value = 0.1  # 10% off
    
    order = Order(discount_service=discount_service)
    order.add_item(Product(price=100))
    
    assert order.calculate_total() == 90
```

### Fakes (Working Implementations)
```python
class FakeUserRepository:
    def __init__(self):
        self.users = {}
    
    def save(self, user):
        self.users[user.id] = user
    
    def find(self, user_id):
        return self.users.get(user_id)

def test_user_registration():
    repo = FakeUserRepository()
    service = UserService(repo)
    
    user = service.register(email="test@example.com")
    
    assert repo.find(user.id) is not None
```

## Edge Cases to Test

```python
# Boundaries
def test_pagination_first_page():
    assert paginate(items, page=1, per_page=10)

def test_pagination_last_page():
    assert paginate(items, page=10, per_page=10)

# Empty/Null
def test_search_with_empty_query():
    assert search("") == []

def test_user_with_no_orders():
    assert user.get_orders() == []

# Invalid Input
def test_negative_quantity_raises_error():
    with pytest.raises(ValueError):
        order.add_item(product, quantity=-1)

# Concurrency (if applicable)
def test_concurrent_updates_dont_lose_data():
    ...
```

## Test Organization

### By Feature
```
tests/
├── test_user_registration.py
├── test_user_login.py
├── test_order_creation.py
└── test_order_checkout.py
```

### By Type
```
tests/
├── unit/
│   ├── test_user.py
│   └── test_order.py
├── integration/
│   ├── test_user_api.py
│   └── test_order_api.py
└── e2e/
    └── test_checkout_flow.py
```

## TDD Anti-Patterns

| Anti-Pattern | Problem | Solution |
|--------------|---------|----------|
| Test after | Less design benefit | Always write test first |
| Too many mocks | Brittle, coupled | Use real objects or fakes |
| Testing implementation | Refactor breaks tests | Test behavior, not internals |
| Slow tests | Developer avoids running | Mock slow dependencies |
| Flaky tests | Lose trust in tests | Fix immediately or delete |
| No refactor step | Accumulating debt | Always refactor after green |
