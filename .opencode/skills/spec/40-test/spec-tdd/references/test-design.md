# Test Design Strategies

## Test Pyramid

```
        /\
       /  \      E2E Tests (Few)
      /────\     - Slow, expensive
     /      \    - Test critical paths
    /        \   
   /──────────\  Integration Tests (Some)
  /            \ - Test component interactions
 /              \ - API contracts, database
/────────────────\ Unit Tests (Many)
                  - Fast, isolated
                  - Test logic, edge cases
```

### Ratio Guidelines
- Unit: ~70%
- Integration: ~20%
- E2E: ~10%

## Unit Test Design

### What to Unit Test
```python
# Pure logic
def test_calculate_discount():
    assert calculate_discount(total=100, tier="gold") == 15

# State transitions
def test_order_can_be_cancelled_when_pending():
    order = Order(status="pending")
    order.cancel()
    assert order.status == "cancelled"

# Edge cases
def test_calculate_discount_with_zero_total():
    assert calculate_discount(total=0, tier="gold") == 0

# Error handling
def test_order_cannot_be_cancelled_when_shipped():
    order = Order(status="shipped")
    with pytest.raises(InvalidStateError):
        order.cancel()
```

### What NOT to Unit Test
```python
# Don't test implementation details
def test_uses_hashmap():  # ❌
    assert isinstance(cache._storage, dict)

# Don't test third-party libraries
def test_json_serialization():  # ❌
    assert json.dumps({"a": 1}) == '{"a": 1}'

# Don't test trivial getters/setters
def test_user_name_getter():  # ❌
    user = User(name="John")
    assert user.name == "John"
```

## Integration Test Design

### Database Integration
```python
@pytest.fixture
def db_session():
    """Create a test database session with rollback."""
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()

def test_user_repository_saves_and_retrieves(db_session):
    repo = UserRepository(db_session)
    user = User(email="test@example.com")
    
    repo.save(user)
    retrieved = repo.find(user.id)
    
    assert retrieved.email == "test@example.com"
```

### API Integration
```python
def test_create_order_api(client, auth_token):
    response = client.post(
        "/api/orders",
        json={"product_id": "prod_123", "quantity": 2},
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    
    assert response.status_code == 201
    assert "order_id" in response.json()
    
    # Verify side effects
    order = Order.query.get(response.json()["order_id"])
    assert order.status == "pending"
```

### External Service Integration
```python
@pytest.fixture
def payment_service(httpx_mock):
    """Mock external payment service."""
    httpx_mock.add_response(
        url="https://api.stripe.com/v1/charges",
        json={"id": "ch_123", "status": "succeeded"}
    )
    return PaymentService()

def test_process_payment(payment_service):
    result = payment_service.charge(amount=1000, token="tok_visa")
    
    assert result.success is True
    assert result.charge_id == "ch_123"
```

## E2E Test Design

### Critical Path Testing
```python
def test_complete_checkout_flow(browser, test_user):
    """Test the happy path of checkout."""
    # Login
    browser.goto("/login")
    browser.fill("email", test_user.email)
    browser.fill("password", "password123")
    browser.click("button[type=submit]")
    
    # Add to cart
    browser.goto("/products/widget-123")
    browser.click("button.add-to-cart")
    
    # Checkout
    browser.click("a.checkout")
    browser.fill("card_number", "4242424242424242")
    browser.fill("card_expiry", "12/25")
    browser.fill("card_cvc", "123")
    browser.click("button.pay")
    
    # Verify
    assert browser.text("h1") == "Order Confirmed"
    assert "order_" in browser.url
```

### E2E Anti-Patterns
```python
# ❌ Testing every minor flow
def test_user_can_change_font_size():  # Too granular for E2E
    ...

# ❌ Unstable selectors
browser.click("div.MuiBox-root > div:nth-child(2) > button")  # Fragile

# ✅ Use test IDs
browser.click("[data-testid='checkout-button']")  # Stable
```

## Test Data Strategies

### Factories
```python
# Using factory_boy
class UserFactory(factory.Factory):
    class Meta:
        model = User
    
    email = factory.Sequence(lambda n: f"user{n}@example.com")
    name = factory.Faker("name")
    created_at = factory.LazyFunction(datetime.now)

def test_user_registration():
    user = UserFactory(email="custom@example.com")
    assert user.email == "custom@example.com"
```

### Builders
```python
class OrderBuilder:
    def __init__(self):
        self._customer = None
        self._items = []
        self._status = "pending"
    
    def with_customer(self, customer):
        self._customer = customer
        return self
    
    def with_item(self, product, quantity=1):
        self._items.append((product, quantity))
        return self
    
    def with_status(self, status):
        self._status = status
        return self
    
    def build(self):
        order = Order(customer=self._customer, status=self._status)
        for product, qty in self._items:
            order.add_item(product, qty)
        return order

def test_order_with_multiple_items():
    order = (OrderBuilder()
        .with_customer(create_customer())
        .with_item(create_product("Widget"), quantity=2)
        .with_item(create_product("Gadget"), quantity=1)
        .build())
    
    assert len(order.items) == 2
```

### Fixtures
```python
@pytest.fixture
def premium_user():
    return User(
        email="premium@example.com",
        tier="premium",
        credits=1000
    )

@pytest.fixture
def order_with_items(premium_user):
    order = Order(customer=premium_user)
    order.add_item(Product(name="Widget", price=100))
    return order

def test_premium_user_gets_discount(order_with_items):
    total = order_with_items.calculate_total()
    assert total < 100  # Premium discount applied
```

## Contract Testing

### Provider Test
```python
# Order service (provider)
def test_get_order_contract():
    """Verify we meet the contract expected by consumers."""
    response = client.get("/api/orders/123")
    
    # Match the schema consumers expect
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "status" in data
    assert data["status"] in ["pending", "completed", "cancelled"]
```

### Consumer Test
```python
# Checkout service (consumer)
def test_order_service_contract(order_service_mock):
    """Verify our expectations of the order service."""
    # Define expected response
    order_service_mock.expect_get(
        "/api/orders/123",
        response={"id": "123", "status": "pending"}
    )
    
    # Use it
    order = order_client.get_order("123")
    assert order.status == "pending"
    
    # Verify mock was called as expected
    order_service_mock.verify()
```

## Property-Based Testing

```python
from hypothesis import given, strategies as st

@given(st.integers(), st.integers())
def test_add_is_commutative(a, b):
    assert add(a, b) == add(b, a)

@given(st.lists(st.integers()))
def test_sort_maintains_length(items):
    sorted_items = sorted(items)
    assert len(sorted_items) == len(items)

@given(st.emails())
def test_email_validation(email):
    # Should not raise
    validate_email(email)
```

## Test Coverage Strategy

### Meaningful Coverage
```python
# Not just line coverage - test behaviors

# ❌ 100% coverage but meaningless
def test_all_lines_executed():
    obj = MyClass()
    obj.do_something()  # Just calling, no assertions

# ✅ Meaningful coverage
def test_do_something_creates_record():
    obj = MyClass()
    obj.do_something()
    assert len(obj.records) == 1
    assert obj.records[0].status == "created"
```

### Coverage Targets
```yaml
coverage:
  minimum: 80%
  branches: true
  exclude:
    - "*/migrations/*"
    - "*/tests/*"
    - "*/__init__.py"
```
