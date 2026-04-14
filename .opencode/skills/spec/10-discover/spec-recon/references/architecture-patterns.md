# Architecture Patterns Recognition

## Layered Architecture

### Detection
```
project/
├── presentation/  or  handlers/  or  api/  or  routes/
├── business/      or  services/  or  domain/
├── persistence/   or  repositories/  or  data/
└── models/        or  entities/
```

### Characteristics
- Clear separation by technical concern
- Each layer only calls the layer below
- Easy to understand, common pattern

### Code Signs
```python
# Handler → Service → Repository chain
@app.post("/users")
def create_user(data: UserCreate):
    return user_service.create(data)  # Calls service

class UserService:
    def create(self, data):
        return self.repository.save(User(**data))  # Calls repo

class UserRepository:
    def save(self, user):
        db.add(user)  # Calls DB
```

---

## Hexagonal Architecture (Ports & Adapters)

### Detection
```
project/
├── core/           # Domain logic, no external deps
│   ├── domain/
│   └── ports/      # Interfaces
├── adapters/       # Implementations
│   ├── http/
│   ├── database/
│   └── messaging/
└── config/
```

### Characteristics
- Core has zero external dependencies
- Ports define interfaces (what we need)
- Adapters implement interfaces (how we get it)
- Easy to swap implementations

### Code Signs
```python
# Port (interface)
class UserRepository(Protocol):
    def find(self, user_id: str) -> User: ...
    def save(self, user: User) -> None: ...

# Adapter (implementation)
class PostgresUserRepository:
    def find(self, user_id: str) -> User:
        return db.query(User).get(user_id)

# Core uses port, doesn't know about Postgres
class UserService:
    def __init__(self, repo: UserRepository):
        self.repo = repo  # Injected
```

---

## Domain-Driven Design (DDD)

### Detection
```
project/
├── domain/
│   ├── user/
│   │   ├── aggregate.py
│   │   ├── entity.py
│   │   ├── value_object.py
│   │   ├── repository.py
│   │   └── service.py
│   └── order/
│       ├── aggregate.py
│       └── ...
├── application/
└── infrastructure/
```

### Characteristics
- Organized by business domain
- Rich domain models (not anemic)
- Ubiquitous language in code
- Aggregates protect invariants

### Code Signs
```python
# Aggregate root
class Order:
    def add_item(self, product: Product, quantity: int):
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        self.items.append(OrderItem(product, quantity))
        self._recalculate_total()
    
    def submit(self):
        if not self.items:
            raise DomainError("Cannot submit empty order")
        self.status = OrderStatus.SUBMITTED
        self.submitted_at = datetime.now()

# Value object (immutable)
@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str
    
    def add(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError("Currency mismatch")
        return Money(self.amount + other.amount, self.currency)
```

---

## Clean Architecture

### Detection
```
project/
├── entities/       # Enterprise business rules
├── use_cases/      # Application business rules
├── interface_adapters/  # Controllers, Presenters, Gateways
└── frameworks/     # Web, DB, external tools
```

### Characteristics
- Dependency rule: outer → inner only
- Use cases orchestrate domain logic
- Entities are the most stable
- Framework is the most volatile

### Code Signs
```python
# Entity (innermost)
class User:
    def can_perform(self, action: str) -> bool:
        return action in self.permissions

# Use case (orchestrates)
class CreateOrderUseCase:
    def __init__(self, user_repo, order_repo, payment_gateway):
        ...
    
    def execute(self, request: CreateOrderRequest) -> CreateOrderResponse:
        user = self.user_repo.find(request.user_id)
        if not user.can_perform("create_order"):
            raise PermissionDenied()
        ...

# Controller (outer)
@app.post("/orders")
def create_order(request: Request):
    use_case = CreateOrderUseCase(...)
    response = use_case.execute(CreateOrderRequest(...))
    return JSONResponse(response)
```

---

## Microservices

### Detection
```
monorepo/
├── services/
│   ├── user-service/
│   │   ├── Dockerfile
│   │   ├── src/
│   │   └── tests/
│   ├── order-service/
│   └── payment-service/
├── shared/
└── docker-compose.yml
```

Or separate repositories per service.

### Characteristics
- Independent deployment
- Service-to-service communication (HTTP/gRPC/messaging)
- Each service owns its data
- Distributed complexity

### Code Signs
```python
# HTTP client to other service
class UserClient:
    def get_user(self, user_id: str) -> User:
        response = httpx.get(f"{USER_SERVICE_URL}/users/{user_id}")
        return User(**response.json())

# Event publishing
class OrderService:
    def create(self, order):
        db.save(order)
        event_bus.publish("order.created", OrderCreatedEvent(order.id))

# Event consuming
@event_handler("order.created")
def handle_order_created(event: OrderCreatedEvent):
    send_confirmation_email(event.order_id)
```

---

## Modular Monolith

### Detection
```
project/
├── modules/
│   ├── users/
│   │   ├── api.py
│   │   ├── service.py
│   │   └── models.py
│   ├── orders/
│   └── payments/
├── shared/
└── main.py
```

### Characteristics
- Single deployable unit
- Strong module boundaries
- Modules communicate via defined interfaces
- Easy path to microservices later

### Code Signs
```python
# Module interface
# modules/users/__init__.py
from .api import router as users_router
from .service import UserService

# Cross-module communication
# modules/orders/service.py
from modules.users import UserService  # Import from public interface

class OrderService:
    def __init__(self, user_service: UserService):
        self.user_service = user_service
```

---

## Pattern Detection Checklist

| Pattern | Look For |
|---------|----------|
| Layered | handlers/ → services/ → repositories/ |
| Hexagonal | ports/, adapters/, core with no imports |
| DDD | aggregates, value objects, domain events |
| Clean | use_cases/, entities/ with dependency inversion |
| Microservices | Multiple Dockerfiles, service-to-service clients |
| Modular Monolith | modules/ with clear boundaries |

## Anti-Pattern Recognition

### Big Ball of Mud
- No clear structure
- Everything imports everything
- Business logic in handlers
- No tests

### Anemic Domain Model
- Domain objects are just data containers
- All logic in services
- No encapsulation

### Distributed Monolith
- Multiple services, but tightly coupled
- Synchronized releases required
- Shared database
