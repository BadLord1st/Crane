# Performance Optimization Patterns

## Caching Patterns

### 1. Read-Through Cache
```python
def get_user(user_id: str) -> User:
    # Try cache first
    cached = cache.get(f"user:{user_id}")
    if cached:
        return User.from_json(cached)
    
    # Cache miss: fetch from DB
    user = db.query(User).get(user_id)
    
    # Store in cache
    cache.set(f"user:{user_id}", user.to_json(), ttl=3600)
    
    return user
```

### 2. Write-Through Cache
```python
def update_user(user_id: str, data: dict) -> User:
    # Update DB
    user = db.query(User).get(user_id)
    user.update(**data)
    db.commit()
    
    # Update cache synchronously
    cache.set(f"user:{user_id}", user.to_json(), ttl=3600)
    
    return user
```

### 3. Cache-Aside (Lazy Loading)
```python
def get_user(user_id: str) -> User:
    cache_key = f"user:{user_id}"
    
    # Application manages cache
    cached = cache.get(cache_key)
    if cached:
        return User.from_json(cached)
    
    user = db.query(User).get(user_id)
    if user:
        cache.set(cache_key, user.to_json(), ttl=3600)
    
    return user

def update_user(user_id: str, data: dict):
    user = db.query(User).get(user_id)
    user.update(**data)
    db.commit()
    
    # Invalidate cache (lazy reload on next read)
    cache.delete(f"user:{user_id}")
```

### 4. Write-Behind (Async Write)
```python
def update_user(user_id: str, data: dict):
    # Update cache immediately
    current = cache.get(f"user:{user_id}")
    updated = {**current, **data}
    cache.set(f"user:{user_id}", updated, ttl=3600)
    
    # Queue async DB write
    queue.enqueue("update_user_db", user_id=user_id, data=data)
```

## Database Optimization

### N+1 Query Prevention
```python
# ❌ N+1 Problem
users = db.query(User).all()
for user in users:
    orders = user.orders  # N additional queries!

# ✅ Eager Loading
users = db.query(User).options(joinedload(User.orders)).all()
for user in users:
    orders = user.orders  # Already loaded

# ✅ Select Related (Django)
users = User.objects.select_related('profile').prefetch_related('orders')
```

### Connection Pooling
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,           # Steady-state connections
    max_overflow=20,        # Burst capacity
    pool_timeout=30,        # Wait time for connection
    pool_recycle=3600,      # Recycle connections hourly
    pool_pre_ping=True,     # Test connections before use
)
```

### Query Optimization
```python
# ❌ Over-fetching
users = db.query(User).all()  # Fetches all columns

# ✅ Select only needed columns
users = db.query(User.id, User.name, User.email).all()

# ❌ Filtering in Python
all_users = db.query(User).all()
active_users = [u for u in all_users if u.is_active]

# ✅ Filtering in database
active_users = db.query(User).filter(User.is_active == True).all()
```

## API Optimization

### Pagination
```python
@app.get("/api/users")
def list_users(
    page: int = 1,
    per_page: int = 20,
    cursor: str = None  # For cursor-based
):
    # Offset pagination (simple but slow for large offsets)
    offset = (page - 1) * per_page
    users = db.query(User).offset(offset).limit(per_page).all()
    
    # Cursor pagination (efficient for large datasets)
    if cursor:
        decoded = decode_cursor(cursor)
        users = db.query(User)\
            .filter(User.id > decoded)\
            .order_by(User.id)\
            .limit(per_page + 1)\
            .all()
        
        has_next = len(users) > per_page
        next_cursor = encode_cursor(users[-2].id) if has_next else None
        users = users[:per_page]
```

### Compression
```python
# FastAPI with gzip middleware
from starlette.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Or in reverse proxy (nginx)
# gzip on;
# gzip_types application/json;
```

### Response Streaming
```python
from fastapi.responses import StreamingResponse

@app.get("/api/large-export")
async def export_data():
    async def generate():
        yield "["
        first = True
        async for row in db.stream_query(LargeTable):
            if not first:
                yield ","
            first = False
            yield row.to_json()
        yield "]"
    
    return StreamingResponse(generate(), media_type="application/json")
```

## Async Patterns

### Parallel API Calls
```python
import asyncio
import httpx

async def fetch_user_data(user_id: str):
    async with httpx.AsyncClient() as client:
        # Sequential: 300ms + 200ms + 100ms = 600ms
        # profile = await client.get(f"/api/profile/{user_id}")
        # orders = await client.get(f"/api/orders/{user_id}")
        # prefs = await client.get(f"/api/preferences/{user_id}")
        
        # Parallel: max(300ms, 200ms, 100ms) = 300ms
        profile, orders, prefs = await asyncio.gather(
            client.get(f"/api/profile/{user_id}"),
            client.get(f"/api/orders/{user_id}"),
            client.get(f"/api/preferences/{user_id}"),
        )
        
        return {
            "profile": profile.json(),
            "orders": orders.json(),
            "preferences": prefs.json(),
        }
```

### Background Tasks
```python
from fastapi import BackgroundTasks

@app.post("/api/orders")
async def create_order(order: Order, background_tasks: BackgroundTasks):
    # Fast response path
    order = db.create_order(order)
    
    # Defer slow operations
    background_tasks.add_task(send_confirmation_email, order.id)
    background_tasks.add_task(update_analytics, order.id)
    background_tasks.add_task(sync_inventory, order.items)
    
    return order
```

## Memory Optimization

### Generator for Large Datasets
```python
# ❌ Loads all into memory
def process_all():
    items = list(db.query(Item).all())  # 1M rows in memory
    for item in items:
        process(item)

# ✅ Streams one at a time
def process_all():
    for item in db.query(Item).yield_per(1000):
        process(item)
```

### Object Pooling
```python
from queue import Queue

class ConnectionPool:
    def __init__(self, max_size=10):
        self.pool = Queue(maxsize=max_size)
        for _ in range(max_size):
            self.pool.put(self._create_connection())
    
    def get(self):
        return self.pool.get()
    
    def release(self, conn):
        self.pool.put(conn)
    
    def _create_connection(self):
        return expensive_connection_create()
```

## Frontend Performance

### Code Splitting
```javascript
// React lazy loading
const Dashboard = React.lazy(() => import('./Dashboard'));

function App() {
  return (
    <Suspense fallback={<Loading />}>
      <Dashboard />
    </Suspense>
  );
}
```

### Image Optimization
```html
<!-- Responsive images -->
<img 
  srcset="image-320.jpg 320w,
          image-640.jpg 640w,
          image-1280.jpg 1280w"
  sizes="(max-width: 320px) 280px,
         (max-width: 640px) 580px,
         1200px"
  src="image-1280.jpg"
  alt="Description"
  loading="lazy"
  decoding="async"
/>
```

### Bundle Analysis
```bash
# Webpack
npx webpack-bundle-analyzer stats.json

# Vite
npx vite-bundle-visualizer

# Next.js
ANALYZE=true npm run build
```
