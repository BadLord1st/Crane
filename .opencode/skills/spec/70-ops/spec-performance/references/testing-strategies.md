# Performance Testing Strategies

## Test Types

| Type | Purpose | When to Run |
|------|---------|-------------|
| Smoke | Verify basic functionality | Every deploy |
| Load | Normal expected load | Weekly, pre-release |
| Stress | Beyond normal load | Monthly, pre-launch |
| Spike | Sudden load increase | Pre-launch, capacity planning |
| Soak | Extended duration | Pre-launch, memory leak detection |

## Performance Budgets

### API Response Time
```yaml
budgets:
  api:
    p50: 50ms
    p95: 200ms
    p99: 500ms
    max: 1000ms  # Hard limit

  database:
    simple_query: 10ms
    complex_query: 100ms
    
  external_calls:
    timeout: 5000ms
    p95: 500ms
```

### Web Vitals (Frontend)
```yaml
core_web_vitals:
  LCP: 2500ms   # Largest Contentful Paint
  FID: 100ms    # First Input Delay
  CLS: 0.1      # Cumulative Layout Shift
  TTFB: 800ms   # Time to First Byte
  FCP: 1800ms   # First Contentful Paint
```

### Resource Budgets
```yaml
resources:
  javascript: 200KB  # gzipped
  css: 50KB          # gzipped
  images: 500KB      # per page
  fonts: 100KB       # total
```

## Load Testing with k6

### Basic Load Test
```javascript
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '1m', target: 50 },   // Ramp up
    { duration: '3m', target: 50 },   // Sustain
    { duration: '1m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<200'],  // 95% under 200ms
    http_req_failed: ['rate<0.01'],    // <1% error rate
  },
};

export default function () {
  const res = http.get('https://api.example.com/users');
  
  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 200ms': (r) => r.timings.duration < 200,
  });
  
  sleep(1);
}
```

### Stress Test
```javascript
export const options = {
  stages: [
    { duration: '2m', target: 100 },
    { duration: '5m', target: 100 },
    { duration: '2m', target: 200 },
    { duration: '5m', target: 200 },
    { duration: '2m', target: 300 },
    { duration: '5m', target: 300 },
    { duration: '5m', target: 0 },
  ],
};
```

### Spike Test
```javascript
export const options = {
  stages: [
    { duration: '10s', target: 100 },
    { duration: '1m', target: 100 },
    { duration: '10s', target: 1000 },  // Spike!
    { duration: '3m', target: 1000 },
    { duration: '10s', target: 100 },
    { duration: '3m', target: 100 },
    { duration: '10s', target: 0 },
  ],
};
```

## Python Benchmarking

### pytest-benchmark
```python
import pytest

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def test_fibonacci_performance(benchmark):
    result = benchmark(fibonacci, 20)
    assert result == 6765

# With custom settings
def test_api_performance(benchmark, client):
    result = benchmark.pedantic(
        client.get,
        args=('/api/users',),
        iterations=10,
        rounds=5,
        warmup_rounds=2
    )
    assert result.status_code == 200
```

### locust
```python
from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 5)
    
    @task(3)
    def get_users(self):
        self.client.get("/api/users")
    
    @task(1)
    def create_user(self):
        self.client.post("/api/users", json={
            "name": "Test User",
            "email": "test@example.com"
        })
    
    def on_start(self):
        # Login
        self.client.post("/api/login", json={
            "username": "testuser",
            "password": "testpass"
        })
```

## Database Performance

### Query Analysis
```sql
-- PostgreSQL: Explain analyze
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT * FROM orders 
WHERE user_id = 123 
AND created_at > NOW() - INTERVAL '30 days';

-- Look for:
-- - Seq Scan (missing index)
-- - High buffer reads
-- - Actual rows >> Estimated rows
```

### Slow Query Detection
```sql
-- Enable slow query log
ALTER SYSTEM SET log_min_duration_statement = '100ms';
SELECT pg_reload_conf();

-- Find slow queries
SELECT query, 
       calls,
       total_time / calls as avg_time,
       rows / calls as avg_rows
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 20;
```

### Index Recommendations
```sql
-- Find missing indexes (PostgreSQL)
SELECT schemaname, relname, 
       seq_scan, seq_tup_read,
       idx_scan, idx_tup_fetch,
       seq_tup_read / seq_scan as avg_seq_rows
FROM pg_stat_user_tables
WHERE seq_scan > 0
ORDER BY seq_tup_read DESC
LIMIT 10;
```

## Profiling

### Python
```python
# cProfile
python -m cProfile -o profile.stats script.py

# Visualize with snakeviz
pip install snakeviz
snakeviz profile.stats

# Line profiler
pip install line_profiler
kernprof -l -v script.py

# Memory profiler
pip install memory_profiler
python -m memory_profiler script.py
```

### Node.js
```javascript
// Built-in profiler
node --prof app.js
node --prof-process isolate-*.log > profile.txt

// clinic.js for flame graphs
npx clinic flame -- node app.js
```

### Go
```go
import "net/http/pprof"

// Add to main
go func() {
    log.Println(http.ListenAndServe("localhost:6060", nil))
}()

// Profile
go tool pprof http://localhost:6060/debug/pprof/profile?seconds=30
```

## CI/CD Integration

### GitHub Actions
```yaml
name: Performance Tests

on:
  pull_request:
    branches: [main]

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Start application
        run: docker-compose up -d
      
      - name: Wait for health
        run: |
          timeout 60 bash -c 'until curl -s http://localhost:8080/health; do sleep 1; done'
      
      - name: Run k6 load test
        uses: grafana/k6-action@v0.3.1
        with:
          filename: tests/load/smoke.js
          
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: k6-results
          path: results/
```

## Performance Regression Detection

### Compare Against Baseline
```python
def test_api_regression(benchmark, baseline_file):
    result = benchmark(call_api)
    
    # Load baseline
    with open(baseline_file) as f:
        baseline = json.load(f)
    
    # Compare
    current_p95 = result.stats.median
    baseline_p95 = baseline['p95']
    
    # Allow 10% regression
    assert current_p95 < baseline_p95 * 1.1, \
        f"Performance regression: {current_p95}ms > {baseline_p95 * 1.1}ms"
```

### Automated Alerts
```yaml
# In k6
thresholds:
  http_req_duration:
    - threshold: 'p(95)<200'
      abortOnFail: true  # Stop if exceeded
      delayAbortEval: '10s'
```
