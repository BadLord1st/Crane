# Atomicity Patterns (Commands/Lua)

## Prefer atomic primitives
- Counters: `INCR`, `INCRBY`, `HINCRBY`.
- Sets: `SADD`, `SREM`, `SISMEMBER`.
- Sorted sets for leaderboards/rate windows: `ZADD`, `ZREMRANGEBYSCORE`.

Avoid: `GET` → compute → `SET` when a single atomic command can do it.

## TTL + counter correctness
If you need “INCR and set TTL on first hit”, use Lua to avoid races.

Template (pseudo):
```lua
local v = redis.call('INCR', KEYS[1])
if v == 1 then redis.call('PEXPIRE', KEYS[1], ARGV[1]) end
return v
```

## Idempotency keys
- Pattern: `SET key token NX EX ttl`.
- If `SET` fails, treat as duplicate request and return the stored result or a conflict.
- TTL must outlive the maximum retry window.

## Locks (only if necessary)
Acquire:
- `SET lock:{name} {token} NX PX {ttl_ms}`

Release (must check token, use Lua):
```lua
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
else
  return 0
end
```

Lock rules
- Always use TTL.
- Keep critical sections short.
- Prefer redesign (idempotency + atomic updates) over locks when possible.

## Transactions
- `WATCH/MULTI/EXEC` is acceptable for small optimistic transactions.
- Lua is often simpler and reduces round trips.
