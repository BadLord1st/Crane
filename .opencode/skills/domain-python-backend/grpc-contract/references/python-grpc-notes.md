# Python gRPC Notes

- Prefer channel reuse (channels are multiplexed); do not create a channel per request.
- Server concurrency:
  - Sync: tune ThreadPoolExecutor(max_workers=...) for expected load.
  - Async: use grpc.aio to integrate with asyncio; avoid blocking calls in handlers.
- Interceptors:
  - Use server interceptors for exception→StatusCode mapping, logging, metrics, tracing hooks.
  - Use client interceptors for deadlines defaults, retries policy enforcement (if used), tracing metadata injection.
- Operability:
  - Health checking and (optionally) reflection can be enabled for internal environments; gate in prod if needed.
