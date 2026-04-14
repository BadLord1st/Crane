# TLS & Proxy Checklist (Granian)

- Where is TLS terminated?
  - [ ] At ingress / reverse proxy (preferred)
  - [ ] In-app (only if explicitly required)

- Reverse proxy basics
  - [ ] Proxy passes `X-Forwarded-For`, `X-Forwarded-Proto` (or equivalent)
  - [ ] App/framework is configured to trust proxy headers appropriately
  - [ ] Reasonable timeouts (connect / read / write) align with app request timeouts
  - [ ] HTTP keep-alive enabled between proxy <-> app

- Security headers (proxy level)
  - [ ] HSTS (if HTTPS-only)
  - [ ] CSP (if browser-facing)

- Rate limiting / WAF (optional)
  - [ ] Basic rate limiting
  - [ ] Bot protection if needed

- Observability
  - [ ] Request ID / Trace ID propagated from proxy to app
  - [ ] Access logs include upstream response time
