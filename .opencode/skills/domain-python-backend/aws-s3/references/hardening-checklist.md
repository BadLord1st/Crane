# Hardening Checklist.md

## Guardrails (must-have)
- Block Public Access enabled (account + bucket), unless exception documented.
- Encryption at rest: default SSE enabled; define SSE-KMS only if compliance/controls require it.
- Least privilege policies: scope to bucket + prefix; no broad principals unless explicit public website case.
- Logging/audit strategy defined (CloudTrail data events and/or server access logs where justified).

## Data protection
- Versioning enabled where recovery from overwrite/delete matters.
- Lifecycle rules in place (expiration/transition); retention aligned with business/compliance.
- If WORM retention required: evaluate Object Lock + governance/compliance mode.

## Operational hygiene
- Key naming conventions + prefixing rules.
- ContentType/metadata rules documented for common object types.
- Large uploads: multipart approach + retries (use SDK defaults where possible).
- Cost notes: requests + egress + storage class choices; caching/CDN for read-heavy workloads.

## Verification checklist (evidence-oriented)
- Public access blocked (attempt public GET without auth fails).
- Encryption enforced (object headers / bucket default encryption confirmed).
- Policies validated (no unintended principals; prefix scoping correct).
- Lifecycle visible (rules attached; sample object transitions/expiry understood).
- Logs present and retained per policy.
