# Security Baseline (Network/ACL/TLS)

## Network isolation
- Redis should not be reachable from the public internet.
- Restrict access by VPC/subnet/security group/firewall.

## AuthN/AuthZ
- Use ACL users with least privilege (separate users per service if feasible).
- Store credentials in secret manager / environment injection (never in code).

## Transport security
- Prefer TLS in transit when crossing trust boundaries.

## Dangerous commands
- In production, restrict/disable operationally dangerous commands (e.g., `FLUSHALL`, `FLUSHDB`, `CONFIG`, `KEYS`).
- Prefer operational runbooks and explicit break-glass procedures.

## Data sensitivity
- Assume Redis may contain sensitive derived data (sessions, tokens, user hints).
- If persistence (RDB/AOF) is enabled, backups and disks become sensitive assets.

## Multi-tenant notes
- If multiple services share one Redis, keyspace discipline is mandatory.
- Prefer separate logical DBs only as a convenience; security boundaries should be at network/ACL level.
