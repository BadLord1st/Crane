---
name: aws-s3
description: "Produce an Amazon S3 usage + hardening spec (buckets, access patterns, guardrails, lifecycle, logging). Plan-only: do not apply changes in AWS."
metadata:
  signature: "spec-s3-storage :: (SpecNode, Repo) -> S3StorageSpec"
---

## When to use
- The system needs blob/object storage (uploads, exports, backups, static assets, logs).
- You must formalize S3 security posture (no accidental public exposure, least-privilege).
- You need a repeatable S3 layout: buckets, prefixes, retention, and access patterns.

## Inputs
- SpecNode: what data goes to S3 (classification, size, read/write patterns, retention).
- Access model requirements: private-by-default vs static website, client direct upload/download, cross-account needs.
- Compliance constraints: encryption mode, retention/WORM, audit requirements.
- Repo context: current AWS/IaC style (Terraform/CloudFormation/CDK), SDK language (boto3, etc).

## Outputs
- A spec node covering:
  - Bucket map (domains), naming/prefix conventions, and ownership assumptions.
  - Access patterns (server-side only vs presigned URLs vs CloudFront).
  - Security baseline (Block Public Access, least-privilege policies, encryption defaults).
  - Data management (versioning, lifecycle/retention, optional Object Lock).
  - Logging/audit plan (CloudTrail + server access logs) and verification checklist.
  - “Risk matrix” of misconfigurations + mitigations.

## Protocol
1) Inventory current usage
   - Scan repo for S3 usage, bucket names, ACL usage, presigned URLs, and credential handling.
2) Choose bucket topology
   - Separate buckets by data domain / access pattern (e.g., uploads vs logs vs backups).
   - Define key prefixes and naming rules to avoid unbounded “hot prefixes” during list operations.
3) Define access patterns (default: private-by-default)
   - Prefer presigned URLs for browser/client upload/download when feasible (avoid proxying payloads).
   - If public distribution is required, prefer CDN in front of S3 (and keep bucket private when possible).
4) Security baseline (guardrails-first)
   - Require S3 Block Public Access at bucket/account level unless explicitly justified.
   - Define least-privilege: IAM role policy + bucket policy (scoped to bucket + prefixes).
   - Forbid broad principals (“*”) unless it is a deliberate public website case with explicit sign-off.
5) Encryption + data protection
   - Set default SSE (SSE-S3 baseline; SSE-KMS when compliance/controls require it).
   - Document KMS ownership (key policy, rotation, who can decrypt).
   - Enable versioning when recovery from delete/overwrite matters; define delete semantics.
   - If regulatory retention/WORM is required: evaluate Object Lock and retention modes.
6) Lifecycle + cost controls
   - Define lifecycle rules per prefix (transition to lower-cost storage, expiration).
   - Add guardrails for request costs and egress; recommend caching/CDN if read-heavy.
7) Logging + auditability
   - Choose logging: CloudTrail data events + S3 server access logs (where needed).
   - Define log destinations and retention for audit evidence.
8) Verification checklist
   - “Prove” controls: public access blocked, encryption enforced, policies least-privilege, lifecycle active, logs visible.

## Deliverables
- [ ] `spec-s3-storage` node linked from spec-index (or your equivalent).
- [ ] Bucket map + prefix conventions + access patterns (with rationale).
- [ ] Draft policies (IAM + bucket policy) as templates (placeholders, no real IDs/ARNs if unknown).
- [ ] Lifecycle/retention rules per bucket/prefix.
- [ ] Logging/audit plan + verification checklist + risk matrix.

## Safety
- This skill is spec-only. Do NOT run AWS CLI / apply IaC / change live resources.
- If the user requests applying changes: require explicit confirmation + dry-run plan + rollback plan.

## Anti-patterns
- Making buckets public “because it’s simpler”.
- Relying on object ACLs for broad access instead of well-scoped policies/presigned URLs.
- Hardcoding AWS keys in code/config; missing IAM role usage.
- No lifecycle rules (unbounded growth) or no versioning where recovery matters.
- Missing content-type/metadata conventions (clients behave inconsistently).
- No audit/log trail for security-sensitive buckets.

## Examples
- Request: “Спроектируй S3 для пользовательских загрузок (до 200MB), приватно, загрузка из браузера.”
  Output: bucket topology + presigned PUT/GET flow + least-privilege policy templates + lifecycle + verification.
- Request: “Нужен S3 для логов/архивов на 1 год хранения, потом удалять, аудит обязателен.”
  Output: retention/lifecycle spec + logging plan + encryption/KMS decision + risk matrix.
- Request: “Нужен публичный статический сайт на S3.”
  Output: explicit exception path: which Block Public Access settings must change, minimal public policy, plus audit notes.

## Edge cases
- Static website hosting requires deliberate public access configuration (treat as exception).
- Cross-account access (partner analytics, centralized logging): prefer scoped access points/policies.
- Very large objects / streaming uploads: enforce multipart strategy + retry semantics.
- Compliance retention: Object Lock may require special bucket settings and operational constraints.
- KMS-heavy workloads: watch for permission boundaries + operational overhead (key policy, grants).

## References
- [S3 Block Public Access](https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-control-block-public-access.html)
- [S3 Server-Side Encryption (SSE)](https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingServerSideEncryption.html)
- [S3 Versioning](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Versioning.html)
- [S3 Lifecycle Management](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lifecycle-mgmt.html)
- [S3 Server Access Logging](https://docs.aws.amazon.com/AmazonS3/latest/userguide/ServerLogs.html)
- [S3 Object Lock](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lock.html)
- [Access patterns & presigned URLs](./references/access-patterns.md)
- [Hardening checklist](./references/hardening-checklist.md)
