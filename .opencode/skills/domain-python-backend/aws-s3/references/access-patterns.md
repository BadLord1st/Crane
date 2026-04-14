# Access Patterns

## Default pattern: private bucket + presigned URLs
- Use presigned URLs for direct client upload/download to keep the backend stateless and avoid proxying payloads.
- Never expose the whole bucket publicly unless there is a strict, reviewed requirement.

## Upload flow (conceptual)
1) Client asks backend for an upload intent (size/type/key constraints).
2) Backend returns presigned PUT (short TTL) + expected ContentType/metadata/tags.
3) Client uploads directly to S3 using presigned PUT.
4) Backend records object key + metadata (and optionally validates via HEAD).

## Download flow (conceptual)
- Backend authorizes user -> returns presigned GET (short TTL), or serves via CDN with signed URLs/cookies.

## Policy implications
- Backend role: s3:PutObject / s3:GetObject limited to specific bucket + prefixes.
- Prefer scoping by prefix (per-tenant, per-domain). Avoid wide wildcard permissions.
- Avoid object ACL-based public-read defaults; treat as an exception-only mechanism.

## When public is required
- Static website buckets: document exception, minimal public policy, and explicit sign-off.
- For public distribution at scale: prefer CDN in front of S3 (and keep bucket private when possible).
