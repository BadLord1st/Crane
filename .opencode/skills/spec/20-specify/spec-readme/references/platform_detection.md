# Deployment Platform Detection

Do not guess the platform. Detect it from repo artifacts, then document what exists.

## Common deployment signals

- Docker
  - `Dockerfile`
  - `docker-compose.yml` / `compose.yaml`

- Kubernetes
  - `k8s/**`, `kubernetes/**`
  - `helm/**`

- Terraform / IaC
  - `terraform/**`, `*.tf`

- Vercel
  - `vercel.json`
  - `.vercel/**`

- Netlify
  - `netlify.toml`

- Fly.io
  - `fly.toml`

- Render
  - `render.yaml`

- Railway
  - `railway.toml`, `railway.json`

- Heroku-like
  - `Procfile`
  - `app.json`

- Google App Engine
  - `app.yaml`

- Serverless
  - `serverless.yml`

## Default fallback

If no deployment target is detectable:
- Prefer documenting Docker-based deployment if a `Dockerfile` exists.
- Otherwise, keep deployment section minimal and point to what would be required (build output, required env vars, and the start command).
