# Docker Compose snippet (Granian)

> Template only. Do NOT bake secrets into the image. Prefer runtime env injection.

```yaml
services:
  app:
    image: __IMAGE__
    command: ["__REPLACE_WITH_GRANIAN_CMD__"]
    ports:
      - "8000:8000"
    environment:
      ENV: "prod"
    # If using an env_file, keep it out of git
    # env_file:
    #   - .env
    deploy:
      resources:
        limits:
          cpus: "__CPUS__"
          memory: "__MEM__"
```
