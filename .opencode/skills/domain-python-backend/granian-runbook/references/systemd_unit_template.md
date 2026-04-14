# systemd unit template (Granian)

> This is a template. Do NOT paste secrets. Prefer environment files with restricted permissions.

```ini
[Unit]
Description=__SERVICE_NAME__ (Granian)
After=network.target

[Service]
Type=simple
User=__USER__
Group=__GROUP__
WorkingDirectory=__WORKDIR__

# Prefer an EnvironmentFile rather than inline secrets
EnvironmentFile=-/etc/__SERVICE_NAME__/env

# Command: replace with your granian invocation or python module entry
ExecStart=__EXEC_START__

# Graceful shutdown
TimeoutStopSec=__SECONDS__
KillSignal=SIGTERM
Restart=on-failure
RestartSec=2

# Hardening (adjust as needed)
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=__RW_PATHS__

[Install]
WantedBy=multi-user.target
```
