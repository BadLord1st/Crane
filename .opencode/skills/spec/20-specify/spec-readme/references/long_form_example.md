# Long-Form Example (Skeleton)

This is a more verbose, end-to-end README skeleton. Use selectively.

```md
# <Project Name>

<Short overview.>

## Table of Contents
- [Quickstart](#quickstart)
- [Local Development](#local-development)
- [Common Commands](#common-commands)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## Quickstart

```bash
<one-liner to run, if the repo supports it>
```

## Local Development

### Prerequisites
- <runtime>
- <package manager>
- <db/cache>

### Setup

```bash
git clone <...>
cd <...>
<install deps>
cp <env example> <env>
<start dependencies>
<run app>
```

### Verify

```bash
<run tests>
```

## Common Commands

| Command | Description |
|---|---|
| `<run>` | Run app |
| `<test>` | Run tests |
| `<lint>` | Run lint |
| `<typecheck>` | Run typecheck |
| `<format>` | Format code |

## Architecture

### Directory Layout

```
<top-level folders and what they mean>
```

### Request / Data Flow (if applicable)

```
<client> -> <api> -> <service> -> <db>
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|---|---:|---|
| `<VAR>` | yes | ... |
| `<VAR>` | no | ... |

### Secrets

- <where secrets come from in dev>
- <where secrets come from in prod>

## Deployment

Document what exists in this repo (CI workflows, Docker build, Helm chart, etc.).
If there is a runbook/spec node, link it here.

## Troubleshooting

### <Symptom>

**What you see:** `<error>`

**Likely cause:** ...

**Fix:**

```bash
<commands>
```
```
