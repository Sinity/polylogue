# Security Model

Polylogue is a local-first archive. The daemon API is designed for
loopback access by the CLI, TUI, web reader, and browser extension.

## Threat Model

The primary threat is a malicious web page loaded in a local browser
making requests to the daemon API. Since browsers allow localhost
requests from any origin by default, any page can reach the daemon.

### Attack Surface

| Endpoint | Method | Risk |
|---|---|---|
| `/api/status` | GET | Read-only, low risk |
| `/api/conversations` | GET | Read-only, low risk |
| `/api/health` | GET | Read-only, low risk |
| `/api/ingest` | POST | **Mutating** — stages files for ingestion |
| `/api/reset` | POST | **Destructive** — resets archive state |
| `/api/backfill` | POST | **Mutating** — triggers rebuild |

### Mitigations

1. **Auth token**: When `--api-auth-token` is configured, ALL clients
   (including localhost) must present a `Bearer` token. Only skip auth
   when no token is configured (local dev default).

2. **Loopback-only by default**: The daemon binds to `127.0.0.1` by
   default. Remote binding requires explicit `--insecure-allow-remote`.

3. **No CORS allowlist by default**: Cross-origin requests are blocked
   unless `--browser-capture-origin` is configured.

### Future

- CSRF tokens for mutating endpoints
- Fine-grained token scopes (read vs write)
- TLS for non-loopback binds
