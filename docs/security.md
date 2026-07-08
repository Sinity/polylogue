# Security Model

Polylogue is a local-first archive. The daemon API is designed for
loopback access by the CLI, TUI, web reader, and browser extension.

## Threat Model

The primary threat is a malicious web page loaded in a local browser
making requests to the daemon API. Browsers allow localhost requests
from any origin by default, so any page the user visits can reach
loopback ports.

### Trust Boundaries

| Boundary | Inside | Outside |
|---|---|---|
| Loopback network trust | Same-machine processes (CLI, hooks, daemon) | Browser tabs the user happens to open |
| Browser-origin trust | The same-origin web shell served by the daemon itself | Every other tab, extension, iframe |
| Remote bind trust | Operator-acknowledged via `--insecure-allow-remote` + token | Default: refused at startup |
| Process trust | The user's own processes (same UID) | Other users on a multi-user box |

### Actors

- **Same-user local process** — CLI, hooks, scripts. Trusted: can read the SQLite archive directly anyway.
- **Hostile browser origin** — A page loaded in any browser tab the user has open. Untrusted: must be blocked from POSTing to mutating endpoints regardless of how it reaches the daemon.
- **Browser extension** — The Polylogue browser-capture extension uses a *separate* receiver with its own token (see `polylogue/browser_capture/server.py`). Aligned by design — different threat surface.
- **LAN neighbor** — Reachable only on `--insecure-allow-remote` with token. Default: refused at startup.

### Assets

- Raw archive data (session content, raw artifacts, blob store)
- Local filesystem paths surfaced via `/api/sources`
- Daemon control operations (`/api/reset`, `/api/ingest`, `/api/maintenance/*`)

### Attack Surface

| Endpoint | Method | Risk | Auth | Origin |
|---|---|---|---|---|
| `/api/status` | GET | Read-only metadata | Token (when configured) | Any |
| `/api/health`, `/api/health/check` | GET | Health probe | Token (when configured) | Any |
| `/api/sessions`, `/api/sessions/:id`, `/.../messages`, `/.../raw` | GET | Read session data | Token | Any |
| `/api/facets` | GET | Read-only aggregations | Token | Any |
| `/api/sources` | GET | Returns absolute filesystem paths to authenticated callers | Token | Any |
| `/api/raw_artifacts/:id` | GET | Returns raw session payload | Token | Any |
| `/api/reset` | POST | **Destructive** — resets archive state | Token | Same-origin |
| `/api/ingest` | POST | **Mutating** — schedules ingestion | Token | Same-origin |
| `/api/maintenance/plan`, `/api/maintenance/run` | POST | **Mutating** — runs maintenance backfills | Token | Same-origin |

## Mitigations

1. **Bearer token auth.** When `--api-auth-token` is configured, every
   request — including from loopback — must present a matching
   `Authorization: Bearer <token>` header. The pure-logic check lives
   at `polylogue/daemon/http.py:_check_auth_logic` and is invoked by
   `_dispatch_get` and `do_POST` before any handler runs.

2. **Loopback-only by default.** The daemon binds to `127.0.0.1` by
   default. The "loopback" predicate is the shared
   `polylogue/core/loopback.py:is_loopback_host` (RFC 5735: full
   `127.0.0.0/8`, `::1`, and `localhost`). Non-loopback bind
   (`--api-host 0.0.0.0` etc.) requires *both* `--insecure-allow-remote`
   (operator opts into the risk) *and* `--api-auth-token`; the
   enforcement lives in `polylogue/daemon/cli.py` and rejects either
   missing flag at startup.

3. **Origin allowlist for mutating endpoints.** POST endpoints reject
   requests whose `Origin` header points to a non-loopback host. The
   loopback host definition is shared with the browser-capture
   receiver via `polylogue/core/loopback.py:is_loopback_host` (which
   the receiver uses for bind validation); the daemon HTTP API
   additionally uses `is_loopback_origin` from the same module to
   parse browser-supplied `Origin` headers. Both follow RFC 5735: the
   entire `127.0.0.0/8` block, `::1`, and the literal `localhost`
   name (both `http` and `https` schemes). The `Origin` parser rejects
   malformed bracketed IPv6 forms such as `http://[::1].evil.com` or
   `http://[::1]:bad`. This is the CSRF boundary: a hostile page
   loaded in the user's browser cannot POST to the daemon even if it
   somehow learned the bearer token, because the browser attaches its
   own `Origin`. The check lives at
   `polylogue/daemon/http.py:_check_cross_origin`.

4. **No CORS preflight.** `OPTIONS` requests return `405 Method Not
   Allowed` by design. The daemon does not advertise `Access-Control-Allow-*`
   headers. Cross-origin browsers cannot meaningfully use the API even
   on read endpoints because the response lacks the CORS headers
   browsers require.

## Explicit Decisions

These decisions are documented here so a regression that quietly
flips one of them is recognized as a security policy change rather
than an implementation tweak.

### Raw artifacts are not content-redacted

`/api/raw_artifacts/:id` returns the full raw payload (JSONL or other
provider artifact). The bearer token implies operator-level trust:
the same operator can read the blob store directly. Redacting at the
API surface would create a false sense of isolation while leaving the
filesystem unguarded.

### `/api/sources` returns absolute paths

`/api/sources` returns absolute filesystem paths in the `root` field
to authenticated callers. The operator needs them for source diagnosis
(why didn't this directory ingest?). Same trust argument as raw
artifacts: same-user filesystem access already exposes these paths.

### `OPTIONS` returns 405

No CORS preflight is offered. The web shell and browser extension are
same-origin or use a dedicated receiver, both of which avoid the
preflight path. Adding CORS to the daemon API would be a deliberate
architectural change.

### Browser capture has its own token

The browser-capture receiver (`polylogue/browser_capture/server.py`)
is intentionally a separate component with its own
`--browser-capture-auth-token`, `--browser-capture-origin`, and
`--browser-capture-allow-remote` flags. The daemon HTTP API and the
browser-capture receiver have different trust models — extension
talking to receiver vs. CLI/hooks/web-shell talking to daemon — and
sharing a single token would conflate them.

Unlike the daemon HTTP API, the receiver requires a bearer token by
default even on plain loopback: without an explicit
`--browser-capture-auth-token`, one is auto-minted into a 0600 file
(`polylogued browser-capture token show` prints it for pairing) so that
no local process other than the paired extension can read receiver
state or post captures. `--browser-capture-allow-no-auth` is the
explicit, logged opt-out (polylogue-gnie).

## Non-Threats

Out of scope:

- **Multi-user access.** Polylogue is single-user; no user isolation.
- **TLS / HTTPS.** Loopback-only by default; remote bind is operator
  opt-in and out of scope for built-in TLS.
- **Encryption at rest.** SQLite is plaintext; disk encryption is the
  OS's responsibility.
- **CSRF tokens (in addition to Origin check).** The Origin allowlist
  is the CSRF boundary; we don't issue per-request CSRF tokens.
- **User accounts / RBAC / multi-user auth.** Single-user model.
- **Token-leak prevention beyond our own logs.** We do not log the
  token; we do not put it in URLs. Operators that paste it into shell
  history, ticket trackers, or screenshots own that risk.

## See Also

- `docs/daemon-threat-model.md` — finer-grained asset/threat tables.
- `polylogue/daemon/http.py` — auth and origin-check implementation.
- `polylogue/daemon/cli.py:309-319` — remote-bind enforcement.
- `polylogue/browser_capture/server.py` — separate receiver auth.
- `tests/unit/daemon/test_daemon_http_security.py` — per-endpoint
  security matrix.
- `tests/unit/daemon/test_daemon_cli_remote_bind.py` — remote-bind
  refusal coverage.
