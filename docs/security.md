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
| `/api/web-auth/session` | POST/DELETE | Mint/rotate or revoke a browser credential | Exact-origin bootstrap; current web credential for revoke | Exact origin |
| `/api/status` | GET | Read-only metadata | Bearer or web credential (when configured) | Credential-bound |
| `/api/health`, `/api/health/check` | GET | Health probe | Bearer or web credential (when configured) | Credential-bound |
| `/api/sessions`, `/api/sessions/:id`, `/.../messages`, `/.../raw` | GET | Read session data | Bearer or web credential | Credential-bound |
| `/api/facets` | GET | Read-only aggregations | Bearer or web credential | Credential-bound |
| `/api/sources` | GET | Returns absolute filesystem paths to authenticated callers | Bearer or web credential | Credential-bound |
| `/api/raw_artifacts/:id` | GET | Returns raw session payload | Bearer or web credential | Credential-bound |
| `/api/reset` | POST | **Destructive** — resets archive state | Bearer only when auth is configured | Exact origin |
| `/api/ingest` | POST | **Mutating** — schedules ingestion | Bearer only when auth is configured | Exact origin |
| `/api/maintenance/plan`, `/api/maintenance/run` | POST | **Mutating** — runs maintenance backfills | Bearer only when auth is configured | Exact origin |

## Mitigations

1. **Separate machine and browser credentials.** When `--api-auth-token`
   is configured, machine clients present `Authorization: Bearer <token>`.
   The first-party shell instead rotates a short-lived `read`/`user_state`/
   `events` credential through `POST /api/web-auth/session`. It can update
   marks, annotations, saved views, recall packs, and workspaces, but archive
   reset, ingest, and maintenance operations remain machine-bearer capabilities.
   The opaque value
   is stored in an `HttpOnly; SameSite=Strict; Path=/` cookie; the daemon keeps
   only its SHA-256 digest. It is never returned in JSON or accepted in a URL.
   Credential-shaped query parameters are rejected with `400
   credential_in_query`; route metadata and disconnect logs omit query strings
   entirely, including values under unrecognized parameter names.
   Missing, invalid, expired, revoked, wrong-origin, and insufficient-scope
   decisions use explicit response codes. The lifecycle lives in
   `polylogue/daemon/web_auth.py`; fetch and EventSource consume the same cookie.
   The digest registry prunes on issue/validate/revoke and enforces hard global
   and per-origin record bounds, so rotation cannot create unbounded retention.

2. **Loopback-only by default.** The daemon binds to `127.0.0.1` by
   default. The "loopback" predicate is the shared
   `polylogue/core/loopback.py:is_loopback_host` (RFC 5735: full
   `127.0.0.0/8`, `::1`, and `localhost`). Non-loopback bind
   (`--api-host 0.0.0.0` etc.) requires *both* `--insecure-allow-remote`
   (operator opts into the risk) *and* `--api-auth-token`; the
   enforcement lives in `polylogue/daemon/cli.py` and rejects either
   missing flag at startup.

3. **Exact origin for browser credentials and mutations.** A loopback origin
   on a different port is still a different origin. Bootstrap therefore
   requires the browser `Origin` authority to match `Host` exactly, credentials
   are bound to that canonical origin, and mutation requests repeat the exact
   authority check. Missing `Origin` remains valid for trusted non-browser
   bearer clients. The checks live in `polylogue/daemon/web_auth.py` and
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

## Excision and Secret Hygiene

"The archive can forget on purpose" (polylogue-27m). Two related but
distinct mechanisms live under `polylogue/security/`:

### Candidate-only secret detection

`polylogue/security/secret_scan.py` finds credential-shaped spans
(AWS/GitHub/Slack/OpenAI/Anthropic key shapes, PEM private-key headers, JWTs,
and an entropy-filtered generic `key=value` rule) in captured content and
records them as `AssertionKind.SECRET_CANDIDATE` assertions. This is a
**triage aid, not a leak-prevention boundary** — it is fully consistent with
"Raw artifacts are not content-redacted" above: the scanner does not gate or
redact reads, it surfaces candidates for an operator to review and, if
warranted, excise.

- The scanner **never returns, stores, or logs the matched literal**. Callers
  only ever see a SHA-256 fingerprint (one-way, for idempotent re-detection),
  a byte length, a pattern id, and span offsets into the source text.
- Written assertions use `author_kind="detector"`, which the shared
  `upsert_assertion` write chokepoint unconditionally coerces to
  `status=CANDIDATE` with `{"inject": false, "promotion_required": true}` —
  a detector can never self-promote a finding to authoritative/injectable
  context (the same invariant `PATHOLOGY`/`TRANSFORM_CANDIDATE` findings use).
- Coverage: `tests/unit/security/test_secret_scan.py` (also the
  `devtools test -k secret_candidate` anchor cited by
  `docs/plans/security-privacy-coverage.yaml`).

### Local excision (standalone/off mode — authoritative)

`polylogue ops excise --session <id> --reason "..."` (`--dry-run` to
preview, `--yes` to apply) removes a session across every local tier:

1. `embeddings.db` — the session's message vectors (if embedded).
2. `index.db` — `sessions` cascades to `messages`/`blocks`/`session_links`
   via `ON DELETE CASCADE`; the FTS triggers clean the contentless search
   index automatically.
3. `source.db` — `blob_refs` and `raw_sessions` rows (cascading to
   `raw_session_memberships`/`raw_membership_census`), then a durable
   removed-hash marker is recorded in the new `excised_content` table
   (migration `010_excised_content.sql`, `SOURCE_SCHEMA_VERSION` 10).
4. `user.db` — content-bearing assertions targeting the excised
   session/messages/blocks are removed, and one durable
   `AssertionKind.EXCISION_RECORD` audit receipt is written (reason, actor,
   removed hashes, per-tier counts).

**Ordinary re-ingest cannot resurrect excised content.** The removed-hash
marker is consulted at the single acquire-time write choke point,
`write_source_raw_session` (`polylogue/storage/sqlite/archive_tiers/
source_write.py`) — shared by the CLI import path and the daemon watch path
via `parse_sources_archive`. A re-acquire attempt whose payload hashes to a
recorded `removed_hash` raises `ContentExcisedError`; the batch orchestration
layer (`pipeline/services/archive_ingest.py:write_pair`) catches this
specifically and skips just that one file (counted in
`ParseResult.excised_skips`) rather than aborting the whole ingest run.

Blob *bytes* are never force-unlinked out from under a lease by excision
itself: removing the `blob_refs`/`raw_sessions` rows un-references the blob,
and the existing reference-counted blob GC (`polylogue/storage/blob_gc.py`)
reclaims the physical bytes on its next run using its own lease discipline.

Coverage: `tests/unit/security/test_excision.py`, including a real
`parse_sources_archive` round trip (synthetic-corpus fixture, not a hand-
rolled JSONL literal) proving the batch-skip behavior end to end.

### Mirror/primary lifecycle (Sinex-backed modes — mechanism only)

When Polylogue is Sinex-backed (mirror or primary mode — see the
Sinex-backed evidence mode work tracked separately), local excision cannot be
authoritative on its own: another replica may still hold the content.
`polylogue/security/lifecycle.py` implements the **local half** of that
lifecycle:

- `polylogue ops excise --mode mirror|primary --yes` writes a durable
  lifecycle-request/outbox row as an `AssertionKind.EXCISION_REQUEST`
  assertion in `user.db` — **never in `ops.db`**, so deleting or resetting
  the disposable ops tier cannot erase a pending request.
- The request state machine (`pending -> acknowledged -> confirmed`, or
  `pending -> rejected`, both terminal) is driven by
  `drive_lifecycle_request` against any `ExcisionLifecycleContract`
  implementation. `SinexContractFake` is a fault-injecting **test-only**
  implementation — it can drop N requests (simulated network loss) or force
  a rejection.
- **Primary mode invalidates the local replica only after a `confirmed`
  state** (`apply_primary_invalidation_if_confirmed`); a rejected, still-
  pending, or unknown request always returns `success=False` with an
  explicit reason and never touches the archive. A network fault leaves the
  request `pending` (retryable), never silently reinterpreted as a
  rejection or a confirmation.
- A "process restart" is modeled in tests by constructing a **new**
  `SinexContractFake` instance and re-driving the same durable request id —
  the outcome is identical because the durable row, not the client's
  in-memory state, is authoritative.

**Explicit non-goal:** this bead (polylogue-27m) does not claim a real Sinex
purge, a clean Sinex rebuild, disconnected-replica closure, or a backup-
restore proof — `polylogue ops excise --mode mirror|primary` only records
the durable local request today; nothing in this repository yet drives it
against a real Sinex confirmation. Binding this mechanism to Sinex's real
`privacy_invalidation_scope`, purge authority, and non-resurrection proof
across backups/replicas is tracked separately (polylogue-303r.6) and a
contract-fake-only test run must never be read as satisfying that scope.

Coverage: `tests/unit/security/test_excision_lifecycle.py` (network-loss
retry, restart-recovers-from-durable-row, `ops.db`-deletion survival,
rejection-cannot-report-success, confirm-gated invalidation).

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
- `polylogue/security/secret_scan.py`, `polylogue/security/excision.py`,
  `polylogue/security/lifecycle.py` — excision and secret-hygiene mechanics
  (polylogue-27m).
- `tests/unit/security/test_secret_scan.py`,
  `tests/unit/security/test_excision.py`,
  `tests/unit/security/test_excision_lifecycle.py`.
