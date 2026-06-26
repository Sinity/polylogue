[← Back to Docs](README.md)

# Daemon (polylogued)

`polylogued` is the long-lived Polylogue service. It watches chat directories,
ingests new sessions, runs periodic maintenance, and exposes optional HTTP
surfaces.

## Quickstart

```bash
polylogued run
```

The daemon auto-discovers standard chat directories and begins ingestion. It
stays running until interrupted.

Check status:

```bash
polylogued status
```

## Auto-Discovery

The daemon watches these directories by default:

```
~/.claude/projects/       Claude Code sessions
~/.codex/sessions/         Codex sessions
```

Custom watch roots with `--root` (repeatable):

```bash
polylogued run --root /path/to/exports --root /another/path
```

## Configuration Flags

By default `polylogued run` enables every component (watch, browser capture, HTTP API). Pass the corresponding `--no-*` flag to disable any one.

| Flag | Default | Description |
|------|---------|-------------|
| `--root` | (auto) | Override watch root (repeatable) |
| `--debounce-s` | `2.0` | Quiet period in seconds before parsing a modified file |
| `--no-watch` | off | Disable the live source watcher |
| `--no-browser-capture` | off | Disable the browser-capture receiver |
| `--no-api` | off | Disable the HTTP API + web reader |

### Browser Capture

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Browser-capture receiver host |
| `--port` | `8765` | Browser-capture receiver port |
| `--spool` | (auto) | Browser-capture artifact spool path |
| `--browser-capture-auth-token` | none | Auth token for non-loopback requests |
| `--browser-capture-origin` | none | Additional allowed origin (repeatable) |
| `--insecure-allow-remote` | off | Allow non-loopback binding |

### HTTP API

| Flag | Default | Description |
|------|---------|-------------|
| `--api-host` | `127.0.0.1` | API server host |
| `--api-port` | `8766` | API server port |
| `--api-auth-token` | auto | API auth token (auto-generated if not provided) |

## HTTP API Endpoints

The HTTP API server runs by default. Pass `--no-api` to disable it. The endpoints below are exposed under the configured `--api-host:--api-port`.

Route contract metadata lives in `polylogue/daemon/route_contracts.py`.
Dispatch still happens in `polylogue/daemon/http.py`, but the dispatcher now
exposes its implemented route patterns from the routing tables used at runtime.
The route-contract tests compare the two surfaces, so stable routes,
shell-supported helpers, operational probes, and mutation routes must stay
classified with explicit auth and response posture.

| Route class | Auth policy | Examples |
|-------------|-------------|----------|
| Browser shell bootstrap | unauthenticated loopback HTML | `GET /`, `GET /s/:id`, `GET /p`, `GET /a` |
| Operational probes | unauthenticated loopback probe/scrape | `GET /healthz/live`, `GET /healthz/ready`, `GET /metrics` |
| Stable read/query API | bearer token when configured | `GET /api/sessions`, `GET /api/query-units`, `GET /api/sessions/:id`, `GET /api/sessions/:id/read`, `GET /api/sessions/:id/recovery`, `GET /api/assertions`, `GET /api/sessions/:id/provenance` |
| User overlay reads | bearer token when configured | `GET /api/user/marks`, `GET /api/user/saved-views/:id` |
| Browser-accessible mutations | bearer token plus same-origin browser request | `POST /api/user/marks`, `DELETE /api/user/saved-views/:id`, `POST /api/maintenance/run` |
| Observability ingest | explicit config flag plus loopback-or-bearer policy | `POST /v1/traces`, `POST /v1/metrics`, `POST /v1/logs` |

User overlay mutation routes return the shared mutation result envelope
(`status`, `operation`, `affected_count`, and resource/target identity fields).
Detailed overlay resources are read through the corresponding `GET
/api/user/...` endpoints.

### GET /api/health

Health check with database statistics.

```json
{"ok": true, "db_size_bytes": 52428800, "wal_size_bytes": 4096, "disk_free_bytes": 107374182400, "quick_check": "pass"}
```

### GET /api/status

Full daemon status including component state, source lag, ingestion throughput,
FTS readiness, and insight freshness.

### GET /api/sessions

List sessions. Query params: `limit`, `origin`, `since`.

### GET /api/sessions/:id

Get a single session by ID.

### GET /api/sessions/:id/messages

Get messages for a session. Query params: `limit`, `offset`.

### GET /api/query-units

Return terminal rows for explicit query-unit expressions. Query params:
`expression`, `limit`, `offset`. `expression` must be a
`messages/actions/blocks/assertions/runs/observed-events/context-snapshots where ...` query and returns the shared
`QueryUnitEnvelope` row envelope or, for SQL-backed
`group by FIELD | count` pipelines, `QueryUnitAggregateEnvelope`.

### GET /api/sessions/:id/read

Execute a supported single-session read profile and return the shared
`SessionReadViewEnvelope`. Query params include `view`, `format`, and
profile-specific options such as `report`, `limit`, `offset`,
`window_hours`, and `since_hours`.

### GET /api/sessions/:id/recovery

Read storage-free recovery evidence for one session. Query params:
`report=digest|work-packet` and `format=json|markdown`. Returns the
shared `RecoveryReadPayload`; markdown is supported for `work-packet`.

### GET /api/assertions

List assertion-backed overlay claims through `AssertionClaimListPayload`.
Query params include `target_ref`, `scope_ref`, `kind`, `status`,
`context_inject`, and `limit`. The default status scope is
`active,candidate`; use `status=all` for an explicit full lifecycle read.

### GET /api/facets

Query faceted aggregations. The route returns `FacetsResponse`, including
`complete_families`, `deferred_families`, `generated_at`, `stale`,
`stale_age_s`, `budget_exceeded`, and `family_status`. Repo and action
families are deferred from first-paint responses by default; request them
with `include_deferred=1` or `families=repos,action_types`, optionally
bounded by `budget_ms`. The daemon also polls the client socket from SQLite
progress handlers while computing facet SQL, so browser aborts/timeouts can
stop already-started archive scans instead of waiting for response write.

### GET /api/sources

List configured watch sources and their availability.

### GET /api/raw_artifacts/:id

Fetch raw (pre-parsed) artifact data by raw ID.

### POST /api/reset

Trigger a daemon reset operation.

### GET /

Web shell (localhost-only, no auth). A lightweight browser interface for
querying the archive through the daemon.

### Authentication

When an API auth token is configured, `/api/*` routes require it even from
loopback. Pass it as a Bearer token:

```bash
curl -H "Authorization: Bearer <token>" http://host:8766/api/status
```

Mutating browser-accessible routes also reject cross-origin `Origin` headers,
even when the Bearer token is valid. `GET /healthz/*` and `GET /metrics` remain
unauthenticated so health checks and Prometheus scrapers can operate without
credentials.

## Browser Capture Receiver

The browser capture receiver accepts chat session payloads from local browser
extensions. It stores artifacts in the spool directory and triggers ingestion
through the normal pipeline.

Enabled by default on `127.0.0.1:8765`. Disable with `--no-browser-capture`.

## Health Monitoring

`polylogued status` reports typed daemon health via the `DaemonStatus` model.
`polylogued health` runs tiered health checks (fast by default,
`--expensive` to include full integrity checks).

### Status Fields

| Field | Description |
|-------|-------------|
| `daemon_liveness` | Whether the daemon process is running (pidfile check) |
| `component_state.watcher` | `running` or `stopped` |
| `component_state.api` | `running` or `stopped` |
| `component_state.browser_capture` | `running` or `stopped` |
| `source_lag` | Per-source file counts and availability |
| `failing_files` | Files that failed ingestion |
| `fts_readiness.messages_ready` | FTS index covers all messages |
| `fts_readiness.actions_ready` | FTS index covers tool-use/tool-result action blocks |
| `insight_freshness` | Sessions with profiles vs. total |
| `embedding_readiness` | Embedding enabled, coverage, pending/stale, failures, cost |
| `db_size_bytes` | Database file size |
| `wal_size_bytes` | WAL file size |
| `blob_dir_size_bytes` | Blob store size |
| `disk_free_bytes` | Free disk space |
| `ingestion_throughput` | Messages and files per second |

### Health Check Tiers

`polylogued health` runs checks grouped by cost:

**Fast** (sub-1s):
- `daemon_liveness` — pidfile check
- `disk_space` — free disk space (warns at 500 MB, critical at 100 MB)
- `wal_size` — WAL file size (warns at 50 MB, errors at 200 MB)
- `source_availability` — watch roots exist and are readable
- `fts_trigger_drift` — checks the three canonical message/block FTS sync
  triggers (`messages_fts_a{i,d,u}`); critical when
  any are missing. With `[health] fts_auto_restore = true` (or
  `POLYLOGUE_HEALTH_FTS_AUTO_RESTORE=1`), the daemon restores triggers
  and rebuilds the FTS index in place, then emits a WARNING-level
  recovery alert so the operator is told the self-heal happened.

**Medium** (sub-10s queries):
- `fts_readiness` — FTS index coverage vs. total messages
- `raw_failures` — parse/validation failure counts
- `stale_ingest_attempts` — running attempts past the stale threshold
- `insight_freshness` — session profile coverage vs. total sessions
- `repeated_stage_failures` — recent ingest attempt failure rate

**Expensive** (potentially >10s):
- `db_integrity` — `PRAGMA integrity_check`
- `blob_integrity` — sample blob store content verification
- `embedding_coverage` — embedding coverage when enabled

Each check produces a typed `HealthAlert` with severity (`ok`, `warning`,
`error`, `critical`), message, and `consecutive_failures` counter for
persistent condition detection.

### Notification Backend

Health alerts are delivered through a configurable notification backend.
The default `log` backend writes alerts to the structured logger.
Configure via `polylogue.toml`:

```toml
[notifications]
backend = "log"
```

Or via environment variable:

```bash
export POLYLOGUE_NOTIFICATION_BACKEND=log
```

#### Webhook backend

Selecting `backend = "webhook"` POSTs a typed JSON envelope to a
user-configured URL on every non-OK health tick. The URL is fed through the
resolved config and must be set; missing or empty URLs raise a typed config
error at daemon startup rather than silently falling back to `log`.

```toml
[notifications]
backend = "webhook"
webhook_url = "https://hooks.example.invalid/polylogue"
```

Or via environment variables:

```bash
export POLYLOGUE_NOTIFICATION_BACKEND=webhook
export POLYLOGUE_NOTIFICATION_WEBHOOK_URL=https://hooks.example.invalid/polylogue
```

Envelope shape:

```json
{
  "alerts": [
    {
      "check_name": "wal_size",
      "tier": "fast",
      "severity": "warning",
      "message": "wal_size warning: 80 MB",
      "checked_at": "2026-05-17T00:00:00+00:00",
      "consecutive_failures": 1
    }
  ],
  "emitted_at": 1747440000.123,
  "daemon_version": "0.6.0+g<sha>"
}
```

The backend uses a bounded 5-second per-attempt timeout and a single retry on
transient failure (network error or HTTP 5xx). 4xx responses are treated as
permanent client errors and are not retried. Persistent failure surfaces
through the existing periodic-loop catch boundary in
`polylogue/daemon/cli.py`; it does not crash the daemon. Dedup and
rate-limiting are out of scope for this backend.

## Maintenance

Maintenance tasks are split between daemon-owned (fast, inline) and
operator-owned (heavy, scheduled externally via systemd timers or cron).

For the operator-facing maintenance surface — `polylogue ops maintenance
preview/plan/run`, the resume/`--operation-id` pattern, scope filters,
the status and failure surfaces, and incident runbooks — see
[maintenance.md](maintenance.md). The daemon's inline convergence
loops below cover the high-frequency, low-cost tasks; anything that
exceeds those bounded windows is operator-driven and documented
there.

### Daemon-Owned Tasks

These run automatically inside the daemon process:

| Task | Frequency | Description |
|------|-----------|-------------|
| WAL checkpoint | Every 5 minutes | Keeps the WAL file bounded via `PRAGMA wal_checkpoint(TRUNCATE)` |
| Heartbeat | Every 15 minutes | Logs session/message counts as structured heartbeat |
| FTS convergence | Every 10 minutes | Verifies FTS coverage, rebuilds if messages are unindexed |
| Health checks | Configurable (default 5 min) | Runs bounded FAST health checks by default, sends notifications on non-OK status. MEDIUM and EXPENSIVE checks are explicit operator diagnostics. |
| FTS startup check | Once at startup | Rebuilds the FTS index if messages exist but aren't indexed (covers gaps from pre-daemon data) |

Health check tier and interval are configurable via `polylogue.toml`:

```toml
[health]
health_check_interval_s = 300
health_check_tiers = "fast"
```

### Operator-Owned Tasks

These are heavier operations that should run outside the daemon via
systemd timers, cron, or manual invocation. The daemon does not own
these — operators are responsible for scheduling them:

| Task | Tool | Frequency | Description |
|------|------|-----------|-------------|
| Durability-tier backup | `polylogue ops backup` | Daily | archives copy `source.db`, `user.db`, `embeddings.db`, and referenced blobs while omitting rebuildable `index.db` and disposable `ops.db`. |
| Blob store backup | `polylogue ops backup` | Weekly | backup copies referenced blob files. For large archives, restic or similar incremental backup tools can also target `blob/` directly. |
| Database vacuum | `sqlite3 <db> "VACUUM"` | Monthly | Reclaims space after large deletes or updates. Requires downtime or `VACUUM INTO` to a new file while the daemon runs. |
| Litestream replication | Litestream | Continuous | Real-time WAL replication to S3-compatible storage. Configure Litestream to watch the database and WAL files; the daemon's periodic WAL checkpoint is compatible with Litestream's replication model. |

### Litestream Integration (Guidance)

Litestream provides continuous SQLite replication by shipping WAL frames
to object storage. Integration guidance:

1. Install Litestream and configure it to watch the Polylogue tier files you
   need to preserve continuously. Prioritize `source.db`, `user.db`, and
   `embeddings.db`; include `index.db` when avoiding reindex time matters.
2. The daemon's periodic WAL checkpoint (`PRAGMA wal_checkpoint(TRUNCATE)`)
   triggers Litestream to create new generations. No daemon changes needed.
3. Test restores periodically: `litestream restore -o /tmp/restore.db <path>`.

A sample Litestream config:

```yaml
dbs:
  - path: /home/user/.local/share/polylogue/source.db
    replicas:
      - type: s3
        bucket: my-backups
        path: polylogue/source
        endpoint: https://s3.amazonaws.com
  - path: /home/user/.local/share/polylogue/user.db
    replicas:
      - type: s3
        bucket: my-backups
        path: polylogue/user
        endpoint: https://s3.amazonaws.com
```

### Blob Store Backup Guidance

The blob store uses content-addressed storage under
`<archive_root>/blob/`. Each blob's filename is its SHA-256 hash, so
identical content is automatically deduplicated. For backup:

- `polylogue ops backup` copies archive blobs referenced by `source.db`
- For large archives, use restic or rsync targeting the `blob/` directory
- Blobs are write-once, read-many — incremental backup tools work well

When a backup sees source rows or attachment references whose content-addressed
blob file is absent, it keeps the backup usable but records the debt instead of
silently hiding it. The warning names the total missing referenced blobs and
the backup directory includes `blob-reference-debt.json` with the exact count,
bounded hash sample, and reference-source counts. Treat that report as
read-only recovery evidence: restore missing blob files from an older backup or
re-ingest the affected raw sources before deleting any source/blob/link rows.

### Vacuum Guidance

SQLite `VACUUM` rebuilds the database file, reclaiming free pages. It
requires the full database size in free disk space. Two approaches:

1. **Online** (preferred): `polylogue ops backup` uses `VACUUM INTO` to
   produce a clean copy without blocking the daemon. Swap the new file
   in during a maintenance window.
2. **Offline**: Stop the daemon, run `sqlite3 <db> "VACUUM"`, then
   restart. Only needed when `VACUUM INTO` is unavailable (SQLite < 3.27).

## Systemd Integration

Example unit file for running `polylogued` as a user service:

```ini
[Unit]
Description=Polylogue daemon
After=network.target

[Service]
Type=simple
ExecStart=%h/.nix-profile/bin/polylogued run --api-port 8766
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

The daemon writes a pidfile to the archive root and uses `fcntl` advisory
locking to prevent concurrent instances.

## Embeddings

The daemon can generate Voyage AI vector embeddings for message content, stored
in the `message_embeddings` vec0 virtual table and consumed by `--similar` search
and the `hybrid` retrieval lane.

Embedding convergence is **opt-in** — ordinary daemon catch-up does not make
provider API calls unless explicitly configured.

### Configuration

Embedding settings live in the `[embedding]` section of `polylogue.toml` (see
`polylogue config` for the resolved path). The daemon reads these keys:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `false` | Enable post-ingest embedding convergence |
| `model` | string | `voyage-4` | Voyage AI model name |
| `dimension` | int | `1024` | Vector dimension (must match the model) |
| `max_cost_usd` | float | `5.0` | Per-run soft cost cap in USD (0 = no limit) |
| `voyage_api_key` | string | (none) | Voyage AI API key |

Example TOML:

```toml
[embedding]
enabled = true
model = "voyage-4"
dimension = 1024
max_cost_usd = 1.00
voyage_api_key = "va-..."
```

Without a `voyage_api_key`, the embedding stage reports "disabled" in daemon
status — this is not an error.

### Model/dimension changes

When the configured model or dimension differs from stored embeddings, the
daemon marks all sessions for re-embedding. A dimension change also drops
and recreates the vec0 virtual table.

### Checking coverage

```bash
polylogue analyze                    # embedding coverage in archive aggregates
polylogue ops embed status               # cheap readiness + latest catch-up run
polylogue ops embed status --detail      # exact pending-message/retrieval accounting
polylogued status                    # daemon status includes embedding readiness
```

`polylogue ops embed backfill` records each bounded catch-up window in the local
archive as `embedding_catchup_runs`. The latest run is shown by
`polylogue ops embed status`, including terminal state, stop reason, processed
sessions, embedded messages, errors, and estimated cost. This is the
operator recovery point after interruption, OOM, restart, or a cost/error
window stop; per-session retry state still lives in `embedding_status`.

## Service Recovery

`polylogued` is typically managed as a systemd user service. Example unit:

```ini
# ~/.config/systemd/user/polylogued.service
[Unit]
Description=Polylogue daemon
After=network.target

[Service]
ExecStart=%h/.local/bin/polylogued run
Restart=on-failure
Environment=POLYLOGUE_CONFIG=%h/.config/polylogue/polylogue.toml

[Install]
WantedBy=default.target
```

Check status: `systemctl --user status polylogued`
View logs: `journalctl --user -u polylogued -f`

Pause or stop the daemon with systemd, not a Polylogue-specific switch:

```bash
systemctl --user stop polylogued
systemctl --user start polylogued
systemctl --user disable --now polylogued
```

Packaged deployments should keep host-friendliness in the service manager.
The NixOS module defaults the daemon to background scheduling (`Nice=10`,
idle I/O scheduling, conservative `IOWeight`, `MemoryHigh`, and `MemoryMax`)
while the daemon itself still converges the full archive: live source ingest,
raw parsing, FTS freshness, insight materialization, and convergence-debt retry.
Operators can override those systemd resource controls in Nix or via a
systemd drop-in without changing daemon behavior.

If the daemon fails to start, check:
- tier files exist and are writable
- `polylogue.toml` is valid (run `polylogue config`)
- No other instance is running (pidfile lock)

## Backup and Recovery

The Polylogue archive consists of:
- `source.db` — raw acquisition/source evidence SQLite database
- `index.db` — parsed sessions, search, graph, and insight read models
- `embeddings.db` — vector index, embedding status, and catch-up metadata
- `user.db` — user marks, corrections, and annotations
- `ops.db` — daemon, cursor, and telemetry SQLite state
- `blob/` — content-addressed blob store

For backups, prioritize `source.db`, `user.db`, `embeddings.db`, and `blob/`.
`index.db` is rebuildable but convenient to keep. `ops.db` is disposable unless
you need operational history.

**Continuous backup** with Litestream:
```bash
litestream replicate source.db s3://my-bucket/polylogue/source
litestream replicate user.db s3://my-bucket/polylogue/user
```

**Blob store backup** with restic:
```bash
restic backup /path/to/archive/blob/
```

**Recovery**:
```bash
# Stop daemon
systemctl --user stop polylogued
# Restore files
cp /backup/source.db /path/to/archive/
cp /backup/user.db /path/to/archive/
cp /backup/embeddings.db /path/to/archive/
cp -r /backup/blob/ /path/to/archive/blob/
# Start daemon
systemctl --user start polylogued
```

The daemon will catch up on any files ingested after the backup was taken.
```
