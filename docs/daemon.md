[← Back to Docs](README.md)

# Daemon (polylogued)

`polylogued` is the long-lived Polylogue service. It watches chat directories,
ingests new conversations, runs periodic maintenance, and exposes optional HTTP
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

### GET /api/health

Health check with database statistics.

```json
{"ok": true, "db_size_bytes": 52428800, "wal_size_bytes": 4096, "disk_free_bytes": 107374182400, "quick_check": "pass"}
```

### GET /api/status

Full daemon status including component state, source lag, ingestion throughput,
FTS readiness, and insight freshness.

### GET /api/conversations

List conversations. Query params: `limit`, `provider`, `since`.

### GET /api/conversations/:id

Get a single conversation by ID.

### GET /api/conversations/:id/messages

Get messages for a conversation. Query params: `limit`, `offset`.

### GET /api/facets

Query faceted aggregations.

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

Auth is required for non-loopback requests. Pass the daemon token as a Bearer
token:

```bash
curl -H "Authorization: Bearer <token>" http://host:8766/api/status
```

## Browser Capture Receiver

The browser capture receiver accepts chat session payloads from local browser
extensions. It stores artifacts in the spool directory and triggers ingestion
through the normal pipeline.

Enabled by default on `127.0.0.1:8765`. Disable with `--no-browser-capture`.

## Health Monitoring

`polylogued status` reports typed daemon health via the `DaemonStatus` model.
`polylogued health` runs tiered health checks (fast + medium by default,
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
| `fts_readiness.action_events_ready` | FTS index covers all action events |
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

### Daemon-Owned Tasks

These run automatically inside the daemon process:

| Task | Frequency | Description |
|------|-----------|-------------|
| WAL checkpoint | Every 5 minutes | Keeps the WAL file bounded via `PRAGMA wal_checkpoint(TRUNCATE)` |
| Heartbeat | Every 15 minutes | Logs conversation/message counts as structured heartbeat |
| FTS convergence | Every 10 minutes | Verifies FTS coverage, rebuilds if messages are unindexed |
| Health checks | Configurable (default 5 min) | Runs tiered health checks (FAST + MEDIUM by default), sends notifications on non-OK status |
| FTS startup check | Once at startup | Rebuilds the FTS index if messages exist but aren't indexed (covers gaps from pre-daemon data) |

Health check tier and interval are configurable via `polylogue.toml`:

```toml
[health]
health_check_interval_s = 300
health_check_tiers = "fast,medium"
```

### Operator-Owned Tasks

These are heavier operations that should run outside the daemon via
systemd timers, cron, or manual invocation. The daemon does not own
these — operators are responsible for scheduling them:

| Task | Tool | Frequency | Description |
|------|------|-----------|-------------|
| Database backup | `polylogue backup` | Daily | Creates a clean SQLite database copy. Use `VACUUM INTO` when available (SQLite >= 3.27) for a defragmented copy. |
| Blob store backup | `polylogue backup --include-blobs` | Weekly | Copies the content-addressed blob store. Integration with restic or similar incremental backup tools is recommended for production use. |
| Database vacuum | `sqlite3 <db> "VACUUM"` | Monthly | Reclaims space after large deletes or updates. Requires downtime or `VACUUM INTO` to a new file while the daemon runs. |
| Litestream replication | Litestream | Continuous | Real-time WAL replication to S3-compatible storage. Configure Litestream to watch the database and WAL files; the daemon's periodic WAL checkpoint is compatible with Litestream's replication model. |

### Litestream Integration (Guidance)

Litestream provides continuous SQLite replication by shipping WAL frames
to object storage. Integration guidance:

1. Install Litestream and configure it to watch the Polylogue database
   (typically `~/.local/share/polylogue/polylogue.db`).
2. The daemon's periodic WAL checkpoint (`PRAGMA wal_checkpoint(TRUNCATE)`)
   triggers Litestream to create new generations. No daemon changes needed.
3. Test restores periodically: `litestream restore -o /tmp/restore.db <path>`.

A sample Litestream config:

```yaml
dbs:
  - path: /home/user/.local/share/polylogue/polylogue.db
    replicas:
      - type: s3
        bucket: my-backups
        path: polylogue
        endpoint: https://s3.amazonaws.com
```

### Blob Store Backup Guidance

The blob store uses content-addressed storage under
`<archive_root>/blob/`. Each blob's filename is its SHA-256 hash, so
identical content is automatically deduplicated. For backup:

- `polylogue backup --include-blobs` copies blobs alongside the database
- For large archives, use restic or rsync targeting the `blob/` directory
- Blobs are write-once, read-many — incremental backup tools work well

### Vacuum Guidance

SQLite `VACUUM` rebuilds the database file, reclaiming free pages. It
requires the full database size in free disk space. Two approaches:

1. **Online** (preferred): `polylogue backup` uses `VACUUM INTO` to
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
ExecStart=%h/.nix-profile/bin/polylogued run --enable-api --api-port 8766
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
| `embedding_enabled` | bool | `false` | Enable post-ingest embedding convergence |
| `embedding_model` | string | `voyage-4` | Voyage AI model name |
| `embedding_dimension` | int | `1024` | Vector dimension (must match the model) |
| `embedding_max_cost_usd` | float | `0.0` | Cost cap in USD (0 = no limit) |
| `voyage_api_key` | string | (none) | Voyage AI API key |

Example TOML:

```toml
[embedding]
embedding_enabled = true
embedding_model = "voyage-4"
embedding_dimension = 1024
embedding_max_cost_usd = 1.00
voyage_api_key = "va-..."
```

Without a `voyage_api_key`, the embedding stage reports "disabled" in daemon
status — this is not an error.

### Model/dimension changes

When the configured model or dimension differs from stored embeddings, the
daemon marks all conversations for re-embedding. A dimension change also drops
and recreates the vec0 virtual table.

### Checking coverage

```bash
polylogue stats                      # embedding coverage in archive stats
polylogued status                    # daemon status includes embedding readiness
```

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
raw parsing, FTS repair, insight materialization, and convergence-debt retry.
Operators can override those systemd resource controls in Nix or via a
systemd drop-in without changing daemon behavior.

If the daemon fails to start, check:
- Archive DB exists and is writable
- `polylogue.toml` is valid (run `polylogue config`)
- No other instance is running (pidfile lock)

## Backup and Recovery

The Polylogue archive consists of:
- `polylogue.db` — SQLite database (WAL mode, includes `-wal` and `-shm` files)
- `blob/` — content-addressed blob store

**Quick backup** (daemon stopped):
```bash
polylogue backup --output-dir /backup/polylogue
```

**Continuous backup** with Litestream:
```bash
litestream replicate polylogue.db s3://my-bucket/polylogue
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
cp /backup/polylogue.db /path/to/archive/
cp -r /backup/blob/ /path/to/archive/blob/
# Start daemon
systemctl --user start polylogued
```

The daemon will catch up on any files ingested after the backup was taken.
```
