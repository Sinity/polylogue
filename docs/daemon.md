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

| Flag | Default | Description |
|------|---------|-------------|
| `--root` | (auto) | Override watch root (repeatable) |
| `--debounce-s` | `2.0` | Quiet period in seconds before parsing a modified file |
| `--no-watch` | off | Disable the live source watcher |
| `--no-browser-capture` | off | Disable the browser-capture receiver |

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
| `--enable-api` | off | Start the daemon HTTP API server |
| `--api-host` | `127.0.0.1` | API server host |
| `--api-port` | `8766` | API server port |
| `--api-auth-token` | auto | API auth token (auto-generated if not provided) |

## HTTP API Endpoints

When `--enable-api` is set, the daemon exposes these endpoints.

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

`polylogued status` reports typed daemon health via the `DaemonStatus` model:

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
| `db_size_bytes` | Database file size |
| `wal_size_bytes` | WAL file size |
| `blob_dir_size_bytes` | Blob store size |
| `disk_free_bytes` | Free disk space |
| `ingestion_throughput` | Messages and files per second |

## Maintenance

The daemon runs periodic maintenance tasks:

- **WAL checkpoint** every 5 minutes (keeps the WAL file bounded)
- **Heartbeat** every 15 minutes (logs conversation/message counts)
- **FTS startup check**: rebuilds the FTS index if messages exist but aren't
  indexed (covers gaps from pre-daemon data)

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

When `VOYAGE_API_KEY` is set and `POLYLOGUE_DAEMON_ENABLE_EMBEDDINGS` is enabled
(`1`, `true`, or `yes`), the daemon can generate vector embeddings for message
content. Embeddings are stored in the `message_embeddings` vec0 virtual table
and used by the `--similar` search and `hybrid` retrieval lane. Embedding
convergence is opt-in so ordinary daemon catch-up does not make provider API
calls.

Check embedding coverage:

```bash
polylogue stats
# Embeddings: 1,234/567 convs, 45,678 msgs (87.3%)
```
