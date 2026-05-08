[← Back to README](../README.md)

# Configuration

## File Layout

Polylogue follows XDG Base Directory specification:

```
~/.local/share/polylogue/           # XDG_DATA_HOME/polylogue
├── polylogue.db                    # SQLite database
├── blobs/                          # Stored attachment/blob payloads
└── browser-capture/                # Browser-capture artifact spool

~/.claude/projects/                  # Auto-discovered: Claude Code sessions
~/.codex/sessions/                   # Auto-discovered: Codex sessions

~/.config/polylogue/                # XDG_CONFIG_HOME/polylogue
└── polylogue-credentials.json      # Google OAuth credentials (if using Drive)

~/.local/state/polylogue/           # XDG_STATE_HOME/polylogue
└── token.json                      # OAuth token cache
```

## Input Conventions

- `polylogued run` watches configured source roots and owns ingestion.
- Use `polylogue ingest PATH` to ask the running daemon to ingest an explicit
  file or directory.
- Directory names are for organization only; providers are detected from content.
- Supported source formats include `.json`, `.jsonl`, and `.zip`.

## Configuration Model

Polylogue has a layered configuration system with four sources (highest precedence first):

1. CLI flag overrides
2. `POLYLOGUE_*` environment variables
3. `polylogue.toml` (resolved from `$POLYLOGUE_CONFIG`, then `$XDG_CONFIG_HOME/polylogue/polylogue.toml`, then `./polylogue.toml`)
4. Built-in defaults (full-featured-by-default: watch + browser-capture + HTTP API all on)

Use `polylogue config` to print the resolved configuration as TOML, or `polylogue config -f json` for a machine-readable form.

### TOML schema

```toml
[archive]
root = "/home/user/.local/share/polylogue"

[daemon]
host = "127.0.0.1"
port = 8766

[daemon.api]
host = "127.0.0.1"
port = 8766
# auth_token = "..."   # optional; required for non-loopback API binding

[daemon.browser_capture]
port = 8765
allowed_origins = "127.0.0.1"

[embedding]
# enabled defaults to true when VOYAGE_API_KEY is set, false otherwise
enabled = true
model = "voyage-4"
dimension = 1024
max_cost_usd = 5.0   # soft monthly cap; 0 = unlimited

[logging]
level = "INFO"
force_plain = false

[notifications]
backend = "log"

[health]
check_interval_s = 300
check_tiers = "fast,medium"
```

Filesystem layout is owned by `polylogue.paths`, which reads directory
environment variables lazily when its path functions are called.

Path-safety helpers are separate from both surfaces. Code that sanitizes
provider or conversation names imports from `polylogue.paths.sanitize`.

## Environment Policy

Environment variable precedence is:

1. XDG roots define the base config, data, cache, and state directories.
2. `POLYLOGUE_ARCHIVE_ROOT` overrides only the archive root; the database still
   defaults to `$XDG_DATA_HOME/polylogue/polylogue.db`.
3. Source discovery is derived from resolved filesystem paths and Drive cache or
   auth files.
4. Drive authentication may override credential and token files through the
   Drive-specific environment variables below.
5. Vector indexing reads `VOYAGE_API_KEY` only when daemon embedding convergence
   is explicitly enabled.

These are the supported runtime overrides:

| Variable | Description |
|----------|-------------|
| `XDG_CONFIG_HOME` | Base directory for `polylogue-credentials.json` |
| `XDG_DATA_HOME` | Base directory for the database, blob store, browser-capture spool, and Drive cache |
| `XDG_CACHE_HOME` | Base directory for cache/index output |
| `XDG_STATE_HOME` | Base directory for OAuth token and runtime state |
| `POLYLOGUE_ARCHIVE_ROOT` | Override the archive root instead of using `$XDG_DATA_HOME/polylogue` |
| `POLYLOGUE_FORCE_PLAIN` | Force non-interactive plain output |
| `VOYAGE_API_KEY` | Voyage AI API key for embeddings |
| `POLYLOGUE_DAEMON_ENABLE_EMBEDDINGS` | Set to `1`, `true`, or `yes` to let daemon convergence call the embedding provider |
| `POLYLOGUE_CREDENTIAL_PATH` | Drive auth override for the OAuth client JSON path |
| `POLYLOGUE_TOKEN_PATH` | Drive auth override for the OAuth token path |

## Backup and Export

The database is a single SQLite file. To backup:

```bash
cp ~/.local/share/polylogue/polylogue.db ~/backups/polylogue-$(date +%Y%m%d).db
```

To export all conversations as JSON:

```bash
polylogue --format json > conversations.json

# Or with filters
polylogue -p claude-ai --format json > claude-conversations.json
```

Polylogue does not copy one-shot input exports into a managed import directory.
Keep the original provider export files if you need a reproducible rebuild from
the exact downloaded payloads.

## Google Drive Integration

For Gemini conversations via Google Drive:

1. Create OAuth credentials at [Google Cloud Console](https://console.cloud.google.com/)
2. Download to `~/.config/polylogue/polylogue-credentials.json`
3. Run `polylogue auth` to complete OAuth flow

Polylogue syncs the fixed `Google AI Studio` folder name used by Gemini exports.

## Observability

Polylogue exposes daemon health through `polylogued status` and
`polylogue status`. The archive database stores ingestion state, live cursors,
and derived read-model freshness so automation does not need to scrape terminal
output.

### Health Checks

- `polylogue doctor` validates config, archive root, DB reachability, index status, and Drive credential/token presence.
- `polylogue doctor --repair` runs safe derived-data and database maintenance.
- `polylogue doctor --cleanup` runs destructive archive cleanup; preview it first.
- `polylogue doctor --repair --vacuum` compacts the database after maintenance.
- Workstation-specific policy such as cgroup slice placement and hard caps belongs in the host environment, not in the product CLI.

---

**See also:** [CLI Reference](cli-reference.md) · [MCP Integration](mcp-integration.md) · [Data Model](data-model.md)
