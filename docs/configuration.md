[← Back to README](../README.md)

# Configuration

## File Layout

Polylogue follows XDG Base Directory specification:

```
~/.local/share/polylogue/           # XDG_DATA_HOME/polylogue
├── source.db                       # raw acquisition log
├── index.db                        # Parsed/searchable/read-model index
├── embeddings.db                   # Vector index and embedding status
├── user.db                         # User marks, corrections, and annotations
├── ops.db                          # Daemon/cursor/telemetry state
├── blob/                           # Stored attachment/blob payloads
└── browser-capture/                # Browser-capture artifact spool

~/.claude/projects/                  # Auto-discovered: Claude Code sessions
~/.codex/sessions/                   # Auto-discovered: Codex sessions

~/.config/polylogue/                # XDG_CONFIG_HOME/polylogue
└── polylogue-credentials.json      # Google OAuth credentials (if using Drive)

~/.local/state/polylogue/           # XDG_STATE_HOME/polylogue
└── token.json                      # OAuth token cache
```

## Canonical Paths

Polylogue resolves all filesystem locations through `polylogue.paths` using
XDG environment variables. The defaults (when no override is set) are:

| Location | Default path | Override |
|---|---|---|
| Archive root | `~/.local/share/polylogue/` | `POLYLOGUE_ARCHIVE_ROOT` |
| Source DB | `~/.local/share/polylogue/source.db` | (follows archive root) |
| Index DB | `~/.local/share/polylogue/index.db` | (follows archive root) |
| Embeddings DB | `~/.local/share/polylogue/embeddings.db` | (follows archive root) |
| User DB | `~/.local/share/polylogue/user.db` | (follows archive root) |
| Ops DB | `~/.local/share/polylogue/ops.db` | (follows archive root) |
| Blob store | `~/.local/share/polylogue/blob/` | (follows data home) |
| Config file | `~/.config/polylogue/polylogue.toml` | `POLYLOGUE_CONFIG` |
| Config dir | `~/.config/polylogue/` | `XDG_CONFIG_HOME` |

Use `polylogue paths` to print the resolved paths for the current
environment, or `polylogue paths --format json` for machine-readable
output. The command also reports any detected bind mounts and
non-canonical files at the archive root.

### Bind Mounts and `/realm/data/captures/`

Some deployments (such as the [sinnix](https://github.com/Sinity/sinnix)
NixOS configuration) expose the archive at two path strings through a
btrfs subvolume bind mount:

```
~/.local/share/polylogue/       → same files as below
/realm/data/captures/polylogue/ → same files as above
```

Both paths refer to the **same physical directory** — they share device
and inode numbers. The bind mount is transparent to Polylogue; the
daemon, CLI, and MCP server all operate on whichever path they resolve.

**Which path to use in `polylogue.toml`:**

- Use `~/.local/share/polylogue` (the XDG default) as the canonical
  path in configuration and MCP server settings. It is portable across
  hosts and does not depend on the sinnix data-lake layout.
- `/realm/data/captures/polylogue` is the same directory via bind mount
  and is equally correct at runtime. It only exists on hosts that run
  the sinnix configuration; referencing it from config files makes them
  host-specific.

If both path strings appear in logs or tool output, they do **not**
indicate two separate archives or a misconfiguration — they are the
same directory tree.

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
check_tiers = "fast"

# Convergence-debt alert thresholds (#1226). The daemon raises a typed
# HealthAlert when the per-family count of `live_convergence_debt` rows
# crosses these levels. Overrides per source family let a stuck
# claude-code-session session escalate sooner than a long-tail chatgpt
# export. dedup_window_s suppresses repeated alerts within the window;
# severity escalations and resolution always fire immediately.
[health.convergence_debt]
default_warning = 1
default_error = 10
dedup_window_s = 3600

[health.convergence_debt.families.claude-code-session]
warning = 1
error = 5

[health.convergence_debt.families.chatgpt-export]
warning = 25
error = 200
```

Filesystem layout is owned by `polylogue.paths`, which reads directory
environment variables lazily when its path functions are called.

Path-safety helpers are separate from both surfaces. Code that sanitizes
provider or session names imports from `polylogue.paths.sanitize`.

## Environment Policy

Environment variable precedence is:

1. XDG roots define the base config, data, cache, and state directories.
2. `POLYLOGUE_ARCHIVE_ROOT` overrides the archive root and the archive
   databases under it (`source.db`, `index.db`, `embeddings.db`, `user.db`,
   and `ops.db`).
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

The archive uses SQLite files with different durability classes:

- `source.db`, `user.db`, `embeddings.db`, and `blob/` are the expensive or
  irreplaceable state to prioritize in backups.
- `index.db` is rebuildable from `source.db`, but a backup avoids a full
  reindex after restore.
- `ops.db` is disposable daemon state; back it up only when preserving
  operational history matters.

For an offline file-level backup, stop the daemon and copy the archive root or
the specific tier files you need. Include matching `-wal` and `-shm` companions
when copying a live WAL-mode SQLite database.

To export all sessions as JSON:

```bash
polylogue --format json > sessions.json

# Or with filters
polylogue -p claude-ai --format json > claude-sessions.json
```

Polylogue does not copy one-shot input exports into a managed import directory.
Keep the original provider export files if you need a reproducible rebuild from
the exact downloaded payloads.

## Google Drive Integration

For Gemini sessions via Google Drive:

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
