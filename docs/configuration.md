[← Back to README](../README.md)

# Configuration

## File Layout

Polylogue follows XDG Base Directory specification:

```
~/.local/share/polylogue/           # XDG_DATA_HOME/polylogue
├── polylogue.db                    # SQLite database
├── blobs/                          # Stored attachment/blob payloads
├── browser-capture/                # Browser-capture artifact spool
└── render/                         # Rendered output
    ├── html/
    │   └── claude/
    │       └── abc123.html
    └── md/
        └── claude/
            └── abc123.md

~/.claude/projects/                  # Auto-discovered: Claude Code sessions
~/.codex/sessions/                   # Auto-discovered: Codex sessions

~/.config/polylogue/                # XDG_CONFIG_HOME/polylogue
└── polylogue-credentials.json      # Google OAuth credentials (if using Drive)

~/.local/state/polylogue/           # XDG_STATE_HOME/polylogue
└── token.json                      # OAuth token cache
```

## Input Conventions

- Import downloaded exports with `polylogue run --input PATH`
- `--input` accepts files, directories, and archives
- Directory names are for organization only; providers are detected from content
- Symlinks are followed
- Files are processed recursively
- Supported formats: `.json`, `.jsonl`, `.zip`

## Configuration Model

Polylogue has no general-purpose config file. Runtime configuration is assembled
by `polylogue.config`: it discovers sources, builds Drive/index settings, and
captures the current database path in the `Config` value. Filesystem layout is
owned by `polylogue.paths`, which reads directory environment variables lazily
when its path functions are called.

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
5. Vector indexing reads `VOYAGE_API_KEY` when building index configuration or
   dispatching embedding commands.

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

Polylogue writes run metadata to disk and keeps a SQLite history so automation can consume results without scraping terminal output.

### Run Ledger

- Every `run` writes `archive_root/runs/run-<timestamp>-<run_id>.json`.
- The same run records are stored in `$XDG_DATA_HOME/polylogue/polylogue.db` (table: `runs`).
- Render output lives under `render_root` (defaults to `archive_root/render`).

### Health Checks

- `polylogue doctor` validates config, archive root, DB reachability, index status, and Drive credential/token presence.
- `polylogue doctor --repair` runs safe derived-data and database maintenance.
- `polylogue doctor --cleanup` runs destructive archive cleanup; preview it first.
- `polylogue doctor --repair --vacuum` compacts the database after maintenance.

### Path Inspection

- `polylogue run --preview` prints resolved sources and output paths.

---

**See also:** [CLI Reference](cli-reference.md) · [MCP Integration](mcp-integration.md) · [Data Model](data-model.md)
