[← Back to README](../README.md)

# Configuration

## File Layout

Polylogue follows XDG Base Directory specification:

```
~/.local/share/polylogue/           # XDG_DATA_HOME/polylogue
├── polylogue.db                    # SQLite database
├── inbox/                          # Drop exports here (or symlink)
│   ├── chatgpt/                    # Organize by provider (optional)
│   │   └── conversations.json
│   └── claude/
│       └── export.jsonl
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

## Inbox Conventions

- Drop provider exports directly in `inbox/` or in subdirectories
- Subdirectory names are for organization only (provider auto-detected from content)
- Symlinks are followed
- Files are processed recursively
- Supported formats: `.json`, `.jsonl`, `.zip`

## Configuration

Polylogue has no general-purpose config file. It follows XDG defaults, auto-discovers supported sources, and exposes only a few operational overrides for archive location, terminal mode, embeddings, and Drive credentials.

## Environment Overrides

These are the supported runtime overrides:

| Variable | Description |
|----------|-------------|
| `POLYLOGUE_ARCHIVE_ROOT` | Override the archive root instead of using `$XDG_DATA_HOME/polylogue` |
| `POLYLOGUE_FORCE_PLAIN` | Force non-interactive plain output |
| `VOYAGE_API_KEY` | Voyage AI API key for embeddings |
| `POLYLOGUE_CREDENTIAL_PATH` | Path to OAuth client JSON |
| `POLYLOGUE_TOKEN_PATH` | Path to OAuth token |

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

The inbox directory contains original exports and can be re-synced to rebuild the database.

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
