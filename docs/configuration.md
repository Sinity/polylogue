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

**No configuration file.** Polylogue is truly zero-config. Paths follow XDG Base Directory specification.

## Environment Overrides

Optional environment variables for vector search and API keys:

| Variable | Alternative | Description |
|----------|-------------|-------------|
| `POLYLOGUE_VOYAGE_API_KEY` | `VOYAGE_API_KEY` | Voyage AI API key for embeddings |
| `POLYLOGUE_FORCE_PLAIN` | | Force non-interactive plain output |
| `POLYLOGUE_LOG` | | Log level: `error`, `warn`, `info`, `debug` |
| `POLYLOGUE_CREDENTIAL_PATH` | | Path to OAuth client JSON |
| `POLYLOGUE_TOKEN_PATH` | | Path to OAuth token |

## Run Ledger

Every `polylogue run` writes a JSON record to `archive_root/runs/run-<timestamp>-<run_id>.json` and to the `runs` table in the database. This enables automation to consume run results without scraping terminal output.

## Backup and Export

The database is a single SQLite file. To backup:

```bash
cp ~/.local/share/polylogue/polylogue.db ~/backups/polylogue-$(date +%Y%m%d).db
```

To export all conversations as JSON:

```bash
polylogue --format json > conversations.json

# Or with filters
polylogue -p claude --format json > claude-conversations.json
```

The inbox directory contains original exports and can be re-synced to rebuild the database.

## Google Drive Integration

For Gemini conversations via Google Drive:

1. Create OAuth credentials at [Google Cloud Console](https://console.cloud.google.com/)
2. Download to `~/.config/polylogue/polylogue-credentials.json`
3. Run `polylogue auth` to complete OAuth flow

The "Google AI Studio" folder is automatically synced (hardcoded).

---

**See also:** [CLI Reference](cli-reference.md) · [MCP Integration](mcp-integration.md) · [Data Model](data-model.md)
