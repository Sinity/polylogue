# Polylogue

Polylogue turns AI chat exports into a local, searchable archive. It ingests exports into SQLite, renders Markdown/HTML views, and builds a full-text index. Everything stays on your machine.

## What it does

- Import ChatGPT, Claude, Claude Code, Codex, and generic JSON/JSONL/ZIP exports (streaming ingestion via `ijson`)
- Ingest Google AI Studio chats from Drive (OAuth, optional, with robust retries)
- Idempotent ingest with incremental index updates (safe to re-run)
- Render Markdown/HTML per conversation with customizable Jinja2 templates
- Search conversations with FTS (SQLite) or Vector Search (Qdrant + Voyage AI) and open results directly

## Install

Polylogue is built for Nix-first development, but you can run it directly from the repo.

- Nix dev shell: `direnv allow` or `nix develop`
- Build a packaged CLI: `nix build .#polylogue`
- Run from source: `python3 polylogue.py --help`

## Quick start

```bash
polylogue config init --interactive
polylogue run --preview
polylogue run
polylogue search "your query"
polylogue search
```

Drop local exports under `~/.local/share/polylogue/inbox` (default), or add custom sources in the config.

## Preview

![Preview output](docs/assets/cli-plan.svg)
![Run output](docs/assets/cli-run.svg)
![Search output](docs/assets/cli-search.svg)

Screenshots are captured from the interactive UI theme.

## Configuration

Config lives at `$POLYLOGUE_CONFIG` or `$XDG_CONFIG_HOME/polylogue/config.json`.

Example:

```json
{
  "version": 2,
  "archive_root": "~/.local/share/polylogue/archive",
  "render_root": "~/.local/share/polylogue/archive/render",
  "sources": [
    {"name": "inbox", "path": "~/.local/share/polylogue/inbox"},
    {"name": "gemini", "folder": "Google AI Studio"}
  ],
  "template_path": "~/.config/polylogue/template.html"
}
```

Notes:

- `POLYLOGUE_ARCHIVE_ROOT` overrides `archive_root` at runtime.
- `POLYLOGUE_RENDER_ROOT` overrides `render_root` at runtime.
- `POLYLOGUE_TEMPLATE_PATH` overrides `template_path`.
- Local sources use `path`; Drive sources use `folder`.
- `--source last` reuses the previous interactive source selection.
- Use `polylogue config show` for a quick summary, or `polylogue config show --json` for raw output.
- `--source NAME` (repeatable) limits `run` to selected sources.
- `POLYLOGUE_FORCE_PLAIN=1` forces the non-interactive UI mode.

## Drive auth

- Create an OAuth client (Desktop app) and place the JSON at `~/.config/polylogue/credentials.json`, or set `POLYLOGUE_CREDENTIAL_PATH`.
- Run any command with `--interactive` to complete auth. Tokens are stored at `~/.config/polylogue/token.json` (or `POLYLOGUE_TOKEN_PATH`).
- The default Drive folder is `Google AI Studio`.

## Vector Search (Qdrant)

Polylogue supports optional vector search using Qdrant and Voyage AI embeddings.

1. Set environment variables:
   ```bash
   export QDRANT_URL="http://localhost:6333"
   export QDRANT_API_KEY="your-key" # Optional
   export VOYAGE_API_KEY="your-voyage-api-key"
   ```
2. Run `polylogue run --stage index` to build/update the vector index.

## Commands

- `polylogue run` - ingest, render, and index
- `polylogue run --preview` - preview counts without writing
- `polylogue run --stage ingest|render|index` - run one stage
- `polylogue index` - rebuild the FTS (and vector) index
- `polylogue search [QUERY]` - full-text search
  - `--source NAME` - filter by source/provider
  - `--since DATE` - filter by date (ISO format)
  - `--open` - open the selected result
- `polylogue export` - export DB to JSONL
- `polylogue health` - cached health checks
- `polylogue state reset` - reset local state (DB + last-source)
- `polylogue config init/show/set/edit` - manage config

## Output layout

- Renders: `render_root/<provider>/<conversation_id>/conversation.md`
- HTML: `render_root/<provider>/<conversation_id>/conversation.html`
- Assets: `archive_root/assets/<prefix>/<attachment_id>`
- Exports: `archive_root/exports/conversations.jsonl`
- Runs: `archive_root/runs/run-<timestamp>-<run_id>.json`
- Database: `$XDG_STATE_HOME/polylogue/polylogue.db`

## Development

- Tests: `pytest -q`
- Refresh screenshots: `python3 scripts/generate_screenshots.py`
- See `docs/` for provider walkthroughs and pipeline details.

See `AGENTS.md` for additional development guidelines.