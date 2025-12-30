# Polylogue

Polylogue is a CLI for ingesting AI chat exports into a local SQLite store and rendering clean Markdown/HTML views. It supports Google Drive (Google AI Studio) plus local exports from ChatGPT, Claude, Claude Code, and Codex via a generic parser.

## Quick Start

- Enter the dev shell: `direnv allow` (or `nix develop`).
- Initialize config: `polylogue config init --interactive`.
- Drop exports under `~/.local/share/polylogue/inbox` (or set a custom source path).
- Preview counts: `polylogue plan`.
- Ingest + render + index: `polylogue run`.
- Search: `polylogue search "your query"`.

## Configuration

Config lives at `$POLYLOGUE_CONFIG` or `$XDG_CONFIG_HOME/polylogue/config.json`.

Schema:

- `version`: config version (current: 1)
- `archive_root`: where renders, exports, runs, and assets live
- `sources`: list of sources
  - Drive: `{ "name": "gemini", "type": "drive", "folder": "Google AI Studio" }`
  - Local: `{ "name": "inbox", "type": "auto", "path": "~/.local/share/polylogue/inbox" }`
- `profiles`: named render profiles
  - `attachments`: `download` | `link` | `skip`
  - `html`: `auto` | `on` | `off`
  - `index`: boolean
  - `sanitize_html`: boolean

`POLYLOGUE_ARCHIVE_ROOT` overrides `archive_root` at runtime. Run `polylogue config show` to see the resolved config and env overrides.

## Drive Auth

- Create an OAuth client (Desktop app) and place the JSON at `~/.config/polylogue/credentials.json`, or set `POLYLOGUE_CREDENTIAL_PATH`.
- Run a command with `--interactive` to complete auth. Tokens are stored at `~/.config/polylogue/token.json` (or `POLYLOGUE_TOKEN_PATH`).
- The default Drive folder is `Google AI Studio`.

## Commands

- `polylogue plan` - show planned counts.
- `polylogue run` - ingest, render, and index.
- `polylogue ingest` - ingest only.
- `polylogue render` - render only.
- `polylogue search QUERY` - full text search (FTS).
- `polylogue export` - export DB to JSONL.
- `polylogue open` - open latest render.
- `polylogue health` - run cached health checks.
- `polylogue config init/show/set` - manage config.

Use `--source NAME` (repeatable) on `plan/run/ingest/render` to limit work to specific sources.

## Output Layout

- Renders: `archive_root/render/<provider>/<conversation_id>/conversation.md`
- HTML: `archive_root/render/<provider>/<conversation_id>/conversation.html`
- Assets: `archive_root/assets/<prefix>/<attachment_id>`
- Exports: `archive_root/exports/conversations.jsonl`
- Runs: `archive_root/runs/run-<timestamp>.json`
- Database: `$XDG_STATE_HOME/polylogue/polylogue.db`

## Testing

- `pytest -q`

See `AGENTS.md` for development guidelines.
