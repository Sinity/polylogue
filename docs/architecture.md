# Architecture

Polylogue is a small pipeline that ingests chat exports into SQLite, renders Markdown/HTML, and builds an FTS index.

## Core modules

- `polylogue/config.py` and `polylogue/paths.py` define config schema, defaults, and XDG paths.
- `polylogue/cli/click_app.py` is the CLI entrypoint and routes commands to the pipeline.
- `polylogue/run.py` coordinates preview planning, ingesting, rendering, and indexing.
- `polylogue/source_ingest.py` loads local JSON/JSONL/ZIP exports and auto-detects provider formats.
- `polylogue/drive_client.py` handles Google Drive OAuth + downloads; `polylogue/drive_ingest.py` turns Drive payloads into conversations.
- `polylogue/store.py` and `polylogue/db.py` handle SQLite storage for conversations, messages, attachments, and runs.
- `polylogue/render.py` writes Markdown/HTML; `polylogue/assets.py` defines attachment paths.
- `polylogue/index.py` builds the SQLite FTS index; `polylogue/search.py` performs queries.
- `polylogue/health.py` runs cached health checks.

## Data flow

1. Inputs are read from local sources (JSON/JSONL/ZIP) or Drive folders.
2. Parsed conversations are stored in SQLite with idempotent upserts.
3. Render uses the DB to write Markdown/HTML per conversation.
4. Index builds FTS records for `polylogue search`.

## Data locations

- Runs: `archive_root/runs/run-<timestamp>.json`
- Renders: `archive_root/render/<provider>/<conversation_id>/conversation.md`
- Assets: `archive_root/assets/<prefix>/<attachment_id>`
- DB: `$XDG_STATE_HOME/polylogue/polylogue.db`
