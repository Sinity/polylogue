# ChatGPT Exports

Polylogue ingests ChatGPT exports via the generic parser.

## Supported Inputs

- `sessions.json` containing a `mapping` graph.
- JSON lists with `messages` arrays.
- JSONL streams containing per-message entries.

## Current Behavior

- Extracts message text from `message.content.parts` and preserves roles.
- Orders mapping nodes by timestamp when available.
- Stores session/message metadata in the SQLite store for daemon-owned
  ingestion and downstream render/read surfaces.

## Attachment Handling

- Extracts attachment metadata (id, name, mime_type, size) from message metadata.
- Stores references in the SQLite attachment table for integration with other tools.
- Note: Attachment binaries are not downloaded from ChatGPT exports; only metadata is recorded.
