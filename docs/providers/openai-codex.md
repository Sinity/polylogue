# OpenAI Codex Sessions

Polylogue ingests Codex JSONL sessions via the generic parser.

## Supported Inputs

- JSONL streams containing message-like records.

## Current Behavior

- Uses `role`/`sender` for roles and `content`/`text` for message bodies.
- Stores data in the SQLite store and renders Markdown/HTML on `polylogue run`.

## Limitations

- Tool call/result pairing and auxiliary logs are not implemented yet.
