# ChatGPT Exports

Polylogue ingests ChatGPT exports via the generic parser.

## Supported Inputs

- `conversations.json` containing a `mapping` graph.
- JSON lists with `messages` arrays.
- JSONL streams containing per-message entries.

## Current Behavior

- Extracts message text from `message.content.parts` and preserves roles.
- Orders mapping nodes by timestamp when available.
- Stores conversation/message metadata in the SQLite store and renders Markdown/HTML on `polylogue run`.

## Limitations

- Attachment extraction from ChatGPT ZIP exports is not implemented yet; metadata is preserved when present.
