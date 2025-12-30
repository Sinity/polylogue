# Claude.ai Exports

Polylogue ingests Claude exports via the generic parser.

## Supported Inputs

- JSON payloads containing `chat_messages` (the Claude export shape).
- JSON lists with `messages` arrays.
- JSONL streams containing per-message entries.

## Current Behavior

- Converts `chat_messages` into ordered message text.
- Preserves roles from `sender`/`role`.
- Captures attachment metadata from `attachments`/`files` lists.

## Limitations

- Attachment binaries are not copied from Claude exports yet; only metadata is recorded.
