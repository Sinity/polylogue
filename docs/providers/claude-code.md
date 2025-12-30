# Claude Code Sessions

Polylogue ingests Claude Code JSONL sessions via the generic parser.

## Supported Inputs

- JSONL files with per-event records.

## Current Behavior

- Reads `role`/`sender` fields for message roles.
- Extracts text from `content`, `text`, or `content.parts`.
- Serializes tool-use and tool-result segments into JSON text when present.

## Limitations

- Full tool pairing and workspace metadata are not implemented yet.
