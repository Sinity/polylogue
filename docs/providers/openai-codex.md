# OpenAI Codex Sessions

Polylogue ingests Codex JSONL sessions via typed validation using the CodexRecord model.

## Supported Formats

Codex supports three format generations (auto-detected):

1. **Envelope Format** (Newest)
   - `{"type":"session_meta","payload":{...}}`
   - `{"type":"response_item","payload":{...}}`

2. **Direct Format** (Intermediate)
   - `{"type":"message","role":"user","content":[...]}`
   - Session metadata on first line with `id` and `timestamp`

3. **Legacy Format**
   - Sessions marked by `record_type:"state"` delimiters
   - Direct message records with role/content fields

## Current Behavior

- Detects format generation via CodexRecord.format_type property
- Normalizes role across all formats via effective_role property
- Extracts text content via text_content property
- Captures git context and system instructions from session metadata
- Builds conversation-level provider_meta with git/instructions

## Limitations

- Tool call/result pairing and workspace metadata are not implemented yet.
