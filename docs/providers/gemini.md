# Gemini (Google AI Studio)

Polylogue ingests Gemini chats through the Drive API and direct file uploads.

## Supported Inputs

- Drive folder containing AI Studio JSON exports (`chunkedPrompt.chunks`).
- Local JSON/JSONL files with `chunkedPrompt` structure or direct chunk lists.

## Provider Detection

- **Path-based detection**: Filename or path containing "gemini" triggers Gemini parsing.
- **Content-based detection** (v5+): Detects `chunkedPrompt` field or `chunks` arrays in payloads.

## Current Behavior

- Reads `chunkedPrompt.chunks` in order and renders user/assistant turns.
- Captures `driveDocument` references as attachments.
- Downloads Drive attachments into the archive assets folder during ingest.

## Defaults

- Drive folder name: `Google AI Studio`.
- Drive auth requires an OAuth client JSON and an interactive run.
