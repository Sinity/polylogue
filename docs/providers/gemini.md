# Gemini (Google AI Studio)

Polylogue ingests Gemini chats through the Drive API.

## Supported Inputs

- Drive folder containing AI Studio JSON exports (`chunkedPrompt.chunks`).

## Current Behavior

- Reads `chunkedPrompt.chunks` in order and renders user/assistant turns.
- Captures `driveDocument` references as attachments.
- Downloads Drive attachments into the archive assets folder during ingest.

## Defaults

- Drive folder name: `Google AI Studio`.
- Drive auth requires an OAuth client JSON and an interactive run.
