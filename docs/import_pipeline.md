# Import Pipeline

Polylogue uses a simple ingest pipeline that feeds a SQLite store, renders Markdown/HTML, and builds a search index.

## Inputs

- Local sources: JSON, JSONL, or ZIP bundles discovered under a source `path`.
- Drive sources: JSON/JSONL files from a Drive folder (default `Google AI Studio`).

## Parsing

- `polylogue/source_ingest.py` handles local payloads and auto-detects provider shapes:
  - ChatGPT `mapping` graphs.
  - Claude `chat_messages` arrays (attachments metadata captured).
  - Generic `messages` lists or flat JSONL message streams.
- `polylogue/drive_ingest.py` downloads Drive payloads and parses `chunkedPrompt.chunks` for Gemini chats.

## Storage

- `polylogue/ingest.py` writes conversations/messages/attachments into SQLite via `polylogue/store.py`.
- Conversations are idempotent based on content hash.

## Rendering

- `polylogue/render.py` writes Markdown (and optional HTML) into `archive_root/render/<provider>/<conversation_id>/`.

## Indexing

- `polylogue/index.py` builds the SQLite FTS index for `polylogue search`.
