# Import Pipeline

Polylogue uses a simple ingest pipeline that feeds a SQLite store, renders Markdown/HTML, and builds a search index.

## Inputs

- Local sources: JSON, JSONL, or ZIP bundles discovered under a source `path`.
- Drive sources: JSON/JSONL files from a Drive folder (default `Google AI Studio`).

## Parsing

- `polylogue/source_ingest.py` handles local payloads and auto-detects provider shapes.
- `polylogue/importers/` contains provider-specific parsers:
  - `chatgpt.py` - ChatGPT `mapping` graphs
  - `claude.py` - Claude AI and Claude Code `chat_messages` arrays
  - `codex.py` - OpenAI Codex CLI sessions
  - `base.py` - Common structures (`ParsedConversation`, `ParsedMessage`, `ParsedAttachment`)
- `polylogue/drive_ingest.py` downloads Drive payloads and parses `chunkedPrompt.chunks` for Gemini chats.

## Storage

- `polylogue/pipeline/ingest.py` prepares bundles with content hashing for idempotent storage.
- `polylogue/store.py` writes conversations/messages/attachments into SQLite.
- Conversations are deduplicated based on content hash (SHA-256).

## Rendering

- `polylogue/render.py` writes Markdown and HTML into `render_root/<provider>/<conversation_id>/`.

## Indexing

- `polylogue/index.py` builds the SQLite FTS5 index for `polylogue search`.
