# Import Pipeline

Polylogue uses a staged pipeline that ingests provider exports, stores them in SQLite, renders Markdown/HTML, and builds a search index.

## Inputs

- Local sources: JSON, JSONL, or ZIP bundles discovered under a source `path` (default: `~/.local/share/polylogue/inbox/`).
- Drive sources: JSON/JSONL files from a Google Drive folder (default `Google AI Studio`).

## Parsing

- `polylogue/sources/source.py` handles local payloads and auto-detects provider shapes via `detect_provider()`.
- `polylogue/sources/parsers/` contains provider-specific parsers:
  - `chatgpt.py` — ChatGPT UUID graph traversal (`mapping` field)
  - `claude.py` — Claude AI (JSONL `chat_messages`) and Claude Code (JSON with `parentUuid`/`sessionId`)
  - `codex.py` — OpenAI Codex CLI sessions
  - `drive.py` — Gemini `chunkedPrompt.chunks`
  - `base.py` — Common structures (`ParsedConversation`, `ParsedMessage`, `ParsedAttachment`, `normalize_role`)
- `polylogue/sources/drive.py` downloads Drive payloads via the OAuth-authenticated `DriveClient`.

## Storage

- `polylogue/pipeline/ingest.py` prepares bundles with NFC-normalized content hashing for idempotent storage.
- `polylogue/storage/store.py` writes conversations/messages/attachments into SQLite under `_WRITE_LOCK`.
- `polylogue/storage/repository.py` coordinates writes via `ConversationRepository`.
- Conversations are deduplicated based on content hash (SHA-256).

## Rendering

- `polylogue/rendering/core.py` orchestrates rendering via `RenderService`.
- `polylogue/rendering/renderers/markdown.py` and `html.py` write output into `render_root/<provider>/<conversation_id>/`.

## Indexing

- `polylogue/storage/index.py` builds the SQLite FTS5 index for full-text search.
- `polylogue/storage/search_providers/qdrant.py` handles optional vector indexing via Qdrant.

## Pipeline Orchestration

The full pipeline is orchestrated by `polylogue/pipeline/runner.py`:

```
Source files → detect_provider() → Parse → Hash (NFC) → Store (under lock) → Render (parallel) → Index
```

Run via CLI: `polylogue run` (or `polylogue run --preview` for dry-run).
