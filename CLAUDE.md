# Polylogue

> Update when adding features or changing behaviors. See `~/.claude/CLAUDE.md` for philosophy.

**Mission**: Local-first AI chat archive (ChatGPT, Claude, Codex, Gemini → SQLite + FTS5/Qdrant)

---

## Quick Reference

```bash
# Dev
uv run pytest -q --ignore=tests/test_qdrant.py
uv run mypy polylogue/
uv run ruff check polylogue/ tests/

# Run
uv run polylogue                              # Stats
uv run polylogue "search terms"               # Query
uv run polylogue -p claude --since "last week"
uv run polylogue sync --preview               # Dry-run sync
uv run polylogue check --repair               # Integrity check
```

**Env vars**: `POLYLOGUE_ARCHIVE_ROOT`, `QDRANT_URL`, `QDRANT_API_KEY`, `VOYAGE_API_KEY`

---

## Core Architecture

| Aspect | Implementation | Behavior |
|--------|----------------|----------|
| **Storage** | `StorageBackend` protocol | SQLiteBackend default, thread-local connections |
| **Search** | `SearchProvider` protocol | FTS5 (local), Qdrant (vector), LRU cache |
| **Rendering** | `OutputRenderer` protocol | Markdown/HTML via `--format` flag |
| **Services** | DI via `dependency-injector` | IngestionService, IndexService, RenderService |
| **Thread safety** | `_WRITE_LOCK` + thread-local | Lock in `store.py`, max 16 parallel ingests, 4 render workers |
| **Deduplication** | SHA-256 + NFC normalization | Same content → same hash → skip |

**Data Flow**:
```
JSON/ZIP/Drive → detect_provider() → Parse + extract content_blocks →
Hash (NFC) → Store (under lock) → Render (parallel) → Index
```

---

## Key Invariants

| Invariant | Details |
|-----------|---------|
| **Content hash includes** | title, timestamps, messages (id/role/text/timestamp), attachments (id/mime_type) |
| **Content hash excludes** | provider_meta, metadata (user-editable fields) |
| **Idempotent** | Safe to re-run imports; unchanged conversations skipped |
| **Thread-local connections** | Each thread gets own SQLite connection via `connection_context()` |
| **Write serialization** | All DB writes go through `_WRITE_LOCK` in `store.py` |

---

## Critical Files

### Core
| File | Purpose |
|------|---------|
| `lib/models.py` | Message/Conversation with `is_thinking`, `is_tool_use`, `is_substantive` |
| `lib/projections.py` | Fluent API: `conv.project().substantive().min_words(50).execute()` |
| `lib/filters.py` | Conversation filter chain: `p.filter().provider("claude").list()` |
| `lib/repository.py` | ConversationRepository (query interface) |
| `protocols.py` | StorageBackend, SearchProvider, OutputRenderer, VectorProvider |
| `container.py` | ApplicationContainer (DI) |
| `types.py` | NewType IDs: ConversationId, MessageId, AttachmentId |

### Storage
| File | Purpose |
|------|---------|
| `storage/store.py` | Record definitions, `_WRITE_LOCK` |
| `storage/repository.py` | StorageRepository (write coordination) |
| `storage/backends/sqlite.py` | SQLiteBackend (schema v5, migrations) |
| `storage/search_providers/fts5.py` | FTS5 search (incremental, query escaping) |
| `storage/search_providers/qdrant.py` | Qdrant vector search (Voyage embeddings) |
| `storage/db.py` | Thread-local connections, `connection_context()` |

### Ingestion
| File | Purpose |
|------|---------|
| `ingestion/source.py` | `detect_provider()`, encoding fallback, ZIP bomb protection |
| `importers/chatgpt.py` | UUID graph traversal, content_blocks from `content.parts` |
| `importers/claude.py` | JSONL (Claude AI) + structured blocks (Claude Code) |
| `importers/codex.py` | Session exports |
| `ingestion/drive.py` | Gemini via Google Drive API |

### Pipeline
| File | Purpose |
|------|---------|
| `pipeline/runner.py` | Orchestrates ingest → render → index |
| `pipeline/services/ingestion.py` | IngestionService (parallel, bounded) |
| `pipeline/services/indexing.py` | IndexService (FTS5/Qdrant management) |
| `pipeline/services/rendering.py` | RenderService (parallel output) |

### CLI
| File | Purpose |
|------|---------|
| `cli/click_app.py` | QueryFirstGroup (positional args → query mode) |
| `cli/query.py` | Filter chain, output formatting, modifiers |
| `cli/commands/sync.py` | Ingest → render → index |
| `cli/commands/check.py` | Integrity checks, `--repair`, `--vacuum` |

---

## Filter Chain API (lib/filters.py)

```python
from polylogue import Polylogue
p = Polylogue()
convs = p.filter().provider("claude").since("2024-01-01").contains("error").limit(10).list()
```

**Filters** (chainable):
| Method | Purpose |
|--------|---------|
| `provider(*names)` / `no_provider()` | Include/exclude providers |
| `tag(*tags)` / `no_tag()` | Include/exclude tags |
| `contains(text)` / `no_contains()` | FTS search include/exclude |
| `has(*types)` | Content types: `thinking`, `tools`, `attachments`, `summary` |
| `since(date)` / `until(date)` | Date range |
| `title(pattern)` / `id(prefix)` | Title contains, ID prefix |
| `sort(field)` | `date`, `tokens`, `messages`, `words`, `longest`, `random` |
| `reverse()` / `limit(n)` / `sample(n)` | Order/limit |
| `where(predicate)` | Custom filter function |

**Terminals**: `list()` → `list[Conversation]`, `first()` → `Conversation|None`, `count()` → `int`, `pick()` → interactive

---

## Projection API (lib/projections.py)

```python
# Message-level filtering within a conversation
msgs = conv.project().substantive().min_words(50).since(date).execute()
for msg in conv.project().contains("error").iter():  # lazy
    ...
```

**Filters** (chainable):
| Method | Purpose |
|--------|---------|
| `substantive()` | No noise/tools |
| `dialogue()` | User + assistant only |
| `user_messages()` / `assistant_messages()` | By role |
| `without_noise()` | Skip system, empty |
| `with_attachments()` | Has attachments |
| `min_words(n)` / `max_words(n)` | Length bounds |
| `since(date)` / `before(date)` | Date range |
| `contains(pattern)` / `matches(regex)` | Text search |
| `where(predicate)` | Custom filter |
| `limit(n)` | Max results |

**Terminals**: `execute()` → `list[Message]`, `iter()` → lazy iterator, `count()` → `int`

---

## Thread Safety

```
✅ Parallel: prepare_ingest() — pure hashing/parsing
✅ Parallel: rendering — ThreadPoolExecutor(max_workers=4)
✅ Bounded: ingestion futures — max 16 in-flight, FIRST_COMPLETED wait
✅ Pattern: with _WRITE_LOCK: save() — atomic writes

❌ NEVER: Direct sqlite3 ops — always use StorageRepository
❌ NEVER: Hold _WRITE_LOCK during I/O — only during commit
❌ NEVER: Mutate shared state without locks
```

---

## Content Hashing

```python
# core/hashing.py - NFC normalization prevents duplicates from equivalent Unicode
normalized = unicodedata.normalize("NFC", text)
hashlib.sha256(normalized.encode("utf-8")).hexdigest()
```

**Conversation hash** (pipeline/ids.py): `json.dumps(normalized, sort_keys=True, separators=(",", ":"))`

---

## Database Schema (v5)

| Table | Key Fields |
|-------|------------|
| **conversations** | conversation_id (PK), content_hash, source_name (GENERATED), metadata (JSON) |
| **messages** | conversation_id (FK CASCADE), content_blocks in provider_meta |
| **attachments** | attachment_id (PK), ref_count |
| **attachment_refs** | Many-to-many linking |
| **runs** | Audit trail (plan_snapshot, counts_json) |

**Metadata precedence** (display title): `metadata.title > original_title > id[:8]`

**Path**: `XDG_DATA_HOME/polylogue/polylogue.db`

---

## Content Blocks

Structured semantic decomposition stored in `provider_meta.content_blocks`:

```python
[
    {"type": "text", "text": "..."},
    {"type": "thinking", "thinking": "..."},
    {"type": "tool_use", "name": "search", "id": "toolu_123", "input": {...}},
    {"type": "tool_result", "tool_use_id": "toolu_123", "content": "..."},
]
```

Powers `is_thinking`, `is_tool_use`, `is_substantive` properties. Extracted at import time per provider.

---

## Provider Quirks

| Provider | Format | Structure | Content Blocks | Importer |
|----------|--------|-----------|----------------|----------|
| **ChatGPT** | conversations.json | UUID graph (`mapping`) | From `content.parts` + `content_type` | `importers/chatgpt.py` |
| **Claude AI** | JSONL | Flat `chat_messages` | None (simple text) | `importers/claude.py` |
| **Claude Code** | JSON array | `parentUuid`/`sessionId` | Structured `content` array | `importers/claude.py` |
| **Gemini** | Drive API | `chunkedPrompt.chunks` | From chunks | `ingestion/drive.py` |
| **Codex** | Session export | Session-based | N/A | `importers/codex.py` |

**Detection**: `ingestion/source.py:detect_provider()` via `looks_like()` functions

---

## External Integrations

| Integration | Config | Behavior |
|-------------|--------|----------|
| **Voyage AI** | `POLYLOGUE_VOYAGE_API_KEY` / `VOYAGE_API_KEY` | voyage-2 model, 1024-dim, 5× backoff |
| **Qdrant** | `POLYLOGUE_QDRANT_URL` / `QDRANT_URL`, `POLYLOGUE_QDRANT_API_KEY` / `QDRANT_API_KEY` | Cosine distance, 3× backoff |
| **Anthropic** | `POLYLOGUE_ANTHROPIC_API_KEY` / `ANTHROPIC_API_KEY` | For LLM annotation (future) |
| **OpenAI** | `POLYLOGUE_OPENAI_API_KEY` / `OPENAI_API_KEY` | For alternative embeddings (future) |
| **Google Gemini** | `POLYLOGUE_GOOGLE_API_KEY` / `GOOGLE_API_KEY` / `GEMINI_API_KEY` | For Gemini API access (future) |
| **Google Drive** | `~/.config/polylogue/polylogue-credentials.json` | OAuth 2.0, browser auth |

**Environment Variable Precedence**: POLYLOGUE_* prefixed variables checked first, then unprefixed versions. Allows project-specific config without affecting global tools.

---

## Security

| Protection | Location | Behavior |
|------------|----------|----------|
| **FTS5 injection** | `escape_fts5_query()` in `storage/search.py` | Quotes special chars, handles operators |
| **Encoding fallback** | `ingestion/source.py` | UTF-8 → UTF-8-sig → UTF-16 → UTF-32 → ignore |
| **ZIP bomb** | `ingestion/source.py` | Max 100:1 ratio, 500MB limit |

---

## Extension Pattern

All major components use protocols for extensibility:

```python
# 1. Implement protocol from protocols.py
class NewBackend:
    def save_conversation(self, conversation, messages, attachments) -> dict[str, int]: ...
    def get_conversation(self, conversation_id) -> ConversationRecord | None: ...
    # ... other protocol methods

# 2. Pass to repository/service
repo = StorageRepository(backend=NewBackend())
```

Same pattern for `SearchProvider`, `OutputRenderer`, `VectorProvider`.

---

## Anti-Patterns

| ❌ Don't | ✅ Do Instead |
|----------|---------------|
| Direct sqlite3 ops | Use `StorageRepository` |
| Skip content_blocks extraction | Extract at import time |
| Hardcode paths | Use `config.archive_root` |
| Manual commits | Let `connection_context()` handle |
| Modify hash logic | Maintain compatibility (breaks idempotency) |
| String matching for thinking/tools | Use `message.is_tool_use`, `is_thinking` |
| Mutate shared state in parallel | Use locks or keep pure |
| Create StorageRepository without backend | Pass `backend=create_default_backend()` |

---

## Common Debugging

| Error | Diagnostic |
|-------|------------|
| "Database locked" | Writes bypassing `_WRITE_LOCK` in `store.py` |
| "ref_count=0 but referenced" | `_prune_attachment_refs()` not called |
| "Content hash mismatch" | NFC normalization changed — check `core/hashing.py` |
| "FTS syntax error" | Unescaped query — use `escape_fts5_query()` |
| "Thinking not detected" | Missing content_blocks — check importer |
| "Config not reflected" | Env vars override — `polylogue config show` |

---

## Pinned Notes

@.claude/scratch/ files transcluded here. Currently empty.
