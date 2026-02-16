# Polylogue

> Update when adding features or changing behaviors. See `~/.claude/CLAUDE.md` for philosophy.

**Mission**: Local-first AI chat archive (ChatGPT, Claude, Codex, Gemini → SQLite + FTS5 + sqlite-vec)

---

## Quick Reference

```bash
# Dev
uv run pytest -q
uv run mypy polylogue/
uv run ruff check polylogue/ tests/

# Run
uv run polylogue                              # Stats
uv run polylogue "search terms"               # Query
uv run polylogue -p claude --since "last week"
uv run polylogue --similar "error handling"   # Semantic search
uv run polylogue run --preview                # Dry-run sync
uv run polylogue check --repair               # Integrity check
uv run polylogue mcp                          # MCP server (stdio)
```

**Env vars**: `POLYLOGUE_ARCHIVE_ROOT`, `POLYLOGUE_RENDER_ROOT`

**Library API**:
```python
async with Polylogue() as archive:
    stats = await archive.stats()  # Returns ArchiveStats
    convs = await archive.get_conversations(["id1", "id2"])
    results = await archive.search("error handling")
    filtered = await archive.filter().contains("python").provider("claude").list()
    await archive.rebuild_index()
```

---

## Core Architecture

| Aspect | Implementation | Behavior |
|--------|----------------|----------|
| **Storage** | `SQLiteBackend` (async-first) | aiosqlite with sync connection helpers |
| **Search** | `SearchProvider` protocol | FTS5 (local), sqlite-vec (vector), LRU cache |
| **Rendering** | Markdown/HTML renderers | Via `--format` flag |
| **Services** | `polylogue.services` module | Singleton factories for backend + repository |
| **Pipeline** | `run_sources` (async-first) | Acquire → Parse → Render → Index |
| **Thread safety** | `_WRITE_LOCK` + thread-local (sync), `asyncio.Lock` (async) | Write serialization |
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
| `facade.py` | `Polylogue` — async-first library API |
| `services.py` | Singleton factories: `get_backend()`, `get_repository()` |
| `protocols.py` | SearchProvider, VectorProvider (storage protocol deleted) |
| `types.py` | NewType IDs: ConversationId, MessageId, AttachmentId |

### Storage
| File | Purpose |
|------|---------|
| `storage/store.py` | Record definitions, `_WRITE_LOCK` |
| `storage/repository.py` | ConversationRepository (write coordination) |
| `storage/backends/async_sqlite.py` | SQLiteBackend (async-first, aiosqlite) |
| `storage/backends/sqlite.py` | Sync utilities, row mappers, connection helpers |
| `storage/search_providers/fts5.py` | FTS5 search (incremental, query escaping) |
| `storage/search_providers/sqlite_vec.py` | sqlite-vec vector search |
| `storage/backends/connection.py` | Thread-local connections, `connection_context()` |

### Sources (ingestion + parsing)
| File | Purpose |
|------|---------|
| `sources/source.py` | `detect_provider()`, encoding fallback, ZIP bomb protection |
| `sources/parsers/chatgpt.py` | UUID graph traversal, content_blocks from `content.parts` |
| `sources/parsers/claude.py` | JSONL (Claude AI) + structured blocks (Claude Code) |
| `sources/parsers/codex.py` | Session exports |
| `sources/drive.py` | Gemini via Google Drive API |

### Pipeline
| File | Purpose |
|------|---------|
| `pipeline/runner.py` | Orchestrates acquire → parse → render → index (async-first) |
| `pipeline/prepare.py` | Record preparation and dedup |
| `pipeline/services/acquisition.py` | AcquisitionService (raw data storage) |
| `pipeline/services/parsing.py` | ParsingService (raw → typed records) |
| `pipeline/services/indexing.py` | IndexService (FTS5 management) |
| `pipeline/services/rendering.py` | RenderService (concurrent render) |

### CLI
| File | Purpose |
|------|---------|
| `cli/click_app.py` | QueryFirstGroup (positional args → query mode) |
| `cli/query.py` | Filter chain, output formatting, modifiers |
| `cli/commands/run.py` | Ingest → render → index |
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

**Terminals** (all async): `await list()` → `list[Conversation]`, `await first()` → `Conversation|None`, `await count()` → `int`, `await pick()` → interactive

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

❌ NEVER: Direct sqlite3 ops — always use ConversationRepository
❌ NEVER: Hold _WRITE_LOCK during I/O — only during commit
❌ NEVER: Mutate shared state without locks
```

---

## Content Hashing

```python
# lib/hashing.py - NFC normalization prevents duplicates from equivalent Unicode
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

| Provider | Format | Structure | Content Blocks | Parser |
|----------|--------|-----------|----------------|--------|
| **ChatGPT** | conversations.json | UUID graph (`mapping`) | From `content.parts` + `content_type` | `sources/parsers/chatgpt.py` |
| **Claude AI** | JSONL | Flat `chat_messages` | None (simple text) | `sources/parsers/claude.py` |
| **Claude Code** | JSON array | `parentUuid`/`sessionId` | Structured `content` array | `sources/parsers/claude.py` |
| **Gemini** | Drive API | `chunkedPrompt.chunks` | From chunks | `sources/drive.py` |
| **Codex** | Session export | Session-based | N/A | `sources/parsers/codex.py` |

**Detection**: `sources/source.py:detect_provider()` via `looks_like()` functions

---

## MCP Server

Polylogue provides a Model Context Protocol (MCP) server for AI assistant integration.

**Start server**: `uv run polylogue mcp` (stdio transport)

**Capabilities**:
- **Tools**: search, list, get conversations
- **Resources**: Dynamic with query parameters
  - `polylogue://stats` - Archive statistics
  - `polylogue://conversations` - All conversations
  - `polylogue://conversations?provider=claude&since=2024-01-01` - Filtered
  - `polylogue://conversation/{id}` - Single conversation
- **Prompts**: Standardized workflows
  - `analyze-errors` - Find error patterns & solutions
  - `summarize-week` - Weekly insights summary
  - `extract-code` - Extract & organize code snippets

**Example resource template**:
```
polylogue://conversations?provider=claude&tag=important&limit=50
```

**Example prompt invocation**:
```json
{
  "method": "prompts/get",
  "params": {
    "name": "analyze-errors",
    "arguments": {"provider": "claude", "since": "2024-01-01"}
  }
}
```

---

## Demo & Synthetic Data

```bash
polylogue demo --seed                    # Full demo environment
polylogue demo --seed --env-only         # Shell-friendly (eval $(...))
polylogue demo --corpus -p chatgpt -n 5  # Raw fixture files
```

Uses `SyntheticCorpus` from `polylogue.sources.synthetic`. Shared with test fixtures (`seeded_db`, `synthetic_source`, `raw_synthetic_samples`). See [docs/demo.md](demo.md).

---

## External Integrations

| Integration | Config | Behavior |
|-------------|--------|----------|
| **Google Drive** | `~/.config/polylogue/polylogue-credentials.json` | OAuth 2.0, browser auth |

**Environment Variable Precedence**: POLYLOGUE_* prefixed variables checked first, then unprefixed versions (see `lib/env.py`). Allows project-specific config without affecting global tools.

---

## Security

| Protection | Location | Behavior |
|------------|----------|----------|
| **FTS5 injection** | `escape_fts5_query()` in `storage/search.py` | Quotes special chars, handles operators |
| **Encoding fallback** | `sources/source.py` | UTF-8 → UTF-8-sig → UTF-16 → UTF-32 → ignore |
| **ZIP bomb** | `sources/source.py` | Max 100:1 ratio, 500MB limit |

---

## Extension Points

Search uses protocol-based extensibility:

```python
# SearchProvider protocol (implementations: FTS5, Hybrid)
# See protocols.py for interfaces
```

Storage uses `SQLiteBackend` directly (single backend, no protocol abstraction).
Vector search uses sqlite-vec (self-contained, no external services).

---

## Anti-Patterns

| ❌ Don't | ✅ Do Instead |
|----------|---------------|
| Direct sqlite3 ops | Use `ConversationRepository` |
| Skip content_blocks extraction | Extract at import time |
| Hardcode paths | Use `config.archive_root` |
| Manual commits | Let `connection_context()` handle |
| Modify hash logic | Maintain compatibility (breaks idempotency) |
| String matching for thinking/tools | Use `message.is_tool_use`, `is_thinking` |
| Mutate shared state in parallel | Use locks or keep pure |
| Create repository without backend | Use `polylogue.services.get_repository()` |

---

## Common Debugging

| Error | Diagnostic |
|-------|------------|
| "Database locked" | Writes bypassing `_WRITE_LOCK` in `store.py` |
| "ref_count=0 but referenced" | `_prune_attachment_refs()` not called |
| "Content hash mismatch" | NFC normalization changed — check `lib/hashing.py` |
| "FTS syntax error" | Unescaped query — use `escape_fts5_query()` |
| "Thinking not detected" | Missing content_blocks — check parser in `sources/parsers/` |
| "Config not reflected" | Env vars override — check `polylogue run --preview` |

---

## Pinned Notes

Session notes are ephemeral and not persisted.
