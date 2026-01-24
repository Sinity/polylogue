# Polylogue Developer Guide

> Update this file when adding features, changing behaviors, or discovering patterns during debugging.
> See `~/.claude/CLAUDE.md` "Project CLAUDE.md Philosophy" for maintenance principles.

**Mission**: Local-first AI chat archive (ChatGPT, Claude, Codex, Gemini → SQLite + FTS5/Qdrant)

## Core Architecture

| Aspect | Implementation | Key Behavior |
|--------|---------------|--------------|
| **Storage** | StorageBackend protocol | SQLiteBackend, thread-local connections, _WRITE_LOCK in repository |
| **Search** | SearchProvider protocol | FTS5Provider (local, BM25), QdrantProvider (Voyage embeddings), LRU cache |
| **Rendering** | OutputRenderer protocol | MarkdownRenderer (.md), HTMLRenderer (.html + .md), --format flag |
| **Services** | DI via dependency-injector | IngestionService (parallel, bounded), IndexService, RenderService |
| **Thread safety** | _WRITE_LOCK + thread-local | repository.py:48 owns lock, max 16 in-flight futures, 4 render workers |
| **Deduplication** | SHA-256 + NFC normalization | Idempotent: same content → same hash → skip re-import |
| **Type safety** | mypy strict mode | NewType IDs, protocol types |

**Key Invariants**:
- Content hash includes: title, timestamps, messages (id/role/text/timestamp), attachments (id/mime_type)
- Content hash excludes: provider_meta (semantic content only)
- Idempotent: Safe to re-run imports, unchanged conversations skipped
- Thread-local connections: Each thread has own SQLite connection via `connection_context()`
- _WRITE_LOCK serializes: All DB writes go through StorageRepository under lock

## Data Flow

```
JSON/ZIP/Drive → Auto-detect provider → Parse + extract content_blocks →
Hash (NFC normalized) → Store (under _WRITE_LOCK) → Render (parallel) →
Index (FTS5 + optional Qdrant)
```

## Critical Files

### Core Abstractions & Models
- **`lib/models.py`**: Message/Conversation with semantic classification (`is_thinking`, `is_tool_use`, `is_substantive`) + metadata properties (`user_title`, `display_title`, `summary`, `tags`)
- **`lib/projections.py`**: Fluent projection API (`conv.project().substantive().min_words(50).execute()`)
- **`lib/repository.py`**: ConversationRepository (primary query interface with semantic projections)
- **`core/hashing.py`**: NFC-normalized SHA-256 hashing
- **`protocols.py`**: Protocol definitions (StorageBackend, SearchProvider, OutputRenderer, VectorProvider)
- **`container.py`**: ApplicationContainer (dependency injection using dependency-injector)
- **`types.py`**: NewType IDs (ConversationId, MessageId, AttachmentId)

### Storage Layer (Refactored)
- **`storage/db.py`**: Thread-local connections, schema v5, migrations
- **`storage/store.py`**: Record definitions (ConversationRecord, MessageRecord, AttachmentRecord)
- **`storage/repository.py`**: StorageRepository owns `_WRITE_LOCK`, requires StorageBackend (no legacy path)
- **`storage/backends/sqlite.py`**: SQLiteBackend implementation (schema, migrations, CRUD)
- **`storage/search_providers/fts5.py`**: FTS5 search provider (incremental indexing, query escaping)
- **`storage/search_providers/qdrant.py`**: Qdrant vector search provider (Voyage AI embeddings, lazy imports)
- **`storage/search_cache.py`**: LRU cache for search results (version-based invalidation)

### Ingestion & Source Management
- **`ingestion/source.py`**: Provider detection, ParsedConversation/ParsedMessage
- **`ingestion/ingest.py`**: ingest_bundle() creates default backend if none provided (backward compatible)
- **`ingestion/drive.py`**: Google Drive client and Gemini parsing
- **`ingestion/drive_client.py`**: OAuth credential management
- **`importers/chatgpt.py`**: Graph traversal, extracts thinking from `content_type`
- **`importers/claude.py`**: JSONL chat_messages (Claude AI) + structured blocks (Claude Code)
- **`importers/codex.py`**: Session exports

### Pipeline & Services (Refactored)
- **`pipeline/runner.py`**: Orchestrates ingest → render → index pipeline (service-based)
- **`pipeline/ingest.py`**: prepare_ingest() creates default backend if none provided (backward compatible)
- **`pipeline/ids.py`**: Content hash generation (NFC-normalized)
- **`pipeline/models.py`**: PlanResult, RunResult (typed result objects)
- **`pipeline/services/ingestion.py`**: IngestionService (parallel ingest with bounded submission)
- **`pipeline/services/indexing.py`**: IndexService (FTS5/Qdrant management, chunked updates)
- **`pipeline/services/rendering.py`**: RenderService (requires OutputRenderer, no legacy fallback)
- **`pipeline/services/__init__.py`**: Service exports

### Rendering (Abstracted)
- **`rendering/renderers/markdown.py`**: MarkdownRenderer (plain .md output)
- **`rendering/renderers/html.py`**: HTMLRenderer (Jinja2-based .html + .md)
- **`rendering/renderers/__init__.py`**: create_renderer() factory (format selection)
- **`render.py`**: Legacy render functions (backward compatibility)

### CLI (Query-First)
- **`cli/click_app.py`**: Main CLI entry with QueryFirstGroup class (parse_args override for query-first routing)
- **`cli/query.py`**: Query mode execution (filter chain, output formatting, aggregation, modifiers)
- **`cli/container.py`**: CLI dependency injection wrappers
- **`cli/helpers.py`**: Stats display, config loading, render path resolution
- **`cli/commands/sync.py`**: Sync command (ingest → render → index)
- **`cli/commands/check.py`**: Health check with repair/vacuum
- **`cli/commands/auth.py`**: OAuth flows (Google Drive)

### Configuration
- **`config.py`**: Config objects (IndexConfig, DriveConfig, Source)
- **`verify.py`**: Data quality checks (orphaned refs, integrity violations)

## Thread Safety Model

| Component | Pattern | Location | Behavior |
|-----------|---------|----------|----------|
| **DB writes** | _WRITE_LOCK | repository.py:48 | Serializes all writes, held only during commit (not I/O) |
| **DB reads** | Thread-local conn | db.py | Each thread gets own connection via `connection_context()` |
| **Ingestion** | Bounded submission | runner.py | Max 16 in-flight futures, FIRST_COMPLETED wait |
| **Rendering** | ThreadPoolExecutor | runner.py | max_workers=4, parallel output generation |
| **Ingest metrics** | IngestResult._lock | pipeline/ingest.py | Protects shared counter updates |

**Parallelization Rules**:
```
✅ Parallel: prepare_ingest() (pure: hashing, parsing, content_blocks extraction)
✅ Parallel: rendering (ThreadPoolExecutor max_workers=4)
✅ Bounded: Ingestion futures (max 16 in-flight prevents memory explosion)
✅ Pattern: with _WRITE_LOCK: save() → Atomically write under lock

❌ NEVER: Direct sqlite3.connect() or sqlite3 ops (bypass repository)
❌ NEVER: Hold _WRITE_LOCK during I/O (deadlock risk, only during commit)
❌ NEVER: Mutate shared state without locks (use thread-local or lock)
```

## Content Hashing (Critical!)

**Unicode Normalization** (core/hashing.py:23):
```python
def hash_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)  # "café" = "café" regardless of composition
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
```

**Why**: Prevents duplicates from visually identical text with different Unicode representations.

**Conversation hash** (pipeline/ids.py):
- Includes: title, timestamps, messages (id/role/text/timestamp), attachments (id/mime_type)
- Excludes: provider_meta (semantic content only)
- Serialization: `json.dumps(normalized, sort_keys=True, separators=(",", ":"))`

**Idempotency**: Same content → same hash → skip on re-import.

## Database Schema (v5)

**Tables**:
1. **conversations** - PK: conversation_id, content_hash for dedup, **source_name** (GENERATED), **metadata** (JSON dict v5)
2. **messages** - FK: conversation_id (CASCADE), stores content_blocks in provider_meta JSON
3. **attachments** - PK: attachment_id, ref_count (manual maintenance)
4. **attachment_refs** - Many-to-many, pruned via `_prune_attachment_refs()`
5. **runs** - Audit trail (plan_snapshot, counts_json, drift_json)

**v5 Metadata Column**:
```sql
-- User-editable metadata stored as JSON
metadata TEXT NOT NULL DEFAULT '{}'
-- Keys: title (user override), summary, tags (list), plus custom keys
```

**Metadata Precedence** (display title): `metadata.title > original_title > truncated_id[:8]`

**v4 Performance Fix** (preserved):
```sql
source_name TEXT GENERATED ALWAYS AS (json_extract(provider_meta, '$.source')) STORED
CREATE INDEX idx_conversations_source_name ON conversations(source_name);
```

**Migrations**: v1→v2 (attachment ref counting), v2→v3 (runs table), v3→v4 (source_name column), v4→v5 (metadata column)

**Database Path**: `XDG_DATA_HOME/polylogue/polylogue.db` (semantic data, not ephemeral state)

## Content Blocks Architecture

**Purpose**: Structured semantic decomposition of message content.

**Structure** (stored in `provider_meta.content_blocks`):
```python
[
    {"type": "text", "text": "Response content"},
    {"type": "thinking", "thinking": "Reasoning trace"},
    {"type": "tool_use", "name": "search", "id": "toolu_123", "input": {...}},
    {"type": "tool_result", "tool_use_id": "toolu_123", "content": "..."},
]
```

**Extraction** (at import time):
- **ChatGPT**: From `message.content.parts` + `content_type`
- **Claude Code**: From structured `content` array
- **Gemini**: From `chunkedPrompt.chunks`
- **Claude AI**: No structured blocks (simple text)

**Usage**:
- Powers `is_thinking`, `is_tool_use`, `is_substantive` properties
- Eliminates brittle text heuristics
- Enables precise semantic filtering

**Multimodal content**: Images/files handled via **attachments table**, NOT content_blocks.

## Metadata System (v5)

**Purpose**: User-editable conversation metadata separate from provider data.

**Storage**: `conversations.metadata` column (JSON dict, default `{}`)

**Reserved Keys**:
| Key | Type | Purpose |
|-----|------|---------|
| `title` | string | User override for display title |
| `summary` | string | User-written summary |
| `tags` | list[str] | User-assigned tags |

**Conversation Properties** (lib/models.py):
```python
conv.user_title   # metadata.get("title") - user override
conv.display_title  # user_title or title or id[:8]
conv.summary      # metadata.get("summary")
conv.tags         # metadata.get("tags", [])
```

**CRUD Operations** (StorageRepository / SQLiteBackend):
```python
# Get full metadata dict
meta = repo.get_metadata(conv_id)

# Update single key
repo.update_metadata(conv_id, "title", "My Custom Title")

# Delete key
repo.delete_metadata(conv_id, "title")

# Tag management
repo.add_tag(conv_id, "important")
repo.remove_tag(conv_id, "archive")
```

**Invariant**: Metadata is user-editable; content_hash excludes metadata (only semantic content).

## Semantic Projections

**Fluent API** (lib/projections.py):
```python
# Lazy-evaluated, composable filters
conv.project()
    .substantive()        # No noise/tools
    .min_words(50)        # Meaningful length
    .since(datetime(...)) # After date
    .limit(10)
    .execute()            # Materialize

# Lazy iteration (memory efficient)
for msg in conv.project().contains("error").iter():
    process(msg)

# Count without materialization
error_count = conv.project().contains("error").count()
```

**Available filters**: `substantive()`, `dialogue()`, `user_messages()`, `assistant_messages()`, `without_noise()`, `with_attachments()`, `min_words(n)`, `max_words(n)`, `since(date)`, `before(date)`, `contains(pattern)`, `matches(regex)`, `where(predicate)`

**Legacy API**: `conversation.substantive_only()`, `iter_pairs()`, `without_noise()`

## Conversation Filter Chain API

**Fluent API** (lib/filters.py):
```python
from polylogue import Polylogue

p = Polylogue()

# Filter chain for conversation-level queries
convs = (
    p.filter()
    .provider("claude")           # Filter by provider
    .since("2024-01-01")          # After date
    .contains("error")            # FTS search
    .tag("important")             # Has tag
    .sort("date")                 # Sort field
    .limit(10)                    # Max results
    .list()                       # Execute query
)

# Terminal methods
first = p.filter().provider("chatgpt").first()   # First match or None
count = p.filter().has("thinking").count()       # Count matches
conv = p.filter().tag("review").pick()           # Interactive picker
```

**Filter methods** (return self for chaining):
| Method | Purpose |
|--------|---------|
| `provider(*names)` | Include providers |
| `no_provider(*names)` | Exclude providers |
| `tag(*tags)` | Include tags |
| `no_tag(*tags)` | Exclude tags |
| `contains(text)` | FTS search |
| `no_contains(text)` | Exclude text |
| `has(*types)` | Content types: thinking, tools, attachments, summary |
| `since(date)` | After date |
| `until(date)` | Before date |
| `title(pattern)` | Title contains |
| `id(prefix)` | ID prefix |
| `sort(field)` | date, tokens, messages, words, longest, random |
| `reverse()` | Ascending instead of descending |
| `limit(n)` | Max results |
| `sample(n)` | Random sample |
| `where(predicate)` | Custom filter function |

**Terminal methods** (execute query):
| Method | Returns |
|--------|---------|
| `list()` | `list[Conversation]` |
| `first()` | `Conversation \| None` |
| `count()` | `int` |
| `pick()` | Interactive selection |

## Archive Stats

```python
stats = p.stats()
print(f"Conversations: {stats.conversation_count}")
print(f"Messages: {stats.message_count}")
print(f"Providers: {stats.providers}")  # dict[str, int]
print(f"Tags: {stats.tags}")            # dict[str, int]
print(f"Recent: {[c.id for c in stats.recent]}")  # 5 most recent
```

## CLI Commands

**Query-First Architecture** (Phase 3.3): Positional args without subcommand prefix = query mode.

### Query Mode (Default)

```bash
polylogue                           # Stats mode (no args, no filters)
polylogue "search terms"            # Query mode (positional = FTS search)
polylogue -p claude --since "last week"  # Filter mode (no query, filters only)
polylogue --latest --output browser # Most recent, open in browser
```

| Category | Flags | Description |
|----------|-------|-------------|
| **Filters** | `-c/--contains`, `-C/--no-contains`, `--regex/--no-regex` | FTS/regex text filters |
| | `-p/--provider`, `-P/--no-provider` | Provider include/exclude |
| | `-t/--tag`, `-T/--no-tag` | Tag filters (comma = OR, key:value) |
| | `--has/--no-has` | Content types: thinking, tools, summary, attachments |
| | `--since/--until`, `--title`, `-i/--id` | Date range, title contains, ID prefix |
| | `-n/--limit`, `--latest`, `--sample` | Result limits, sampling |
| | `--sort`, `--reverse`, `--similar` | Ordering: date, tokens, messages, words, longest, random |
| **Output** | `-o/--output` | Destinations: stdout, browser, clipboard (comma-sep) |
| | `-f/--format` | Formats: markdown, json, html, obsidian, org |
| | `--fields`, `--list`, `--stats` | Field selection, force list, stats-only |
| | `--by-month/--by-provider/--by-tag` | Aggregation histograms |
| | `--open`, `--csv`, `--pick` | Browser open, CSV export, interactive picker |
| **Modifiers** | `--set KEY VALUE`, `--unset KEY` | Metadata modification |
| | `--add-tag/--rm-tag`, `--annotate` | Tag management, LLM annotation |
| | `--delete` | Delete matched (requires filter) |
| **Global** | `--plain/--interactive`, `--config`, `-v/--verbose` | Output mode, config path |

**Exit Codes**: 0 = success, 1 = error, 2 = no results

### Subcommands

| Command | Purpose | Key Flags | Example |
|---------|---------|-----------|---------|
| **`sync`** | Ingest → render → index | `--preview`, `--stage`, `--source`, `--format` | `polylogue sync --stage all` |
| **`check`** | Integrity checks/repair | `--verbose`, `--repair`, `--vacuum` | `polylogue check --repair` |
| **`reset`** | Clear database/cache | `--database`, `--cache`, `--all`, `--force` | `polylogue reset --database` |
| **`mcp`** | Start MCP server | `--transport` | `polylogue mcp` |
| **`config`** | Configuration | `show`, `init`, `set` | `polylogue config show` |
| **`auth`** | OAuth flows | `-p/--provider`, `--refresh`, `--revoke` | `polylogue auth` |
| **`serve`** | API server | `--host`, `--port` | `polylogue serve` |
| **`completions`** | Shell completions | `bash`, `zsh`, `fish` | `polylogue completions bash` |

**Implementation**: `cli/click_app.py` - `QueryFirstGroup` class overrides `parse_args()` to convert positional args to `--query-term` options, enabling both query mode and subcommand dispatch.

## Provider Quirks

| Provider | Format | Structure | Content Blocks | Attachments | Importer |
|----------|--------|-----------|----------------|-------------|----------|
| **ChatGPT** | conversations.json | UUID graph (`mapping`), parent/child traversal | From `content.parts` + `content_type` | Tool outputs in `metadata.attachments` | importers/chatgpt.py |
| **Claude AI** | JSONL | Flat `chat_messages` array | Simple text only (no structured blocks) | N/A | importers/claude.py |
| **Claude Code** | JSON array | `parentUuid`/`sessionId` markers | `[{type: "text\|thinking\|tool_use"}]` | Via content array | importers/claude.py (same file) |
| **Gemini** | Drive API | `chunkedPrompt.chunks` | From chunk structure | Via Drive metadata | ingestion/drive.py |
| **Codex** | Session export | Session-based | N/A | N/A | importers/codex.py |

**Detection**: `ingestion/source.py:detect_provider()` auto-detects from payload structure (`looks_like()` functions)

## External Integrations

| Integration | Purpose | Config | Auth | Retry Logic | Behavior |
|-------------|---------|--------|------|-------------|----------|
| **Voyage AI** | Embeddings for Qdrant | `VOYAGE_API_KEY` (required) | API key | 5× exponential backoff 1s→10s | voyage-2 model, 1024-dim vectors, batch API |
| **Qdrant** | Vector search | `QDRANT_URL` (default: localhost:6333), `QDRANT_API_KEY` (optional) | Optional API key | 3× exponential backoff 1s→5s | Cosine distance, batch upsert |
| **Google Drive** | Gemini chat import | `~/.config/polylogue/polylogue-credentials.json`, `~/.config/polylogue/polylogue-token.json` | OAuth 2.0 (interactive) | 3× (configurable via `ENV_DRIVE_RETRIES`) | Auto-refresh token, requires browser for initial auth |

**Notes**:
- Qdrant: storage/search_providers/qdrant.py, lazy imports (no deps if unused)
- Drive: ingestion/drive_client.py, OAuth flow blocks until user grants access
- Voyage: Single batch call (no chunking), handles all messages at once

## Security & Safety

### FTS5 Query Escaping (search.py:62-99)
Hardened against injection:
- Asterisk-only queries (`*`, `***`) → empty phrase
- Trailing/consecutive operators (`test AND`, `a AND AND b`) → quoted
- Special chars (`"*^(){}[]|&!+-`) → individual token quoting
- Recent fix: Added `+` operator, asterisk-only, operator position checks

### Encoding Fallback (source_ingest.py:17-49)
**Chain**: UTF-8 → UTF-8-sig → UTF-16 (LE/BE) → UTF-32 (LE/BE) → UTF-8 w/ `errors='ignore'`
- Removes null bytes (`\x00`) from decoded strings
- Returns `None` if all attempts fail

### ZIP Bomb Protection (source_ingest.py:28-30)
- Max compression ratio: 100:1
- Max uncompressed size: 500MB

## Performance Characteristics

### What's Fast ✅
- **Parallel ingestion**: prepare_ingest() hashes in parallel (max 16 in-flight)
- **Parallel rendering**: ThreadPoolExecutor max_workers=4
- **Bounded submission**: Prevents memory explosion
- **Incremental FTS**: Only updates changed conversations (chunked batches of 200)
- **Source filtering**: Uses indexed `source_name` column (v4)
- **Content-hash checks**: Skips unchanged conversations

### What's Expensive
- **SHA-256 hashing**: O(n) for message text (acceptable in parallel phase)
- **FTS rebuild**: Deletes + re-inserts all messages (chunked, rarely needed)
- **Qdrant**: Network I/O + API rate limits (incremental updates only)

### Optimization Patterns
```python
# ✅ Good: Incremental updates
update_index_for_conversations(changed_ids)

# ❌ Bad: Full rebuild
rebuild_index()

# ✅ Good: Bounded submission
while len(futures) > 16:
    done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
    for fut in done: handle(fut); del futures[fut]

# ❌ Bad: Unbounded
for item in items: futures.add(executor.submit(process, item))
```

## Storage Backend Abstraction

**Protocol** (protocols.py):
```python
class StorageBackend(Protocol):
    """Abstract interface for storage backends."""
    def save_conversation(
        self,
        conversation: ConversationRecord,
        messages: list[MessageRecord],
        attachments: list[AttachmentRecord],
    ) -> dict[str, int]: ...

    def get_conversation(self, conversation_id: str) -> ConversationRecord | None: ...

    def iter_conversations(self, **filters) -> Iterator[ConversationRecord]: ...

    # Metadata CRUD (v5)
    def get_metadata(self, conversation_id: str) -> dict[str, object]: ...
    def update_metadata(self, conversation_id: str, key: str, value: object) -> None: ...
    def delete_metadata(self, conversation_id: str, key: str) -> None: ...
    def add_tag(self, conversation_id: str, tag: str) -> None: ...
    def remove_tag(self, conversation_id: str, tag: str) -> None: ...
```

**SQLiteBackend** (storage/backends/sqlite.py):
- **Schema**: v5 with migrations support
- **Performance**: Indexed lookups on provider_name, provider_conversation_id, source_name
- **Deduplication**: Content hash checks before insert
- **Transactions**: Atomic operations via connection context manager

**Extension Points**:
- Add PostgreSQL backend: Implement StorageBackend protocol in `storage/backends/postgres.py`
- Add DuckDB backend: Implement StorageBackend protocol in `storage/backends/duckdb.py`
- Pass backend to StorageRepository via constructor: `repo = StorageRepository(backend=PostgresBackend())`
- **Backend is required**: StorageRepository always requires a backend parameter
- Use `create_default_backend()` for convenience: `StorageRepository(backend=create_default_backend())`

## Search Provider Abstraction

| Provider | Type | Storage | Indexing | Performance | Retry |
|----------|------|---------|----------|-------------|-------|
| **FTS5Provider** | Full-text (BM25) | SQLite virtual table | Incremental (delete+reinsert 200/batch) | Fast, local, no API | N/A |
| **QdrantProvider** | Vector semantic | Remote Qdrant/Docker | Batch (Voyage API bulk) | Network I/O, rate-limited | Voyage 5×, Qdrant 3× exp backoff |
| **Search Cache** | LRU wrapper | In-memory (128 entries) | Version-based invalidation | Fast (sub-millisecond) | N/A |

**Protocol Methods** (protocols.py):
- `index(messages: list[MessageRecord])` → Batch index/reindex
- `search(query: str, limit: int)` → Ranked results with scores
- `delete(message_ids: list[str])` → Remove from index

**IndexConfig**:
| Field | Default | Env Override | Purpose |
|-------|---------|--------------|---------|
| `enabled` | True | - | Enable/disable indexing |
| `provider` | "fts5" | - | "fts5" \| "qdrant" \| "hybrid" |
| `qdrant_url` | None | QDRANT_URL | `http://localhost:6333` |
| `qdrant_api_key` | None | QDRANT_API_KEY | Optional auth |
| `voyage_api_key` | None | VOYAGE_API_KEY | Required for Qdrant |

**Extension**: Implement SearchProvider → Add to factory → Configure via IndexConfig

## Service Layer

| Service | File | Responsibility | Lifecycle | Key Methods | Thread Safety |
|---------|------|---------------|-----------|-------------|---------------|
| **IngestionService** | pipeline/services/ingestion.py | Orchestrate ingestion from sources | Factory | `ingest_source()`, `ingest_all()`, `ingest_drive()` | Via StorageRepository |
| **IndexService** | pipeline/services/indexing.py | Manage FTS5/Qdrant index updates | Factory | `index_incremental()`, `rebuild_index()` | Via SearchProvider |
| **RenderService** | pipeline/services/rendering.py | Generate Markdown/HTML/JSON outputs | Factory | `render_markdown()`, `render_html()`, `render_json()` | Parallel (4 workers) |

**Pipeline Flow**: `IngestionService.ingest_all()` → `IndexService.index_incremental(changed_ids)` → `RenderService.render_all()`
**Orchestration**: pipeline/runner.py coordinates service calls, aggregates results into RunResult

## Dependency Injection

**Framework**: `dependency-injector>=4.41.0`

| Provider | Type | Dependencies | File |
|----------|------|--------------|------|
| `config` | Singleton | load_config() | container.py |
| `storage` | Singleton | SQLiteBackend (singleton) | container.py |
| `ingestion_service` | Factory | storage, config | container.py |
| `indexing_service` | Factory | storage, config | container.py |
| `rendering_service` | Factory | renderer (OutputRenderer), config.render_root | container.py |

**Behavior**:
- **Singleton**: Same instance across calls (`repo1 is repo2`)
- **Factory**: New instance per call (`svc1 is not svc2`)
- **CLI Integration**: cli/container.py wraps container with `create_*()` factory functions
- **Testing**: `container.override()` enables easy mocking (test_container.py)
- **Benefits**: Explicit deps, type-safe, single source of truth, backward compatible

## Renderer Abstraction

**Protocol** (protocols.py): `OutputRenderer` with `render(conversation_id, output_path) → Path`, `supports_format(format) → bool`

| Renderer | File | Output | Dependencies | Performance |
|----------|------|--------|--------------|-------------|
| **MarkdownRenderer** | rendering/renderers/markdown.py | .md only | None (stdlib only) | Fast, minimal |
| **HTMLRenderer** | rendering/renderers/html.py | .html + .md | Jinja2, markdown-it | Custom templates via template_path |

**Factory**: `create_renderer(format, config)` in rendering/renderers/__init__.py
**CLI**: `polylogue run --format markdown` or `--format html` (default)
**Extension**: Implement OutputRenderer → Add to factory → Use via `--format pdf`


## Configuration Objects

| Object | File | Fields | Defaults | Env Override |
|--------|------|--------|----------|--------------|
| **Config** | config.py | archive_root, index, drive, sources | YAML-based (optional, defaults if missing) | `POLYLOGUE_ARCHIVE_ROOT` |
| **IndexConfig** | config.py | enabled, provider, qdrant_url, qdrant_api_key, voyage_api_key | enabled=True, provider="fts5" | `QDRANT_URL`, `QDRANT_API_KEY`, `VOYAGE_API_KEY` |
| **DriveConfig** | config.py | enabled, credential_path, retries, backoff_factor | enabled=False, retries=3, backoff=1.0 | `POLYLOGUE_CREDENTIAL_PATH`, `ENV_DRIVE_RETRIES` |
| **Source** | config.py | name, path, provider, enabled | enabled=True | - |

**Precedence** (highest → lowest): CLI flags → Env vars (`POLYLOGUE_*`) → YAML config → Defaults
**Providers**: "chatgpt", "claude", "codex", "gemini" (auto-detected from payload structure)

## Error Handling

| Exception | Category | Pipeline Behavior | Example |
|-----------|----------|-------------------|---------|
| `ConfigError` | Configuration | Fail-fast (abort) | Invalid YAML, missing archive_root |
| `DatabaseError` | Storage | Fail-fast (abort) | Schema corruption, locked database |
| `DriveAuthError` | External | Fail-fast (abort) | OAuth token expired, invalid credentials |
| `DriveNotFoundError` | External | Graceful (log, skip) | Missing Drive file, deleted conversation |
| `QdrantError` | External | Graceful (log, set indexed=False) | Network timeout, Qdrant unavailable |
| `UIError` | Presentation | Graceful (log, fallback) | Template rendering failure |

**Pipeline Strategy** (pipeline/runner.py):
- **Ingest**: Fail-fast (abort on error, raise exception)
- **Render**: Graceful degradation (log, continue, track in RunResult.render_failures)
- **Index**: Graceful degradation (log, set `indexed=False`, continue)

## Testing

**Key Test Suites**:
| Suite | Focus |
|-------|-------|
| test_importers_chatgpt.py | Provider parsing, graph traversal, content_blocks |
| test_importers_claude.py | JSONL parsing, Claude Code structured blocks |
| test_projections.py | Fluent API, lazy evaluation, composability |
| test_store.py | Storage layer, _WRITE_LOCK, transactions |
| test_db.py | Database ops, migrations, schema v5 |
| test_hashing.py | NFC normalization, content hash stability |
| test_properties.py | Hypothesis property-based (hash stability, idempotency) |
| test_pipeline_concurrent.py | Thread safety, bounded submission |
| test_container.py | DI singleton vs factory behavior |
| test_renderers.py | OutputRenderer protocol compliance |
| tests/benchmarks/ | Performance baselines and regression detection |

**Fixtures** (tests/conftest.py): `workspace_env` (isolated temp dirs), `test_conn` (DB connection), `DbFactory` (conversation factory)

### Test Coverage Rationale

**Overall Target**: 75% coverage (current: 69%)
**Core Business Logic**: 85%+ target (storage, pipeline, models, protocols)

**Low-Coverage Modules** (Deliberate):

| Module | Coverage | Rationale |
|--------|----------|-----------|
| ui/facade.py | 44% | Presentation layer wrapping Rich/questionary. Hard to test (ANSI, TTY). Tested via integration. |
| ui/interactive.py | 37% | Interactive prompts require TTY mocking. Manual testing preferred. |
| cli/formatting.py | 68% | Cosmetic formatting. Visual QA more valuable. |
| ingestion/drive_client.py | 26% | External OAuth API. Target: 60-65% (Sprint 3). Priority: HIGH VALUE. |
| cli/commands/* | 25-70% | User-facing commands. Target: 70-80% (Sprint 2). Priority: MEDIUM-HIGH. |

**Coverage Priorities**:

1. **Business logic** (80%+ target): storage, pipeline, models, protocols
2. **External integrations** (60%+ target): Drive, Qdrant, OAuth
3. **Security-critical** (100% target): SQL injection, command validation
4. **User-facing commands** (70%+ target): CLI, search, rendering

**Testing Philosophy**:
- Focus on high-value business logic and external integrations
- Accept low coverage on presentation layers (UI, formatting)
- Prioritize security-critical paths for 100% coverage
- Use property-based testing (Hypothesis) for invariants

### Test Isolation Patterns

**Module Reload Class Identity Issue**:
When tests reload modules (e.g., to pick up new XDG_DATA_HOME), class identity breaks:
```python
# Problem: After reload, DatabaseError class has different identity
import importlib
importlib.reload(polylogue.storage.db)  # Creates NEW DatabaseError class
# Other modules still have OLD DatabaseError reference

# Solution 1: Check type name instead of class identity
with pytest.raises(Exception) as exc_info:
    risky_operation()
assert exc_info.type.__name__ == "DatabaseError"
assert "expected message" in str(exc_info.value)

# Solution 2: Import after reload for consistent references
importlib.reload(polylogue.storage.db)
from polylogue.storage.db import DatabaseError as CurrentDatabaseError
with pytest.raises(CurrentDatabaseError, match="message"):
    risky_operation()
```

**Autouse Fixture** (conftest.py): Clears thread-local DB state, search cache, and env vars before each test.

**Workspace Fixtures**: `workspace_env`, `cli_workspace` provide isolated temp directories with XDG_DATA_HOME set correctly.

## Development Patterns

### Adding a Storage Backend

**Goal**: Support a new storage engine (PostgreSQL, DuckDB, etc.)

1. Create `polylogue/storage/backends/newbackend.py`
2. Implement `StorageBackend` protocol from `protocols.py`:
   ```python
   class NewBackend:
       def save_conversation(
           self,
           conversation: ConversationRecord,
           messages: list[MessageRecord],
           attachments: list[AttachmentRecord],
       ) -> dict[str, int]:
           # Check content_hash for deduplication
           # Insert conversation, messages, attachments atomically
           # Return counts: {"conversations": N, "messages": M, ...}
   ```
3. Implement query methods:
   - `get_conversation(conversation_id: str) → ConversationRecord | None`
   - `iter_conversations(**filters) → Iterator[ConversationRecord]`
4. Register in `storage/__init__.py` for import discovery
5. Pass to StorageRepository: `repo = StorageRepository(backend=NewBackend())`

**Key Requirements**:
- Content-hash deduplication (skip if hash exists)
- Atomic transactions (all or nothing)
- Idempotent (safe to re-run)
- Return standardized counts dict

### Adding a Search Provider

**Goal**: Support new search engine (Milvus, Elasticsearch, etc.)

1. Create `polylogue/storage/search_providers/newprovider.py`
2. Implement `SearchProvider` protocol from `protocols.py`:
   ```python
   class NewProvider:
       def index(self, messages: list[MessageRecord]) -> None:
           # Load messages, convert to documents
           # Handle embeddings if vector-based (e.g., via Voyage AI)
           # Bulk insert to index
           # Be idempotent (delete then re-insert, or upsert)

       def search(self, query: str, limit: int = 100) -> list[SearchResult]:
           # Parse query (plain text, semantic, hybrid)
           # Return ranked results with relevance scores

       def delete(self, message_ids: list[str]) -> None:
           # Remove messages from index by ID
   ```
3. Add configuration to `IndexConfig` (config.py):
   ```python
   @dataclass
   class IndexConfig:
       provider: Literal["fts5", "qdrant", "newprovider"] = "fts5"
       newprovider_url: str | None = None  # New config option
   ```
4. Update `IndexService` in `pipeline/services/indexing.py` to instantiate provider
5. Register in search provider factory

**Key Requirements**:
- Idempotent indexing (safe to re-index)
- Handle incremental updates (only changed messages)
- Return SearchResult objects with `message_id`, `score`, `preview`
- Graceful error handling (log, set `indexed=False`, continue)

### Adding a Service

**Goal**: Create a new business-logic service (e.g., ExportService, AnalysisService)

1. Create `polylogue/pipeline/services/newservice.py`
2. Define result class:
   ```python
   @dataclass
   class NewServiceResult:
       success: bool
       count: int
       errors: list[str] = field(default_factory=list)
   ```
3. Implement service class:
   ```python
   class NewService:
       def __init__(self, repository: StorageRepository, config: Config):
           self.repository = repository
           self.config = config

       def execute(self, **params) -> NewServiceResult:
           # Use repository for DB operations
           # Return result object
   ```
4. Add to `pipeline/services/__init__.py` exports
5. Instantiate in `pipeline/runner.py` and call during pipeline

**Key Requirements**:
- Accept `StorageRepository` for coordinated DB access
- Accept `Config` for configuration
- Return result object with counts + errors
- Use repository for all DB operations (never direct SQLite)
- Thread-safe (repository handles locks)

### Adding an Importer

1. Create `polylogue/importers/newprovider.py`
2. Implement `looks_like(payload) → bool` (format detection)
3. Implement `parse(payload, fallback_id) → ParsedConversation`
4. **Extract content_blocks** from structured message content
5. Extract attachments via `attachment_from_meta()`
6. Register in `ingestion/source.py:detect_provider()`

**Example** (claude.py):
```python
def parse_code(payload: list, fallback_id: str) -> ParsedConversation:
    messages = []
    for item in payload:
        content_blocks = []
        if isinstance(item.get("content"), list):
            for seg in item["content"]:
                if seg.get("type") == "thinking":
                    content_blocks.append({
                        "type": "thinking",
                        "thinking": seg.get("thinking"),
                    })
                # ... handle tool_use, text, etc.

        meta = {"content_blocks": content_blocks} if content_blocks else None
        messages.append(ParsedMessage(..., provider_meta=meta))

    return ParsedConversation(...)
```

### Adding Semantic Projections

**Approach 1: Properties** (simple boolean classification):
```python
@property
def is_thinking(self) -> bool:
    if self.provider_meta:
        blocks = self.provider_meta.get("content_blocks", [])
        if any(b.get("type") == "thinking" for b in blocks):
            return True
    return False
```

**Approach 2: Fluent API** (composable filters):
```python
def min_words(self, n: int) -> ConversationProjection:
    return self.where(lambda m: m.word_count >= n)
```

### Anti-Patterns to Avoid

1. **Direct sqlite3 ops without StorageRepository** → Use `repository.save_conversation()`
2. **Skip content_blocks extraction** → Extract structured blocks at import
3. **Hardcode paths** → Use `config.archive_root`
4. **Manual commits in pipeline** → Let StorageRepository + connection_context() handle it
5. **Modify hash logic carelessly** → Breaks idempotency, maintain compatibility
6. **Bypass semantic projections** → Use `message.is_tool_use` not string matching
7. **Mutate shared state in parallel ops** → Use locks or keep operations pure
8. **Bypass StorageRepository for writes** → All writes must go through repository with `_WRITE_LOCK`
9. **Assume backend is SQLite** → Use StorageBackend protocol, support any backend
10. **Hardcode search provider** → Use SearchProvider protocol, support FTS5/Qdrant/others
11. **Create StorageRepository without backend** → Always pass `backend=create_default_backend()` or custom backend

## Type Safety

**Configuration**: Mypy strict mode enabled (`strict = true`), type marker (`polylogue/py.typed`)

**Type Annotations**:
| Category | Types | Usage Example |
|----------|-------|---------------|
| NewType IDs | ConversationId, MessageId, AttachmentId | `def get(id: ConversationId) → Conversation` |
| Protocols | StorageBackend, SearchProvider, OutputRenderer, VectorProvider | `repo: StorageRepository[StorageBackend]` |
| Variance | Mapping (covariant), dict (invariant) | `def format(data: Mapping[str, object])` |
| Unions | Path \| None, str \| None | `config_path: Path \| None = None` |
| Forward refs | TYPE_CHECKING imports | `if TYPE_CHECKING: from .projections import ...` |

**Type Safety Patterns**:
- Runtime type narrowing: `isinstance(blocks, list) and isinstance(b, dict)` before iteration
- NewType conversions: `ConversationId(str(row["conversation_id"]))` not bare `str`
- Variance correctness: `Mapping[str, object]` for function params (covariant)
- Method shadowing: `builtins.list` to avoid conflict with `self.list()`
- Full annotations: All functions have parameter and return types

## Build Commands

```bash
# Tests
uv run pytest -q --ignore=tests/test_qdrant.py
uv run pytest -v tests/test_lib.py
uv run pytest --cov=polylogue --cov-report=html

# Quality
uv run mypy polylogue/
uv run ruff check polylogue/ tests/
uv run ruff format --check polylogue/ tests/

# Benchmarks
uv run python tests/benchmarks/run_all.py

# Run (query-first CLI)
uv run polylogue --help                              # Help
uv run polylogue                                     # Stats mode (no args)
uv run polylogue "python error"                      # Query mode
uv run polylogue -p claude --since "last week"       # Filter mode
uv run polylogue --latest -f json                    # Most recent, JSON
uv run polylogue sync --format html                  # Sync subcommand
POLYLOGUE_FORCE_PLAIN=1 uv run polylogue sync --preview
uv run polylogue check --verbose
```

## Pinned Notes

Active scratch files transcluded at session startup. Use `@.claude/scratch/NNN-topic.md` syntax (bare line, not in code block).

*Currently empty - add as needed*

---

## Common Debugging

**"Database locked"**: Multiple write attempts → Ensure all writes go through StorageRepository with `_WRITE_LOCK` (storage/repository.py:48)

**"Attachments ref_count=0 but referenced"**: `_prune_attachment_refs()` not called → Check storage/repository.py or backend

**"Content hash mismatch"**: Normalization changed → Verify NFC normalization (core/hashing.py:23)

**"FTS syntax error"**: Unescaped query → Use `escape_fts5_query()` (tests: test_search.py)

**"Thinking not detected"**: Missing content_blocks → Verify importer extracts blocks (lib/models.py:is_thinking)

**"Config not reflected"**: Env vars override config → Check `POLYLOGUE_*` vars, use `polylogue config show --json`

**"Backend not being used"**: StorageRepository initialized without backend → Pass `backend=YourBackend()` to constructor

**"Search provider not indexing"**: IndexService not instantiated → Check pipeline/runner.py service creation

**"Service thread safety issue"**: Direct DB access bypassing repository → Use `StorageRepository` for all writes

**"Search provider mismatch"**: Wrong provider instantiated → Verify IndexConfig.provider setting and service initialization

