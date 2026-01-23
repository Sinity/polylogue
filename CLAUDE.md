# Polylogue Developer Guide

Polylogue is a local-first AI chat archive that ingests exports from ChatGPT, Claude AI, Claude Code, Codex, and Google Drive into SQLite with full-text search and optional vector search.

## Core Architecture

**Mission**: Privacy-preserving, searchable archive of AI chat history.

**Design Principles** (Post-Refactoring):
- **Layered abstraction**: Storage (backends) → Ingestion → Rendering → Pipeline services
- **Backend agnostic**: StorageBackend protocol enables SQLite/PostgreSQL/other implementations
- **Search provider agnostic**: SearchProvider protocol supports FTS5/Qdrant/vector alternatives
- **Service-oriented**: IngestionService, IndexService, RenderService encapsulate business logic
- **Repository pattern**: StorageRepository owns the write lock and coordinates DB operations

**Key Invariants**:
- **Content-hash deduplication**: SHA-256 with NFC Unicode normalization prevents duplicates
- **Idempotent ingestion**: Safe to re-run imports; unchanged conversations skipped
- **Thread-safe**: Thread-local connections + `_WRITE_LOCK` in StorageRepository for writes
- **Structured content**: Content blocks decompose messages into text/thinking/tool_use segments
- **Semantic projections**: Fluent API for filtering (substantive, dialogue-only, pairs, etc.)

## Data Flow

```
JSON/ZIP/Drive → Auto-detect provider → Parse + extract content_blocks →
Hash (NFC normalized) → Store (under _WRITE_LOCK) → Render (parallel) →
Index (FTS5 + optional Qdrant)
```

## Critical Files

### Core Abstractions & Models
- **`lib/models.py`**: Message/Conversation with semantic classification (`is_thinking`, `is_tool_use`, `is_substantive`)
- **`lib/projections.py`**: Fluent projection API (`conv.project().substantive().min_words(50).execute()`)
- **`core/hashing.py`**: NFC-normalized SHA-256 hashing
- **`protocols.py`**: Protocol definitions (StorageBackend, SearchProvider)

### Storage Layer (Refactored)
- **`storage/db.py`**: Thread-local connections, schema v4, migrations
- **`storage/store.py`**: Record definitions (ConversationRecord, MessageRecord, AttachmentRecord)
- **`storage/repository.py`**: StorageRepository owns `_WRITE_LOCK`, coordinates DB operations
- **`storage/backends/sqlite.py`**: SQLiteBackend implementation (schema, migrations)
- **`storage/search_providers/fts5.py`**: FTS5 search provider (incremental indexing)
- **`storage/search_providers/qdrant.py`**: Qdrant vector search provider (Voyage AI embeddings)

### Ingestion & Source Management
- **`ingestion/source.py`**: Provider detection, ParsedConversation/ParsedMessage
- **`ingestion/ingest.py`**: Source ingestion pipeline (Drive, files, etc.)
- **`ingestion/drive.py`**: Google Drive client and Gemini parsing
- **`ingestion/drive_client.py`**: OAuth credential management
- **`importers/chatgpt.py`**: Graph traversal, extracts thinking from `content_type`
- **`importers/claude.py`**: JSONL chat_messages (Claude AI) + structured blocks (Claude Code)
- **`importers/codex.py`**: Session exports

### Pipeline & Services (Refactored)
- **`pipeline/runner.py`**: Orchestrates ingest → render → index pipeline
- **`pipeline/ingest.py`**: Prepares conversations, checks content hashes
- **`pipeline/ids.py`**: Content hash generation
- **`pipeline/services/ingestion.py`**: IngestionService orchestrates conversation ingest
- **`pipeline/services/indexing.py`**: IndexService manages search index updates
- **`pipeline/services/rendering.py`**: RenderService generates output formats
- **`pipeline/services/__init__.py`**: Service exports

### Configuration
- **`config.py`**: Config objects (IndexConfig, DriveConfig, Source)
- **`verify.py`**: Data quality checks (orphaned refs, integrity violations)

## Thread Safety Model

**Architecture**:
```python
ThreadPoolExecutor(max_workers=4)
    ├─ prepare_ingest() - Pure (hashing, parsing) → runs in parallel
    │  Bounded submission: max 16 in-flight futures
    └─ StorageRepository.save_conversation() - Writes DB → serialized under _WRITE_LOCK

Rendering: Parallel (max_workers=4)
```

**Locks**:
- `_WRITE_LOCK` (StorageRepository:48): Protects all DB writes, owned by repository
- Thread-local connections: Each thread has own SQLite connection via `connection_context()`
- `IngestResult._lock`: Protects shared ingest metrics during parallel preparation

**Rules**:
- ✅ Parallelize pure ops (hashing, parsing, rendering)
- ✅ All DB writes go through StorageRepository with `_WRITE_LOCK` held
- ✅ Use thread-local connections via `connection_context()` context manager
- ✅ Bound in-flight futures (prevents memory explosion, max 16 in-flight)
- ✅ Pure operations (hashing, parsing) can run unbounded in parallel
- ❌ DON'T bypass StorageRepository for DB writes
- ❌ DON'T hold `_WRITE_LOCK` during I/O (only during commit)
- ❌ DON'T mutate shared state without locks (use thread-local state when possible)

**Verified**: storage/repository.py `with _WRITE_LOCK: save_conversation(...)`

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

## Database Schema (v4)

**Tables**:
1. **conversations** - PK: conversation_id, content_hash for dedup, **source_name** (GENERATED v4)
2. **messages** - FK: conversation_id (CASCADE), stores content_blocks in provider_meta JSON
3. **attachments** - PK: attachment_id, ref_count (manual maintenance)
4. **attachment_refs** - Many-to-many, pruned via `_prune_attachment_refs()`
5. **runs** - Audit trail (plan_snapshot, counts_json, drift_json)

**v4 Performance Fix**:
```sql
-- OLD: json_extract() can't use index
SELECT * FROM conversations WHERE json_extract(provider_meta, '$.source') = 'x';

-- NEW: Generated column with index
source_name TEXT GENERATED ALWAYS AS (json_extract(provider_meta, '$.source')) STORED
CREATE INDEX idx_conversations_source_name ON conversations(source_name);
```

**Migrations**: v1→v2 (attachment ref counting), v2→v3 (runs table), v3→v4 (source_name column)

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

## CLI Commands

- **`run`**: Ingest → render → index pipeline (with `--preview`, `--stage`, `--source`)
- **`search`**: FTS5 full-text search with query escaping
- **`view`**: Semantic projection viewer
  - Projections: `full`, `dialogue`, `clean` (default), `pairs`, `user`, `assistant`, `thinking`, `stats`
  - Filters: `--provider`, `--since`, `--until`, `--query` (FTS)
  - Output: text, `--json`, `--json-lines`, `--list`
  - Example: `polylogue view --provider claude --query "python" -p stats --json`
- **`verify`**: Data integrity checks (orphaned refs, missing files)
- **`config`**: View/edit configuration

## Provider Quirks

### ChatGPT
- **Format**: `conversations.json` with UUID-keyed graph (`mapping`)
- **Traversal**: Reconstruct order via parent/child relationships
- **Content blocks**: From `message.content.parts` + `content_type`
- **Attachments**: Tool outputs in `message.metadata.attachments`

### Claude AI
- **Format**: JSONL with `chat_messages` array
- **Structure**: Flat messages with `uuid`, `sender`, `text`, `created_at`
- **Content blocks**: Simple text, no structured blocks

### Claude Code
- **Format**: JSON array with `parentUuid`/`sessionId` markers
- **Content blocks**: `[{"type": "text|thinking|tool_use", ...}]`
- **Detection**: `isThought: true` or `type: "thinking"`

### Gemini (Drive)
- **Format**: `chunkedPrompt.chunks`
- **Content blocks**: From chunk structure

## External Integrations

### Qdrant (Vector Search)
- **Embedding**: Voyage AI (`voyage-2`, 1024-dim vectors, cosine distance)
- **API Key**: `VOYAGE_API_KEY` env var (required)
- **URL**: `QDRANT_URL` env (default: `http://localhost:6333`)
- **API Key**: `QDRANT_API_KEY` env (optional)
- **Retry logic**:
  - Voyage embeddings: 5 attempts, exponential backoff 1s→10s
  - Qdrant ops: 3 attempts, exponential backoff 1s→5s
- **Batch**: Processes all messages in single API call (no chunking)

### Google Drive OAuth
- **Credentials**: `~/.config/polylogue/polylogue-credentials.json` (or `POLYLOGUE_CREDENTIAL_PATH`)
- **Token**: `~/.config/polylogue/polylogue-token.json` (plain JSON, not encrypted)
- **Refresh**: Automatic via google-auth library
- **Headless**: ❌ Requires interactive browser flow for initial auth
- **Retry**: Configurable via `ENV_DRIVE_RETRIES` (default: 3), exponential backoff

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
```

**SQLiteBackend** (storage/backends/sqlite.py):
- **Schema**: v4 with migrations support
- **Performance**: Indexed lookups on provider_name, provider_conversation_id, source_name
- **Deduplication**: Content hash checks before insert
- **Transactions**: Atomic operations via connection context manager

**Extension Points**:
- Add PostgreSQL backend: Implement StorageBackend protocol in `storage/backends/postgres.py`
- Add DuckDB backend: Implement StorageBackend protocol in `storage/backends/duckdb.py`
- Pass backend to StorageRepository via constructor: `repo = StorageRepository(backend=PostgresBackend())`
- Backend is optional: `StorageRepository(backend=None)` uses direct SQLite for backward compatibility

## Search Provider Abstraction

**Protocol** (protocols.py):
```python
class SearchProvider(Protocol):
    """Abstract interface for search providers."""
    def index(self, messages: list[MessageRecord]) -> None: ...

    def search(self, query: str, limit: int = 100) -> list[SearchResult]: ...

    def delete(self, message_ids: list[str]) -> None: ...
```

**FTS5Provider** (storage/search_providers/fts5.py):
- **Type**: Full-text search (BM25 ranking)
- **Storage**: Virtual table in SQLite
- **Incremental**: Deletes + re-inserts affected conversations
- **Query escaping**: Hardened against injection (asterisk-only, operator position checks)
- **Performance**: Fast, local, no external API

**QdrantProvider** (storage/search_providers/qdrant.py):
- **Type**: Vector semantic search
- **Embeddings**: Voyage AI (1024-dim, cosine distance)
- **Storage**: Remote Qdrant instance or local Docker
- **Indexing**: Batch embeddings via Voyage API, bulk insert to Qdrant
- **Retry logic**: Exponential backoff (Voyage 5×, Qdrant 3×)

**Configuration** (IndexConfig):
```python
@dataclass
class IndexConfig:
    """Configuration for search indexing."""
    enabled: bool = True
    provider: Literal["fts5", "qdrant"] = "fts5"  # Or "hybrid" for both
    qdrant_url: str | None = None  # Override: env QDRANT_URL
    qdrant_api_key: str | None = None  # Override: env QDRANT_API_KEY
    voyage_api_key: str | None = None  # Override: env VOYAGE_API_KEY
```

**Extension Points**:
- Add hybrid search: Create HybridProvider combining FTS5 + Qdrant
- Add Milvus: Implement SearchProvider in `storage/search_providers/milvus.py`
- Add Elasticsearch: Implement SearchProvider in `storage/search_providers/elasticsearch.py`

## Service Layer

**IngestionService** (pipeline/services/ingestion.py):
- **Responsibility**: Orchestrate conversation ingestion from sources
- **Inputs**: Config, Source objects, archive_root
- **Outputs**: IngestResult with counts and changed_ids
- **Thread-safe**: Uses StorageRepository for coordinated writes
- **Methods**:
  - `ingest_source(source: Source) → IngestResult`: Ingest from single source
  - `ingest_all() → IngestResult`: Ingest all configured sources
  - `ingest_drive() → IngestResult`: Ingest from Google Drive (with OAuth flow)

**IndexService** (pipeline/services/indexing.py):
- **Responsibility**: Manage search index updates
- **Inputs**: Changed conversation IDs, SearchProvider
- **Workflow**: Load messages → request embeddings (if vector) → batch index
- **Methods**:
  - `index_conversations(conversation_ids: list[str]) → IndexResult`
  - `index_incremental(changed_ids: set[str]) → IndexResult`
  - `rebuild_index() → IndexResult`

**RenderService** (pipeline/services/rendering.py):
- **Responsibility**: Generate output formats (Markdown, HTML, JSON)
- **Inputs**: Conversations with semantic projections applied
- **Rendering**: Jinja2 templates, parallel execution
- **Methods**:
  - `render_markdown(conversation, options) → str`
  - `render_html(conversation, options) → str`
  - `render_json(conversations) → str`

**Pipeline Orchestration** (pipeline/runner.py):
```python
# Typical flow
ingest_service = IngestionService(repo, archive_root, config)
index_service = IndexService(repo, config.index)
render_service = RenderService(config)

# Run pipeline
ingest_result = ingest_service.ingest_all()
index_result = index_service.index_incremental(ingest_result.changed_ids)
render_result = render_service.render_all(conversations)
```

## Configuration Objects

**Config** (config.py):
Top-level configuration object (YAML-based):
```python
@dataclass
class Config:
    archive_root: Path
    index: IndexConfig
    drive: DriveConfig | None = None
    sources: list[Source] = field(default_factory=list)
```

**IndexConfig**:
```python
@dataclass
class IndexConfig:
    enabled: bool = True
    provider: Literal["fts5", "qdrant"] = "fts5"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    voyage_api_key: str | None = None
```

**DriveConfig**:
```python
@dataclass
class DriveConfig:
    enabled: bool = False
    credential_path: Path | None = None
    retries: int = 3
    backoff_factor: float = 1.0
```

**Source**:
```python
@dataclass
class Source:
    name: str
    path: str
    provider: str  # "chatgpt", "claude", "codex", "gemini"
    enabled: bool = True
```

**Env var precedence** (highest to lowest):
1. Command-line flags
2. Environment variables (`POLYLOGUE_*`)
3. Config file YAML
4. Built-in defaults

## Error Handling

**Exception Hierarchy**:
- `ConfigError` - Configuration errors
- `DatabaseError` - Database operations
- `DriveError` - Google Drive operations
  - `DriveAuthError` - OAuth failures
  - `DriveNotFoundError` - Missing resources
- `QdrantError` - Vector index errors
- `UIError` - UI rendering errors

**Pipeline Strategy** (runner.py):
- **Ingest**: Fail-fast (abort on error, raise exception)
- **Render**: Graceful degradation (log, continue, report failures)
- **Index**: Log, set `indexed=False`, continue

## Testing

**Coverage**: **85-90%** estimated
- 43 test files
- 17,371 lines of test code
- 9,022 lines of source code
- Ratio: 1.93:1 (exceptionally high)

**Key suites**:
- `test_importers_*.py` - Provider parsing (828 lines chatgpt, 826 claude)
- `test_projections.py` - Fluent API (714 lines)
- `test_store.py` - Storage layer (785 lines)
- `test_db.py` - Database ops (637 lines)
- `test_hashing.py` - Content hash verification (253 lines)
- `test_properties.py` - Hypothesis property-based (349 lines)
- `test_pipeline_concurrent.py` - Thread safety (232 lines)

**Fixtures** (tests/conftest.py):
- `workspace_env` - Isolated temp directories (config/state/archive)
- `test_conn` - Test database connection
- `DbFactory` - Test conversation creation helper

**Property-based testing** (Hypothesis):
```python
@given(text())
def test_hash_stability(input_text):
    assert hash_text(input_text) == hash_text(input_text)
```

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

## Build Commands

```bash
# Tests
uv run pytest -q
uv run pytest -v tests/test_lib.py
uv run pytest --cov=polylogue --cov-report=html

# Quality
uv run mypy polylogue/
uv run ruff check polylogue/ tests/
uv run ruff format --check polylogue/ tests/

# Run
uv run polylogue --help
POLYLOGUE_FORCE_PLAIN=1 uv run polylogue run --preview
uv run polylogue verify --verbose
uv run polylogue view --provider claude-code --since 2024-01-01
```

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

## Recent Changes (Last 5 Commits)

1. **767e272**: Relax rich version constraint
2. **e807710**: Structured content_blocks for semantic decomposition
3. **0fe2801**: Add verify command, improve claude-code parsing
4. **b7a452d**: Provider-aware classification for tool/thinking detection
5. **71aa06d**: Add semantic projection API for conversation analysis
