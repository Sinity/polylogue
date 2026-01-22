# Polylogue Developer Guide

Polylogue is a local-first AI chat archive that ingests exports from ChatGPT, Claude AI, Claude Code, Codex, and Google Drive into SQLite with full-text search and optional vector search.

## Core Architecture

**Mission**: Privacy-preserving, searchable archive of AI chat history.

**Key Invariants**:
- **Content-hash deduplication**: SHA-256 with NFC Unicode normalization prevents duplicates
- **Idempotent ingestion**: Safe to re-run imports; unchanged conversations skipped
- **Thread-safe**: Thread-local connections + `_WRITE_LOCK` for writes
- **Structured content**: Content blocks decompose messages into text/thinking/tool_use segments
- **Semantic projections**: Fluent API for filtering (substantive, dialogue-only, pairs, etc.)

## Data Flow

```
JSON/ZIP/Drive → Auto-detect provider → Parse + extract content_blocks →
Hash (NFC normalized) → Store (under _WRITE_LOCK) → Render (parallel) →
Index (FTS5 + optional Qdrant)
```

## Critical Files

### Core Abstractions
- **`lib/models.py`**: Message/Conversation with semantic classification (`is_thinking`, `is_tool_use`, `is_substantive`)
- **`lib/projections.py`**: Fluent projection API (`conv.project().substantive().min_words(50).execute()`)
- **`lib/repository.py`**: Primary query interface (`ConversationRepository`)
- **`core/hashing.py`**: NFC-normalized SHA-256 hashing

### Pipeline
- **`pipeline/runner.py`**: Orchestrates ingest (parallel) → render (parallel) → index (chunked)
- **`pipeline/ingest.py`**: Prepares conversations, checks content hashes
- **`pipeline/ids.py`**: Content hash generation

### Storage
- **`db.py`**: Thread-local connections, schema v4, migrations
- **`store.py`**: Record definitions, `_WRITE_LOCK`, attachment ref counting

### Importers (with content_blocks)
- **`importers/chatgpt.py`**: Graph traversal, extracts thinking from `content_type`
- **`importers/claude.py`**: JSONL chat_messages (Claude AI) + structured blocks (Claude Code)
- **`importers/drive.py`**: Gemini chunkedPrompt parsing
- **`importers/codex.py`**: Session exports

### Search & Rendering
- **`search.py`**: FTS5 with hardened query escaping
- **`index.py`**: FTS updates (chunked), Qdrant integration
- **`render.py`**: Jinja2 Markdown/HTML rendering
- **`verify.py`**: Data quality checks (orphaned refs, integrity violations)

## Thread Safety Model

**Architecture**:
```python
ThreadPoolExecutor(max_workers=4)
    ├─ prepare_ingest() - Pure (hashing, parsing) → runs in parallel
    │  Bounded submission: max 16 in-flight futures
    └─ ingest_bundle() - Writes DB → serialized under _WRITE_LOCK

Rendering: Parallel (max_workers=4)
```

**Locks**:
- `_WRITE_LOCK` (store.py:21): Protects writes + commits atomically
- `_counts_lock` (runner.py): Protects shared counter updates

**Rules**:
- ✅ Parallelize pure ops (hashing, parsing, rendering)
- ✅ Use `_WRITE_LOCK` for all DB writes
- ✅ Thread-local connections (`db.py:12`)
- ✅ Bound in-flight futures (prevents memory explosion)
- ❌ DON'T mutate shared state without locks
- ❌ DON'T hold `_WRITE_LOCK` during I/O

**Verified**: store.py:362 `with connection_context(conn) as db_conn, _WRITE_LOCK:`

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

### Adding an Importer

1. Create `polylogue/importers/newprovider.py`
2. Implement `looks_like(payload) → bool` (format detection)
3. Implement `parse(payload, fallback_id) → ParsedConversation`
4. **Extract content_blocks** from structured message content
5. Extract attachments via `attachment_from_meta()`
6. Register in `source_ingest.py:detect_provider()`

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

1. **Direct sqlite3 ops without context managers** → Use `with open_connection(path):`
2. **Skip content_blocks extraction** → Extract structured blocks at import
3. **Hardcode paths** → Use `config.archive_root`
4. **Manual commits in pipeline** → Let `connection_context()` handle it
5. **Modify hash logic carelessly** → Breaks idempotency, maintain compatibility
6. **Bypass semantic projections** → Use `message.is_tool_use` not string matching
7. **Mutate shared state in parallel ops** → Use locks or keep operations pure
8. **Skip _WRITE_LOCK for DB writes** → All writes must be inside lock

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

**"Database locked"**: Multiple write attempts → Ensure `_WRITE_LOCK` held (store.py:362)

**"Attachments ref_count=0 but referenced"**: `_prune_attachment_refs()` not called → Check store.py:380

**"Content hash mismatch"**: Normalization changed → Verify NFC normalization (core/hashing.py:23)

**"FTS syntax error"**: Unescaped query → Use `escape_fts5_query()` (tests: test_search.py)

**"Thinking not detected"**: Missing content_blocks → Verify importer extracts blocks (models.py:is_thinking)

**"Config not reflected"**: Env vars override config → Check `POLYLOGUE_*` vars, use `polylogue config show --json`

## Recent Changes (Last 5 Commits)

1. **767e272**: Relax rich version constraint
2. **e807710**: Structured content_blocks for semantic decomposition
3. **0fe2801**: Add verify command, improve claude-code parsing
4. **b7a452d**: Provider-aware classification for tool/thinking detection
5. **71aa06d**: Add semantic projection API for conversation analysis
