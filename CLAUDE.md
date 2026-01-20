# Polylogue Developer Guide

Polylogue turns AI chat exports into a local, searchable archive. It ingests exports from ChatGPT, Claude, Claude Code, Codex, and generic JSON/JSONL/ZIP sources into SQLite, renders Markdown/HTML views, and builds a full-text index (with optional vector search via Qdrant).

## Project Overview

**Core Mission**: Provide a local-first, privacy-preserving way to organize and search AI chat history.

**Key Characteristics**:
- Content-hash based deduplication: conversations and messages are deduplicated using SHA-256 hashes of normalized content
- Idempotent ingestion: safe to re-run imports without duplicating data
- Provider-agnostic: auto-detects format (ChatGPT, Claude AI, Claude Code, Codex, generic)
- Streaming parsing: uses `ijson` for memory-efficient large file handling
- Incremental indexing: FTS index updates only for changed conversations
- Semantic projections: domain models support rich filtering and analysis views

## Architecture Summary

### Core Layers

1. **CLI Layer** (`polylogue/cli/`)
   - `click_app.py`: Main entrypoint routing commands
   - `commands/`: Individual commands (run, search, config, verify, view, etc.)
   - `helpers.py`, `types.py`: Utilities and type definitions
   - `ui/facade.py`: Interactive/plain output abstraction

2. **Configuration & Paths** (`config.py`, `paths.py`)
   - XDG-compliant config/state/data directories
   - Source definitions (local paths or Drive folders)
   - Template path for custom HTML rendering

3. **Pipeline** (`polylogue/pipeline/`)
   - `runner.py`: Orchestrates ingest → render → index flow
   - `ingest.py`: Prepares conversations for storage
   - `ids.py`: Content-hash generation and ID logic
   - `models.py`: Pipeline data structures (PlanResult, RunResult)

4. **Data Ingestion** (`importers/`, `source_ingest.py`, `drive_ingest.py`)
   - `importers/base.py`: Common structures (ParsedConversation, ParsedMessage, ParsedAttachment)
   - Provider-specific parsers (chatgpt.py, claude.py, codex.py)
   - Local file detection and streaming
   - Drive OAuth and download logic

5. **Storage** (`db.py`, `store.py`)
   - SQLite database with schema versioning (SCHEMA_VERSION = 4)
   - Thread-local connections with context managers
   - Content-hash based upserts
   - Attachment ref counting for cleanup

6. **Domain Models** (`lib/models.py`, `lib/repository.py`)
   - `Message`, `Conversation`, `DialoguePair`, `Attachment` classes
   - Semantic classification: `is_user`, `is_assistant`, `is_tool_use`, `is_thinking`, `is_substantive`
   - Semantic projections: `substantive_only()`, `iter_pairs()`, `without_noise()`
   - `ConversationRepository`: primary interface for querying conversations

7. **Rendering & Search** (`render.py`, `search.py`, `index.py`)
   - Jinja2-based Markdown/HTML rendering per conversation
   - SQLite FTS5 full-text search
   - Optional Qdrant vector search (with Voyage AI embeddings)
   - Asset management (attachments stored by hash)

8. **Server** (`server/`)
   - FastAPI-based REST API for programmatic access
   - Web interface support (WIP)

### Data Flow

```
JSON/ZIP/Drive Sources
    ↓
[Auto-detect provider format] (detect_provider)
    ↓
[Parse to ParsedConversation] (provider importers)
    ↓
[Hash content, check for duplicates] (conversation_content_hash, prepare_ingest)
    ↓
[Store in SQLite] (ingest_bundle)
    ↓
[Render to Markdown/HTML] (render_conversation)
    ↓
[Build/update FTS index] (update_index_for_conversations)
    ↓
[Optional: Update Qdrant vector index]
```

## Key Patterns to Follow

### Adding a New Importer

1. Create `polylogue/importers/newprovider.py`
2. Implement `looks_like(payload)` → bool to detect format
3. Implement `parse(payload, fallback_id)` → ParsedConversation
4. Handle message extraction using `ParsedMessage` with role, text, timestamp
5. Extract attachments using `attachment_from_meta(meta, message_id, index)` helper
6. Register in `source_ingest.py:detect_provider()` and `_parse_json_payload()`

**Example pattern from claude.py**:
```python
def looks_like_code(payload: list) -> bool:
    # Check for provider-specific markers
    return any(key in item for key in ("parentUuid", "sessionId"))

def parse_code(payload: list, fallback_id: str) -> ParsedConversation:
    messages, attachments = extract_messages_from_chat_messages(payload)
    return ParsedConversation(
        provider_name="claude-code",
        provider_conversation_id=conv_id,
        title=title,
        messages=messages,
        attachments=attachments,
    )
```

### Adding a New CLI Command

1. Create `polylogue/cli/commands/yourcommand.py`
2. Define a Click command with `@click.command()`
3. Use `@click.pass_obj` to access `AppEnv` (contains UI, config_path)
4. Import and register in `click_app.py:cli.add_command()`
5. Handle errors by raising `ConfigError`, `DriveError`, or `DatabaseError` — catch in command and call `fail()`

**Example from run.py**:
```python
@click.command("run")
@click.option("--preview", is_flag=True)
@click.pass_obj
def run_command(env: AppEnv, preview: bool) -> None:
    try:
        config = load_effective_config(env)
    except ConfigError as exc:
        fail("run", str(exc))
```

### Adding Semantic Projections

1. Extend `Message` or `Conversation` in `lib/models.py`
2. Add a property that returns a boolean classification (e.g., `is_thinking`)
3. Implement filtering using `filter(lambda m: condition)`
4. Document with docstring and example

**Example (thinking detection)**:
```python
@property
def is_thinking(self) -> bool:
    """Message contains reasoning/thinking traces."""
    if self.provider_meta and self.provider_meta.get("isThought"):
        return True
    if not self.text:
        return False
    return (
        text.lower().startswith("here's a thinking process")
        or any(re.search(pattern, text) for pattern in _THINKING_MARKERS)
    )
```

### Testing Conventions

1. Use `pytest` with fixtures from `conftest.py`
2. Use `workspace_env` fixture for temporary config/state/archive directories
3. Use `DbFactory` from `tests/factories.py` to create test conversations
4. Test both parsing (unit) and pipeline (integration)
5. Import from `polylogue.*` directly, not relative paths

**Example**:
```python
def test_semantic_models():
    msg = Message(id="1", role="user", text="hello world")
    assert msg.is_user
    assert msg.is_substantive

def test_repository(mock_db):
    repo = ConversationRepository(mock_db)
    factory = DbFactory(mock_db)
    factory.create_conversation(
        id="c1", provider="test",
        messages=[{"id": "m1", "role": "user", "text": "test"}]
    )
    conv = repo.get("c1")
    assert conv is not None
```

### Error Handling Patterns

Use domain-specific exceptions and handle at command level:

```python
from polylogue.config import ConfigError
from polylogue.db import DatabaseError
from polylogue.drive_client import DriveError, DriveAuthError

try:
    # operation
except ConfigError as exc:
    fail("command", str(exc))
except DatabaseError as exc:
    fail("command", f"Database error: {exc}")
except DriveAuthError as exc:
    fail("command", f"Drive auth required: {exc}")
```

## Important Invariants

### Content-Hash Based Deduplication

- **Conversation hash**: `conversation_content_hash()` hashes title, timestamps, message list (id, role, text, timestamp), and attachments list
- **Message hash**: `message_content_hash()` hashes message id, role, text, timestamp
- **Storage**: Upserts based on (provider_name, provider_conversation_id) + content_hash comparison
- **Idempotency**: If hash matches existing conversation, skip; if different, update

**Key files**: `pipeline/ids.py`, `pipeline/ingest.py`

### Attachment Ref Counting

- **Attachment table**: Stores metadata with `ref_count`
- **Attachment_refs table**: One row per (attachment, conversation, message) tuple
- **Cleanup**: `_prune_attachment_refs()` removes orphaned refs when conversation is updated
- **Garbage collection**: Attachments with `ref_count == 0` can be deleted

**Key files**: `store.py:_prune_attachment_refs()`, `store.py:store_records()`

### Thread-Local Database Connections

- **Global state**: `_LOCAL = threading.local()` in `db.py`
- **Context manager**: `open_connection(path)` or `connection_context(conn)`
- **Reuse**: Same thread reuses connection if already open
- **Nesting**: Supports nested context managers with depth tracking
- **Transactions**: Auto-commits on normal exit, rolls back on exception

**Key file**: `db.py`

### FTS Sync Requirements

- **Rebuild**: `rebuild_index()` clears and rebuilds FTS from all messages
- **Incremental**: `update_index_for_conversations(ids)` updates only changed conversations
- **Chunking**: Large updates are chunked (size=200) to avoid query size limits
- **Qdrant**: If `QDRANT_URL` set, also updates vector index

**Key file**: `index.py`

## Don't Do List

### Anti-Patterns to Avoid

1. **Direct sqlite3 operations without context managers**
   - Use `with open_connection(path) as conn: conn.execute(...)`

2. **Creating importers that skip attachment handling**
   - Use `attachment_from_meta()` for all attachment fields

3. **Hardcoding paths instead of using config**
   - Use `config.archive_root` or environment overrides

4. **Committing transactions manually in pipeline**
   - Let `connection_context()` handle commit on exit

5. **Modifying content hash logic without careful consideration**
   - Always maintain backward compatibility for hashing

6. **Bypassing semantic projections for filtering**
   - Use `message.is_tool_use` property instead of string matching

7. **Ignoring provider_meta in message/attachment models**
   - Extract key fields, preserve raw in provider_meta

8. **Not handling encoding issues in JSON parsing**
   - Use `_decode_json_bytes()` with encoding fallbacks

9. **Adding new commands without plain/interactive mode support**
   - Check `env.ui.plain` and provide non-interactive paths

10. **Forgetting to update FTS index after ingest**
    - Call `update_index_for_conversations()` in pipeline

## Build/Test Commands

```bash
# Run tests
uv run pytest -q
uv run pytest -v tests/test_lib.py

# Type checking
uv run mypy polylogue/

# Linting
uv run ruff check polylogue/ tests/

# Format check
uv run ruff format --check polylogue/ tests/

# Run from source
uv run polylogue --help

# Force plain mode
POLYLOGUE_FORCE_PLAIN=1 uv run polylogue run --preview
```

## Key Files to Know

### Entry Points
- **CLI**: `polylogue/cli/__main__.py`
- **Python API**: `polylogue/lib/repository.py` for programmatic access
- **Server**: `polylogue/server/app.py` (FastAPI app)

### Core Abstractions
- **Message/Conversation models**: `polylogue/lib/models.py`
- **Repository**: `polylogue/lib/repository.py`
- **ParsedConversation**: `polylogue/importers/base.py`
- **Config**: `polylogue/config.py`

### Critical Modules
- **Database**: `polylogue/db.py` — Connection pooling, schema, migrations
- **Storage**: `polylogue/store.py` — Record definitions, upsert logic
- **Pipeline**: `polylogue/pipeline/runner.py` — Orchestrates full flow
- **Hashing**: `polylogue/core/hashing.py` — Unified hash utilities
