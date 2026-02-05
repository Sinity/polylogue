# Polylogue Architecture

**Version**: Post-refactoring (2026-01-23)
**Status**: ✅ Production-ready layered architecture (Priorities 1-4 complete)

This document describes Polylogue's architecture after the comprehensive refactoring completed in January 2026. The codebase follows a **layered architecture** with clear separation of concerns, **protocol-based abstraction**, and **dependency injection** throughout.

---

## Table of Contents

1. [Overview](#overview)
2. [Design Principles](#design-principles)
3. [Package Structure](#package-structure)
4. [Layer Architecture](#layer-architecture)
5. [Key Abstractions](#key-abstractions)
6. [Data Flow](#data-flow)
7. [Extension Points](#extension-points)
8. [Dependencies](#dependencies)
9. [Thread Safety Model](#thread-safety-model)
10. [Testing Architecture](#testing-architecture)

---

## Overview

Polylogue is a **local-first AI chat archive** that ingests exports from multiple providers (ChatGPT, Claude AI, Claude Code, Codex, Google Drive/Gemini) into SQLite with full-text search (FTS5) and optional vector search (Qdrant).

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         CLI Layer                                │
│  (commands/, click_app.py, query.py)                             │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Pipeline Layer                              │
│  (runner.py, services/{ingestion,indexing,rendering})           │
└──┬───────────────────┬───────────────────┬───────────────────────┘
   │                   │                   │
   ▼                   ▼                   ▼
┌─────────────┐  ┌──────────────┐  ┌────────────────┐
│  Ingestion  │  │   Storage    │  │   Rendering    │
│   Layer     │  │    Layer     │  │     Layer      │
└─────────────┘  └──────────────┘  └────────────────┘
   │                   │                   │
   │                   ▼                   │
   │            ┌──────────────┐          │
   │            │ Domain Layer │          │
   │            │(lib/models,  │          │
   │            │ projections) │          │
   │            └──────────────┘          │
   │                                      │
   └──────────────────┬───────────────────┘
                      │
                      ▼
              ┌──────────────┐
              │  Core Layer  │
              │ (hashing,    │
              │  json, log)  │
              └──────────────┘
```

### Layer Responsibilities

1. **CLI Layer**: Command parsing, user interaction, output formatting
2. **Pipeline Layer**: Orchestration of ingestion → indexing → rendering
3. **Ingestion Layer**: Source discovery, parsing, content extraction
4. **Storage Layer**: Database operations, search providers, repositories
5. **Rendering Layer**: Markdown/HTML generation from domain models
6. **Domain Layer**: Business logic, semantic projections, query API
7. **Core Layer**: Utilities (hashing, JSON, timestamps, logging)

---

## Design Principles

### 1. Protocol-Based Abstraction

All external dependencies are defined as **runtime-checkable protocols** (`polylogue/protocols.py`):

- `StorageBackend`: Database operations (SQLite, future: PostgreSQL, DuckDB)
- `SearchProvider`: Full-text search (FTS5, future: Elasticsearch)
- `VectorProvider`: Semantic search (Qdrant, future: pgvector)
- `Renderer`: Output generation (Jinja2 templates, future: custom renderers)

This enables:

- **Testing**: Mock implementations via protocols
- **Flexibility**: Swap backends without code changes
- **Type safety**: Runtime interface validation

### 2. Dependency Injection

Services receive dependencies via constructor injection:

```python
class IngestionService:
    def __init__(
        self,
        repository: StorageRepository,
        archive_root: Path,
        config: Config,
    ):
        self.repository = repository
        self.archive_root = archive_root
        self.config = config
```

Benefits:

- **Testability**: Easy to inject test doubles
- **Explicitness**: Dependencies visible in signatures
- **Lifetime management**: Clear ownership of resources

### 3. Separation of Concerns

**Vertical slicing**: Each package owns a specific concern (storage, ingestion, rendering)

**Horizontal layering**: Higher layers depend on lower layers, not vice versa

```
CLI → Pipeline → Services → Domain → Core
 ↓       ↓          ↓
Storage  Ingestion  Rendering
```

### 4. Idempotency

All operations are designed to be safely re-runnable:

- **Content hashing**: SHA-256 with NFC Unicode normalization prevents duplicates
- **Upsert semantics**: Database operations update existing records
- **Incremental indexing**: Only changed conversations re-indexed

### 5. Thread Safety

**Bounded parallelism** with explicit locking:

- Thread-local database connections
- `StorageRepository._write_lock` serializes writes
- Parallel pure operations (hashing, parsing, rendering)
- Bounded in-flight futures prevent memory exhaustion

---

## Package Structure

```
polylogue/
├── cli/                      # CLI Layer
│   ├── commands/             # Command implementations
│   │   ├── run.py           # Main pipeline command (ingest → render → index)
│   │   ├── check.py         # Integrity checks, repair, vacuum
│   │   ├── dashboard.py     # TUI dashboard
│   │   ├── mcp.py           # MCP server command
│   │   ├── auth.py          # OAuth flow for Google Drive
│   │   ├── reset.py         # Database reset
│   │   └── completions.py   # Shell completions
│   ├── click_app.py         # QueryFirstGroup (positional args → query mode)
│   ├── query.py             # Filter chain, output formatting, modifiers
│   ├── helpers.py           # CLI utilities
│   └── editor.py            # Shell command validation, browser/editor opening
│
├── sources/                  # Source Detection & Parsing
│   ├── parsers/             # Provider-specific parsers
│   │   ├── base.py          # ParsedConversation types, normalize_role
│   │   ├── chatgpt.py       # ChatGPT UUID graph traversal
│   │   ├── claude.py        # Claude AI (JSONL) + Claude Code (JSON)
│   │   ├── codex.py         # Codex sessions
│   │   └── drive.py         # Gemini chunkedPrompt
│   ├── providers/           # Pydantic models for raw provider JSON
│   │   ├── chatgpt.py       # ChatGPT data models
│   │   ├── claude_ai.py     # Claude AI data models
│   │   ├── claude_code.py   # Claude Code data models
│   │   ├── codex.py         # Codex data models
│   │   └── gemini.py        # Gemini data models
│   ├── source.py            # detect_provider(), local file ingestion
│   ├── drive.py             # Google Drive ingestion
│   └── drive_client.py      # Drive OAuth + API client
│
├── pipeline/                 # Pipeline Layer
│   ├── services/            # Service implementations
│   │   ├── ingestion.py    # IngestionService (parallel, bounded)
│   │   ├── indexing.py     # IndexService (FTS5/Qdrant)
│   │   ├── rendering.py    # RenderService (parallel output)
│   │   └── acquisition.py  # Source acquisition
│   ├── runner.py            # Pipeline orchestrator
│   ├── ingest.py            # Ingest preparation
│   ├── ids.py               # Content hash generation
│   └── models.py            # Pipeline result types
│
├── storage/                  # Storage Layer
│   ├── backends/            # Backend implementations
│   │   ├── sqlite.py       # SQLiteBackend (only backend, schema v5)
│   │   └── async_sqlite.py # AsyncSQLiteBackend
│   ├── search_providers/    # Search implementations
│   │   ├── fts5.py         # FTS5Provider
│   │   └── qdrant.py       # QdrantProvider (vector search)
│   ├── store.py             # Record types, _WRITE_LOCK
│   ├── repository.py        # ConversationRepository (write coordination)
│   ├── async_repository.py  # Async facade
│   ├── index.py             # FTS5 indexing
│   └── search.py            # Search utilities, FTS5 escaping
│
├── schemas/                  # Schema Layer
│   ├── unified.py           # Unified schema with glom transforms
│   ├── common.py            # Shared schema types
│   ├── claude_code_records.py # Claude Code record schemas
│   ├── schema_inference.py  # JSON schema inference from samples
│   └── validator.py         # Schema validation utilities
│
├── lib/                      # Domain Layer
│   ├── models.py            # Conversation, Message, Attachment
│   ├── messages.py          # MessageCollection, MessageSource protocol
│   ├── projections.py       # ConversationProjection (fluent API)
│   ├── filters.py           # ConversationFilter (chainable queries)
│   ├── hashing.py           # NFC-normalized SHA-256
│   ├── json.py              # orjson utilities
│   ├── dates.py             # Timestamp parsing
│   ├── roles.py             # Role enum/literal
│   └── log.py               # structlog setup
│
├── rendering/                # Rendering Layer
│   ├── renderers/           # Output renderers
│   │   ├── markdown.py     # Markdown renderer
│   │   └── html.py         # HTML renderer
│   ├── core.py              # Rendering orchestration
│   └── render_paths.py      # Output path resolution
│
├── ui/                       # UI Components
│   ├── facade.py            # ConsoleFacade, ConsoleLike protocol, PlainConsole
│   ├── tui/                 # Textual TUI
│   │   ├── app.py          # Textual application
│   │   ├── screens/        # TUI screens (browser, search, dashboard)
│   │   └── widgets/        # Custom widgets
│   └── __init__.py          # UI class (high-level abstraction)
│
├── mcp/                      # MCP Server
│   └── server.py            # Model Context Protocol implementation
│
├── config.py                 # Configuration (Config, IndexConfig, DriveConfig)
├── services.py               # Singleton factories: get_backend(), get_repository()
├── facade.py                 # Polylogue — top-level library API
├── async_facade.py           # AsyncPolylogue — async library API
├── paths.py                  # XDG path resolution
├── protocols.py              # SearchProvider, VectorProvider protocols
├── types.py                  # NewType definitions (ConversationId, MessageId, etc.)
└── templates/                # Jinja2 Templates
    ├── conversation.md.j2   # Markdown template
    └── conversation.html.j2 # HTML template
```

---

> **Note**: The detailed layer descriptions, protocol references, and extension points below are from a previous architecture revision. Some module paths and abstractions (e.g., `StorageBackend` protocol, `ingestion/`, `importers/`, `core/`, `container.py`) have been superseded by the current structure shown in the Package Structure section above. For the authoritative current reference, see [docs/CLAUDE.md](./CLAUDE.md).

## Layer Architecture

### 1. CLI Layer (`cli/`)

**Responsibility**: User interface, command parsing, output formatting

**Key Components**:

- `click_app.py`: Click application setup, command registration
- `commands/*.py`: Individual command implementations
- `formatting.py`: Rich/plain text formatting

**Dependencies**: Pipeline, Storage, Lib

**Example**:

```python
# cli/commands/run.py
from polylogue.container import get_container
from polylogue.pipeline.runner import run_pipeline

@click.command()
def run(source_names, preview, stage):
    container = get_container()
    config = container.config()
    repository = container.storage()
    result = run_pipeline(
        config=config,
        repository=repository,
        source_names=source_names,
        preview_only=preview,
        stage=stage,
    )
    display_result(result)
```

### 2. Pipeline Layer (`pipeline/`)

**Responsibility**: Orchestration of ingestion → indexing → rendering workflow

**Key Components**:

- `runner.py`: Top-level orchestrator (plan → ingest → index → render)
- `services/ingestion.py`: IngestionService (parallel ingest with bounded submission)
- `services/indexing.py`: IndexService (FTS5 + Qdrant coordination)
- `services/rendering.py`: RenderService (parallel Markdown/HTML generation)

**Dependencies**: Storage, Ingestion, Rendering, Domain

**Service Pattern**:

```python
class IngestionService:
    def __init__(
        self,
        repository: StorageRepository,
        archive_root: Path,
        config: Config,
    ):
        self.repository = repository
        self.archive_root = archive_root
        self.config = config

    def ingest_sources(
        self,
        sources: list[Source],
        ui: object | None = None,
    ) -> IngestResult:
        # Parallel hashing (pure) → Serial writes (under lock)
        ...
```

### 3. Storage Layer (`storage/`)

**Responsibility**: Persistence, search, indexing

**Key Components**:

#### Database Operations

- `db.py`: Thread-local connections, schema v4, migrations
- `store.py`: Record types (ConversationRecord, MessageRecord, AttachmentRecord), low-level CRUD
- `repository.py`: StorageRepository (high-level API, write lock)

#### Backends

- `backends/sqlite.py`: SQLiteBackend (implements StorageBackend protocol)
- Future: `backends/postgresql.py`, `backends/duckdb.py`

#### Search Providers

- `search_providers/fts5.py`: FTS5Provider (implements SearchProvider protocol)
- `search_providers/qdrant.py`: QdrantProvider (implements VectorProvider protocol)

**Protocol Example**:

```python
@runtime_checkable
class StorageBackend(Protocol):
    def get_conversation(self, id: str) -> ConversationRecord | None: ...
    def save_conversation(self, record: ConversationRecord) -> None: ...
    def get_messages(self, conversation_id: str) -> list[MessageRecord]: ...
    def save_messages(self, records: list[MessageRecord]) -> None: ...
    # ... attachment operations, transactions
```

**Repository Pattern**:

```python
class StorageRepository:
    def __init__(self):
        self._write_lock = threading.Lock()

    def save_conversation(
        self,
        conversation: ConversationRecord,
        messages: list[MessageRecord],
        attachments: list[AttachmentRecord],
    ) -> bool:
        with self._write_lock:
            # Atomic save of conversation + messages + attachments
            ...
```

### 4. Ingestion Layer (`ingestion/`)

**Responsibility**: Source discovery, parsing, content extraction

**Key Components**:

- `source.py`: Local file ingestion (JSON/JSONL/ZIP), auto-detection
- `drive.py`: Google Drive ingestion (folder traversal, export detection)
- `drive_client.py`: OAuth flow, Drive API client

**Provider Detection** (`source.py`):

```python
def detect_provider(payload: Any, fallback_id: str) -> ParsedConversation:
    """Auto-detect provider format and parse."""
    if chatgpt.looks_like(payload):
        return chatgpt.parse(payload, fallback_id)
    if claude.looks_like_code(payload):
        return claude.parse_code(payload, fallback_id)
    # ... codex, claude-ai, drive
    raise ValueError("Unknown format")
```

### 5. Domain Layer (`lib/`)

**Responsibility**: Business logic, semantic projections, query API

**Key Components**:

- `models.py`: Conversation, Message (with semantic properties)
- `projections.py`: ConversationProjection (fluent filtering API)
- `repository.py`: ConversationRepository (high-level query interface)

**Semantic Properties** (`models.py`):

```python
class Message:
    @property
    def is_thinking(self) -> bool:
        """Claude extended thinking detection."""
        if self.provider_meta:
            blocks = self.provider_meta.get("content_blocks", [])
            return any(b.get("type") == "thinking" for b in blocks)
        return False

    @property
    def is_substantive(self) -> bool:
        """Meaningful content (excludes tools, noise)."""
        return (
            not self.is_tool_use
            and not self.is_tool_result
            and self.word_count >= 3
        )
```

**Fluent Projection API** (`projections.py`):

```python
# Lazy-evaluated, composable filters
conversation.project()
    .substantive()          # No noise/tools
    .min_words(50)          # Meaningful length
    .since(datetime(...))   # After date
    .limit(10)
    .execute()              # Materialize

# Lazy iteration (memory efficient)
for msg in conversation.project().contains("error").iter():
    process(msg)
```

### 6. Rendering Layer (transitional)

**Responsibility**: Markdown/HTML generation from domain models

**Location**: `rendering/renderers/` package

**Templates**: Jinja2 templates in `templates/`

**Example**:

```python
from polylogue.rendering.renderers import HTMLRenderer

def render_conversation(conversation_id: str, output_path: Path, archive_root: Path) -> Path:
    renderer = HTMLRenderer(archive_root)
    return renderer.render(conversation_id, output_path)
```

### 7. Core Layer (`core/`)

**Responsibility**: Shared utilities (no business logic)

**Key Components**:

- `hashing.py`: NFC-normalized SHA-256 for content deduplication
- `json.py`: JSON serialization utilities
- `timestamps.py`: ISO 8601 parsing
- `log.py`: Structured logging setup

**Critical Utility** (`hashing.py`):

```python
import unicodedata
import hashlib

def hash_text(text: str) -> str:
    """Content hash with Unicode normalization.

    NFC normalization ensures "café" = "café" regardless of
    composition (U+00E9 vs U+0065 U+0301).
    """
    normalized = unicodedata.normalize("NFC", text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
```

---

## Key Abstractions

### 1. StorageBackend Protocol

**Purpose**: Decouple application from database implementation

**Interface**:

```python
@runtime_checkable
class StorageBackend(Protocol):
    # Read operations
    def get_conversation(self, id: str) -> ConversationRecord | None: ...
    def list_conversations(self, source: str | None = None) -> list[ConversationRecord]: ...
    def get_messages(self, conversation_id: str) -> list[MessageRecord]: ...
    def get_attachments(self, conversation_id: str) -> list[AttachmentRecord]: ...

    # Write operations
    def save_conversation(self, record: ConversationRecord) -> None: ...
    def save_messages(self, records: list[MessageRecord]) -> None: ...
    def save_attachments(self, records: list[AttachmentRecord]) -> None: ...

    # Transaction control
    def begin(self) -> None: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...
```

**Implementations**:

- `storage.backends.sqlite.SQLiteBackend` (default)
- Future: `storage.backends.postgresql.PostgreSQLBackend`
- Future: `storage.backends.duckdb.DuckDBBackend`

**Factory**:

```python
from polylogue.storage.backends import create_backend

# Use default SQLite
backend = create_backend()

# Use specific path
backend = create_backend(db_path=Path("/custom/path.db"))

# Use configuration
backend = create_backend(config=config)
```

### 2. SearchProvider / VectorProvider Protocols

**SearchProvider** (full-text search):

```python
@runtime_checkable
class SearchProvider(Protocol):
    def index(self, messages: list[MessageRecord]) -> None:
        """Index messages for full-text search."""
        ...

    def search(self, query: str) -> list[str]:
        """Execute search, return message IDs."""
        ...
```

**VectorProvider** (semantic search):

```python
@runtime_checkable
class VectorProvider(Protocol):
    def upsert(self, conversation_id: str, messages: list[MessageRecord]) -> None:
        """Upsert message embeddings (idempotent)."""
        ...

    def query(self, text: str, limit: int = 10) -> list[tuple[str, float]]:
        """Find semantically similar messages."""
        ...
```

**Implementations**:

- `storage.search_providers.fts5.FTS5Provider` (default)
- `storage.search_providers.qdrant.QdrantProvider` (optional, requires Voyage AI)

**Factory**:

```python
from polylogue.storage.search_providers import create_search_provider, create_vector_provider

# FTS5 (always available)
search = create_search_provider()

# Qdrant (if configured)
vector = create_vector_provider(config)  # Returns None if not configured
```

### 3. OutputRenderer Protocol (✅ implemented Priority 3.3)

**Purpose**: Decouple rendering from template engine

**Interface**:

```python
@runtime_checkable
class OutputRenderer(Protocol):
    def render(self, conversation_id: str, output_path: Path) -> Path:
        """Render conversation to output format."""
        ...

    def supports_format(self) -> str:
        """Return the format this renderer supports."""
        ...
```

**Implementations**:

- `rendering.renderers.markdown.MarkdownRenderer` (stdlib only, .md output)
- `rendering.renderers.html.HTMLRenderer` (Jinja2-based, .html + .md output)

**Factory**: `create_renderer(format, config)` in rendering/renderers/**init**.py

### 4. Service Pattern

**Pattern**: Focused services with constructor DI

**Example** (IngestionService):

```python
class IngestionService:
    """Service for ingesting conversations from sources."""

    def __init__(
        self,
        repository: StorageRepository,
        archive_root: Path,
        config: Config,
    ):
        self.repository = repository
        self.archive_root = archive_root
        self.config = config

    def ingest_sources(
        self,
        sources: list[Source],
        ui: object | None = None,
    ) -> IngestResult:
        """Ingest conversations from configured sources.

        Returns:
            IngestResult with counts and changed conversation IDs.
        """
        # Implementation
        ...
```

**Benefits**:

- **Testability**: Easy to inject mocks
- **Explicitness**: Dependencies visible in signature
- **Composability**: Services can depend on other services

---

## Data Flow

### Ingestion Flow

```
Source Discovery → Parse → Hash → Store → Index → Render
     │              │       │       │       │       │
     ▼              ▼       ▼       ▼       ▼       ▼
Local files    Provider  NFC    SQLite   FTS5    Jinja2
Google Drive   parsers   SHA-256 (upsert) Qdrant  MD/HTML
```

**Detailed Steps**:

1. **Source Discovery** (`ingestion/source.py`, `ingestion/drive.py`)
   - Scan local files (JSON/JSONL/ZIP)
   - Query Google Drive folders
   - Auto-detect provider format

2. **Parse** (`importers/*.py`)
   - ChatGPT: Graph traversal of `mapping` structure
   - Claude Code: Extract `content_blocks` from structured array
   - Claude AI: JSONL `chat_messages` parsing
   - Codex: Session export parsing
   - Gemini: `chunkedPrompt.chunks` extraction

3. **Hash** (`pipeline/ids.py`, `core/hashing.py`)
   - NFC Unicode normalization
   - SHA-256 of conversation content
   - Excludes provider_meta (semantic content only)
   - Idempotency check: skip if hash unchanged

4. **Store** (`storage/repository.py`)
   - Atomic transaction (conversation + messages + attachments)
   - Under `_write_lock` for thread safety
   - Upsert semantics (update if exists)

5. **Index** (`pipeline/services/indexing.py`)
   - FTS5: Incremental updates (chunked batches of 200)
   - Qdrant: Batch embeddings via Voyage AI
   - Retry logic: 5 attempts for embeddings, 3 for Qdrant

6. **Render** (`pipeline/services/rendering.py`)
   - Parallel Markdown/HTML generation (ThreadPoolExecutor)
   - Jinja2 templates
   - Output to `render_root/<provider>/<conversation_id>/`

### Query Flow

```
User Query → Search → Retrieve → Project → Render
     │          │         │          │         │
     ▼          ▼         ▼          ▼         ▼
FTS5 query  Message   Conversation  Filters  Format
Qdrant      IDs       objects       (fluent) (text/JSON)
```

**Detailed Steps**:

1. **Search** (`storage/search_providers/fts5.py`, `cli/commands/search.py`)
   - FTS5 full-text query (hardened escaping)
   - Returns message IDs

2. **Retrieve** (`lib/repository.py`)
   - ConversationRepository.get_by_ids()
   - Hydrate Conversation objects with messages

3. **Project** (`lib/projections.py`)
   - Fluent API: `.substantive().min_words(50).since(...)`
   - Lazy evaluation (iterator or materialized list)

4. **Render** (`cli/formatting.py`)
   - Text output (default)
   - JSON (`--json`)
   - JSON Lines (`--json-lines`)

### Configuration Flow

```
Env Vars → Config File → Config Objects → Services
    │           │              │              │
    ▼           ▼              ▼              ▼
Fallback    JSONC/JSON   Config,         IngestionService,
values      parsing      IndexConfig,    IndexService,
                         DriveConfig     RenderService
```

**Priority**: Explicit args > Config file > Env vars > Defaults

**Config Objects** (`config.py`):

```python
@dataclass
class IndexConfig:
    fts_enabled: bool = True
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    voyage_api_key: str | None = None

@dataclass
class DriveConfig:
    credentials_path: Path
    token_path: Path
    retry_count: int = 3
    timeout: int = 30

@dataclass
class Config:
    archive_root: Path
    render_root: Path
    db_path: Path
    sources: list[Source]
    index_config: IndexConfig
    drive_config: DriveConfig
```

---

## Extension Points

### 1. Adding a Storage Backend

**Example**: PostgreSQL backend

**Steps**:

1. Create `polylogue/storage/backends/postgresql.py`
2. Implement `StorageBackend` protocol:

```python
from polylogue.protocols import StorageBackend
from polylogue.storage.store import ConversationRecord, MessageRecord, AttachmentRecord

class PostgreSQLBackend:
    """PostgreSQL storage backend."""

    def __init__(self, connection_string: str):
        self.conn_string = connection_string
        self._conn = None

    def get_conversation(self, id: str) -> ConversationRecord | None:
        # SELECT FROM conversations WHERE conversation_id = ?
        ...

    def save_conversation(self, record: ConversationRecord) -> None:
        # INSERT ... ON CONFLICT UPDATE
        ...

    # ... implement remaining protocol methods
```

1. Update factory (`storage/backends/__init__.py`):

```python
def create_backend(config: Config | None = None) -> StorageBackend:
    if config and config.backend_type == "postgresql":
        return PostgreSQLBackend(config.db_connection_string)
    return SQLiteBackend(db_path=config.db_path)
```

1. Runtime verification:

```python
from polylogue.protocols import StorageBackend

backend = create_backend(config)
assert isinstance(backend, StorageBackend)  # Protocol check
```

### 2. Adding a Search Provider

**Example**: Elasticsearch provider

**Steps**:

1. Create `polylogue/storage/search_providers/elasticsearch.py`
2. Implement `SearchProvider` protocol:

```python
from polylogue.protocols import SearchProvider
from polylogue.storage.store import MessageRecord

class ElasticsearchProvider:
    """Elasticsearch full-text search provider."""

    def __init__(self, url: str, index_name: str = "polylogue_messages"):
        self.client = Elasticsearch(url)
        self.index = index_name

    def index(self, messages: list[MessageRecord]) -> None:
        bulk_ops = [
            {"index": {"_id": msg.message_id}},
            {"text": msg.text, "conversation_id": msg.conversation_id}
            for msg in messages
        ]
        self.client.bulk(index=self.index, operations=bulk_ops)

    def search(self, query: str) -> list[str]:
        result = self.client.search(
            index=self.index,
            query={"match": {"text": query}},
        )
        return [hit["_id"] for hit in result["hits"]["hits"]]
```

1. Update factory (`storage/search_providers/__init__.py`):

```python
def create_search_provider(config: Config | None = None) -> SearchProvider:
    if config and config.search_type == "elasticsearch":
        return ElasticsearchProvider(config.elasticsearch_url)
    return FTS5Provider(db_path=config.db_path)
```

### 3. Adding a Renderer

**Example**: Custom Markdown renderer

**Steps**:

1. Create `polylogue/rendering/renderers/custom.py`
2. Implement `Renderer` protocol:

```python
from pathlib import Path
from polylogue.protocols import Renderer
from polylogue.lib.models import Conversation

class CustomRenderer:
    """Custom Markdown renderer with special formatting."""

    def render_markdown(self, conversation: Conversation, output_path: Path) -> Path:
        output_file = output_path / "conversation.md"

        with output_file.open("w") as f:
            f.write(f"# {conversation.title}\n\n")
            for msg in conversation.messages:
                f.write(f"**{msg.role}**: {msg.text}\n\n")

        return output_file

    def render_html(self, conversation: Conversation, output_path: Path) -> Path:
        # Convert markdown to HTML
        ...
```

1. Use in RenderService:

```python
from polylogue.rendering.renderers.custom import CustomRenderer

renderer = CustomRenderer()
renderer.render_markdown(conversation, output_path)
```

### 4. Adding an Importer

**Example**: New provider format

**Steps**:

1. Create `polylogue/importers/newprovider.py`
2. Implement detection + parsing:

```python
from polylogue.importers.base import ParsedConversation, ParsedMessage

def looks_like(payload: Any) -> bool:
    """Detect new provider format."""
    return isinstance(payload, dict) and "new_provider_key" in payload

def parse(payload: dict, fallback_id: str) -> ParsedConversation:
    """Parse new provider export."""
    messages = []
    for item in payload["messages"]:
        content_blocks = []
        # Extract structured content_blocks
        if item.get("thinking"):
            content_blocks.append({
                "type": "thinking",
                "thinking": item["thinking"],
            })
        # ... extract text, tool_use, etc.

        messages.append(ParsedMessage(
            message_id=item["id"],
            role=item["role"],
            text=item["text"],
            timestamp=item["timestamp"],
            provider_meta={"content_blocks": content_blocks} if content_blocks else None,
        ))

    return ParsedConversation(
        conversation_id=payload.get("id", fallback_id),
        title=payload["title"],
        messages=messages,
        attachments=[],
        provider_name="newprovider",
    )
```

1. Register in `ingestion/source.py`:

```python
import polylogue.importers.newprovider as newprovider

def detect_provider(payload: Any, fallback_id: str) -> ParsedConversation:
    if newprovider.looks_like(payload):
        return newprovider.parse(payload, fallback_id)
    # ... existing checks
```

---

## Dependencies

### Layer Dependencies

**Dependency Graph** (higher layers depend on lower layers):

```
┌───────────────┐
│   CLI Layer   │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│Pipeline Layer │
└───┬───────┬───┘
    │       │
    ▼       ▼
┌─────────┐ ┌─────────┐ ┌──────────┐
│Ingestion│ │ Storage │ │Rendering │
└────┬────┘ └────┬────┘ └────┬─────┘
     │           │            │
     └───────────┼────────────┘
                 │
                 ▼
          ┌──────────┐
          │  Domain  │
          └────┬─────┘
               │
               ▼
          ┌──────────┐
          │   Core   │
          └──────────┘
```

### Package Dependencies

**CLI** (`cli/`):

- Depends on: Pipeline, Storage, Lib, Core
- No dependents (top-level)

**Pipeline** (`pipeline/`):

- Depends on: Storage, Ingestion, Lib, Core
- Depended on by: CLI

**Storage** (`storage/`):

- Depends on: Lib (models), Core
- Depended on by: Pipeline, CLI

**Ingestion** (`ingestion/`):

- Depends on: Importers, Core
- Depended on by: Pipeline

**Lib** (`lib/`):

- Depends on: Core, Storage (records only)
- Depended on by: Storage, Pipeline, CLI

**Core** (`core/`):

- Depends on: Standard library only
- Depended on by: All layers

### External Dependencies

**Required**:

- `click`: CLI framework
- `rich`: Terminal formatting
- `jinja2`: Template engine
- `sqlite3`: Database (stdlib)

**Optional**:

- `google-auth`, `google-auth-oauthlib`, `google-api-python-client`: Google Drive
- `qdrant-client`: Vector search
- `voyageai`: Embeddings for Qdrant
- `fastapi`, `uvicorn`: Web server (experimental)

---

## Thread Safety Model

### Architecture

```
ThreadPoolExecutor (max_workers=4)
    │
    ├─ prepare_ingest() - Pure operations (parallel)
    │  ├─ Hash computation (NFC normalization)
    │  ├─ Content parsing
    │  └─ Attachment extraction
    │
    └─ ingest_bundle() - Database writes (serialized)
       └─ under StorageRepository._write_lock
```

### Locks

1. **StorageRepository._write_lock** (storage/repository.py)
   - Protects: All database writes + commits
   - Scope: Per-repository instance
   - Pattern: `with self._write_lock: ...`

2. **IngestResult._lock** (pipeline/services/ingestion.py)
   - Protects: Shared counter updates
   - Scope: Per-ingestion operation

### Rules

✅ **DO**:

- Parallelize pure operations (hashing, parsing, rendering)
- Use `_write_lock` for all database writes
- Use thread-local connections (`db.get_connection()`)
- Bound in-flight futures (max 16 to prevent memory explosion)

❌ **DON'T**:

- Mutate shared state without locks
- Hold `_write_lock` during I/O operations
- Share database connections across threads

### Example

**Correct** (parallel hashing, serial writes):

```python
def ingest_sources(self, sources: list[Source]) -> IngestResult:
    result = IngestResult()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}

        for conversation in iter_conversations(sources):
            # Pure operation - run in parallel
            future = executor.submit(prepare_ingest, conversation)
            futures[future] = conversation.conversation_id

            # Bound submission (max 16 in-flight)
            if len(futures) > 16:
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    bundle = fut.result()
                    # Serial write under lock
                    self.repository.save_conversation(bundle)
                    del futures[fut]

    return result
```

**Incorrect** (unbounded submission, no lock):

```python
# ❌ Memory explosion + race conditions
futures = [executor.submit(process, conv) for conv in conversations]
for fut in as_completed(futures):
    bundle = fut.result()
    save_to_db(bundle)  # No lock!
```

---

## Testing Architecture

### Coverage

**Metrics**:

- Test lines: 17,371
- Source lines: 9,022
- Ratio: 1.93:1 (exceptionally high)
- Coverage: 85-90% estimated

### Test Organization

```
tests/
├── conftest.py                   # Shared fixtures
├── test_lib.py                   # Domain model tests
├── test_projections.py           # Projection API (714 lines)
├── test_db.py                    # Database ops (637 lines)
├── test_store.py                 # Storage layer (785 lines)
├── test_hashing.py               # Content hash verification
├── test_properties.py            # Hypothesis property-based (349 lines)
├── test_pipeline_concurrent.py   # Thread safety (232 lines)
├── test_importers_chatgpt.py     # ChatGPT parser (828 lines)
├── test_importers_claude.py      # Claude parser (826 lines)
└── ...

43 test files
```

### Key Fixtures

**From `tests/conftest.py`**:

```python
@pytest.fixture
def workspace_env(tmp_path):
    """Isolated temp directories for config/state/archive."""
    config_dir = tmp_path / "config"
    state_dir = tmp_path / "state"
    archive_dir = tmp_path / "archive"

    for d in [config_dir, state_dir, archive_dir]:
        d.mkdir()

    os.environ["XDG_CONFIG_HOME"] = str(config_dir)
    os.environ["XDG_STATE_HOME"] = str(state_dir)
    os.environ["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_dir)

    yield WorkspaceEnv(config=config_dir, state=state_dir, archive=archive_dir)

@pytest.fixture
def test_conn(workspace_env):
    """Test database connection with schema."""
    from polylogue.storage.db import connection_context, init_db

    init_db()
    with connection_context(None) as conn:
        yield conn

@pytest.fixture
def db_factory(test_conn):
    """Factory for creating test conversations."""
    from tests.helpers import DbFactory

    return DbFactory(test_conn)
```

### Testing Patterns

**Unit Tests** (pure functions):

```python
def test_hash_stability():
    text = "café"
    assert hash_text(text) == hash_text(text)

def test_nfc_normalization():
    # U+00E9 (single char) vs U+0065 U+0301 (e + combining acute)
    composed = "café"  # U+00E9
    decomposed = "cafe\u0301"  # e + combining
    assert hash_text(composed) == hash_text(decomposed)
```

**Integration Tests** (with database):

```python
def test_save_and_retrieve_conversation(db_factory):
    conversation = db_factory.create_conversation(
        title="Test",
        message_count=5,
    )

    retrieved = ConversationRepository().get_by_id(conversation.conversation_id)
    assert retrieved.title == "Test"
    assert len(retrieved.messages) == 5
```

**Property-Based Tests** (Hypothesis):

```python
from hypothesis import given
from hypothesis.strategies import text

@given(text())
def test_hash_determinism(input_text):
    """Hash must be deterministic."""
    assert hash_text(input_text) == hash_text(input_text)

@given(text(min_size=1))
def test_hash_collision_resistance(input_text):
    """Different inputs should produce different hashes."""
    modified = input_text + " "
    assert hash_text(input_text) != hash_text(modified)
```

**Concurrency Tests**:

```python
def test_parallel_ingestion_thread_safety():
    """Verify no race conditions during parallel ingest."""
    conversations = [create_test_conversation() for _ in range(100)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(ingest, conv) for conv in conversations]
        results = [fut.result() for fut in futures]

    # Verify no duplicates, no missing records
    stored = list_all_conversations()
    assert len(stored) == 100
    assert len(set(c.conversation_id for c in stored)) == 100
```

---

## Summary

Polylogue's architecture achieves:

- **Clear separation of concerns**: 9 packages with distinct responsibilities (cli, sources, pipeline, storage, schemas, lib, rendering, ui, mcp)
- **Protocol-based search**: SearchProvider and VectorProvider protocols for FTS5 and Qdrant backends
- **Direct storage**: SQLiteBackend as the single storage implementation (no protocol abstraction overhead)
- **Singleton services**: `polylogue.services` module provides factory functions for backend + repository
- **Thread safety**: Bounded parallelism, `_WRITE_LOCK` in store.py, thread-local connections
- **Idempotency**: NFC-normalized content hashing enables safe re-runs
- **Type safety**: mypy strict mode, 0 errors across 100 source files

For the authoritative current reference, see:

- [docs/CLAUDE.md](./CLAUDE.md) — Developer guide with critical patterns, invariants, and API reference
- [Architecture Roadmap](./architecture-roadmap.md) — Future enhancements
- [Performance Guide](./performance.md) — Benchmarking results
- Historical documents archived in [docs/archive/](./archive/)
