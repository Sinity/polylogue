# Polylogue Architecture

**Version**: Post-refactoring (2026-02-16)
**Status**: Production-ready layered architecture (Priorities 1-4 complete)

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

Polylogue is a **local-first AI chat archive** that ingests exports from multiple providers (ChatGPT, Claude AI, Claude Code, Codex, Google Drive/Gemini) into SQLite with full-text search (FTS5) and sqlite-vec vector search.

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

- `SearchProvider`: Full-text search (FTS5, Hybrid)
- `VectorProvider`: Semantic search (sqlite-vec + Voyage AI embeddings)

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
│   │   ├── parsing.py      # ParsingService (provider detection + parsing)
│   │   ├── indexing.py     # IndexService (FTS5)
│   │   ├── rendering.py    # RenderService (parallel output)
│   │   └── acquisition.py  # Source acquisition
│   ├── runner.py            # Pipeline orchestrator
│   ├── prepare.py           # Ingest preparation (hashing, dedup)
│   ├── enrichment.py        # Post-ingest enrichment
│   ├── ids.py               # Content hash generation
│   ├── events.py            # Watch sync event definitions + handlers
│   └── watch.py             # File watching for continuous sync
│
├── storage/                  # Storage Layer
│   ├── backends/            # Backend implementations
│   │   ├── __init__.py     # Factory (create_backend)
│   │   ├── async_sqlite.py # SQLiteBackend (async-first, aiosqlite)
│   │   ├── connection.py   # Sync utilities, connection pool, open_connection
│   │   └── schema.py       # DDL, migrations
│   ├── search_providers/    # Search implementations
│   │   ├── fts5.py         # FTS5Provider
│   │   ├── sqlite_vec.py   # sqlite-vec vector search
│   │   └── hybrid.py       # HybridProvider (FTS5 + vector)
│   ├── store.py             # Record types, _WRITE_LOCK
│   ├── repository.py        # ConversationRepository (async-first write coordination)
│   ├── index.py             # FTS5 indexing (sync)
│   ├── async_index.py       # FTS5 indexing (async)
│   └── search.py            # Search utilities, FTS5 escaping
│
├── schemas/                  # Schema Layer
│   ├── unified.py           # Unified schema with glom transforms
│   ├── code_detection.py    # Language/code block detection
│   ├── registry.py          # Schema version registry
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
├── services.py               # Invocation-scoped runtime service container
├── facade.py                 # Polylogue — async-first library API
├── paths.py                  # XDG path resolution
├── protocols.py              # SearchProvider protocol
├── types.py                  # NewType definitions (ConversationId, MessageId, etc.)
├── errors.py                 # PolylogueError hierarchy
├── export.py                 # Export utilities
├── health.py                 # System health checks
└── version.py                # Version detection (git / importlib.metadata)
```

---

## Layer Architecture

Each layer has a defined responsibility and communicates only with adjacent layers.

### CLI Layer (`cli/`)

Entry point for all user interaction. Uses Click with a custom `QueryFirstGroup` that treats positional arguments as implicit search queries. Commands are thin wrappers that translate CLI flags into domain operations.

- **click_app.py**: Custom Click group that routes bare `polylogue "search terms"` to query mode
- **query.py**: Translates filter flags (`--provider`, `--since`, `--has`) into `ConversationFilter` chains
- **commands/**: Each subcommand (`run`, `check`, `dashboard`, `mcp`, `auth`, `reset`) is a self-contained module

### Pipeline Layer (`pipeline/`)

Orchestrates the full ingestion lifecycle. The `runner.py` module coordinates the pipeline services in sequence (`acquire -> validate -> parse -> render -> index`) and reports stage progress via callbacks; watch-mode notifications flow through `events.py` and `watch.py`.

- **runner.py**: `run_sources()` is the top-level async entry point
- **services/**: Stateless service classes (acquisition, validation, parsing, rendering, indexing) injected with repositories
- **events.py** / **watch.py**: Watch-mode sync events and downstream notification handlers

### Ingestion Layer (`sources/`)

Source discovery and provider-specific parsing. Each provider parser converts raw wire format into `ParsedConversation`/`ParsedMessage` intermediates.

- **source.py**: `detect_provider()` probes file content via `looks_like()` functions — no filename heuristics
- **parsers/**: One module per provider, all producing the same `ParsedConversation` type
- **drive.py** / **drive_client.py**: Google Drive OAuth + API for Gemini conversations

### Storage Layer (`storage/`)

Database operations, search, and write coordination. The `SQLiteBackend` is async-first (aiosqlite) with sync connection helpers for thread-pool operations.

- **repository.py**: `ConversationRepository` — the single write coordination point
- **store.py**: Record types and row mappers, `_WRITE_LOCK` for write serialization
- **search_providers/**: Protocol implementations for FTS5, sqlite-vec, and hybrid search

### Rendering Layer (`rendering/`)

Generates output files from domain models. Renderers are stateless functions that accept a `Conversation` and produce files.

- **core.py**: `RenderService` orchestrates parallel rendering via `ThreadPoolExecutor`
- **renderers/**: Markdown and HTML renderers, each producing `render_root/<format>/<provider>/<id>.<ext>`

### Domain Layer (`lib/`)

Pure business logic with no I/O dependencies. Contains the core data models, filter chains, projections, and hashing.

- **models.py**: `Conversation`, `Message`, `Attachment` with semantic classification (`is_thinking`, `is_tool_use`, `is_substantive`)
- **filters.py**: `ConversationFilter` — chainable, lazy builder with async terminals (`.list()`, `.first()`, `.count()`)
- **projections.py**: `ConversationProjection` — message-level filtering within a single conversation

### Core Layer

Shared utilities with no domain knowledge: NFC-normalized SHA-256 hashing (`lib/hashing.py`), orjson helpers (`lib/json.py`), timestamp parsing (`lib/dates.py`), and structured logging (`lib/log.py`).

---

## Key Abstractions

### Protocols (`protocols.py`)

```python
@runtime_checkable
class SearchProvider(Protocol):
    def index(self, messages: list[MessageRecord]) -> None: ...
    def search(self, query: str) -> list[str]: ...

@runtime_checkable
class VectorProvider(Protocol):
    def upsert(self, conversation_id: str, messages: list[MessageRecord]) -> None: ...
    def query(self, text: str, limit: int = 10) -> list[tuple[str, float]]: ...
```

Implementations: `FTS5Provider`, `SqliteVecProvider`, `HybridProvider` (combines both).

### Domain Models (`lib/models.py`)

`Conversation` is the primary aggregate: it holds `messages: list[Message]`, `metadata: dict`, and computed properties like `display_title`, `total_cost_usd`, `word_count`, and `token_count`.

`Message` carries semantic classification via content blocks:
- `is_thinking` — reasoning traces (Claude, ChatGPT, Gemini)
- `is_tool_use` — tool invocations and results
- `is_substantive` — real dialogue (not noise, not thinking)
- `is_noise` — tool use, context dumps, or system prompts

### Filter Chain (`lib/filters.py`)

Lazy builder pattern — no database queries until a terminal method is called:

```python
# Building the filter is synchronous and cheap
chain = ConversationFilter(repo).provider("claude").since("2024-01-01").contains("error")

# Terminal triggers the actual query
results = await chain.list()       # Full conversations
count = await chain.count()        # SQL-optimized count
summaries = await chain.list_summaries()  # Lightweight (no messages)
```

### Projections (`lib/projections.py`)

Message-level filtering within a conversation:

```python
# Get only substantive messages with 50+ words
msgs = conv.project().substantive().min_words(50).execute()
```

---

## Data Flow

### Pipeline Stages

The import pipeline is orchestrated by `pipeline/runner.py`:

```
Source files → detect_provider() → Parse → Hash (NFC) → Store (under lock) → Render (parallel) → Index
```

#### 1. Acquisition

Source discovery scans configured paths for files matching supported formats. Local sources live under `inbox/` (default: `~/.local/share/polylogue/inbox/`). Drive sources are downloaded from Google Drive via OAuth. Each file is read with encoding fallback (UTF-8 → UTF-8-sig → UTF-16 → UTF-32).

#### 2. Parsing

`sources/source.py:detect_provider()` probes file content to determine the provider. Provider-specific parsers in `sources/parsers/` convert raw JSON/JSONL into `ParsedConversation` intermediates:

- **ChatGPT**: UUID graph traversal of the `mapping` field
- **Claude AI**: JSONL with `chat_messages` arrays
- **Claude Code**: JSON arrays with `parentUuid`/`sessionId` markers, structured content blocks
- **Codex**: Session-based JSONL exports (envelope, direct, and legacy formats)
- **Gemini**: `chunkedPrompt.chunks` from Drive API payloads

#### 3. Storage

`pipeline/prepare.py` prepares ingest bundles with NFC-normalized content hashing (SHA-256) for idempotent storage. `storage/store.py` writes conversations, messages, and attachments into SQLite under `_WRITE_LOCK`. `storage/repository.py` coordinates writes via `ConversationRepository`. Conversations are deduplicated by content hash — re-importing unchanged data is a no-op.

#### 4. Rendering

`rendering/core.py` orchestrates parallel rendering via `ThreadPoolExecutor(max_workers=4)`. Markdown and HTML renderers write output into `render_root/<format>/<provider>/<conversation_id>.<ext>`.

#### 5. Indexing

`storage/index.py` builds the SQLite FTS5 index for full-text search. `storage/search_providers/sqlite_vec.py` handles optional vector indexing via sqlite-vec with Voyage AI embeddings.

### Content Hashing

```python
# lib/hashing.py — NFC normalization prevents duplicates from equivalent Unicode
normalized = unicodedata.normalize("NFC", text)
hashlib.sha256(normalized.encode("utf-8")).hexdigest()
```

**Conversation hash** (`pipeline/ids.py`): deterministic JSON serialization with sorted keys and compact separators. Hash includes: title, timestamps, messages (id/role/text/timestamp), attachments (id/mime_type). Hash excludes: `provider_meta`, `metadata` (user-editable fields).

---

## Extension Points

### Search Providers

Search uses protocol-based extensibility. Any class implementing `SearchProvider` can be used:

```python
class SearchProvider(Protocol):
    def index(self, messages: list[MessageRecord]) -> None: ...
    def search(self, query: str) -> list[str]: ...
```

Current implementations:
- **FTS5Provider**: SQLite FTS5 full-text search with smartcase and query escaping
- **SqliteVecProvider**: sqlite-vec vector search with Voyage AI embeddings
- **HybridProvider**: Combines FTS5 and vector results with score normalization

### Rendering

Renderers are stateless functions. Adding a new output format means implementing a renderer module in `rendering/renderers/` and registering it. The CLI `--format` flag maps directly to renderer selection.

### Provider Parsers

Adding a new provider requires:
1. A Pydantic model in `sources/providers/` for the raw wire format
2. A parser in `sources/parsers/` producing `ParsedConversation`
3. A `looks_like()` function in the parser for auto-detection

---

## Dependencies

| Dependency | Purpose |
|------------|---------|
| **aiosqlite** | Async SQLite access (wraps sqlite3) |
| **Click** | CLI framework with custom group for query-first design |
| **orjson** | Fast JSON serialization/deserialization |
| **Rich** | Terminal output formatting (tables, panels, progress) |
| **Textual** | TUI framework for the interactive dashboard |
| **Pydantic** | Validation models for provider wire formats |
| **dateparser** | Natural language date parsing (`"last week"`, `"2 days ago"`) |
| **structlog** | Structured logging |
| **sqlite-vec** | Vector similarity search (self-contained, no external services) |
| **httpx** | HTTP client for Google Drive API |
| **mcp** | Model Context Protocol server SDK |

All dependencies are pure Python or self-contained native extensions. No external services required for core functionality (vector search + Drive are opt-in).

---

## Thread Safety Model

Polylogue uses **bounded parallelism** with explicit locking to balance throughput and safety:

```
✅ Parallel: prepare_ingest() — pure hashing/parsing, no shared state
✅ Parallel: rendering — ThreadPoolExecutor(max_workers=4)
✅ Bounded: ingestion futures — max 16 in-flight, FIRST_COMPLETED wait
✅ Pattern: with _WRITE_LOCK: save() — atomic writes

❌ NEVER: Direct sqlite3 ops — always use ConversationRepository
❌ NEVER: Hold _WRITE_LOCK during I/O — only during commit
❌ NEVER: Mutate shared state without locks
```

### Connection Management

- **Async path**: `SQLiteBackend` uses aiosqlite — one connection per backend instance, serialized by asyncio's event loop
- **Sync path**: `connection_context()` provides thread-local connections via `threading.local()`, each thread gets its own SQLite connection
- **Write serialization**: All writes go through `_WRITE_LOCK` in `store.py`, ensuring atomic commits

### Parallel Boundaries

| Operation | Parallelism | Safety |
|-----------|-------------|--------|
| Content hashing | Fully parallel | Pure functions, no shared state |
| Provider parsing | Fully parallel | Stateless parsers |
| Database writes | Serialized | `_WRITE_LOCK` |
| File rendering | Thread pool (4) | Independent output files |
| FTS indexing | Serialized | Single writer |
| In-flight futures | Bounded (16) | `FIRST_COMPLETED` wait prevents memory exhaustion |

---

## Testing Architecture

### Test Organization

Tests live under `tests/` with clear separation:

```
tests/
├── unit/          # Fast, isolated tests (mocked dependencies)
├── integration/   # Full-stack tests (real SQLite, real pipeline)
├── conftest.py    # Shared fixtures
└── fixtures/      # Static test data files
```

### Key Fixtures

| Fixture | Scope | Purpose |
|---------|-------|---------|
| `seeded_db` | session | Pre-populated database for integration tests |
| `synthetic_source` | function | Temporary source directory with generated files |
| `raw_synthetic_samples` | session | Raw conversation data for unit tests |
| `repository` | function | Fresh `ConversationRepository` with in-memory backend |

All fixtures use the same `SyntheticCorpus` infrastructure as `polylogue demo --seed/--corpus`.
Note: `polylogue demo --showcase` currently seeds from packaged static fixtures.

### Test Markers

| Marker | Purpose |
|--------|---------|
| `@pytest.mark.slow` | Long-running tests (Drive integration, large corpus) |
| `@pytest.mark.integration` | Full-stack tests requiring database |

### Coverage

The test suite enforces ≥90% coverage. As of this writing, 4200+ tests with 1 skip (SQLiteBackend partial-ID limitation).

---

**See also:** [Internals Reference](internals.md) · [Library API](library-api.md) · [Data Model](data-model.md)
