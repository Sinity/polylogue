# Polylogue Architecture

**Version**: Post-refactoring (2026-01-23)
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
│   ├── event_bus.py         # Pipeline event system
│   ├── events.py            # Event type definitions
│   └── watch.py             # File watching for continuous sync
│
├── storage/                  # Storage Layer
│   ├── backends/            # Backend implementations
│   │   ├── sqlite.py       # Sync utilities, row mappers, connection helpers
│   │   └── async_sqlite.py # SQLiteBackend (async-first, aiosqlite)
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
├── services.py               # Singleton factories: get_backend(), get_repository()
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

*For current architecture details, see [CLAUDE.md](../CLAUDE.md).*
