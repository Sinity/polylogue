## Architecture

### Component Layout

```
polylogue/
├── lib/               # Core domain (models, filters, projections, hashing, dates, env)
│   ├── roles.py       # normalize_role() — never raises; returns canonical role string
│   ├── timestamps.py  # parse_timestamp() — never raises; returns None on failure
│   ├── hashing.py     # Deterministic content hashing (NFC normalization)
│   ├── json.py        # orjson wrapper (Decimal/datetime support)
│   ├── query_plan.py  # Query plan building
│   ├── query_plan_description.py  # Human-readable query descriptions
│   ├── query_plan_execution.py    # Query execution engine
│   └── query_spec.py  # Query specification types
├── storage/           # Storage layer
│   ├── backends/      # SQLite (sync + async), connection management, schema DDL
│   │   ├── async_sqlite.py   # Async backend with bulk_connection support
│   │   ├── connection.py      # Sync connection with performance pragmas
│   │   ├── schema_ddl.py      # Schema DDL fragments
│   │   └── schema_upgrade.py  # Schema version management
│   ├── search_providers/      # FTS5, hybrid (RRF fusion), sqlite-vec
│   ├── session_product_*.py   # Session-derived data (profiles, work events, phases, threads)
│   └── store.py               # Record types and JSON helpers
├── sources/           # Source detection and provider parsers
│   ├── source.py      # _decode_json_bytes() — multi-encoding probe, BOM stripping
│   ├── cursor.py      # Source file walking with ZIP mtime skip
│   ├── parsers/       # Per-provider parsers (chatgpt, claude, codex, drive)
│   └── providers/     # Pydantic models for provider JSON schemas
├── pipeline/          # Ingestion → rendering → indexing orchestration
│   └── services/      # Acquisition, parsing (batch), validation, indexing
├── schemas/           # Schema inference, pinning, verification
├── showcase/          # QA exercise catalog and runner
├── cli/               # CLI commands (14 total: run, check, qa, generate, etc.)
├── mcp/               # Model Context Protocol server and tools
├── operations/        # High-level archive operations
├── site/              # Static site generation
├── rendering/         # Output rendering (Markdown, HTML)
├── facade.py          # Polylogue — top-level library API
├── archive_products.py     # Pydantic models for session-derived products
├── archive_product_*.py    # Product builders, rollups, summaries
├── config.py          # Configuration
└── types.py           # NewType IDs (ConversationId, MessageId, AttachmentId)
```

### Key Abstractions

- **SQLiteBackend**: Single storage backend with async (aiosqlite) and sync paths
- **SearchProvider**: Protocol for search (FTS5, Hybrid with RRF)
- **Polylogue**: Main entry point wrapping storage + search + pipeline
- **ConversationFilter**: Fluent filter chain API
- **Session Products**: Derived materialized data (profiles, work events, phases, threads, aggregates)

### Data Flow

```
Provider Exports → detect_provider() → Parse → Hash (NFC) → Store → Session Products → Index (FTS5)
                                                                            ↓
                                                   CLI/MCP ← Query ← Filter Chain
```

### Database

- **Schema v4** (fresh-only, no migration chain)
- **32.5 GB** SQLite with WAL mode
- Performance pragmas: 512MB cache, NORMAL sync, 1GB mmap, deferred WAL checkpoint
- FTS5 with unicode61 tokenizer (porter stemmer NOT compiled)
- Covering indices for analytics queries
- **6,650 conversations**, **2.1M messages** across 5 providers

### Provider Detection

| Provider | Format | Auto-detected By |
|----------|--------|------------------|
| ChatGPT | `conversations.json` | `mapping` field with message graph |
| Claude (web) | `.jsonl` | `chat_messages` array |
| Claude Code | `.json` array | `parentUuid`/`sessionId` markers |
| Codex | `.jsonl` | Session envelope structure |
| Gemini | Google Drive API | `chunkedPrompt.chunks` structure |
