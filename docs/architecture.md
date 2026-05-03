# Polylogue Architecture

Polylogue is a local archive for AI conversations. The system has four rings:

1. archive substrate
2. derived read models
3. user and machine surfaces
4. verification and maintenance

## Rings

### 1. Archive Substrate

Owns stored meaning:

- source acquisition and provider detection
- provider parsing and normalization
- SQLite persistence and search indexes
- archive-level query and runtime operations

Primary modules:

- `polylogue/sources/`
- `polylogue/pipeline/`
- `polylogue/storage/`
- `polylogue/archive/`
- `polylogue/operations/archive.py`

### 2. Derived Read Models

Stored insights computed over the archive:

- session profiles
- work events, phases, threads
- day and week summaries
- provider-level analytics and tag rollups

Primary modules:

- `polylogue/insights/`
- `polylogue/storage/insights/session/`
- `polylogue/storage/repository/insight/`

### 3. Surfaces

These expose the archive and its insights:

- CLI: `polylogue/cli/`
- Python API: `polylogue/api/__init__.py`
- MCP server: `polylogue/mcp/`
- site generation: `polylogue/site/`
- dashboard and TUI: `polylogue/ui/`
- renderers: `polylogue/rendering/`

Leaf adapters over archive operations and derived insights.

### 4. Verification and Maintenance

- schema inference and verification
- synthetic corpus generation
- showcase and deterministic acceptance exercises
- validation lanes, mutation campaigns, benchmark campaigns

Primary modules:

- `polylogue/schemas/`
- `polylogue/showcase/`
- `devtools/`
- `tests/`

## Data Flow

```
source files (JSON/JSONL/ZIP)
  → detect_provider()          # dispatch.py — shape-based, not filename
  → provider parser            # parsers/{chatgpt,claude,codex,drive}.py
  → content hash (NFC)         # pipeline/ids.py — SHA-256 over normalized payload
  → store (upsert-if-changed)  # storage/ — idempotent by content hash
  → session insights           # storage/insights/session/ — profiles, work events, phases, threads
  → FTS index                  # search_providers/fts5.py — unicode61 tokenizer

           CLI / MCP / Python API
                   ↑
             filter chain → query → storage
```

The `all` pipeline stage runs: acquire → parse → materialize → render → site → index.
`reprocess` runs: parse → materialize → render → index (skips acquire).

## Provider Detection

| Provider | Detected by | Parser |
|----------|-------------|--------|
| ChatGPT | `mapping` dict with message graph | `parsers/chatgpt.py` |
| Claude web | `chat_messages` list | `parsers/claude.py` |
| Claude Code | `parentUuid`/`sessionId` in record array | `parsers/claude.py` (code path) |
| Codex | Session envelope structure | `parsers/codex.py` |
| Gemini | `chunkedPrompt.chunks` structure | `parsers/drive.py` |

`detect_provider()` calls each parser's `looks_like()` in order.

## Key Abstractions

| Abstraction | Location | Role |
|-------------|----------|------|
| `Polylogue` | `facade.py` | Async entry point. Wraps storage + search + pipeline. |
| `ConversationRepository` | `storage/repository/__init__.py` | Mixin-composed async repository (9 mixins: archive reads/writes, action reads, insight readers for profile/timeline/thread/summary, raw, vectors). |
| `SearchProvider` protocol | `protocols.py` | FTS5 and Hybrid (RRF fusion) implementations. |
| `ConversationFilter` | `archive/filter/filters.py` | Fluent filter chain used by CLI, MCP, and facade. |
| `Session Insights` | `storage/insights/session/` | Materialized read models: profiles, work events, phases, threads, aggregates. |
| `ContentHash` | `pipeline/ids.py` | SHA-256 over NFC-normalized conversation payload. Title, timestamps, messages, attachments are hashed. User metadata (tags, summaries) is excluded — editable metadata doesn't trigger re-import. |
| `Provider` enum | `types.py` | 6 known providers + UNKNOWN. All provider identity flows through this enum. |

## Database

- Single SQLite file, WAL mode.
- Schema is fresh-first: version mismatches are rejected unless an explicit,
  reviewed in-place upgrade exists for that exact transition. `SCHEMA_VERSION`
  lives in `storage/sqlite/schema_ddl.py`.
- FTS5 with `unicode61` tokenizer (no porter stemmer in this SQLite build).

## Placement Rules

### Substrate (archive meaning)
- `lib/` — domain types, invariants, shared primitives (no I/O, no storage)
- `storage/` — SQLite backends, repositories, FTS, search providers
- `sources/` — provider detection, parsing, acquisition
- `pipeline/` — stage execution, ingestion, validation, rendering pipeline
- `insights/` — derived read models, session insights, analytics
- `operations/` — operation specs, artifact graph, declared runtime contracts

### Surfaces (presentation only)
- `cli/` — Click commands, shared helpers, output formatting
- `mcp/` — MCP server tools
- `api/` — async library API
- `site/` — static site generation
- `rendering/` — markdown/HTML renderers
- `ui/` — TUI, dashboard

### Verification (repo health)
- `proof/` — proof obligations, subject discovery, claim catalog, witnesses
- `devtools/` — operator tooling, lints, campaigns, rendering
- `showcase/` — QA exercises, deterministic acceptance tests
- `tests/` — pytest suite, property tests, integration tests

### Cross-cutting
- `schemas/` — provider schemas, schema inference, validation
- `scenarios/` — synthetic corpus, scenario families

### Key rules
- Surfaces may not import substrate internals directly (see layering.yaml).
- New semantics go into substrate or insights first, then surfaces adapt.
- Proof subjects and claims live in `proof/`; devtools commands that exercise
  them live in `devtools/`.
