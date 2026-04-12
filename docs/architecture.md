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
- `polylogue/lib/`
- `polylogue/operations/archive.py`

### 2. Derived Read Models

Stored products computed over the archive:

- session profiles
- work events, phases, threads
- day and week summaries
- provider-level analytics and tag rollups

Primary modules:

- `polylogue/products/`
- `polylogue/storage/session_product_*.py`
- `polylogue/storage/repository_product_*.py`

### 3. Surfaces

These expose the archive and its products:

- CLI: `polylogue/cli/`
- Python API: `polylogue/facade.py`
- MCP server: `polylogue/mcp/`
- site generation: `polylogue/site/`
- dashboard and TUI: `polylogue/ui/`
- renderers: `polylogue/rendering/`

Leaf adapters over archive operations and derived products.

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
  → session products           # session_product_*.py — profiles, work events, phases, threads
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
| `ConversationRepository` | `storage/repository.py` | Mixin-composed async repository (10 mixins for reads, writes, products, vectors, raw). |
| `SearchProvider` protocol | `protocols.py` | FTS5 and Hybrid (RRF fusion) implementations. |
| `ConversationFilter` | `lib/filters.py` | Fluent filter chain used by CLI, MCP, and facade. |
| `Session Products` | `storage/session_product_*.py` | Materialized read models: profiles, work events, phases, threads, aggregates. |
| `ContentHash` | `pipeline/ids.py` | SHA-256 over NFC-normalized conversation payload. Title, timestamps, messages, attachments are hashed. User metadata (tags, summaries) is excluded — editable metadata doesn't trigger re-import. |
| `Provider` enum | `types.py` | 6 known providers + UNKNOWN. All provider identity flows through this enum. |

## Database

- Single SQLite file, WAL mode.
- Schema is fresh-only: no migration chain. On version mismatch the database is
  wiped and rebuilt. `SCHEMA_VERSION` lives in `storage/backends/schema_ddl.py`.
- FTS5 with `unicode61` tokenizer (no porter stemmer in this SQLite build).

## Placement Rules

- If it changes archive meaning, it belongs in `lib/`, `storage/`, `sources/`,
  `pipeline/`, or `products/`.
- If it only presents existing archive meaning, it belongs in `cli/`, `mcp/`,
  `site/`, `rendering/`, or `ui/`.
- If it exists to prove, refresh, benchmark, or audit the repo, it belongs in
  `devtools/`, `showcase/`, or `tests/`.
- If a surface needs a new concept, define the concept in the substrate or
  product layer first.
