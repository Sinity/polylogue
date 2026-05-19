# Polylogue Architecture

For the target shape, guardrails, and architectural decision log, see
[Architecture Spine](architecture-spine.md). For current sequencing and active
workstreams, see [Execution Plan](execution-plan.md).

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
- daemon web reader: `polylogue/daemon/web_shell.py`
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

The daemon-owned ingest path acquires source payloads, parses provider records,
writes archive rows, and refreshes derived read models through explicit
convergence stages.

## Provider Detection

| Provider | Detected by | Parser |
|----------|-------------|--------|
| ChatGPT | `mapping` dict with message graph | `parsers/chatgpt.py` |
| Claude web | `chat_messages` list | `parsers/claude.py` |
| Claude Code | `parentUuid`/`sessionId` in record array | `parsers/claude.py` (code path) |
| Codex | Session envelope structure | `parsers/codex.py` |
| Gemini | `chunkedPrompt.chunks` structure | `parsers/drive.py` |
| Drive | Google Takeout format with OAuth | `parsers/drive.py` |
| Antigravity | Brain artifact metadata (`*.metadata.json`) or language-server Markdown export envelope | `parsers/antigravity.py` |

`detect_provider()` calls each parser's `looks_like()` in order.

## Dual Vocabulary Period: Provider and Source

The codebase is in a transition between two overlapping vocabularies
for conversation origins:

- **`Provider`** (`polylogue/types.py`): the legacy enum carried by
  every public surface — `provider_name` storage column, CLI
  `--provider` filter, MCP `provider` parameter, daemon facet labels.
  Mixes lab identity (OpenAI, Anthropic, Google), product/runtime
  identity (Claude Code, Codex), and source-family identity
  (claude-code-session vs claude-ai-export) into one token. See the
  vocabulary table in `polylogue/core/provider_identity.py` for the
  detailed conflation surface.
- **`Source`** (`polylogue/core/sources.py`): the source-centered
  replacement. A `Source` is an immutable dataclass with three fields
  — `family` (e.g. `claude-code-session`), `runtime_root` (e.g.
  `~/.claude/projects`), and `originating_lab` (e.g. `anthropic`).
  Every `Provider` has a canonical `Source` via
  `provider_to_source(Provider) -> Source`; `source_to_provider`
  performs the reverse lookup.

This PR introduces only the typed `Source` surface alongside the
existing `Provider` enum. **It does not rename any storage column,
CLI flag, MCP parameter, or public field**: those renames are
deliberately staged into later PRs so each one can land with a
focused review and migration plan. The two vocabularies coexist for
the duration of the transition.

Planned migration sequencing (each is a separate later PR under
#1022):

1. CLI/MCP/API surfaces gain `source` parameter aliases that accept
   source-family tokens; the existing `provider` parameter stays as a
   compatibility alias.
2. Internal callers switch from `Provider` to `Source` at boundaries
   where lab identity and runtime identity need to be distinguished
   (e.g. analytics, cost rollups, source-discovery).
3. Storage column `provider_name` either stays as a physical-schema
   compatibility artifact (documented as legacy) or is renamed via an
   explicit schema-version transition.
4. Provider-wire schemas under `schemas/providers/` are retained as
   lab/provider-scope artifacts — they describe raw export shapes and
   stay keyed by lab/product, not by source family.

Anti-goal: a half-renamed surface where some flags say `provider` and
others say `source`. Each surface flips wholesale or not at all.

## Antigravity Language-Server Export Path

Antigravity persists its conversation transcripts as opaque non-protobuf
`conversations/*.pb` and `implicit/*.pb` blobs that cannot be statically
decoded. The installed Antigravity language server binary
(`language_server_linux_x64`) exposes two endpoints over a local HTTP loopback
port that together form the supported export surface:

| Endpoint | Purpose |
|----------|---------|
| `/exa.language_server_pb.LanguageServerService/SearchConversations` | Returns cascade IDs, titles, workspace names, snippets, and `lastModifiedTime` for stored conversations. |
| `/exa.language_server_pb.LanguageServerService/ConvertTrajectoryToMarkdown` | Returns a complete Markdown export for a given cascade ID — user inputs, planner responses, tool/command events. |

The adapter lives in `polylogue/sources/parsers/antigravity.py`:

- `AntigravityLanguageServerClient` spawns the binary with
  `-standalone -persistent_mode -http_server_port=<port>` against the user's
  Antigravity data root, waits until the search endpoint answers, and tears the
  process down on close.
- `discover_language_server()` resolves the binary in this order:
  `POLYLOGUE_ANTIGRAVITY_LANGUAGE_SERVER` env var, `$PATH`, then the highest
  matching `/nix/store/*-antigravity-*` extension bundle.
- `iter_language_server_exports(root)` drives `SearchConversations` and
  `ConvertTrajectoryToMarkdown` and yields `ParsedConversation` objects through
  `parse_markdown_export()`.

The ingest path is layered in `polylogue/sources/source_parsing.py`: when the
source is `antigravity` and a `conversations/` subdirectory exists, the
language-server export runs first; any `AntigravityExportError` (binary not
found, connection failure, malformed response) is logged and the source falls
back to the existing brain-artifact metadata walk. Both paths emit normalized
`Provider.ANTIGRAVITY` conversations.

## Key Abstractions

| Abstraction | Location | Role |
|-------------|----------|------|
| `Polylogue` | `facade.py` | Async entry point. Wraps storage + search + pipeline. |
| `ConversationRepository` | `storage/repository/__init__.py` | Mixin-composed async repository (9 mixins: archive reads/writes, action reads, insight readers for profile/timeline/thread/summary, raw, vectors). |
| `SearchProvider` protocol | `protocols.py` | FTS5 and Hybrid (RRF fusion) implementations. |
| `ConversationFilter` | `archive/filter/filters.py` | Fluent filter chain used by CLI, MCP, and facade. |
| `Session Insights` | `storage/insights/session/` | Materialized read models: profiles, work events, phases, threads, aggregates. |
| `ContentHash` | `pipeline/ids.py` | SHA-256 over NFC-normalized conversation payload. Title, timestamps, messages, attachments are hashed. User metadata (tags, summaries) is excluded — editable metadata doesn't trigger re-import. |
| `Provider` enum | `types.py` | Legacy source identifier — 9 known providers + UNKNOWN. Public surfaces still flow through this enum during the dual-vocabulary period. |
| `Source` dataclass | `core/sources.py` | Source-centered identity (`family`, `runtime_root`, `originating_lab`). Parallel to `Provider`; see "Dual Vocabulary Period" above. |

## Artifact Taxonomy

Acquired files are classified by `ArtifactKind` before ingestion:

| Kind | Description |
|------|-------------|
| `conversation_document` | A single conversation (Claude Code JSONL, ChatGPT JSON) |
| `conversation_record_stream` | Stream of conversation events |
| `subagent_conversation_stream` | Sidechain sub-agent conversation |
| `agent_sidecar_meta` | Session metadata (history.jsonl, sessions-index.json) |
| `session_index` | Provider-level session index |
| `bridge_pointer` | Pointer from a parent session to a sub-agent session |
| `metadata_document` | Supplementary metadata |
| `unknown` | Unclassified artifact |

`classify_artifact()` in `sources/artifact_taxonomy/` assigns each acquired file
a kind. The daemon uses artifact classification to route files to the correct
ingestion path.

## Hook Integration

Polylogue integrates with AI coding agents via hook scripts that fire on session
lifecycle events:

- **Claude Code**: 16 hook events available (SessionStart, SessionEnd,
  PreToolUse, PostToolUse, Notification, Stop, SubagentStart/Stop, PreCompact,
  PermissionRequest). See [#802](https://github.com/Sinity/polylogue/issues/802).
- **Codex**: 6 hook events available (SessionStart, SessionEnd, PreToolUse,
  PostToolUse, Error, Warning).

Hook scripts call `polylogue-hook` which ingests session data at event
granularity, providing 100% data coverage vs. ~79% from post-hoc JSONL
discovery. Hooks are the enabling infrastructure for real-time context injection
(SessionStart), session completion processing (SessionEnd), and accurate paste
detection (PreToolUse/PostToolUse).

## Embedding Pipeline

Vector embeddings for semantic search, powered by Voyage AI (`voyage-4`,
1024 dimensions) via SQLite-vec (`vec0` virtual table):

- **Storage**: `message_embeddings` (vec0), `embeddings_meta`, `embedding_status`
- **Search**: `--similar` flag triggers pure vector search; hybrid mode combines
  FTS5 + vector via Reciprocal Rank Fusion
- **Integration**: Daemon-side post-ingest embedding is opt-in via
  `embedding_enabled = true` in `polylogue.toml` with a valid `voyage_api_key`;
  default live convergence does not call the embedding provider during catch-up
  ([#828](https://github.com/Sinity/polylogue/issues/828)).

### Activation flow (#1217)

The `polylogue embed` group is the operator-facing onboarding surface:

| Command | Purpose |
|---------|---------|
| `polylogue embed preflight` | Count pending messages + Voyage cost estimate without contacting the provider. |
| `polylogue embed enable` (alias `activate`) | Verify `sqlite-vec`, capture the Voyage key, print the cost preflight, and on confirmation persist `[embedding] enabled = true` (and the API key unless `--no-store-key`) into the user `polylogue.toml`. |
| `polylogue embed backfill` | Run the first embedding batch with per-conversation cost feedback; honours `embedding_max_cost_usd` as a soft cap and stops on overshoot. |
| `polylogue embed disable` | Flip `embedding.enabled = false` without dropping existing embeddings — previously-embedded messages remain queryable via `--similar`. |
| `polylogue embed status` | Coverage / freshness snapshot via `embedding_status_payload`. |

The CLI orchestrates substrate primitives under
`polylogue.storage.embeddings` (`iter_pending_conversations`,
`embed_conversation_sync`) and the cost constants
`ESTIMATED_TOKENS_PER_MESSAGE` / `VOYAGE_4_COST_PER_1M_TOKENS` from
`polylogue.storage.search_providers.sqlite_vec_support`.

### Search defaults (#1217)

Once `embedding_enabled = true` and at least one message is embedded,
`polylogue` searches automatically promote `retrieval_lane=auto` to
`hybrid` (FTS5 + vector RRF) when an FTS query is present. The
elevation lives in
`polylogue/cli/query.py:_maybe_elevate_to_hybrid`. Two ergonomic
overrides land on the root query surface:

- `--lexical` — force `retrieval_lane=dialogue` (FTS-only).
- `--semantic` — promote the query string into `similar_text` so the
  request runs as a vector-only similarity probe (no FTS leg).

See [docs/search.md § Retrieval Lanes](search.md#retrieval-lanes) for the
full lane semantics, ranking policy, and `SearchEnvelope` contract.

`polylogue status` includes an `Embeddings:` line whenever any
messages are embedded, so the operator can see coverage at a glance.

## Blob Store

Content-addressed storage for large binary artifacts (message content,
attachments, exports):

- **Addressing**: SHA-256 hash over content, stored in 256 prefix-sharded
  subdirectories (`blob/ab/cdef...`)
- **Dedup**: Identical content produces identical hashes — automatic
  deduplication
- **Linking**: `artifact_observations.link_group_key` groups blobs by session
  for lifecycle management (there is no separate `blob_links` table; the name
  is a historical alias for this row-group view of `artifact_observations`)
- **Scale**: ~24K blobs, ~42 GB in production archive
- **GC**: Blob garbage collection is tracked in
  [#818](https://github.com/Sinity/polylogue/issues/818)

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
- `pipeline/` — stage execution, daemon ingestion, validation, and indexing
- `insights/` — derived read models, session insights, analytics
- `operations/` — operation specs, artifact graph, declared runtime contracts

### Surfaces (presentation only)
- `cli/` — Click commands, shared helpers, output formatting
- `mcp/` — MCP server tools
- `api/` — async library API
- `rendering/` — markdown/HTML renderers
- `ui/` — TUI, dashboard
- `daemon/` — daemon convergence, HTTP API, and web reader

### Verification (repo health)
- `proof/` — verification catalog internals, subject discovery, claim catalog, witnesses
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
