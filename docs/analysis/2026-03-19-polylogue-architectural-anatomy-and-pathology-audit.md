# Polylogue architectural anatomy and pathology audit

Status: analysis input, not the live execution queue
Role: external-style code-outward audit feeding later program docs

Current execution entrypoint:

- `planning-and-analysis-map-2026-03-21.md`
- `intentional-forward-program-2026-03-21.md`

## Executive judgment

Polylogue is materially more coherent than its repo topology first suggests: there is a **single, real operational spine** that governs most meaningful behavior—**Acquire → Validate → Parse → Prepare/Save → Render → Index**—and that spine is wired end-to-end through the CLI “run” command into a SQLite-backed archive. The system’s “center of gravity” is not any one folder name, but the interplay between **`polylogue.pipeline.*` orchestration**, **`polylogue.sources.source` ingestion/parsing**, and **`polylogue.storage.*` persistence/querying**. fileciteturn43file0L1-L1 fileciteturn16file0L1-L1 fileciteturn62file0L1-L1 fileciteturn64file0L1-L1

That said, the codebase is also carrying **visible architectural scar tissue**: multiple overlapping “search/index” stacks, multiple overlapping “provider identity” surfaces across raw ingestion vs parsed truth, and several subsystems whose names imply stronger boundaries or runtime authority than they actually have (e.g., “vector provider” protocol semantics, drive/gemini structured parsing, indexing service scope). These aren’t local bugs; they’re structural mismatches and partial-migrations that add capability *and* add drift-prone complexity. fileciteturn61file0L1-L1 fileciteturn36file0L1-L1 fileciteturn27file0L1-L1

Bottom line: **the architecture hangs together**, but it’s not clean—there are a few disproportionately-large “truthy” modules (notably `sources/source.py`) and a few duplicated “platform” layers (search/index; schema+semantic extraction) that look like they were built to support competing futures. The highest-value simplification would come from **choosing one canonical path** for (a) indexing/search and (b) provider/schema/semantic truth, and then pruning the alternates. fileciteturn62file0L1-L1 fileciteturn53file0L1-L1 fileciteturn50file0L1-L1

## Structural atlas

### Operational core: pipeline orchestration

The most “real” subsystem boundary in practice is the pipeline itself, because it defines **runtime phases, persisted state transitions, and CLI-exposed stage selection**. The pipeline is structured as a stage runner plus discrete services:

- Stage driver: `polylogue.pipeline.runner` (planning + execution surfaces) fileciteturn16file0L1-L1
- Planning: `polylogue.pipeline.services.planning` (builds the “what would run”/“what should run” plan by scanning sources + querying DB state) fileciteturn25file0L1-L1
- Acquisition: `polylogue.pipeline.services.acquisition` (visits sources, yields/stores raw records) fileciteturn19file0L1-L1
- Validation: `polylogue.pipeline.services.validation` (schema validation of raw payloads; persists validation status/drift; gates parseability) fileciteturn24file0L1-L1
- Parsing: `polylogue.pipeline.services.parsing` (decode raw blobs, classify artifact/provider, parse to conversations, “prepare” and persist canonical records) fileciteturn20file0L1-L1
- Prepare/save mechanics: `polylogue.pipeline.prepare` (build record bundles, compute IDs/hashes, compute semantic fields, attachment materialization) fileciteturn29file0L1-L1
- Rendering: `polylogue.pipeline.services.rendering` (concurrent render loop on conversation IDs using an `OutputRenderer`) fileciteturn26file0L1-L1
- Indexing: `polylogue.pipeline.services.indexing` (FTS table maintenance; described as broader than it is) fileciteturn27file0L1-L1

Cohesion: Strong. The stage services correspond to actual DB state fields (`raw_conversations.validated_at`, `raw_conversations.parsed_at`, etc.) and CLI entrypoints. fileciteturn51file0L1-L1 fileciteturn43file0L1-L1

Fragmentation: Moderate. “Prepare” and “Source parsing” have responsibilities that spill across boundaries (see below), and indexing/search is duplicated across multiple layers. fileciteturn29file0L1-L1 fileciteturn62file0L1-L1 fileciteturn53file0L1-L1

### Storage and “truth surfaces”: SQLite schema + repository + hydrators

Polylogue is opinionated about a **single canonical persistence substrate**: a SQLite database containing raw payload blobs plus normalized conversations/messages/blocks/attachments, plus run logs and derived stats. The actual truth surfaces include:

- `raw_conversations`: raw bytes + acquisition metadata + validation/parse status fields fileciteturn51file0L1-L1
- `conversations`: canonical conversation records, including `provider_meta` and generated `source_name` extracted from `provider_meta.source` fileciteturn51file0L1-L1
- `messages`: canonical text rows with precomputed analytics flags (`has_tool_use`, `has_thinking`, `word_count`) fileciteturn51file0L1-L1
- `content_blocks`: structured segments and semantic tags (`semantic_type`) used for SQL pushdowns and higher-level semantics fileciteturn51file0L1-L1
- `conversation_stats`: derived aggregates used for SQL pushdown filters fileciteturn51file0L1-L1
- `attachments` + `attachment_refs`: attachment identity + reference tracking fileciteturn51file0L1-L1
- `runs`: pipeline run audit log fileciteturn51file0L1-L1

Runtime role ownership is split between:

- `polylogue.storage.backends.async_sqlite.SQLiteBackend`: the “DB API,” connection lifecycle, locking/transaction strategy, schema initialization, and query dispatch into `storage/backends/queries/*`. fileciteturn50file0L1-L1
- `polylogue.storage.repository.ConversationRepository`: the “application repository” that hydrates domain models and implements higher-level search/filter facilities, plus vector embedding hooks. fileciteturn64file0L1-L1
- `polylogue.storage.hydrators` (referenced by repository): turns storage records into domain `Conversation`/`Message` models. fileciteturn64file0L1-L1

Cohesion: Strong at the persistence level; the schema and backend methods line up closely, and the pipeline services persist explicit state. fileciteturn50file0L1-L1 fileciteturn51file0L1-L1

Pathology: Search/index is “over-platformed” (multiple ways to do the same thing), and vector search is partially bolted onto an async system using a synchronous provider. fileciteturn64file0L1-L1 fileciteturn59file0L1-L1

### Source ingestion and parsing: a monolithic “source platform” boundary

Despite being under `sources/`, `polylogue.sources.source` is not a narrow “provider boundary.” It is effectively:

- **Filesystem traversal + filtering + ZIP processing** (including skip rules, zip-bomb heuristics, mtime-based skipping) fileciteturn62file0L1-L1
- **Provider detection** across dict/list/JSONL payloads and raw bytes sniffing fileciteturn62file0L1-L1
- **Artifact taxonomy routing** via `classify_artifact` / `classify_artifact_path` to decide whether a payload “counts” as a conversation fileciteturn62file0L1-L1
- **Stream parsing strategies** for JSONL vs JSON vs nested structures using `ijson` and a multi-strategy reader fileciteturn62file0L1-L1
- **Parser dispatch** into provider parsers (chatgpt/claude/codex/drive), including special handling of grouped JSONL providers fileciteturn62file0L1-L1
- **A persistence-adjacent “bundle” abstraction** (`RecordBundle`, `save_bundle`) that directly depends on storage record types and the repository fileciteturn62file0L1-L1

This file is a *real* subsystem boundary because so much of ingestion depends on it, but it’s also the clearest instance of **responsibility concentration and boundary bleed** (source concerns + parsing concerns + storage glue). fileciteturn62file0L1-L1

### Domain semantics and query surfaces: models + filters + query planning

There is a distinct “semantic/query plane” built around:

- `polylogue.lib.models`: domain `Conversation` / `Message` models embedding semantic classification logic (`is_tool_use`, `is_thinking`, noise filtering, etc.), and a “harmonized” extraction hook into schemas for provider_meta normalization. fileciteturn65file0L1-L1
- `polylogue.lib.filters`: fluent query builder that mixes SQL pushdown (via repository parameters and conversation_stats/content_blocks) with in-memory post-filters and sorting. fileciteturn47file0L1-L1
- `polylogue.lib.query_spec`: “typed selection intent” shared by CLI and other surfaces, compiling into a `ConversationFilter`. fileciteturn46file0L1-L1
- CLI query routing and planning (`polylogue.cli.query`, `polylogue.cli.query_plan`) that chooses between summary-list, streaming, SQL stats, full list, etc. fileciteturn44file0L1-L1 fileciteturn45file0L1-L1

This is a real subsystem: it governs most non-ingest user workflows (list/search/stats/modify). It is also where **semantic truth becomes ambiguous**, because message-level provider metadata is not persisted, so domain heuristics sometimes fall back to provider_meta assumptions that are only valid in some construction paths. fileciteturn65file0L1-L1 fileciteturn51file0L1-L1

### CLI/runtime composition: explicit service scope + multi-surface entrypoints

There are (at least) two operational entrypoints:

- CLI: `polylogue.cli.click_app` (query-first group + subcommands, including `run`, `qa`, `schema`, `site`, `mcp`) fileciteturn40file0L1-L1
- Library facade: `polylogue.facade.Polylogue` (async user-facing entrypoint for querying and parsing sources/files) fileciteturn37file0L1-L1

The CLI composes runtime dependencies through `polylogue.services.RuntimeServices` (config/backend/repository) exposed via `AppEnv`. fileciteturn42file0L1-L1 fileciteturn41file0L1-L1

This explicit service scope is a healthy boundary (it replaces “ambient singleton service locator” per the module docstring), but it coexists with other composition styles (the facade constructs its own backend/repository/config, the pipeline runner can accept injected backend/repository, etc.), so the “center” is real but not exclusive. fileciteturn42file0L1-L1 fileciteturn37file0L1-L1 fileciteturn43file0L1-L1

## Runtime and dataflow map

### Acquisition: source discovery, raw byte capture, and raw storage

**Where truth originates:** on disk (source paths, ZIP entries), read by `iter_source_raw_data` and `iter_source_conversations_with_raw` style iterators. In the pipeline, acquisition specifically uses `iter_source_raw_data`, which yields `RawConversationData` blobs that contain raw bytes and metadata such as `source_path`, `file_mtime`, and a detected `provider_hint`. fileciteturn19file0L1-L1 fileciteturn62file0L1-L1 fileciteturn32file0L1-L1

**Where coordination happens:** `AcquisitionService` scans configured sources, tracking cursors, and persists raw records into SQLite via the async backend (commonly grouped via bulk connection patterns). fileciteturn19file0L1-L1 fileciteturn50file0L1-L1

**Stored truth surface:** `raw_conversations` rows store `raw_content` BLOB plus acquisition metadata and processing state fields. fileciteturn51file0L1-L1

### Validation: schema gating between raw bytes and parsing

Polylogue inserts a dedicated VALIDATE stage which is operationally real: it can be configured (via an environment variable), it produces drift/invalid/error counts, and it persists validation status back onto raw rows. fileciteturn24file0L1-L1

Mechanically, validation:

1. Builds a “raw payload envelope” (decode + provider + artifact classification) and rejects artifacts that are “schema ineligible.” fileciteturn24file0L1-L1
2. Attempts to construct a `SchemaValidator` for the provider/payload; missing schema becomes a “skipped_no_schema” outcome. fileciteturn24file0L1-L1
3. Runs validation in a thread pool (explicitly justified by `orjson` GIL behavior and jsonschema checks), then serializes DB writes. fileciteturn24file0L1-L1

**Invariants enforcement:** this is where raw payloads can be marked as non-parseable and where schema drift is tracked. But because missing schemas are treated as “skip,” the schema system is not fully authoritative—it’s a gate only when a schema exists and strict mode is enabled. fileciteturn24file0L1-L1

### Parsing: decode, provider detection, artifact routing, parse dispatch

Parsing is split between:

- `ParsingService` orchestration (pipeline stage) fileciteturn20file0L1-L1
- `sources.source` provider detection and parse dispatch (`detect_provider`, `parse_payload`) and artifact taxonomy filtering (`classify_artifact`) fileciteturn62file0L1-L1
- Provider-specific parsers (`sources/parsers/*.py`) that normalize wire formats into `ParsedConversation` / `ParsedMessage` / `ParsedContentBlock`. fileciteturn32file0L1-L1 fileciteturn33file0L1-L1 fileciteturn34file0L1-L1 fileciteturn35file0L1-L1 fileciteturn36file0L1-L1

**Where transformations happen:** provider parsers convert heterogeneous payloads into a canonical parsed shape; then `prepare_records` converts that parsed shape into persistent record bundles with deterministic IDs and hashes. fileciteturn32file0L1-L1 fileciteturn29file0L1-L1 fileciteturn30file0L1-L1

### Prepare/persist: canonical IDs, change detection, semantic extraction, attachments

This is the highest-density “real mechanics” zone.

**ID and hash truth:**

- `conversation_id = "{provider}:{provider_conversation_id}"` is deterministic. fileciteturn30file0L1-L1
- `conversation_content_hash` hashes title/timestamps/messages/attachments using sentinel values to distinguish `None` vs empty. fileciteturn30file0L1-L1
- Message IDs are deterministic (`"{conversation_id}:{provider_message_id}"`), and messages have `message_content_hash`. fileciteturn30file0L1-L1

**Change detection:** `PrepareCache` bulk-loads existing conversation hashes and message ID mappings to decide whether content changed and to reuse stable IDs, avoiding per-conversation DB chatter. fileciteturn29file0L1-L1

**Semantic extraction is done *at ingest time*:** `transform_to_records` emits `ContentBlockRecord` rows and sets `semantic_type` and semantic metadata for tool calls (git/file ops/subagent spawns) by classifying tools and extracting structured tool metadata. fileciteturn29file0L1-L1 fileciteturn31file0L1-L1

**Attachment materialization:** attachment IDs prefer content hashes (sha256) if a file exists; otherwise derive from a seed. The pipeline can move attachments into an archive asset path and de-duplicate duplicates. fileciteturn30file0L1-L1 fileciteturn29file0L1-L1

**Persistence surface:** `ConversationRepository.save_conversation` uses a lightweight pre-read of existing `content_hash`, then an UPSERT transaction; unchanged content avoids expensive work, while changed content writes messages, upserts stats, writes content blocks, prunes attachment refs, and saves attachments. fileciteturn64file0L1-L1

### Query, projection, and operator workflows

**Query selection intent** is compiled into `ConversationFilter` chains (`ConversationQuerySpec.build_filter`), which are executed either as full conversation loads or via lightweight summaries depending on route selection and filter compatibility. fileciteturn46file0L1-L1 fileciteturn47file0L1-L1 fileciteturn44file0L1-L1

**SQL pushdown exists, but only for certain dimensions:** it is mediated through repository parameters and a stats/content_blocks-aware query builder, while other filters are applied in Python. fileciteturn47file0L1-L1 fileciteturn48file0L1-L1 fileciteturn52file0L1-L1

**Streaming path exists as a separate operator surface:** repository supports streaming messages (`iter_messages`) and CLI supports `--stream` as a distinct route. fileciteturn50file0L1-L1 fileciteturn44file0L1-L1

### Rendering and indexing

Rendering is an explicit stage that runs DB reads + rendering concurrently, with timeouts and slow-render logging. It can optionally use a backend “read pool” to reduce connection churn. fileciteturn26file0L1-L1 fileciteturn50file0L1-L1

Indexing is also an explicit stage, but the database schema already creates `messages_fts` and triggers on message insert/update/delete. This makes indexing a likely redundancy for fresh DBs, while remaining relevant as a rebuild/repair tool for existing DBs or schema evolutions. fileciteturn51file0L1-L1 fileciteturn27file0L1-L1

## Boundary audit

### Best boundaries

**Pipeline stage boundaries are real and enforced.** The plan/validate/parse gating is not decorative; it corresponds to persisted fields and to the CLI’s `--stage` selection. This is a meaningful separation of acquisition integrity, schema conformance signaling, and canonicalization. fileciteturn43file0L1-L1 fileciteturn25file0L1-L1 fileciteturn24file0L1-L1 fileciteturn51file0L1-L1

**Explicit runtime dependency scope (`RuntimeServices`) is a good boundary.** It gives an invocation-scoped composition root and avoids hidden global state by lazily initializing config/backend/repository. fileciteturn42file0L1-L1 fileciteturn41file0L1-L1

**The SQLite schema + backend query layer is cohesive.** `SQLiteBackend` is a clear “DB boundary,” with explicit transaction and connection reuse strategies (transaction connection, bulk connection, read pool). fileciteturn50file0L1-L1

### Weakest boundaries

**`polylogue.sources.source` is a boundary failure disguised as a boundary.** It is simultaneously an ingestion engine, parser router, archive-file walker, ZIP security layer, streaming JSON reader, and (via `save_bundle`) a persistence adapter. This makes it hard to reason about “source responsibilities” vs “storage responsibilities,” and it increases the blast radius for modifications. fileciteturn62file0L1-L1

**Domain semantics are split across three representations.** There is a parsed representation (`ParsedContentBlock`), a persisted representation (`content_blocks` table + semantic_type), and a domain model representation (`Message.content_blocks` as dicts + heuristics + optional harmonized schema extraction). The coexistence is not inherently wrong, but it can become incoherent when one representation is missing or not populated (see drive/gemini). fileciteturn32file0L1-L1 fileciteturn51file0L1-L1 fileciteturn65file0L1-L1

### Over-abstracted or nominal boundaries

**Search provider protocols look more “platform-like” than their usage warrants.** The `SearchProvider` / `VectorProvider` protocols exist, but the dominant runtime search path is repository + backend SQL (FTS queries), not pluggable search providers. Meanwhile, vector search integration is synchronous and semantically mismatched to protocol docs. The abstraction exists, but it isn’t consistently authoritative. fileciteturn61file0L1-L1 fileciteturn64file0L1-L1 fileciteturn57file0L1-L1

**Indexing is a nominal “stage boundary” with duplicated implementation centers.** There is an indexing service, schema-level triggers, a sync FTS provider, and a “search” module that can execute FTS queries directly—multiple overlapping layers that suggest partial migrations. fileciteturn27file0L1-L1 fileciteturn51file0L1-L1 fileciteturn60file0L1-L1 fileciteturn53file0L1-L1

## Redundancy and accidental complexity findings

**Finding: Four overlapping FTS/index stacks**

Severity: High

Evidence:
- Schema creates `messages_fts` and defines triggers to keep it updated. fileciteturn51file0L1-L1
- IndexService also creates/rebuilds/updates `messages_fts` explicitly via SQL. fileciteturn27file0L1-L1
- `FTS5Provider` implements its own index and search logic via sync sqlite connections. fileciteturn60file0L1-L1
- `polylogue.storage.search` can build ranked FTS queries and execute them (including cached sync entrypoints). fileciteturn53file0L1-L1 fileciteturn54file0L1-L1

Why it is redundant/accidental: This is classic “multiple centers of truth” for one capability. Fresh DBs already have trigger-based indexing, so “indexing stage” becomes repair-only; yet parallel providers and search implementations remain. That increases maintenance cost and makes it unclear which indexing path is intended to be canonical. fileciteturn51file0L1-L1

Simpler shape that is plausible: Choose exactly one canonical indexing mechanism:
- Either (A) **schema triggers + optional one-shot rebuild tool** (remove FTS5Provider indexing and/or collapse IndexService into “repair”), or (B) **explicit rebuild/update service without triggers** (remove triggers and make indexing stage mandatory). The current hybrid is the worst of both worlds. fileciteturn51file0L1-L1 fileciteturn27file0L1-L1

**Finding: Vector search is bolted onto an async system with synchronous network I/O**

Severity: High

Evidence:
- Repository calls `vector_provider.upsert()` and `vector_provider.query()` inside async methods without offloading. fileciteturn64file0L1-L1
- The sqlite-vec provider uses synchronous `httpx.Client` calls to the Voyage API, plus synchronous sqlite connections. fileciteturn59file0L1-L1

Why it is redundant/accidental: It violates the architectural promise that storage/repository operations are async-friendly; calling blocking HTTP within an async call path risks event-loop pauses and makes concurrency behavior unpredictable. fileciteturn64file0L1-L1

Simpler shape that is plausible: Move embeddings and vector queries behind an async boundary (async http client or explicit thread offload), or separate embedding/indexing into a dedicated command/process that writes to SQLite, making runtime query path pure-local. fileciteturn59file0L1-L1 fileciteturn43file0L1-L1

**Finding: Provider identity has too many partially-overlapping “truth surfaces”**

Severity: High

Evidence:
- Acquisition yields raw payloads with a `provider_hint`, derived from file sniffing and/or `Provider.from_string(source.name)` logic. fileciteturn62file0L1-L1
- Raw storage has `raw_conversations.provider_name`, plus `payload_provider`, plus separate validation fields for provider (`validation_provider`) and parseability. fileciteturn51file0L1-L1 fileciteturn24file0L1-L1
- Canonical conversations store provider name separately, and source name is derived from `provider_meta.source`. fileciteturn51file0L1-L1 fileciteturn29file0L1-L1

Why it is redundant/accidental: You can end up with raw rows where `provider_name` is effectively “source name” when detection is uncertain, while parsed conversations treat provider as canonical. Add schema validation’s “canonical_provider” and “payload_provider,” and there is no single, easy-to-explain provider truth. fileciteturn24file0L1-L1 fileciteturn62file0L1-L1

Simpler shape that is plausible: Define a strict vocabulary:
- `source_name`: configuration/operator name (“where it came from”)
- `provider_detected`: what content actually is (post-decode)
- `provider_hint`: unstable heuristic used only during acquisition
…and store only the ones that matter for downstream logic. Right now, the schema and store imply this separation, but the naming/layout doesn’t force it. fileciteturn51file0L1-L1

**Finding: Schema/semantic extraction has “old format” support embedded in the core path**

Severity: Medium

Evidence:
- `pipeline.semantic` explicitly retains an “old-format API (raw dict-based)” alongside the “canonical home for semantic extraction.” fileciteturn31file0L1-L1
- Domain models (`Message.harmonized`) reach into `schemas.unified.extract_from_provider_meta`, indicating yet another semantic normalization path. fileciteturn65file0L1-L1

Why it is redundant/accidental: The system already persists structured `content_blocks` and semantic types at ingest time. Keeping parallel extraction APIs increases the chance of semantic divergence (“this feature works when provider_meta exists but not when loaded from DB,” etc.). fileciteturn29file0L1-L1 fileciteturn51file0L1-L1

Simpler shape that is plausible: Treat the DB `content_blocks` + `semantic_type` as authoritative for operational features, and relegate dict-based extraction to test-only utilities or explicit compatibility modules. fileciteturn51file0L1-L1 fileciteturn31file0L1-L1

**Finding: Two parallel “API front doors” construct runtime dependencies differently**

Severity: Medium

Evidence:
- CLI uses `RuntimeServices` to provide config/backend/repository to commands and MCP server. fileciteturn42file0L1-L1 fileciteturn40file0L1-L1
- The library facade constructs its own minimal `Config` and creates its own `SQLiteBackend` and `ConversationRepository`. fileciteturn37file0L1-L1

Why it is redundant/accidental: This can easily lead to feature drift: CLI might pick up config-driven behavior (sources, index config, etc.) while the facade cannot unless manually wired. It’s not wrong to have both, but it enlarges the surface area for “same behavior, different wiring.” fileciteturn37file0L1-L1

Simpler shape that is plausible: Make `RuntimeServices` the single composition primitive and have the facade wrap it (or accept it), reducing duplicated wiring and configuration handling. fileciteturn42file0L1-L1 fileciteturn37file0L1-L1

## Misfit and miswiring findings

**Finding: Drive/Gemini structured parsing is not integrated into the canonical content-block persistence path**

Severity: High

Evidence:
- `drive.parse_chunked_prompt` builds rich metadata and “content_blocks” but stores them in `provider_meta` and does not populate `ParsedMessage.content_blocks`. fileciteturn36file0L1-L1
- The prepare/persist path generates persistent `ContentBlockRecord` rows from `ParsedConversation.messages[].content_blocks`. If those are empty, semantic fields and content blocks are dropped. fileciteturn29file0L1-L1
- The DB schema does not store message-level provider_meta in `messages`, so the fallback “provider_meta content_blocks” checks in domain `Message` won’t help for DB-loaded messages. fileciteturn51file0L1-L1 fileciteturn65file0L1-L1

Why it does not fit well: The code *looks* like it supports Gemini-style structured blocks, but the core persistence mechanics don’t carry those structures through, making the capability effectively “paper real” unless messages are kept in memory with provider_meta. fileciteturn36file0L1-L1

Consequence: Features like “has thinking,” “has tool use,” and semantic SQL filters will undercount or ignore Gemini/Drive conversations, and any downstream rendering/query logic that depends on persisted content blocks will be inconsistent across providers. fileciteturn47file0L1-L1 fileciteturn51file0L1-L1

**Finding: Protocol semantics do not match implementation semantics for vector search**

Severity: High

Evidence:
- `VectorProvider.query` protocol docstring claims it returns `(conversation_id, similarity_score)` tuples. fileciteturn61file0L1-L1
- Repository code treats vector results as `(message_id, distance)` and then maps message IDs to conversation IDs in SQL. fileciteturn64file0L1-L1
- SqliteVecProvider returns `(message_id, distance)` and queries `message_embeddings` by `message_id`. fileciteturn59file0L1-L1

Why it does not fit well: This is a contract mismatch that invites incorrect future implementations (a different vector provider might “correctly” implement the protocol and break repository assumptions). fileciteturn61file0L1-L1

Consequence: The abstraction boundary is fragile; it will fail silently at integration time because types are compatible but semantics are not. fileciteturn61file0L1-L1 fileciteturn64file0L1-L1

**Finding: `sources/source.py` mixes persistence glue into parsing infrastructure**

Severity: Medium

Evidence:
- The file defines `RecordBundle` and `save_bundle` that imports storage record types and calls repository.save_conversation. fileciteturn62file0L1-L1
- The prepare pipeline imports `RecordBundle`/`save_bundle` from `sources.source`, pulling storage-facing orchestration into the sources layer. fileciteturn29file0L1-L1

Why it does not fit well: Source parsing becomes harder to re-use independently (e.g. for “parse-only tooling” or for provider development) because the module is coupled to persistence shape. fileciteturn62file0L1-L1

Consequence: Any attempt to refactor storage or change record bundling risks destabilizing ingestion/parsing traversal logic (and vice versa). fileciteturn62file0L1-L1

**Finding: Indexing service scope is overstated**

Severity: Low to Medium

Evidence:
- `IndexService` docstring claims “full-text and vector search indices,” but its implementation is only FTS5 table creation/rebuild/update. fileciteturn27file0L1-L1

Why it does not fit well: This inflates perceived subsystem responsibility and hides the real vector indexing pathway (sqlite-vec provider + embeddings tables). fileciteturn59file0L1-L1

Consequence: Operators/developers will likely assume “index stage” covers vector readiness when it does not, increasing operational confusion. fileciteturn27file0L1-L1 fileciteturn43file0L1-L1

## Simplification opportunities

### Collapse and unify

**Unify indexing/search into one canonical mechanism.** The biggest deletion win is reducing to one FTS lifecycle and one search access pattern:
- If triggers are canonical, treat “indexing” as *repair-only* and delete or heavily demote FTS5Provider indexing + redundant sync search helpers. fileciteturn51file0L1-L1 fileciteturn60file0L1-L1 fileciteturn53file0L1-L1
- If explicit rebuild is canonical, remove triggers and concentrate all FTS logic in the indexing service + backend queries. fileciteturn27file0L1-L1 fileciteturn52file0L1-L1

**Split `sources/source.py` into three modules.** A plausible cut:
- `sources.walk` (filesystem/zip traversal, mtime skipping, ZIP safety) fileciteturn62file0L1-L1
- `sources.streams` (JSON/JSONL streaming strategies, decoding) fileciteturn62file0L1-L1
- `sources.dispatch` (provider detection + parse dispatch + artifact taxonomy gating) fileciteturn62file0L1-L1
…and move `RecordBundle/save_bundle` into pipeline/storage where it belongs. fileciteturn62file0L1-L1

**Fix the provider identity model by naming and enforcing it.** The code already hints at distinctions (provider_name vs payload_provider vs source_name), but the acquisition path can collapse them. Pick one “canonical provider” field for raw rows and make it consistent. fileciteturn51file0L1-L1 fileciteturn62file0L1-L1

### Narrow or remove weak abstractions

**Repair the VectorProvider contract and async integration.** Either:
- Make vector provider methods async and non-blocking, or
- quarantine embeddings behind an explicit offline indexing stage (so runtime query remains local). fileciteturn64file0L1-L1 fileciteturn59file0L1-L1

**Decide whether “harmonized schema extraction” is a first-class runtime feature.** Right now it is used as a fallback in domain models for message classification. If you intend content_blocks to be authoritative, push harmonization into ingest/persist instead, or remove it from the hot path. fileciteturn65file0L1-L1 fileciteturn51file0L1-L1

### What should stay as-is

**The pipeline planning/validation split is worth keeping.** Planning service is doing real work (scan + DB state diff + backlog merge), and validation is a meaningful gate with drift reporting and strict/advisory modes. This division is justified. fileciteturn25file0L1-L1 fileciteturn24file0L1-L1

**PrepareCache and deterministic IDs/hashes are good structural primitives.** They turn ingestion into an idempotent-ish process and reduce query storms. This is central complexity that earns its keep. fileciteturn29file0L1-L1 fileciteturn30file0L1-L1

## Design intent, ranked actions, and confidence

**Design-intent vs code-reality comparison**

Supported by code:
- “There is a canonical ingest pipeline.” True in practice: stage runner + services + persisted state fields align. fileciteturn16file0L1-L1 fileciteturn51file0L1-L1
- “Runtime dependency composition is explicit.” True for CLI/MCP via `RuntimeServices`. fileciteturn42file0L1-L1 fileciteturn40file0L1-L1
- “Semantic extraction is canonical at ingest.” Largely true: prepare computes semantic types/tool metadata and persists content blocks. fileciteturn29file0L1-L1 fileciteturn31file0L1-L1

Overstated or drifted in code:
- “Vector provider protocol defines semantics.” It does not; the contract docs disagree with real behavior (message IDs vs conversation IDs). fileciteturn61file0L1-L1 fileciteturn64file0L1-L1
- “Drive/Gemini parsing yields structured content blocks.” It does, but it doesn’t travel through the canonical persistence path, so runtime semantics will not match. fileciteturn36file0L1-L1 fileciteturn29file0L1-L1
- “Indexing is one coherent subsystem.” Reality is multiple overlapping layers (schema triggers, indexing service, search module, sync providers). fileciteturn51file0L1-L1 fileciteturn27file0L1-L1 fileciteturn53file0L1-L1 fileciteturn60file0L1-L1

**Ranked action list**

1) **Choose one canonical FTS/index mechanism and delete the others**
Why it matters: It removes the largest duplicated subsystem and reduces operator confusion. fileciteturn51file0L1-L1
What it would simplify: Index stage semantics, search implementation count, maintenance surface. fileciteturn27file0L1-L1 fileciteturn53file0L1-L1 fileciteturn60file0L1-L1
Risk: High—could break compatibility with existing DBs or workflows that assume a specific indexing path. fileciteturn53file0L1-L1
How to validate: Create a DB fixture with messages, run ingest, verify (a) search returns same conversation IDs for a query, (b) FTS table is populated after parse, (c) rebuild/repair path still works.

2) **Fix drive/gemini content block integration (or explicitly demote it)**
Why it matters: Currently a “silent capability failure” that will poison semantic tooling and filters for at least one provider family. fileciteturn36file0L1-L1
What it would simplify: The semantic layer becomes consistent across providers, and SQL pushdowns become trustworthy. fileciteturn47file0L1-L1 fileciteturn51file0L1-L1
Risk: Medium—requires deciding the canonical representation (ParsedContentBlock vs provider_meta dict blocks) and ensuring persistence. fileciteturn32file0L1-L1 fileciteturn29file0L1-L1
How to validate: Add an ingestion test that parses a gemini/drive sample, then asserts content_blocks rows exist and `has_thinking`/`has_tool_use` flags match expectation.

3) **Split `sources/source.py` and relocate persistence glue**
Why it matters: This is the biggest cohesion failure; modularity improvements here will reduce refactor risk elsewhere. fileciteturn62file0L1-L1
What it would simplify: Reasoning about ingestion, testing provider parsing independently, reducing cyclic dependencies. fileciteturn62file0L1-L1 fileciteturn29file0L1-L1
Risk: Medium—large diff; easy to introduce regressions in traversal/zip/stream handling. fileciteturn62file0L1-L1
How to validate: Golden-file tests for (a) directory traversal skip rules, (b) JSONL grouped providers, (c) ZIP entry filtering, (d) provider detection parity.

4) **Make vector search either truly async-safe or explicitly offline**
Why it matters: Prevents event-loop blocking and clarifies operational cost/perf behavior. fileciteturn64file0L1-L1 fileciteturn59file0L1-L1
What it would simplify: Concurrency expectations and protocol correctness. fileciteturn61file0L1-L1
Risk: Medium to High—touches embedding workflows and external API usage. fileciteturn59file0L1-L1
How to validate: Add async performance tests (event loop responsiveness) plus functional tests for similarity search determinism.

5) **Normalize provider/source naming in raw ingestion tables**
Why it matters: Provider identity is foundational for schema selection, drift tracking, and parse routing; ambiguity here creates long-term correctness debt. fileciteturn24file0L1-L1 fileciteturn51file0L1-L1
What it would simplify: Less confusion about what `provider_name` means in raw rows; fewer “fix-ups” downstream. fileciteturn62file0L1-L1
Risk: Medium—requires migration or compatibility logic for existing DBs. fileciteturn51file0L1-L1
How to validate: Re-run acquisition/validate/parse on a mixed-provider test corpus; confirm validation provider/schema selection is stable.

**Confidence and unknowns**

Highly confident (directly evidenced by code):
- The real operational spine is pipeline stage orchestration + SQLite backend persistence + repository hydration. fileciteturn16file0L1-L1 fileciteturn50file0L1-L1 fileciteturn64file0L1-L1
- Search/index is duplicated across schema triggers, indexing service, sync providers, and search helpers. fileciteturn51file0L1-L1 fileciteturn27file0L1-L1 fileciteturn60file0L1-L1 fileciteturn53file0L1-L1
- Drive/Gemini structured metadata does not flow into persisted content blocks through the canonical prepare path. fileciteturn36file0L1-L1 fileciteturn29file0L1-L1
- Vector provider integration is sync-blocking inside async repository methods and protocol semantics are mismatched. fileciteturn59file0L1-L1 fileciteturn64file0L1-L1 fileciteturn61file0L1-L1

Lower confidence / incomplete due to missing mandatory local inspection:
- I could not inspect the runtime schema store under `/home/sinity/.local/share/polylogue/schemas` or the actual runtime DB under `/home/sinity/.local/share/polylogue`, so I cannot corroborate schema versioning behavior or real-world provider mix/drift frequency against produced artifacts.
- I did not deeply inspect `polylogue/rendering/*`, `polylogue/site/*`, `polylogue/showcase/*`, or `polylogue/mcp/*` beyond the fact that CLI wires them in; those areas may contain additional duplication or miswiring, but they weren’t examined at code depth here. fileciteturn40file0L1-L1
- I did not examine `flake.nix` packaging surfaces or the `sinnix` repo, so I cannot validate deployment/runtime boundary assumptions or whether something that looks “optional” in code is actually mandatory in deployment.

**What I would try to delete, merge, or flatten first**

- Delete or heavily demote `polylogue.storage.search_providers.fts5.FTS5Provider` if it is not a true runtime integration point (it duplicates FTS lifecycle already present elsewhere). fileciteturn60file0L1-L1
- Delete or demote the trigger-based FTS machinery *or* the explicit indexing service—pick one; keeping both is architectural debt. fileciteturn51file0L1-L1 fileciteturn27file0L1-L1
- Flatten `polylogue.storage.search` vs backend search logic so there is one canonical ranked conversation search builder and one execution style (async vs sync), not both. fileciteturn53file0L1-L1 fileciteturn52file0L1-L1
- Split `polylogue.sources.source` into traversal/streaming/dispatch modules and move `RecordBundle/save_bundle` to pipeline/storage to restore subsystem coherence. fileciteturn62file0L1-L1
- Remove `parse_drive_payload` if it is no longer a real call path (it looks like an alternate dispatch path overlapping with `parse_payload`). fileciteturn62file0L1-L1
- Remove the “old-format API” block in `pipeline.semantic` (or quarantine it behind a compatibility module) once no internal callers rely on it. fileciteturn31file0L1-L1
- Normalize and possibly rename raw-table provider fields so `raw_conversations.provider_name` cannot unintentionally become a “source name fallback.” fileciteturn51file0L1-L1 fileciteturn62file0L1-L1
- Make vector providers async-safe (or make embedding a separate offline stage) and update the protocol contract to match real semantics. fileciteturn64file0L1-L1 fileciteturn61file0L1-L1
- Remove duplicated schema initialization logic between sync and async schema paths if one is no longer used (currently schema DDL/vec0 logic exists in multiple places). fileciteturn51file0L1-L1 fileciteturn50file0L1-L1
- Consider flattening “facade vs services” composition by making the facade wrap `RuntimeServices`, reducing parallel wiring paths. fileciteturn37file0L1-L1 fileciteturn42file0L1-L1
