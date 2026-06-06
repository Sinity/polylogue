# Schema Inventory: Universal Archive Semantics vs Provider Metadata

<!-- Generated from code audit of schema DDL, parsers, materialization, insight projections, and mappers. -->
<!-- Ref #840. -->

## Purpose

Classify every schema field, `provider_meta` key, insight projection, and storage mapper across six axes. Each classification includes rationale and the owning follow-up issue or explicit "intentional" label.

## Axes

- **A1 — Already canonical**: Universal archive semantics modeled as first-class columns/rows.
- **A2 — Trapped universal**: Universal-ish semantics currently in `provider_meta` / JSON payloads / ad hoc projection slices.
- **A3 — Provider-specific (OK)**: Provider metadata that can remain in `provider_meta`.
- **A4 — Accidental provider leakage**: Provider-specific columns in universal tables, or generated columns bound to provider_meta.
- **A5 — Raw-only**: Wire artifacts that should live only in raw blobs.
- **A6 — Mapper repetition**: Repeated column lists, SQL tuple builders, row mappers where a local descriptor layer could reduce drift.

---

## 1. Session Table (`sessions`)

### A1 — Already canonical

| Field | Rationale |
|-------|-----------|
| `session_id` | Primary key. |
| `provider_name` | Provider enum string. Universal. |
| `provider_session_id` | Native ID within provider. |
| `title` | Canonical title. |
| `created_at`, `updated_at` | Timestamps. Universal. |
| `sort_key` | Numeric sort key for ordering. |
| `content_hash` | SHA-256 for idempotent write. |
| `metadata` | User-editable metadata (tags, summaries). Excluded from content hash. |
| `version` | Schema version tag. |
| `parent_session_id` | Continuation/sidechain/subagent parent ref. |
| `branch_type` | Enum: continuation, sidechain, fork, subagent. |
| `raw_id` | FK to `raw_sessions`. |

### A4 — Accidental provider leakage

| Field | Rationale | Issue |
|-------|-----------|-------|
| `source_name TEXT GENERATED ALWAYS AS (json_extract(provider_meta, '$.source')) STORED` | This is a generated column bound to `provider_meta`. Currently only two values in practice: "claude" and "codex". Should become a real stored column or reference a source identity table. | [#864](https://github.com/Sinity/polylogue/issues/864) |

### A2 — Trapped universal (in `provider_meta` JSON column)

| Key | Source parser(s) | Consumer(s) | Rationale | Issue |
|-----|------------------|-------------|-----------|-------|
| `source` | Injected by `_merged_session_provider_meta()` in `materialization_runtime.py:150` and `prepare_enrichment.py:58` | `source_name` generated column, CLI source selection | Universal: every session has a source. Currently forced through provider_meta | [#864](https://github.com/Sinity/polylogue/issues/864) |
| `cwd` | Not set by current parsers directly | `rebuild.py:107` via `json_extract(provider_meta, '$.cwd')`; `attribution.py:188` | Session working directory is a universal context fact. May be present from legacy data or specific providers. Read by insight rebuild. | [#864](https://github.com/Sinity/polylogue/issues/864) |
| `gitBranch` | Not set by current parsers | `rebuild.py:108` via `json_extract(provider_meta, '$.gitBranch')`; `attribution.py:199` | Git branch is universal session context. Same path as `cwd`. | [#864](https://github.com/Sinity/polylogue/issues/864) |
| `git` | Codex parser (`codex.py:443`): `{"repository_url": ..., "branch": ...}` | `rebuild.py:109` via `json_extract(provider_meta, '$.git')`; `attribution.py:203` | Git context is universal session metadata. | [#864](https://github.com/Sinity/polylogue/issues/864) |
| `working_directories` | Claude Code (`code_parser.py:251`), Codex (`codex.py:447`) | `attribution.py:192`; capability docs reference `Session.provider_meta.working_directories` | Working directories are universal session context. Currently a list of strings in provider_meta. | [#864](https://github.com/Sinity/polylogue/issues/864) |
| `total_cost_usd` | Claude Code (`code_parser.py:247`) | `SessionRuntimeMixin.total_cost_usd` (fallback); pricing code | Cost is a universal archive fact but currently parsed ad-hoc from provider_meta. | [#803](https://github.com/Sinity/polylogue/issues/803) |
| `total_duration_ms` | Claude Code (`code_parser.py:249`) | `SessionRuntimeMixin.total_duration_ms` (fallback) | Duration is a universal archive fact. | [#803](https://github.com/Sinity/polylogue/issues/803) |
| `models_used` | Claude Code (`code_parser.py:253`) | Capability docs only | Model metadata is universal but currently only captured for Claude Code. | [#803](https://github.com/Sinity/polylogue/issues/803) |

### A3 — Provider-specific (OK to stay in provider_meta)

| Key | Source parser | Rationale |
|-----|---------------|-----------|
| `default_model` | ChatGPT (`chatgpt.py:200`) | Provider-specific model slug. |
| `gizmo_id`, `gizmo_type` | ChatGPT (`chatgpt.py:202-206`) | ChatGPT GPT/gizmo identifiers. Provider-specific. |
| `is_archived` | ChatGPT (`chatgpt.py:209`) | Provider UI state flag. |
| `title_source` | Drive/Gemini (`drive.py:226`) | Hint about where title was derived from. Provider-specific. |
| `instructions` | Codex (`codex.py:445`) | Native Codex session instructions. Provider-specific text. |
| `display_label` | Various (read in `SessionRuntimeMixin.display_title`) | Provider display hint, used only as fallback label. |
| `session` / `capture` | Browser capture (`browser_capture.py:24-27`) | Raw capture metadata. Provider-specific. |

---

## 2. Messages Table (`messages`)

### A1 — Already canonical

| Field | Rationale |
|-------|-----------|
| `message_id` | Primary key. |
| `session_id` | FK to sessions. |
| `provider_message_id` | Native message ID within provider. |
| `role` | Normalized role string. |
| `text` | Canonical message text. FTS-indexed. |
| `sort_key` | Numeric ordering. |
| `content_hash` | SHA-256 for idempotency. |
| `version` | Schema version tag. |
| `parent_message_id` | Threading parent. |
| `branch_index` | Branch position. |
| `provider_name` | Denormalized for query performance. |
| `word_count` | Precomputed word count. |
| `has_tool_use`, `has_thinking`, `has_paste` | Precomputed boolean flags for filter pushdown. |
| `message_type` | Enum: message, tool_use, tool_result, thinking. |

### Note: No message-level `provider_meta`

The `messages` table intentionally does NOT have a `provider_meta` column. Message semantics are stored in `content_blocks`. The domain model `Message` has `provider_meta=None` (set in `hydrators.py:106`). However, the parser (`ParsedMessage`) does produce message-level `provider_meta` which is used during materialization to build `content_blocks` and precomputed flags, then discarded.

### A2 — Trapped universal (message-level, in ParsedMessage.provider_meta only — NOT persisted)

| Key | Source parser | Issue |
|-----|---------------|-------|
| `content_blocks` (in message provider_meta) | `model_runtime.py:179` reads `provider_meta.content_blocks` as fallback | [#839](https://github.com/Sinity/polylogue/issues/839) — make canonical content_blocks authoritative |
| `isSidechain`, `isMeta` | `model_runtime.py:182` reads for message_type fallback | [#839](https://github.com/Sinity/polylogue/issues/839) |
| `isThought` | `model_runtime.py:206` reads for thinking detection fallback | [#839](https://github.com/Sinity/polylogue/issues/839) |
| `raw` (message-level) | `model_runtime.py:148` reads `provider_meta.raw` for various fallback extraction | [#839](https://github.com/Sinity/polylogue/issues/839) — remove fallbacks once canonical rows are authoritative |
| `model` (ChatGPT message) | ChatGPT parser sets `meta["model"]` from `model_slug` | [#842](https://github.com/Sinity/polylogue/issues/842) — decide persisted contract for ChatGPT message metadata |

### A5 — Raw-only

| Content | Rationale |
|---------|-----------|
| `ParsedMessage.provider_meta["raw"]` (entire provider-native message dict) | Parser artifact used during materialization. Not persisted in archive messages. The complete wire payload is preserved in `raw_sessions`. |

---

## 3. Content Blocks (`content_blocks`)

### A1 — Already canonical

| Field | Rationale |
|-------|-----------|
| `block_id` | Primary key. |
| `message_id`, `session_id` | FKs. |
| `block_index` | Position within message. |
| `type` | Content block type (text, tool_use, tool_result, thinking, image, code). |
| `text` | Block text content. |
| `tool_name`, `tool_id` | Tool identity for tool_use blocks. |
| `tool_input` | Tool input as JSON string. |
| `metadata` | Extracted metadata as JSON string (carries `media_type` for image/document blocks since #1240). |
| `semantic_type` | Semantic classification enum. |

All fields are canonical and properly modeled. No provider_meta column — this table is clean.

---

## 4. Action Events (`action_events`)

### A1 — Already canonical

All fields are universal action semantics: `event_id`, `session_id`, `message_id`, `source_block_id`, `timestamp`, `sort_key`, `sequence_index`, `provider_name`, `action_kind`, `tool_name`, `normalized_tool_name`, `tool_id`, `affected_paths_json`, `cwd_path`, `branch_names_json`, `command`, `query_text`, `url`, `output_text`, `search_text`.

The table correctly models working directory (`cwd_path`) and branch names (`branch_names_json`) as first-class columns — these are the same semantics that remain trapped in session `provider_meta` for non-action-event contexts.

---

## 5. Provider Events (`provider_events`)

### A1 — Already canonical

`event_id`, `session_id`, `provider_name`, `event_index`, `event_type`, `timestamp`, `sort_key`, `payload_json`, `materializer_version`.

### A4 — Columns that may not be consistently populated

| Column | Issue |
|--------|-------|
| `source_message_id` | Not all parsers set this. Should be required or removed. |
| `raw_id` | Set by batch ingest but not universally available. Should document coverage. |

Decision: These columns are useful provenance when available but their nullable status is correct. Classified as intentional optional provenance. No follow-up issue needed unless a parser should set them and doesn't.

---

## 6. Attachments (`attachments`, `attachment_refs`)

### A1 — Already canonical

| Field | Rationale |
|-------|-----------|
| `attachment_id` | Primary key. |
| `mime_type`, `size_bytes`, `path` | Universal attachment facts. |
| `ref_count` | Reference counting for GC. |
| `ref_id`, `attachment_id`, `session_id`, `message_id` (attachment_refs) | Relationship linking. |
| `provider_attachment_id`, `provider_file_id`, `provider_drive_id` | First-class native identifier columns on both `attachments` and `attachment_refs`. Lookup queries resolve against these stored TEXT columns; no `json_extract` on the hot path. Landed by [#1252](https://github.com/Sinity/polylogue/issues/1252) (#864 slice B). |
| `upload_origin` | Closed vocabulary classification of how the attachment entered the archive (`drive` / `paste` / `url` / `oauth`, or NULL). Indexed via `idx_attachment_refs_upload_origin` for the attachment-library UI ([#1199](https://github.com/Sinity/polylogue/issues/1199)) grouping. Landed by [#1252](https://github.com/Sinity/polylogue/issues/1252). |

### A2 — Trapped universal (in `provider_meta` JSON)

Native identity indexes were dropped in [#1240](https://github.com/Sinity/polylogue/issues/1240) (audit found zero SQL usage) and the typed columns were promoted in [#1252](https://github.com/Sinity/polylogue/issues/1252) so the hot-path identity lookup (`search_attachment_identity_evidence_hits`) reads stored columns instead of `json_extract` against `provider_meta`. Nothing remaining trapped in this section.

### A3 — Provider-specific attachment metadata

The `provider_meta` column on attachments correctly holds provider-specific metadata (e.g. `attachment_kind`, raw provider document metadata from Drive). This is intentional. Native identifiers (`id`, `provider_id`, `fileId`, `driveId`) may still be mirrored in `provider_meta` for rendering fallbacks, but the canonical lookup surface is the typed columns above.

---

## 7. Session Stats (`session_stats`)

### A1 — Already canonical

`session_id`, `provider_name`, `message_count`, `word_count`, `tool_use_count`, `thinking_count`, `paste_count`. All are precomputed aggregate facts. Clean.

---

## 8. Session Insight Tables

### 8.1 `session_profiles`

#### A1 — Already canonical

First-class columns: `session_id`, `provider_name`, `title`, `first_message_at`, `last_message_at`, `canonical_session_date`, `repo_paths_json`, `repo_names_json`, `tags_json`, `auto_tags_json`, `message_count`, `substantive_count`, `attachment_count`, `work_event_count`, `phase_count`, `word_count`, `tool_use_count`, `thinking_count`, `total_cost_usd`, `total_duration_ms`, `engaged_duration_ms`, `tool_active_duration_ms`, `wall_duration_ms`, `cost_is_estimated`.

Typed payload fields: `evidence_payload_json`, `inference_payload_json`, `enrichment_payload_json` with corresponding `*_search_text` and version/family fields.

The table **properly** has `total_cost_usd`, `total_duration_ms`, `tool_active_duration_ms`, `wall_duration_ms` as first-class columns on the insight read model — these are derived during profile materialization, not read from provider_meta at query time.

#### A2 — Trapped universal (derivation path)

The insight rebuild SQL (`rebuild.py:107-109`) extracts `cwd`, `gitBranch`, `git` from `provider_meta` via `json_extract()` to pass through to `SessionRecord.provider_meta`, which then feeds into attribution and profile building. These three keys represent the only `provider_meta` dependency in the insight rebuild path, and they are used to build: `cwd_paths`, `branch_names`, `repo_paths`, `repo_names`.

**All four of these insight fields already have first-class columns** in `session_profiles` (`repo_paths_json`, `repo_names_json`) and `action_events` (`cwd_path`, `branch_names_json`). The `provider_meta` extraction is a **legacy bootstrapping path** for sessions that pre-date action_events. Owned by [#864](https://github.com/Sinity/polylogue/issues/864).

### 8.2 `session_work_events`, `session_phases`

#### A1 — Already canonical

All fields are derived insight columns. No `provider_meta` dependency.

### 8.3 `work_threads`

#### A4 (legacy payload pattern)

The `payload_json TEXT NOT NULL` field holds a `WorkThreadPayload` — this is a typed Pydantic model serialized to JSON, consistent with the `*_payload_json` pattern used by profiles. However, unlike profiles which split into `evidence_payload_json`/`inference_payload_json`/`enrichment_payload_json`, work_threads uses a single undifferentiated `payload_json`. This is a **legacy pattern**, not a bug. Classified as intentional for now; could be split in a follow-up if thread evidence/inference/enrichment needs separate search indexes.

### 8.4 `session_tag_rollups`

Clean. No `provider_meta` or legacy payload issues.

---

## 9. Raw/Provenance Tables

### 9.1 `raw_sessions`

All fields are raw provenance. `provider_name`, `payload_provider`, `source_name`, `source_path` are properly first-class. No provider_meta column. Clean.

### 9.2 `artifact_observations`

All fields are observation metadata. Clean. No provider_meta column.

---

## 10. Mapper Repetition Audit (A6)

### Current pattern

For each archive table, column lists are repeated in **5 locations**:

1. **DDL** (`schema_ddl_*.py`) — CREATE TABLE column order
2. **Pydantic Record model** (`runtime/*/records.py`) — typed field list
3. **Row mapper** (`queries/mappers_*.py`) — `_row_to_*()` functions
4. **Tuple builder** (`services/ingest_worker.py`) — `_*_tuple()` functions
5. **SQL statement** (`core/common.py`) — explicit column list in INSERT/UPSERT

Adding or renaming a column requires synchronized changes across all 5.

### Affected tables

| Table | DDL | Record | Mapper | Tuple | SQL |
|-------|-----|--------|--------|-------|-----|
| sessions | `schema_ddl_archive.py:45` | `archive/records.py:39` | `mappers_archive.py:41` | `ingest_worker.py:783` | `common.py:22` |
| messages | `schema_ddl_archive.py:78` | `archive/records.py:103` | `mappers_archive.py:64` | `ingest_worker.py:802` | `common.py:52` |
| content_blocks | `schema_ddl_archive.py:117` | `archive/records.py:72` | `mappers_archive.py:88` | `ingest_worker.py:827` | `common.py:85` |
| action_events | `schema_ddl_actions.py:5` | `action/records.py:11` | `mappers_archive.py:169` | `ingest_worker.py:1008+` | `common.py:115` |
| provider_events | `schema_ddl_provider_events.py:5` | `archive/records.py:186` | `mappers_archive.py:195` | `ingest_worker.py:893` | `common.py:125` |
| attachments | `schema_ddl_archive.py:181` | `archive/records.py:150` | (inline in rebuild.py) | `ingest_worker.py:916` | `common.py:133` |

### Assessment

A full ORM is the wrong abstraction for this archive (SQLite/FTS/generated columns/sync+async paths/bulk ingest). However, a **local table descriptor** that:

1. Declares column name, type, and position once per table
2. Generates DDL fragments, Record field validators, mapper assignments, tuple builders, and INSERT column lists from that single declaration

...would eliminate 4 of the 5 repetition points while keeping explicit SQL and full control over SQLite-specific features. The descriptor would be a small data structure, not an ORM.

### Recommendation

Do not pursue a table descriptor layer until at least one schema column promotion has been painful enough to demonstrate the concrete value. Track this as a decision, not an implementation issue. The repetition is a **smell**, not an active bug.

---

## 11. Cross-Cutting: `provider_meta` in Domain Models

### Session domain model

`Session.provider_meta: dict[str, object] | None` — this is the canonical interface. Readers access it for:
- Attribution (cwd, gitBranch, git, working_directories)
- Cost/duration fallback
- Display label
- Pricing
- Repo source filter

### Message domain model

`Message.provider_meta: dict[str, object] | None` — intentionally set to `None` by `message_from_record()`. Fallback readers (`model_runtime.py`) access `provider_meta` on the *domain model instance*, which for archived messages will be `None`. These fallbacks exist for:
- Content block extraction fallback
- Thinking detection fallback
- Sidechain/meta detection fallback
- Cost/token extraction
- ChatGPT message metadata

These fallbacks should be removed once canonical storage is authoritative. See [#839](https://github.com/Sinity/polylogue/issues/839).

### Attachment domain model

`Attachment.provider_meta` — used for provider-specific identity fields and metadata. The native identity keys (`id`, `provider_id`, `fileId`, `driveId`) should be promoted. The rest is provider-specific OK.

---

## 12. Cross-Cutting: `schemas/unified/provider_meta_*`

The unified harmonization layer (`schemas/unified/`) provides:
- Extraction from provider_meta to `HarmonizedMessage` (`extract_from_provider_meta`)
- Content block coercion (`_coerce_content_blocks`)
- Tool call and reasoning trace extraction
- Fallback handling for malformed data

This layer is **currently useful as parser/compatibility glue**, but should not become the durable semantic API after materialization. After canonical fields exist for all universal semantics, this layer should shrink to only handle runtime extraction from raw provider payloads during parsing — not serve as a permanent runtime query path for already-materialized data.

Tracked in [#864](https://github.com/Sinity/polylogue/issues/864) (shrink after canonical fields exist).

---

## 13. Closeout Matrix

| Semantic bucket | Status | Owning issue |
|-----------------|--------|-------------|
| `source_name` (generated column → real column) | Open | [#864](https://github.com/Sinity/polylogue/issues/864) |
| `cwd` / `working_directories` → typed session context | Open | [#864](https://github.com/Sinity/polylogue/issues/864) |
| `gitBranch` / `git` → typed session context | Open | [#864](https://github.com/Sinity/polylogue/issues/864) |
| Attachment native identity (`id`, `provider_id`, `fileId`, `driveId`) → columns | Open | [#864](https://github.com/Sinity/polylogue/issues/864) |
| `provider_meta` allowlist / canonical-vs-provider boundary | Open | [#864](https://github.com/Sinity/polylogue/issues/864) |
| ChatGPT message metadata persisted contract | Open | [#842](https://github.com/Sinity/polylogue/issues/842) |
| Context/protocol artifacts stored as rows/types | Open | [#839](https://github.com/Sinity/polylogue/issues/839) |
| Message-level `provider_meta` fallback removal | Open | [#839](https://github.com/Sinity/polylogue/issues/839) |
| Model/cost/duration read model | Open | [#803](https://github.com/Sinity/polylogue/issues/803) |
| `provider_meta`-driven insight rebuild bootstrapping | Open | [#864](https://github.com/Sinity/polylogue/issues/864) |
| Write bundle/hash honesty | Open | [#841](https://github.com/Sinity/polylogue/issues/841) |
| Provider schema regeneration | Open | [#800](https://github.com/Sinity/polylogue/issues/800) |
| ChatGPT gizmo/model fields | Provider-specific | Intentional |
| Drive/Gemini `title_source` | Provider-specific | Intentional |
| Codex `instructions` text | Provider-specific | Intentional |
| Browser capture `session`/`capture` metadata | Provider-specific | Intentional |
| `display_label` fallback | Provider-specific | Intentional |
| `is_archived` (ChatGPT UI state) | Provider-specific | Intentional |
| `provider_events.source_message_id` / `raw_id` | Optional provenance | Intentional |
| `work_threads.payload_json` (legacy single payload) | Legacy, intentional | None (revisit if thread search needs split) |
| Mapper/tuple/DDL repetition (table descriptor) | Decision deferred | This issue (#840) |

## Provider-Meta Allowlist (#864)

After archive0 promotion, the following classification applies to all `provider_meta` keys:

### Promoted to canonical columns
| Key | Canonical column | Status |
|-----|-----------------|--------|
| `source` | `sessions.source_name` | ✓ Populated during materialization (#884, #909) |
| `working_directories` | `sessions.working_directories_json` | ✓ Populated + backfill (#909, #925) |
| `cwd` | `sessions.working_directories_json` | ✓ Populated (single-element array) |
| `gitBranch` | `sessions.git_branch` | ✓ Column exists, needs backfill |
| `git.repository_url` | `sessions.git_repository_url` | ✓ Column exists, needs backfill |
| `git.branch` | `sessions.git_branch` | ✓ Column exists, needs backfill |
| `id` (attachments) | `attachments.provider_attachment_id` | ✓ Populated during materialization |
| `provider_id` (attachments) | `attachments.provider_file_id` | ✓ Populated during materialization |
| `fileId` (attachments) | `attachments.provider_file_id` | ✓ Populated during materialization |
| `driveId` (attachments) | `attachments.provider_drive_id` | ✓ Populated during materialization |

### Retained as provider-specific (OK in provider_meta)
- `display_label`, `instructions`, `is_archived`, `title_source`, `gizmo_id`, `gizmo_type`
- `moderation`, `safe_urls`, `blocked_urls`, `internal_status_flags`
- Browser capture `session`/`capture` metadata

### Raw-only (exist in raw_sessions, not needed in canonical tables)
- `raw` — full wire payload, preserved in raw_sessions + blob store

### Parser-side typed surface (in flight, #864)

`ParsedSession` now exposes typed `working_directories`, `git_branch`,
and `git_repository_url` fields populated by the Claude Code, Codex, and
Claude session-index parsers in addition to the legacy `provider_meta`
entries. Storage writes, attribution
(`polylogue/archive/session/attribution.py`), and insight rebuild
(`polylogue/storage/insights/session/rebuild.py`) still read from
`provider_meta` for these fields; switching those readers to the typed
parser surface and the matching `SessionRecord`/canonical columns is
the next graduation step under #864.

### Deferred to owning issues
- Model/token/cost/duration → #803
- ChatGPT message-level metadata → promoted to content_block metadata (#842)
- Action event cwd/branch/path context → #866 (lineage graph)
| Full ORM adoption | Rejected | This issue (#840) |
