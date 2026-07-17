[← Back to README](../README.md)

# Data Model

This page describes the **public Python domain models** returned by the library
API, CLI, and MCP surfaces. These are read models hydrated from the archive;
for the on-disk storage shape see [Schema](schema.md), and for the conceptual
rings see [Architecture](architecture.md). The authoritative field definitions
live in `polylogue/archive/session/domain_models.py`,
`polylogue/archive/message/models.py`, and
`polylogue/archive/attachment/models.py` — read those if a field here is
ambiguous.

## Origin, not provider

Public read surfaces are keyed by **`origin`** (the `Origin` enum in
`polylogue/core/enums.py`): `claude-code-session`, `claude-ai-export`,
`chatgpt-export`, `codex-session`, `gemini-cli-session`, `aistudio-drive`,
`hermes-session`, `antigravity-session`, `unknown-export`. The provider-wire
`Provider` enum (`chatgpt`, `claude-code`, …) is retained only at the
parsing/schema boundary and is not the public filter token. Filter and query
surfaces use `origin`.

## Session

`Session` (and its message-less sibling `SessionSummary`) is the primary read
entity.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `SessionId` (str) | Composite ID, `origin:native_id` |
| `origin` | `Origin` | Source-origin token (see above) |
| `title` | `str?` | Parsed session title |
| `created_at` | `datetime?` | Creation timestamp |
| `updated_at` | `datetime?` | Last update timestamp |
| `messages` | `MessageCollection` | Eagerly or lazily materialized messages (`Session` only) |
| `metadata` | `dict[str, object]` | User-metadata overlay (see below) |
| `tags_m2m` | `tuple[str, ...]` | Tags hydrated from the `session_tags` table |
| `working_directories` | `tuple[str, ...]` | Working dirs observed for the session |
| `git_branch` | `str?` | Git branch, when known |
| `git_repository_url` | `str?` | Git remote URL, when known |
| `parent_id` | `SessionId?` | Parent session (continuation/fork/sidechain/subagent) |
| `branch_type` | `BranchType?` | `continuation`, `sidechain`, `fork`, `subagent` |
| `session_events` | `tuple[SessionEvent, ...]` | Structured session events (e.g. compaction) — `Session` only |
| `attachments` | `list[Attachment]` | Session-level attachments not bound to a message |

`SessionSummary` carries the same identity/metadata fields plus precomputed
`message_count` and `dialogue_count`, but omits `messages`, `session_events`,
and `attachments`.

### Metadata overlay and derived properties

`metadata` is the user-owned key/value overlay projected from `user.db`
metadata assertions. It holds the user title override, a user/LLM summary, and
any custom keys. It is **excluded from the content hash**, so editing it never
triggers re-import. Tags are not stored in this overlay: user tags are tag
assertions, while auto-tags are rebuildable `index.db.session_tags` rows and
surface through `tags_m2m`.

Convenience properties resolve these:

- `display_title` → `metadata["title"]` (user override) if set, else `title`, else `id[:8]`.
- `tags` → `tags_m2m` when hydrated, else any tags in `metadata`.
- `summary` → `metadata["summary"]`.
- `is_continuation` → `branch_type == continuation`.

## Message

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Message ID, `session_id:native_id` (or `session_id:position.variant`) |
| `role` | `Role` | `user`, `assistant`, `system`, `tool`, `unknown` |
| `text` | `str?` | Flattened message text |
| `timestamp` | `datetime?` | Message timestamp |
| `message_type` | `MessageType` | `message`, `summary`, `tool_use`, `tool_result`, `thinking`, `context`, `protocol` |
| `material_origin` | `MaterialOrigin` | Authoredness/material source, separate from provider role: `human_authored`, `assistant_authored`, `operator_command`, `runtime_protocol`, `runtime_context`, `tool_result`, `generated_context_pack`, `generated_analysis_pack`, `unknown`. `generated_context_pack` is a persisted legacy/source-marker value for provider-generated context bundles; renaming it is a schema-bump/rebuild decision, not a docs-only cleanup. |
| `provider` | `Provider?` | Provider-wire identity (parse-boundary only; prefer the session `origin`) |
| `content_blocks` | `list[dict]` | Structured content blocks (text/thinking/tool_use/tool_result/image/code/document) |
| `attachments` | `list[Attachment]` | File attachments referenced by the message |

`role` preserves provider/display envelope truth. It must not be used as a
proxy for human-authored prose: Claude Code, for example, carries command
wrappers, task notifications, provider-generated context bundles, and tool-result
protocol envelopes through `role=user`. Use `material_origin` for authoredness
accounting and prose projection.
| `parent_id` | `str?` | Parent message, for branching |
| `branch_index` | `int` | Branch position among sibling variants |
| `has_tool_use` / `has_thinking` / `has_paste_evidence` | `bool` | Precomputed content/evidence flags projected from storage |
| `input_tokens` / `output_tokens` | `int` | Token counts, when reported |
| `cache_read_tokens` / `cache_write_tokens` | `int` | Cache token counts, when reported |
| `duration_ms` | `int` | Reported generation duration |
| `model_name` | `str?` | Model that produced the message |

Cost and per-model token rollups are **not** message-level properties. They are
materialized at the session level in `session_model_usage`,
`session_reported_costs`, and `session_profiles`, and surfaced through the
session insight reads (e.g. `session_costs`, `cost_rollups`). See
[Architecture § Derived Read Models](architecture.md#2-derived-read-models).

### Semantic classification

`Message` exposes derived boolean properties (in
`polylogue/archive/message/model_runtime.py`):

| Property | Meaning |
|----------|---------|
| `is_user` / `is_assistant` / `is_system` | Role-based |
| `is_dialogue` | User or assistant turn |
| `is_tool_use` | Tool call or result |
| `is_thinking` | Reasoning/thinking content |
| `is_context_dump` | Pasted file content / context dump |
| `is_protocol_artifact` | Provider protocol noise |
| `is_noise` | Tool use, context dump, protocol, or system |
| `is_substantive` | Real dialogue (not noise, not thinking) |
| `is_branch` | Has a parent message (non-linear) |

## Attachment

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Attachment ID |
| `name` | `str?` | Filename / display name |
| `mime_type` | `str?` | MIME type |
| `size_bytes` | `int?` | Byte size |
| `path` | `str?` | Local path, if downloaded |
| `source_url` | `str?` | Upload/source URL, if known |
| `caption` | `str?` | Caption, if provided |

Attachments are content-addressed in the blob store and joined to messages
through `attachment_refs`. For Drive/Gemini sources, attachment lookup keys on
explicit identity fields preserved at ingest (the provider attachment id, plus
`fileId`/`driveId`); those identifiers are indexed for exact lookup and search
reports the match as attachment evidence.

## Branching and topology

Sessions form trees and cross-session lineages: continuations, forks,
sidechains, and subagent sessions. Within a session, `Message.parent_id` links
messages into a branch tree, while the materialized `messages.text` is the
flattened active-path content. Across sessions, parent/child references are
persisted as typed rows in `session_links` (even when the parent has not been
ingested yet), and the resolved logical root is materialized as
`session_profiles.logical_session_id`. See
[Internals § Topology Edges](internals.md#topology-edges-1258) and
[Internals § Logical Session Identity](internals.md#logical-session-identity-866).

## Tags

Tags support `key:value` notation for namespacing:

```
important              # Simple tag
repo:polylogue         # Namespaced
status:wip             # Namespaced
```

User tags are assertion rows in `user.db`; auto-tags remain heuristic
`session_tags` rows in `index.db`. The user side is irreplaceable durable
overlay state; the auto side is rebuildable read-model state.

## Typed annotations and seed ontologies

Typed labels reuse the user-tier assertion substrate rather than a parallel
annotation store. `annotation_schemas` holds immutable, versioned construct
definitions; `annotation_batches` records label-run provenance; each label is
an `assertions` row with `kind="annotation"`. Agent-authored labels are always
candidate-scoped and non-injected until an operator judgment accepts them.
Schema vocabulary can therefore grow by inserting immutable registry rows;
the five v1 seed families do not change the user-tier DDL or
`USER_SCHEMA_VERSION`.

| Schema | Grain | Required construct fields | Authority model |
|--------|-------|---------------------------|-----------------|
| `seed.activity@v1` | session, phase, message, block | `activity`, `confidence` | Evidence-linked candidate label; activities are debugging, design, implementation, research, writing, ideation, ops, or procurement. |
| `seed.goal-event@v1` | message, block, work event, observed event | `event_type`, `goal_ref`, `declared_by_ref`, `declaration_authority`, `confidence` | Prospective actor declaration only. Events are opened, blocked, resumed, declared resolved, superseded, or explicitly abandoned. Inactive unresolved goals are a separate, horizon-bound goal-graph derivation and are never emitted as abandonment annotations. |
| `seed.outcome-evidence@v1` | session, work/observed event, commit, check run, pull request, delegation | `outcome_type`, `authority`, `authority_ref`, `temporal_mode`, `confidence` | Structural, rule, and judged evidence remain distinct. Historical backfill is explicit. Outcomes are test passed, commit observed, deployment observed, user accepted, answer declared, or unknown. |
| `seed.knowledge-artifact@v1` | session, message, block, work event, assertion | `artifact_type`, `statement`, `authority`, `authority_ref`, `confidence` | Decision, lesson, preference, fact candidate, fact established, or commitment under a named agent, actor, structural, rule, or operator authority. |
| `seed.reusability@v1` | session, phase, message, block, work event, assertion | `purpose`, `worthy`, `authority`, `authority_ref`, `confidence` | Purpose-specific snippet, recipe, or demo judgment; agent labels remain candidates and operator judgments are explicit. |

Every seed also permits optional `abstain` and `rationale` fields and requires
evidence refs. A schema row is identified by `(schema_id, schema_version)` and
its canonical definition fingerprint; a conflicting re-registration fails
closed rather than mutating the existing definition.

### Governed archive-local bootstrap

Informal tags and affinity scores may nominate an archive-specific draft
schema, but cannot activate it. A nomination is an agent-authored
`ontology_candidate` assertion with `inject:false`. It separately records tag
affinity, nomination confidence, classifier and definition, version crosswalk,
analysis frame and epoch, content/action-pattern/temporal-cost/outcome view
proposals, cross-view disagreement, residue, rare samples, evidence, and privacy
exclusions.

An operator then accepts, renames, splits, or rejects the candidate. The
preserve/read/write sequence runs under `BEGIN IMMEDIATE`; the judgment,
immutable active schema row or rows, and typed `ontology_governance` receipt
are committed atomically. Acceptance does not itself manufacture formal
ontology labels. A subsequent annotation batch writes labels against the
active schema, and those agent-authored rows remain candidates until separately
judged.

Batch annotators consume `AnnotationBatchImportRequest` and
`import_annotation_batch` (also exposed by the archive facade). The importer
resolves target and evidence refs, verifies the durable schema fingerprint,
persists batch provenance, and writes through the assertion chokepoint inside
`BEGIN IMMEDIATE`. This same interface supports built-in seed schemas and
operator-promoted archive-local schemas without adding them to the process-wide
registry.

---

**See also:** [Schema](schema.md) · [Library API](library-api.md) · [CLI Reference](cli-reference.md) · [Configuration](configuration.md)
