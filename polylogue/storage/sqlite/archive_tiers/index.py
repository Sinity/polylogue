"""Archive parsed/search DDL fragment for archive."""

from __future__ import annotations

from polylogue.archive.topology.edge import topology_status_composes_sql, topology_status_excluded_sql
from polylogue.storage.fts.sql import (
    FTS_BULK_SESSION_WRITE_GUARD,
    FTS_MESSAGES_IDENTITY_TABLE_SQL,
    FTS_MESSAGES_TABLE_SQL,
    FTS_TRIGGER_DDL,
    FTS_UNICODE_TOKENIZER,
)
from polylogue.storage.sqlite.action_pairs import action_pairs_refresh_sql
from polylogue.storage.sqlite.archive_tiers.archive_tiers_specs import TABLE_SPECS
from polylogue.storage.sqlite.delegation_facts import delegation_facts_insert_sql

# polylogue-2qx.4: v46 lands the unread-wire batch (polylogue-cgfy/cuxz.8/
# 9x22 field-landing decisions):
#   - messages.stop_reason (StopReason) -- 608,608 Claude assistant messages
#     carry this on the wire; three derived outcome columns guess at the same
#     fact today and are 85-99% 'unknown'.
#   - blocks.tool_result_outcome_unknown_reason (ToolResultUnknownReason) --
#     distinguishes "provider emitted nothing" / "parser distrusts it" /
#     "parser doesn't read this origin's field" instead of one flat NULL.
#   - sessions.display_name, sessions.run_settings_json -- subagent slug
#     display name and per-session provider run-config (aistudio-drive
#     temperature/topP/topK/... ), landed as a JSON column since decomposing
#     it into columns would couple the schema to one provider.
#   - session_links.parent_tool_use_block_id -- the real join-key column
#     replacing delegation_facts' cardinality-gated ordinal dispatch<->child
#     pairing (parentToolUseID, 842,819 records on the wire).
#   - file_edits (new table) -- structuredPatch/originalFile/oldString/
#     newString/filePath/userModified/replaceAll, keyed by the tool_use
#     block that made the edit (105,123 structured diffs, 92,313 pre-edit
#     captures, currently 100% discarded).
#   - session_refs (new table) -- pr-link and future issue-tracker
#     references (20,702 occurrences), tracker-agnostic by design.
#
# Every one of these values depends on parser semantics to populate
# correctly (a shape-only copy-forward would leave every row NULL, exactly
# the v42/v44/v45 precedent) -- SEMANTIC_REPARSE is the honest
# classification, matching those versions, and routes through
# `polylogue ops reset --index && polylogued run` rather than a fast-forward.
#
# index.db is rebuildable derived state, but "rebuildable" is not the same as
# "always rebuilt": every bump above INDEX_FAST_FORWARD_COMPATIBILITY_FLOOR
# must declare its delta class in storage/sqlite/lifecycle.py, and a declared
# non-semantic delta upgrades an existing generation in place through
# index_fast_forward_plan()/apply_index_fast_forward() on connect. Only a
# SEMANTIC_REPARSE delta -- one whose result depends on parser semantics --
# routes to `polylogue ops reset --index && polylogued run`.
#
# A bump without a declaration is a policy violation, not a free rebuild:
# `devtools verify schema-versioning` fails, and the archive silently
# falls back to full raw replay. See polylogue-9rw0 / polylogue-b5l.
#
# polylogue-o4j2: v47 adds sessions.pending_drafts_json -- aistudio-drive's
# chunkedPrompt.pendingInputs non-blank entries (unsent textbox drafts, 7/397
# real sessions with draft text on the live archive). Landed as a session-row
# field rather than a session_event on purpose: a draft is mutable CURRENT
# state (edited in place, gone entirely once submitted), and session_events
# participate in session_revision_projection's append-only comparison axes
# (polylogue-aggz Invariant 1) -- putting mutable state there reproduces the
# exact defect class polylogue-bu1i (acquisition state) and polylogue-nuec
# (provider-remeasurement) were fixed for. See sessions table comment.
#
# polylogue-lzh8: v48 declares -- retroactively, with no DDL change of its
# own -- the semantic classification change 1e0246d77 (#3088, "admit Claude
# Workflow artifacts through OriginSpec", merged 2026-07-18) already shipped
# without a version bump. That commit changed origin_specs.py's artifact
# rules so workflow_run_snapshot / workflow_journal / agent_sidecar_meta /
# adopt_manifest paths correctly classify as parse_policy="fact"
# (parse_as_session=False) instead of silently defaulting to a session, the
# same v42/v44/v45/v46 "values depend on parser semantics" precedent -- a
# shape-only fast-forward cannot un-classify already-materialized rows, so
# there is nothing here for a clone-safe SQL delta to do. Measured live
# impact at declaration time (index.db, 2026-07-31, read-only query): 5,193
# zero-message claude-code-session rows total, of which 172 join to a
# `workflows/` artifact family under the pre-fix classification (164
# agent_sidecar_meta, 7 workflow_run_snapshot, 1 other), all acquired
# 2026-07-14..07-26 -- entirely before the deployed daemon build contained
# 1e0246d77 (sinnix's polylogue pin only advanced past it 2026-07-29). Like
# every other SEMANTIC_REPARSE version here, resolving this requires
# `polylogue ops reset --index && polylogued run`, which an agent must not
# trigger against the operator's live archive without explicit scheduling.
# v49 batches two related cost-accounting fixes from the same forensic audit
# (polylogue-shnc, polylogue-gt1z) onto one bump, matching the v46 "one bump
# covering every field" precedent:
#
#   - polylogue-gt1z: sessions.reported_cost_usd -- the exact provider-
#     reported session total already parsed onto ParsedSession
#     (claude-code-session/hermes-session) but previously discarded at write
#     time (the session_reported_costs table that once mirrored it was
#     dropped in polylogue-v2mg as a zero-consumer table;
#     ParsedSession.reported_cost_usd itself stayed a legitimate
#     parsed-domain field -- see write.py's _seed_session_model_usage_rows
#     docstring). This gives `_session_level_estimate`/`estimate_session_cost`
#     a real value to read so the ``status == "exact"`` cost path (audited
#     polylogue-gt1z: zero production callers behind it, a phantom-verified
#     feature pinned only by hand-built-payload tests) becomes reachable
#     through a real caller -- `polylogue/insights/cost_enrichment.py`'s
#     `enrich_session_cost_insight`, which backs the `insights costs` /
#     `SessionCostInsight` surface.
#   - polylogue-shnc: two new session_model_usage CHECK constraints enforcing
#     that 'priced' implies cost_usd/priced_with are both set, and
#     'origin_reported' implies cost_usd is set -- see that column's comment
#     for the forensic evidence (5,016 self-contradictory 'priced' rows,
#     3,417 NULL-cost 'origin_reported' rows) and write.py's
#     _price_provider_usage_tokens for the writer-side fix that makes the
#     invariant hold going forward.
#
# Both depend on parser/pricing semantics to populate/repair existing rows --
# SEMANTIC_REPARSE, matching the v42/v44/v45/v46/v47/v48 precedent above.
#
# polylogue-vf9x: v50 adds blocks.signature (nullable TEXT) and fixes two
# independent reasoning/thinking-content-loss defects found via a full-corpus
# audit:
#   - Claude Code (base_support.py `content_blocks_from_segments`): since
#     roughly 2026-06 the wire ships THINKING segments with an empty
#     `thinking` body and a `signature` only. An `if text:` guard dropped the
#     whole block, silently zeroing `thinking_count` archive-wide for every
#     2026-06+ session -- a false "reasoning declined" signal, not a real
#     absence. The block is now always recorded (text=NULL when the wire
#     carries none, signature captured when present).
#   - Codex (codex.py): standalone `reasoning` response_item records were
#     read only by the generic session_event compactor, which has no
#     `reasoning`-specific branch -- summary/content were never read at all
#     (not merely char-counted), so 100% of Codex reasoning text was
#     discarded and unreachable from FTS/search. `reasoning` records are now
#     materialized as a THINKING-block message (summary text -- the ~24%
#     recoverable case -- or content text when present; text=NULL when only
#     encrypted_content survives).
# Both are pure parser-semantics changes over the same already-declared
# `blocks.block_type='thinking'` vocabulary plus one additive nullable
# column -- the v42/v44/v45/v46/v48/v49 "values depend on parser semantics,
# no clone-safe SQL delta" precedent. `signature` is deliberately excluded
# from `_block_content_hash` and the lineage prefix signature (write.py)
# because providers re-sign on every replay; including it would break
# citation-anchor and fork-prefix matching for otherwise-identical replayed
# content. Resolving existing rows (recovering historical thinking/reasoning
# content) requires `polylogue ops reset --index && polylogued run` --
# deliberately NOT executed by this declaration.
#
# polylogue-u6tl: v51 wires delegation_facts.mapping_state/result_status onto
# the `literal_check` generator (storage/sqlite/archive_tiers/common.py),
# which previously had zero call sites despite CLAUDE.md documenting it by
# name as the mechanism that keeps `typing.Literal` types and their SQL CHECK
# lists in lockstep. Both columns already only ever receive values from their
# typed counterparts (DelegationMappingState / DelegationResultStatus in
# archive_tiers/archive.py) at the sole writer (delegation_facts_insert_sql);
# this only adds the CHECK the Python type already implied. Measured on the
# live archive (index.db, read-only, 2026-07-31) before declaring this
# version: mapping_state in {edge_only, resolved, unresolved} (0 rows outside
# the 4-value vocabulary), result_status in {error, unknown} (0 rows outside
# the 3-value vocabulary) -- a copy-forward rejects nothing. CONSTRAINT_ONLY,
# matching the v33/v36 precedent (widening/adding a CHECK over unchanged
# values), not SEMANTIC_REPARSE.
# polylogue-rlvj: v52 adds a table-level CHECK to fts_freshness_state --
# state='ready' now requires missing_rows=0 AND excess_rows=0 AND
# duplicate_rows=0 AND source_rows=indexed_rows in the same row. Surface-
# coherence audit 2026-07-31 found a live archive reporting messages_fts
# ready=true / missing_rows=0 while an independent count showed 12,659
# blocks missing from the index: a targeted (session-scoped) repair's
# correct scoped verdict was being written into the single global freshness
# row as an unconditional READY, discarding whatever accurate missing_rows
# an earlier exact snapshot had recorded (daemon/convergence_stages.py's
# `_mark_message_fts_ready_after_targeted_repair`, fixed in the same change
# to always source its row from the real archive-wide invariant, never a
# cheap existence check). The CHECK makes the specific contradiction that
# incident exhibited unrepresentable going forward: any writer that tries to
# assert `ready` alongside a nonzero counter now fails at the database layer
# instead of silently lying to every reader of the ledger. Existing rows are
# already consistent with this shape (every writer that reaches `state=ready`
# already zeroes/aligns these counters -- audited across every
# `record_fts_surface_state_sync`/`_async` call site for this change), so
# this is a pure constraint widening: CONSTRAINT_ONLY, REPLACE_TABLE on
# fts_freshness_state, no values change and no reparse. A pre-existing
# archive can already carry a row that violates this (the very bug being
# fixed) -- the fast-forward executor's `_REPLACE_TABLE_SANITIZERS` entry
# downgrades any such row to 'stale' before the copy runs rather than
# aborting the migration.
#
# polylogue-jc4q: v53 changes how Claude Code identity is derived for a
# resume/fork/usage-limit boundary carryover -- a run of records physically
# stamped with an ANCESTOR session's sessionId even though it is not that
# ancestor's own content (sources/dispatch.py's
# `_claude_code_grouped_record_specs` / `_claude_code_stream_sessions`, plus
# `sources/parsers/claude/code_parser.py`'s `trust_fallback_id`). Before this
# version, such a carryover composed its `provider_session_id` from the bare
# ancestor id, so every fork/resume/quirk descendant of one ancestor collided
# on that ancestor's own `logical_source_key` -- measured live: only 56/185
# (30.3%) of claude-code-session ambiguous revision cohorts resolved cleanly,
# far below chatgpt-export/claude-ai-export (>90%) after the sibling
# polylogue-aggz containment fix. This changes `sessions.native_id` (a
# generated identity column) for every affected raw acquisition and the
# `parent_session_id`/`session_links` lineage edges recorded for it --
# SEMANTIC_REPARSE, not a free fast-forward: only re-parsing already-acquired
# raw evidence recovers the corrected identity split. `polylogue ops reset
# --index && polylogued run` is required; deliberately NOT executed by this
# declaration.

# PR #3537: tightened `claude.looks_like_ai` (polylogue/sources/parsers/
# claude/ai_parser.py) to require at least one `chat_messages` entry
# carrying a role/sender field plus a text/content field, instead of
# accepting a bare (even empty) `chat_messages` key -- the same
# positive-evidence-required shape as PR #3428's `looks_like_code` fix.
# This moves the classifier's decision boundary for identical input
# bytes: some payloads previously admitted as claude-ai-export sessions
# are now refused. SEMANTIC_REPARSE, not a free fast-forward: only
# re-parsing already-acquired raw evidence applies the corrected
# admission decision to existing rows. `polylogue ops reset --index &&
# polylogued run` is required; deliberately NOT executed by this
# declaration.
#
# polylogue-0cn3 / polylogue-5dfu: v55 bundles a derived-tier vocabulary
# cleanup across two related columns:
#  - sessions.title_source/title_ref/title_confidence drop the COALESCE
#    ratchet in write.py's session upsert (`= excluded.X` instead of
#    `COALESCE(excluded.X, sessions.X)`) -- a purely derived provenance
#    triple should always reflect the current parse, not preserve a stale
#    verdict. TitleSource.UNKNOWN is also deleted: it was a second, redundant
#    spelling of "no title evidence" on an already-nullable column (every
#    read site branching on title_source already treated NULL and 'unknown'
#    identically), so parsers now leave title_source unset instead of
#    stamping UNKNOWN. TitleSource.USER is deleted too (zero producers,
#    write- or read-time). TitleSource.PATH is KEPT for compatibility with
#    legacy rows, but current read-time display-label projection leaves
#    title_source NULL when a session has neither a real title nor a display
#    name.
#  - session_links.link_type drops LinkType.REPAIRED (zero producers, and a
#    same-string collision with the unrelated TopologyEdgeStatus.REPAIRED on
#    the `status` column of the same table). FORK and RESUME are kept
#    despite also being 0 rows live: FORK has a real producer
#    (sources/parsers/hermes_state.py's `_branch_type`, just never yet hit
#    by an ingested session) and RESUME is the documented cross-repo
#    "resume lineage edge" fixture in docs/material-protocol-v1.md, expected
#    from the Sinex side (sinex-4j2.1.1) once implemented.
#  - session_links.status (TopologyEdgeStatus) narrows from 4 declared
#    members to the 2 the DDL CHECK and `_status_value` already only ever
#    stored (REPAIRED/QUARANTINED) -- UNRESOLVED/RESOLVED were never
#    constructed outside a Pydantic field default; resolvedness is already
#    carried by resolved_dst_session_id IS NOT NULL.
# Both title_source's CHECK and status's CHECK are also switched from a
# hand-written literal IN (...) list to the generated `nullable_check()`
# form, closing the CLAUDE.md-documented drift gap ("most hand-written
# CHECK(col IN (...)) lists ... still have no generator tie"). A live archive
# can carry `title_source='unknown'` today (14,915 rows measured against
# index.db user_version=46 before this bead's fix) or `link_type='repaired'`
# rows the new, narrower CHECKs would reject on a plain copy-forward, and
# the write-path behavior for title_source genuinely changes (no longer
# ever recomputes 'unknown') -- SEMANTIC_REPARSE, not a free fast-forward.
# `polylogue ops reset --index && polylogued run` is required; deliberately
# NOT executed by this declaration.

# polylogue-fuky: v56 stops materializing `agent_reasoning` rows into
# `session_events` for Codex sessions. Producer/consumer audit found 318,474
# live rows under this wire name with zero code reference anywhere in the
# repo; reading raw wire samples across three sessions confirmed
# `agent_reasoning.text` is a live-streamed echo of the same reasoning-
# summary bullet text the paired `reasoning` record already carries in full
# (`summary[].text`) and that `reasoning` is already materialized as a
# THINKING-block message (v50) -- `agent_reasoning` brings no incremental
# evidence, matching the v42 `agent_message` precedent (a twin
# `ParsedMessage` already exists). See
# `storage/sqlite/archive_tiers/write.py`'s `_SESSION_EVENTS_REDUNDANT_TYPES`
# comment for the full writeup and `sources/parsers/codex.py`'s
# `_CODEX_KNOWN_RESPONSE_ITEM_TYPES` for the sibling classification of every
# other previously-unaudited Codex response_item/event_msg type (those keep
# passing through unchanged -- only `agent_reasoning` is dropped). Like v42,
# this is a writer-materialization change with no DDL delta on
# `session_events` -- SEMANTIC_REPARSE, not a free fast-forward: only
# re-parsing recovers the dropped rows on already-acquired raw evidence.
# `polylogue ops reset --index && polylogued run` is required; deliberately
# NOT executed by this declaration.

# polylogue-ioz7: adds `agent_meta_sidecar_purge_receipts`, an immutable
# per-row audit record for the retired one-shot sidecar purge actuator
# -- the targeted actuator that deletes the ~4,945 pre-2026-07-28
# `agent-<hash>.meta.json` subagent-sidecar phantom sessions found by the
# polylogue-b508 audit (a fixed producer bug materialized a per-subagent
# metadata sidecar file into its own empty session, duplicate of the real,
# correctly-ingested subagent transcript). Brand-new table, no existing
# archive has any rows to migrate -- CONSTRAINT_ONLY/REPLACE_TABLE fast-
# forwards to a plain create, same shape as v33's `insight_materialization`.
# polylogue-omsw: v58 declares the classification change made to
# `archive.artifact_taxonomy.classify_artifact`/`classify_artifact_path`:
# (1) a `tool-results/<name>.json` sidecar now refuses session
# classification even when the caller-supplied provider hint is not
# Claude Code (a generic/ad-hoc acquisition route resolving "unknown"), and
# (2) a `projects/<proj>/<uuid>.jsonl` file whose every record is a
# `file-history-snapshot`/`progress` checkpoint now refuses session
# classification via positive content evidence overriding the path-only
# `coordinator_session_stream` rule. Both are pure classification-decision
# changes over identical input bytes -- the same v42/v44/v45/v46/v48
# "SEMANTIC_REPARSE, no clone-safe SQL delta" shape: only re-parsing
# recovers the correct classification for already-acquired raw evidence.
# Read-only live census at declaration time (source.db, 2026-08-03): 3
# tool-results rows (toolu_013w7YLBZHwsaNtDn9RVdHKG et al., matching the
# operator's own prior investigation) found via
# `devtools workspace tool-result-history-sweep`; 0 file-history-snapshot-
# only raw rows found in the same scan (that estimate's ~214 count applied
# to index.db zero-message sessions, not raw_sessions rows -- some or all
# may already have been swept by an intervening full reindex). `polylogue
# ops reset --index && polylogued run` is required to apply this;
# deliberately NOT executed by this declaration.
# polylogue-4987i: v59 makes Claude Code session_events ordering deterministic
# and chunk-count-invariant (order_session_events/reconcile_code_session_chunks,
# code_parser.py). Previously, eager parsing (full-corpus reindex) and
# streaming/chunked parsing (live incremental ingest of a session interleaved
# with another session's records in the same JSONL -- a common shape for
# subagent/resume sessions) could produce the SAME session_events content in a
# DIFFERENT order for byte-identical raw input, because each streaming chunk
# independently appended its own end-of-chunk summary events (coverage,
# background-completion, delegation-progress) instead of a session-wide
# accumulation. content_hash embeds each event's list position
# (`event_index`, pipeline/ids.py:_session_hash_components), so this was a
# genuine content-hash instability: reindexing a session that was originally
# ingested incrementally could compute a different hash purely from which
# code path produced it, triggering spurious "content changed" reprocessing.
# Fix: both paths now sort session_events by (timestamp, event-type tier,
# encounter index) and streaming's chunk-merge fully accumulates coverage/
# delegation-progress across chunks (not just background-completion) before
# that sort, so the order depends only on parsed content, never on chunk
# count. Read-only live spot-check (source.db, 2026-08-03, 25 smallest
# claude-code-session rows sampled): eager-parsed content_hash is BYTE-
# IDENTICAL before/after this change for every non-interleaved (single-
# chunk) sample -- the fix is a no-op for the common case. Hash drift is
# expected and acknowledged ONLY for sessions whose raw source file
# genuinely interleaves two sessions' records (forcing multi-chunk
# streaming) and that were originally ingested via the streaming path with
# the old buggy order -- same v42/v44/v45/v46/v48/v58 "SEMANTIC_REPARSE, no
# clone-safe SQL delta" shape.
# polylogue-taj0o Stage 2: unifies Claude Code eager and streaming parsing
# into ONE incremental multi-way merge (dispatch.py's
# `_claude_code_multiway_parse`, replacing the former
# `_claude_code_grouped_record_specs` eager grouping and
# `_claude_code_stream_sessions` streaming chunker, plus
# `merge_parsed_session_chunks`'s Claude-Code branch and
# `reconcile_code_session_chunks`, both deleted). Folds every record for one
# session id into the SAME `_SessionAccumulator` across the whole record
# stream, finalizing once at true end of input -- eliminates the
# eager-vs-streaming duplication v59 already fixed the *ordering* symptom
# of. Also resolves the "three-way duplication" identity-conflict finding
# (bd polylogue-taj0o notes): the canonical "primary" group for a file
# containing more than one interleaved Claude Code session is now ALWAYS
# the group whose own sessionId equals the caller's fallback_id (streaming's
# former rule), never eager's former max-by-record-count rule, which could
# pick the wrong group as primary for a small main session sharing a file
# with a much larger subagent transcript. `provider_session_id`/
# `parent_session_id`/`branch_type` can change for any already-materialized
# claude-code-session whose raw file both interleaves more than one session
# AND has a non-fallback_id-matching group with the largest record count --
# SEMANTIC_REPARSE, not a free fast-forward: no DDL change, only re-parsing
# already-acquired raw evidence recovers the corrected identity split.
# `polylogue ops reset --index && polylogued run` is required; deliberately
# NOT executed by this declaration.
# v61 (polylogue-resk): drops session_model_usage.priced_with/priced_at_ms
# and narrows the CHECK that referenced priced_with -- both were write-only
# outside tests (zero production SELECTs), and the FK target price_catalogs
# is dropped in the same change via the index-tier benign-DDL registry (see
# index_convergence.py). CONSTRAINT_ONLY/dead-column-removal, same shape as
# v33/v36/v38/v41/v44: no raw reparse, existing session_model_usage rows
# copy-forward on every other column via the fast-forward executor's
# REPLACE_TABLE path. Does NOT touch session_profiles.priced_with/
# priced_at_ms, a different pair of columns with a real production reader
# (daemon/http.py's session-insights panel) -- see the price_catalogs
# removal comment above session_model_usage's DDL for the full distinction.
# polylogue-664l: v62 batches two column-level dead-weight removals found by
# the 2026-07-31 producer/consumer audit, both clone-safe (no re-parse of raw
# evidence needed -- remaining/unchanged columns carry byte-identical values):
#  - session_provider_usage_events drops 9 write-only columns:
#    model_context_window plus the 8 Hermes billing-provenance columns
#    (estimated_cost_usd, actual_cost_usd, cost_status, cost_source,
#    pricing_version, billing_provider, billing_base_url, billing_mode).
#    Every write site (`_provider_usage_event_row` in write.py) populates
#    them from `ParsedSessionEvent.payload`, but no reader anywhere in the
#    repo ever selects them -- confirmed by re-auditing `storage/usage.py`
#    (the only production reader of this table), which reads only the
#    `last_*`/`total_*` token counters. Same "cache/copy with zero readers"
#    shape as v41's `action_pairs` text-copy removal.
#  - attachment_native_ids.id_kind's CHECK narrows from 5 members to 4,
#    dropping 'source'. Re-auditing confirmed 'url' IS read (the generic
#    `search_attachment_identity_evidence_hits` identity-search query in
#    `storage/sqlite/queries/attachment_records.py` joins ALL id_kind rows,
#    not just attachment/file/drive, contrary to the audit's original
#    "readers pull attachment/file/drive only" claim for 'url') -- so 'url'
#    is kept. 'source' has zero producers anywhere in the repo
#    (`_write_attachment_native_ids` in write.py only ever writes
#    attachment/file/drive/url) and zero live rows can exist by
#    construction, so narrowing the CHECK rejects nothing on any real
#    archive -- CONSTRAINT_ONLY, the same v33/v36/v51/v52 "tightening over
#    unchanged values" precedent.
#  - The other two column-level findings from the same audit (bead
#    polylogue-664l) were re-verified and found stale on current source, so
#    no DDL change accompanies them here: `insight_materialization`'s CHECK
#    vocabulary (9 values) is fully live (every value has a real writer in
#    `rebuild.py`/`write.py`); the registry entries with no per-session
#    ledger row are either `readiness_exempt=True` (query-time aggregates
#    with no separate materialized state to go stale: archive_coverage,
#    tool_usage, session_costs, cost_rollups, usage_timeline, archive_debt)
#    or `session_tag_rollups`, which has its own dedicated non-session-scoped
#    staleness query (`STALE_SESSION_TAG_ROLLUP_COUNT_SQL` in
#    `storage/insights/session/status.py`) because it isn't keyed by
#    session_id at all. `input_high_water_mark_source` is read broadly
#    across public surfaces (`insights/archive.py`'s `time_confidence_for_
#    source`, `daemon/http.py` status payloads), not confined to
#    repair/status internals. price_catalogs metadata is out of scope here
#    (table-level finding, tracked separately by polylogue-resk).
# polylogue-eizc: v63 drops `threads_fts`, a maintained FTS surface with real
# triggers and rebuild/repair/freshness machinery but no application-layer
# consumers. Its only MATCH reader had zero production callers; the live
# thread-search path already uses a LIKE substring scan. The sibling
# `blocks_command_trigram` was kept at v63 because `devtools/affordance_usage.py`'s
# `_cli_action_rows` was a real consumer with a documented archive-scale
# speedup; that devtools module was deleted 2026-08-25 (polylogue-9m6ry,
# completed campaign analytics) with no product-surface replacement, so
# whether `blocks_command_trigram` still has a live consumer is now open --
# see the v63 declaration in lifecycle.py.
# polylogue-xselt: v64 adds parser/lowering semantic stamps consumed by the
# reindex acceptance gate. They remain nullable only so pre-bootstrap index
# generations can be opened long enough to undergo the semantic replay.
# polylogue-cuxz.5: v65 adds the actions view-only ``result_state`` projection.
# It is computed entirely from existing action_pairs outcome columns, so a
# fast-forward only replaces the derived view. No raw session is reparsed and
# no persisted source/blob value changes.
# polylogue-fix-blob-reference-closure: v66 separates native message ids from
# positional coordinates in the generated message_id expression. Existing
# derived rows remain readable as opaque legacy ids, but new materialization
# must replay raw sessions to regenerate message/block/reference identities.
# polylogue-o5smo: v67 assigns typed origin provenance to provider-authored
# Grok, Antigravity, browser-capture, and Hermes titles. Existing parsed rows
# require raw replay because an in-place schema operation cannot recover title
# authorship from the stored untyped title text.
# polylogue-terra-p1: v70 persists the accepted frontier proven by each raw
# revision application. These values are derived at application time and old
# rows cannot be reconstructed safely from the mutable head, so v70 routes
# existing index generations through semantic raw replay.
# n6 n6rkz: v71 widens raw application identity to include every immutable
# accepted-evidence field. Existing application rows must be replayed so their
# decision ids and duplicate constraints are rebuilt from that complete shape.
# polylogue-tas4: v72 makes READY a source-verified FTS verdict. A READY row
# records an exact snapshot bound to the current index schema generation;
# bounded repairs remain STALE and preserve the last exact counters.
# polylogue-h6r: v74 persists the declared worker role alongside actor and
# execution-context refs in work-evidence nodes. Existing graph rows cannot
# recover role from their prior shape, so the normal derived rebuild route
# must regenerate them from projected-run evidence.
# polylogue-oqc6c: session-kind admission is parser-derived and existing rows
# must be replayed so primary/subagent/prompt-suggestion values are recovered
# from durable source evidence.
INDEX_SCHEMA_VERSION = 80

# polylogue-v6i3: shared WHEN-clause fragment gating the blocks_command_trigram
# trigger BODIES on the same dedicated bulk-build guard row messages_fts's
# triggers already use (see storage/fts/sql.py's _FTS_BULK_GUARD_NOT_SET).
# Additive/inert on existing archives -- CREATE TRIGGER IF NOT EXISTS keeps
# the prior (ungated) trigram trigger body on an archive until its next
# rebuild, matching the messages_fts precedent from #3152 (no
# INDEX_SCHEMA_VERSION bump needed for an additive, inert trigger-body change).
_TRIGRAM_BULK_GUARD_NOT_SET = (
    f"NOT EXISTS (SELECT 1 FROM derived_refresh_guard WHERE guard_name = '{FTS_BULK_SESSION_WRITE_GUARD}')"
)

FTS_FRESHNESS_STATE_DDL = f"""
CREATE TABLE IF NOT EXISTS fts_freshness_state (
    {TABLE_SPECS["fts_freshness_state"].ddl_body}
) STRICT;
"""

INDEX_DDL = f"""
-- Candidate construction authority is generation-local.  The source seal is
-- copied here when an inactive generation is created; resume derives pending
-- work from this table, never from an ops cursor or a later source scan.
CREATE TABLE IF NOT EXISTS candidate_source_membership (
    raw_id TEXT PRIMARY KEY,
    blob_hash BLOB NOT NULL,
    blob_size INTEGER NOT NULL CHECK(blob_size >= 0),
    source_snapshot TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'committed')),
    committed_at_ms INTEGER
) STRICT;

CREATE INDEX IF NOT EXISTS idx_candidate_source_membership_pending
ON candidate_source_membership(status, blob_hash, raw_id);

-- Continuation frames are deliberately tier-local: query-unit continuations
-- combine this counter with the user-tier counter, so a write in either
-- relation family invalidates an offset resume without inventing an archive-
-- wide generation for unrelated tiers.
CREATE TABLE IF NOT EXISTS query_unit_frame_state (
    {TABLE_SPECS["query_unit_frame_state"].ddl_body}
) STRICT;

INSERT OR IGNORE INTO query_unit_frame_state(singleton, epoch) VALUES (1, 0);

CREATE TABLE IF NOT EXISTS raw_revision_applications (
    {TABLE_SPECS["raw_revision_applications"].ddl_body}
) STRICT;

DROP INDEX IF EXISTS idx_raw_revision_applications_identity;

CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_revision_applications_identity
ON raw_revision_applications(
    raw_id, session_id, logical_source_key, decision, source_revision,
    COALESCE(accepted_source_revision, ''), accepted_content_hash,
    accepted_frontier_kind, accepted_frontier, acquisition_generation,
    COALESCE(append_end_offset, -1), COALESCE(baseline_raw_id, ''),
    COALESCE(predecessor_raw_id, '')
);

CREATE INDEX IF NOT EXISTS idx_raw_revision_applications_logical
ON raw_revision_applications(logical_source_key, acquisition_generation, raw_id);

CREATE TABLE IF NOT EXISTS raw_revision_heads (
    {TABLE_SPECS["raw_revision_heads"].ddl_body}
) STRICT;

CREATE TABLE IF NOT EXISTS sessions (
    {TABLE_SPECS["sessions"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_sessions_origin_sort
ON sessions(origin, sort_key_ms DESC);

CREATE INDEX IF NOT EXISTS idx_sessions_sort_key
ON sessions(sort_key_ms DESC);

CREATE INDEX IF NOT EXISTS idx_sessions_parent
ON sessions(parent_session_id)
WHERE parent_session_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_sessions_root
ON sessions(root_session_id)
WHERE root_session_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_sessions_raw_id
ON sessions(raw_id)
WHERE raw_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS messages (
    {TABLE_SPECS["messages"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_messages_session_position
ON messages(session_id, position, variant_index);

-- Serves the sort-key ordering used by iter_messages keyset pagination,
-- get_messages_batch, and get_messages_paginated. Those reads order by
-- `(occurred_at_ms IS NULL), occurred_at_ms, message_id` (NULL-timestamp rows
-- sort last). The leading IS NULL expression must itself be an indexed column or
-- the planner ignores the index and sorts the whole session in a temp B-tree on
-- every chunk — verified via EXPLAIN QUERY PLAN: a plain
-- (session_id, occurred_at_ms, message_id) index still triggers
-- `USE TEMP B-TREE FOR ORDER BY`, while the expression index below plans as a
-- covering-index scan with no sort (#2467 / #2475 perf audit).
CREATE INDEX IF NOT EXISTS idx_messages_session_sortkey
ON messages(session_id, (occurred_at_ms IS NULL), occurred_at_ms, message_id);

CREATE INDEX IF NOT EXISTS idx_messages_parent
ON messages(parent_message_id)
WHERE parent_message_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_messages_session_role
ON messages(session_id, role);

CREATE INDEX IF NOT EXISTS idx_messages_role
ON messages(role);

CREATE INDEX IF NOT EXISTS idx_messages_session_material_origin
ON messages(session_id, material_origin);

CREATE INDEX IF NOT EXISTS idx_messages_message_type
ON messages(message_type);

CREATE INDEX IF NOT EXISTS idx_messages_material_origin
ON messages(material_origin);

-- Serves the paid embedding selector and status/preflight counts. The predicate
-- matches authored prose only: user/assistant message rows with human- or
-- assistant-authored material and positive word count. Keeping this as a
-- partial index avoids scanning tool, protocol, context-pack, and generated
-- runtime rows when computing cost windows over large archives.
CREATE INDEX IF NOT EXISTS idx_messages_embedding_prose
ON messages(session_id, position, variant_index, message_id, content_hash)
WHERE message_type = 'message'
  AND role IN ('user', 'assistant')
  AND material_origin IN ('human_authored', 'assistant_authored')
  AND word_count > 0;

CREATE INDEX IF NOT EXISTS idx_messages_active_path
ON messages(session_id, is_active_path, position)
WHERE is_active_path = 1;

CREATE INDEX IF NOT EXISTS idx_messages_active_leaf
ON messages(session_id, is_active_leaf)
WHERE is_active_leaf = 1;

CREATE TABLE IF NOT EXISTS blocks (
    {TABLE_SPECS["blocks"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_blocks_session_position
ON blocks(session_id, message_id, position);

CREATE INDEX IF NOT EXISTS idx_blocks_content_hash
ON blocks(content_hash);

CREATE INDEX IF NOT EXISTS idx_blocks_type
ON blocks(block_type);

-- Serves structured failure/outcome reports and action predicates that anchor
-- on provider-reported tool-result failure rather than prose. The predicate
-- remains on the tool_result block because the public actions relation is a
-- view over paired tool_use/tool_result blocks; starting from failed result
-- rows avoids scanning every tool invocation in large archives.
CREATE INDEX IF NOT EXISTS idx_blocks_tool_result_outcome
ON blocks(block_type, tool_result_is_error, tool_result_exit_code, session_id, tool_id, message_id)
WHERE block_type = 'tool_result';

CREATE INDEX IF NOT EXISTS idx_blocks_type_tool
ON blocks(
    block_type,
    COALESCE(NULLIF(LOWER(tool_name), ''), 'unknown')
);

CREATE INDEX IF NOT EXISTS idx_blocks_tool_id
ON blocks(tool_id)
WHERE tool_id IS NOT NULL;

-- Serves message FTS readiness and repair. Search readiness compares the
-- text-bearing block set with messages_fts_docsize; without this partial index
-- the source-side count scans every block in large archives before the user can
-- run even a simple MATCH query.
CREATE INDEX IF NOT EXISTS idx_blocks_search_text_populated
ON blocks(message_id, position)
WHERE search_text != '';

CREATE TABLE IF NOT EXISTS web_content_constructs (
    {TABLE_SPECS["web_content_constructs"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_web_constructs_session_type
ON web_content_constructs(session_id, construct_type);

-- polylogue-rgbj: the only messages(message_id) FK child table that lacked a
-- leading index on its foreign key column. Every other referencing table
-- (blocks, attachment_refs, paste_spans, session_events,
-- session_agent_policies, session_provider_usage_events) already has one
-- via its PRIMARY KEY or a dedicated index. Without this index, SQLite's FK
-- `ON DELETE CASCADE` enforcement falls back to a full table scan of
-- web_content_constructs for EVERY deleted message row (SQLite does not
-- require a child-key index for FK enforcement, it just scans if one is
-- missing). `_replace_full_session_messages_and_blocks`'s bare
-- `DELETE FROM messages WHERE session_id = ?` therefore cost
-- O(deleted_messages x web_content_constructs_rows): confirmed live at
-- 132,796 web_content_constructs rows, `EXPLAIN QUERY PLAN` showed
-- `SCAN web_content_constructs` before this index, `SEARCH ... USING INDEX`
-- after. See tests/unit/storage/test_schema_safety.py and
-- tests/benchmarks/test_full_session_replace.py for the query-plan and
-- timing proof.
CREATE INDEX IF NOT EXISTS idx_web_constructs_message
ON web_content_constructs(message_id);

CREATE INDEX IF NOT EXISTS idx_web_constructs_url
ON web_content_constructs(url)
WHERE url IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_web_constructs_query
ON web_content_constructs(query)
WHERE query IS NOT NULL;

-- polylogue-2qx.4 (v46): file-edit tool-call evidence (Claude Code Edit/
-- Write toolUseResult): 105,123 structured unified diffs (structuredPatch),
-- 92,313 pre-edit file captures (originalFile), and the oldString/newString/
-- replaceAll/filePath/userModified fields around them -- entirely discarded
-- today (polylogue-cgfy). This is a RELATION (one edit per tool call), keyed
-- by the tool_use block that performed the edit, not a blocks.metadata bag
-- (polylogue-9x22's rejected shape) and not folded into the block row
-- itself. originalFile captures the pre-edit state polylogue-cijx's
-- file-trajectory grading declares unavailable ("observed" only) --
-- this is what raises it to "checkpointed".
CREATE TABLE IF NOT EXISTS file_edits (
    {TABLE_SPECS["file_edits"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_file_edits_session
ON file_edits(session_id);

-- polylogue-rgbj precedent (see web_content_constructs above): every
-- messages(message_id) FK child table needs a leading index on the FK
-- column, or SQLite's ON DELETE CASCADE enforcement falls back to a full
-- table scan of file_edits for every deleted message row.
CREATE INDEX IF NOT EXISTS idx_file_edits_message
ON file_edits(message_id);

CREATE INDEX IF NOT EXISTS idx_file_edits_file_path
ON file_edits(file_path)
WHERE file_path IS NOT NULL;

-- polylogue-2qx.4 (v46): tracker-agnostic external references observed in a
-- session (Claude Code pr-link, 20,702 occurrences on the wire today;
-- generalizes to issue refs so a second tracker never needs its own table).
CREATE TABLE IF NOT EXISTS session_refs (
    {TABLE_SPECS["session_refs"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_refs_kind
ON session_refs(kind, repo, ref_number);

{FTS_MESSAGES_TABLE_SQL}

-- polylogue-1xc.12: rowid-to-block_id identity ledger for the contentless
-- messages_fts table above -- see FTS_MESSAGES_IDENTITY_TABLE_SQL in
-- storage/fts/sql.py (single source of truth, imported here) for the full
-- rationale. Declared as a plain f-string field (not `{{}}`-escaped) because
-- FTS_MESSAGES_IDENTITY_TABLE_SQL is a Python constant substituted at
-- INDEX_DDL build time, not a SQL brace literal.
{FTS_MESSAGES_IDENTITY_TABLE_SQL}

-- FTS triggers for messages_fts table are now dynamically composed from sql.py
-- (polylogue-a7xr.5: consolidate FTS trigger DDL to single source)

-- ohbx: trigram-tokenized (not unicode61) so `detail LIKE '%pattern%'` gets
-- SQLite's built-in trigram LIKE-acceleration for arbitrary substrings, not
-- just whole tokens -- unicode61/messages_fts would miss a match like
-- "notpolyloguefile.txt" (no token boundary around the substring), and
-- searches a much larger candidate set (the word can appear in ordinary
-- prose, not just tool invocations). External content (not contentless,
-- unlike messages_fts) is required: contentless trigram tables silently
-- return zero rows for LIKE queries because SQLite needs to re-read the
-- real text from the content table to verify a candidate trigram match
-- (verified locally against the SQLite forum's own explanation of this
-- exact mechanism before relying on it).
--
-- Query-shape note for callers: SQLite's planner does NOT automatically
-- drive a `blocks_command_trigram JOIN blocks` from the trigram table's
-- LIKE index -- a plain join lets it choose to scan `blocks` as the outer
-- loop and probe the trigram table per row, which is *slower* than the old
-- raw scan (measured: 26s vs 0.15s at 300K rows). The trigram index must
-- drive the query explicitly via `blocks.rowid IN (SELECT rowid FROM
-- blocks_command_trigram WHERE tool_detail_text LIKE ...)`, which forces
-- the trigram LIKE-optimization to run first and reduces the outer table
-- to an indexed rowid lookup (measured: 900x+ faster than the raw scan at
-- 915K rows, using this exact shape).
CREATE VIRTUAL TABLE IF NOT EXISTS blocks_command_trigram USING fts5(
    tool_detail_text,
    tokenize='trigram',
    content='blocks',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS blocks_command_trigram_ai
AFTER INSERT ON blocks
WHEN new.block_type = 'tool_use' AND new.tool_detail_text != ' ' AND {_TRIGRAM_BULK_GUARD_NOT_SET} BEGIN
    INSERT INTO blocks_command_trigram(rowid, tool_detail_text)
    VALUES (new.rowid, new.tool_detail_text);
END;

-- External-content FTS5 tables require the special 'delete' command form
-- with the OLD column value supplied (not a plain DELETE by rowid) --
-- verified locally: a bare `DELETE FROM blocks_command_trigram WHERE rowid
-- = old.rowid` leaves stale trigram postings that later raise "fts5:
-- missing row N from content table" once the real row is gone from
-- `blocks`, because FTS5 needs the old text to locate the exact postings
-- to remove rather than re-reading it from the (already-deleted) content
-- row.
CREATE TRIGGER IF NOT EXISTS blocks_command_trigram_ad
AFTER DELETE ON blocks
WHEN old.block_type = 'tool_use' AND old.tool_detail_text != ' ' AND {_TRIGRAM_BULK_GUARD_NOT_SET} BEGIN
    INSERT INTO blocks_command_trigram(blocks_command_trigram, rowid, tool_detail_text)
    VALUES ('delete', old.rowid, old.tool_detail_text);
END;

CREATE TRIGGER IF NOT EXISTS blocks_command_trigram_au
AFTER UPDATE ON blocks WHEN {_TRIGRAM_BULK_GUARD_NOT_SET} BEGIN
    INSERT INTO blocks_command_trigram(blocks_command_trigram, rowid, tool_detail_text)
    SELECT 'delete', old.rowid, old.tool_detail_text
    WHERE old.block_type = 'tool_use' AND old.tool_detail_text != ' ';
    INSERT INTO blocks_command_trigram(rowid, tool_detail_text)
    SELECT new.rowid, new.tool_detail_text
    WHERE new.block_type = 'tool_use' AND new.tool_detail_text != ' ';
END;

{FTS_FRESHNESS_STATE_DDL}

-- polylogue-2i2w: deliberately NO tool_input/output_text columns here. This
-- table is a join/rank/outcome index over paired tool_use/tool_result
-- blocks -- the text itself already lives once in blocks.tool_input /
-- blocks.text. Storing it again (index schema <= v40) made every tool
-- interaction exist ~3x on disk (868K rows, ~4.7KB/row overflow-chained,
-- ~4.1GB) and turned per-session DELETE into scattered random IO (measured:
-- hours on a whale session replace). The actions VIEW below re-joins blocks
-- by block_id to serve tool_input/output_text to readers at query time.
CREATE TABLE IF NOT EXISTS action_pairs (
    {TABLE_SPECS["action_pairs"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_action_pairs_session_order
ON action_pairs(session_id, message_id, tool_use_block_id);
CREATE INDEX IF NOT EXISTS idx_action_pairs_message
ON action_pairs(message_id);
CREATE INDEX IF NOT EXISTS idx_action_pairs_tool
ON action_pairs(tool_name, session_id, message_id);
CREATE INDEX IF NOT EXISTS idx_action_pairs_semantic
ON action_pairs(semantic_type, session_id, message_id);
CREATE INDEX IF NOT EXISTS idx_action_pairs_path
ON action_pairs(tool_path, session_id, message_id)
WHERE tool_path IS NOT NULL AND tool_path != '';
CREATE INDEX IF NOT EXISTS idx_action_pairs_outcome
ON action_pairs(is_error, exit_code, session_id, message_id)
WHERE is_error IS NOT NULL OR exit_code IS NOT NULL;

-- polylogue-623q: tool_result_block_id is ON DELETE SET NULL against
-- blocks(block_id) but had no supporting index. SQLite must find every
-- action_pairs row whose tool_result_block_id equals each block being
-- deleted; without an index that is a full action_pairs table SCAN per
-- deleted block row -- for a whale session's full-replace block DELETE
-- (thousands of blocks) against an archive-wide action_pairs table, that is
-- effectively O(deleted_blocks x action_pairs_row_count). Verified with a
-- synthetic whale session (2,000 blocks) against a 32K-row action_pairs
-- table: the single ``DELETE FROM blocks WHERE session_id = ?`` dropped
-- from 6.87s to 0.034s (~200x) after adding this index. Partial (excludes
-- NULL) since unpaired tool_use rows never populate this column and NULL
-- never matches an equality FK lookup.
CREATE INDEX IF NOT EXISTS idx_action_pairs_tool_result_block
ON action_pairs(tool_result_block_id)
WHERE tool_result_block_id IS NOT NULL;

-- xnkf: a plain equality join on tool_id fans out when a provider re-emits
-- the same tool_id on distinct messages (verified live: identical toolu_
-- ids as 2 tool_use + 2 tool_result blocks at different positions, NOT
-- variants). Rank each side by transcript order (message position, THEN
-- variant_index, then block position -- messages are only unique on
-- (position, variant_index), so omitting variant_index would leave ties
-- between regenerated variant messages, letting SQLite assign ranks
-- independently/arbitrarily across the two CTEs and re-introduce
-- cross-pairing) within (session_id, tool_id), and pair same-rank rows --
-- the Nth use in the transcript gets the Nth result, never a cross
-- product. Uses with no tool_id (NULL or '') are never rank-paired -- SQL
-- equality never matches NULL, so a NULL tool_id use was already always
-- unpaired under the old plain-equality join; the second branch below
-- preserves that (still surfaced, just with NULL result columns) while the
-- empty-string guard stops '' specifically from cross-joining as if it
-- were a real shared id (parsers currently only ever emit NULL, not '').
-- polylogue-2i2w: tool_input/output_text are no longer materialized on
-- action_pairs itself -- they are re-joined from blocks (by
-- tool_use_block_id / tool_result_block_id) here, at read time, so every
-- consumer of this view keeps byte-identical payloads without any caller
-- change. tu is an INNER JOIN because tool_use_block_id is a NOT NULL FK
-- ON DELETE CASCADE (the action_pairs row cannot outlive its use block); tr
-- is a LEFT JOIN because tool_result_block_id is nullable (unpaired uses)
-- and ON DELETE SET NULL.
CREATE VIEW IF NOT EXISTS actions AS
SELECT
    ap.session_id, ap.message_id, ap.tool_use_block_id, ap.tool_name, ap.semantic_type,
    ap.tool_command, ap.tool_path, tu.tool_input AS tool_input, tr.text AS output_text,
    ap.is_error, ap.exit_code, ap.tool_result_block_id,
    CASE
        WHEN ap.tool_result_block_id IS NULL THEN 'no_result'
        WHEN ap.is_error IS NULL AND ap.exit_code IS NULL THEN 'outcome_unknown'
        WHEN ap.exit_code IS NOT NULL AND ap.exit_code != 0 THEN 'outcome_error'
        WHEN ap.exit_code IS NULL AND ap.is_error = 1 THEN 'outcome_error'
        ELSE 'outcome_success'
    END AS result_state
FROM action_pairs ap
JOIN blocks tu ON tu.block_id = ap.tool_use_block_id
LEFT JOIN blocks tr ON tr.block_id = ap.tool_result_block_id;

CREATE TABLE IF NOT EXISTS session_events (
    {TABLE_SPECS["session_events"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_events_source_message
ON session_events(source_message_id)
WHERE source_message_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS session_agent_policies (
    {TABLE_SPECS["session_agent_policies"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_agent_policies_source_message
ON session_agent_policies(source_message_id)
WHERE source_message_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS session_links (
    {TABLE_SPECS["session_links"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_links_dst
ON session_links(resolved_dst_session_id)
WHERE resolved_dst_session_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_session_links_dst_native
ON session_links(dst_origin, dst_native_id);

-- Resolver writes can change terminal ``delegations`` without replacing a
-- transcript row, so they belong to the query-unit relation frame.
CREATE TRIGGER IF NOT EXISTS query_unit_frame_session_links_insert
AFTER INSERT ON session_links BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;
CREATE TRIGGER IF NOT EXISTS query_unit_frame_session_links_update
AFTER UPDATE ON session_links BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;
CREATE TRIGGER IF NOT EXISTS query_unit_frame_session_links_delete
AFTER DELETE ON session_links BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;

CREATE TABLE IF NOT EXISTS threads (
    {TABLE_SPECS["threads"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_threads_time
ON threads(end_time DESC, start_time DESC);

-- polylogue-eizc: threads_fts (a MATCH index over threads.search_text) was
-- dropped in INDEX_SCHEMA_VERSION 62 -- its only MATCH reader
-- (session_insight_thread_queries.list_threads) had zero production
-- callers; the live "analyze threads" search path
-- (list_thread_insights below) already does a manual LIKE substring scan
-- over thread_id/session title/repo/branch and never touched threads_fts.
-- list_threads now does the same LIKE scan over threads.search_text
-- instead of an FTS5 MATCH. See lifecycle.py's v63 declaration.

CREATE TABLE IF NOT EXISTS thread_sessions (
    {TABLE_SPECS["thread_sessions"].ddl_body}
) STRICT;

CREATE TABLE IF NOT EXISTS session_working_dirs (
    {TABLE_SPECS["session_working_dirs"].ddl_body}
) STRICT;

-- polylogue-cijx.4 decision 1: a repository is keyed on its normalized
-- remote -- every worktree checkout of the same remote is one repo, not
-- one row per checkout. `repo_id` used to be `origin_url || root_path`
-- (`GENERATED ALWAYS`), which made two worktrees of the identical remote
-- two distinct rows purely because their checkout paths differed. `repo_id`
-- is now a plain stored column populated by the writer
-- (`archive_tiers/write.py:_repo_identity_key`) from a canonicalized form
-- of `origin_url` when one is known (`remote:<host>/<path>`, unifying
-- `git@host:owner/repo`, `https://host/owner/repo.git`, `ssh://host/...`),
-- falling back to the resolved checkout root only when no remote exists
-- (`dir:<root_path>` -- decision 1's "outermost git root" fallback for a
-- repository with no remote). It is a plain column rather than a SQLite
-- `GENERATED` expression because remote-URL canonicalization needs real
-- string logic (scheme/userinfo stripping, SCP-vs-URL unification, case
-- folding) that SQL cannot express without duplicating that logic.
-- `origin_url`/`root_path` on this row are now *representative* display
-- values (first-seen wins), not part of the identity key -- every checkout
-- root actually observed for a repo_id is recorded in `repo_checkouts`.
CREATE TABLE IF NOT EXISTS repos (
    {TABLE_SPECS["repos"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_repos_root_path
ON repos(root_path)
WHERE root_path != '';

-- Every distinct checkout root ever observed for a repo identity (decision
-- 1: "a worktree is a checkout of a repository ... every
-- /realm/worktrees/polylogue-* and .claude/worktrees/agent-* is one
-- checkout of polylogue"). Purely additive evidence; `repos.root_path`
-- keeps a single representative value for existing readers, this table is
-- the complete enumeration for readers that need it.
CREATE TABLE IF NOT EXISTS repo_checkouts (
    {TABLE_SPECS["repo_checkouts"].ddl_body}
) STRICT;

CREATE TABLE IF NOT EXISTS session_repos (
    {TABLE_SPECS["session_repos"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_repos_repo
ON session_repos(repo_id);

CREATE TABLE IF NOT EXISTS session_commits (
    {TABLE_SPECS["session_commits"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_commits_hash
ON session_commits(commit_sha);

CREATE INDEX IF NOT EXISTS idx_session_commits_repo
ON session_commits(repo_id)
WHERE repo_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS attachments (
    {TABLE_SPECS["attachments"].ddl_body}
) STRICT;

CREATE TABLE IF NOT EXISTS attachment_refs (
    {TABLE_SPECS["attachment_refs"].ddl_body}
) STRICT;

CREATE TABLE IF NOT EXISTS attachment_native_ids (
    {TABLE_SPECS["attachment_native_ids"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_attachment_refs_session
ON attachment_refs(session_id);

CREATE INDEX IF NOT EXISTS idx_attachment_refs_message
ON attachment_refs(message_id)
WHERE message_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_attachment_refs_attachment
ON attachment_refs(attachment_id);

CREATE INDEX IF NOT EXISTS idx_attachment_refs_upload_origin
ON attachment_refs(upload_origin, session_id)
WHERE upload_origin IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_attachment_native_ids_native
ON attachment_native_ids(id_kind, native_id);

CREATE TABLE IF NOT EXISTS paste_spans (
    {TABLE_SPECS["paste_spans"].ddl_body}
) STRICT;

-- polylogue-623q: paste_spans has no index leading with session_id (its PK
-- is (message_id, position)), so the per-session full-replace DELETE FROM
-- paste_spans WHERE session_id = ? in _clear_session_projection_rows was a
-- full table SCAN (verified via EXPLAIN QUERY PLAN) that gets slower every
-- session as the table grows archive-wide -- every other projection table
-- cleared there already has session_id as a leading PK/index column.
CREATE INDEX IF NOT EXISTS idx_paste_spans_session
ON paste_spans(session_id);

-- model_prices and session_reported_costs were dropped (polylogue-v2mg):
-- zero-consumer tables converged away by the index-tier same-version
-- benign-DDL registry (archive_tiers/index_convergence.py) rather than kept
-- in canonical DDL. price_catalogs was dropped the same way (polylogue-resk,
-- v61): v2mg's stated justification for keeping it ("session_model_usage.
-- priced_with FK, active_price_catalog_id" are genuine reads) measured false
-- in all three particulars -- priced_with/priced_at_ms had zero production
-- SELECTs (write-only outside tests), and active_price_catalog_id's only
-- caller was a test. Cost computation resolves per-model rates from the
-- in-process pricing catalog (polylogue.archive.semantic.pricing.PRICING),
-- never round-tripping through a DB-backed mirror or catalog-identity table.
-- session_profiles no longer persists pricing snapshots; public readers use
-- the canonical session_model_usage/session-reported-cost projection.

CREATE TABLE IF NOT EXISTS session_model_usage (
    {TABLE_SPECS["session_model_usage"].ddl_body}
) STRICT;

CREATE TABLE IF NOT EXISTS session_provider_usage_events (
    {TABLE_SPECS["session_provider_usage_events"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_provider_usage_events_session
ON session_provider_usage_events(session_id, position);

CREATE INDEX IF NOT EXISTS idx_session_provider_usage_events_source_message
ON session_provider_usage_events(source_message_id)
WHERE source_message_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS session_tags (
    {TABLE_SPECS["session_tags"].ddl_body}
) STRICT;

CREATE TRIGGER IF NOT EXISTS query_unit_frame_sessions_insert
AFTER INSERT ON sessions BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;
CREATE TRIGGER IF NOT EXISTS query_unit_frame_sessions_update
AFTER UPDATE ON sessions BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;
CREATE TRIGGER IF NOT EXISTS query_unit_frame_sessions_delete
AFTER DELETE ON sessions BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;
CREATE TRIGGER IF NOT EXISTS query_unit_frame_messages_insert
AFTER INSERT ON messages BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;
CREATE TRIGGER IF NOT EXISTS query_unit_frame_messages_update
AFTER UPDATE ON messages BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;
CREATE TRIGGER IF NOT EXISTS query_unit_frame_messages_delete
AFTER DELETE ON messages BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;
CREATE TRIGGER IF NOT EXISTS query_unit_frame_blocks_insert
AFTER INSERT ON blocks BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;
CREATE TRIGGER IF NOT EXISTS query_unit_frame_blocks_update
AFTER UPDATE ON blocks BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;
CREATE TRIGGER IF NOT EXISTS query_unit_frame_blocks_delete
AFTER DELETE ON blocks BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;
CREATE TRIGGER IF NOT EXISTS query_unit_frame_session_tags_insert
AFTER INSERT ON session_tags BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;
CREATE TRIGGER IF NOT EXISTS query_unit_frame_session_tags_update
AFTER UPDATE ON session_tags BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;
CREATE TRIGGER IF NOT EXISTS query_unit_frame_session_tags_delete
AFTER DELETE ON session_tags BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;

CREATE TABLE IF NOT EXISTS insight_materialization (
    {TABLE_SPECS["insight_materialization"].ddl_body}
) STRICT;

CREATE TABLE IF NOT EXISTS session_work_events (
    {TABLE_SPECS["session_work_events"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_work_events_session
ON session_work_events(session_id, position);

CREATE INDEX IF NOT EXISTS idx_session_work_events_type
ON session_work_events(work_event_type, session_id);

CREATE VIRTUAL TABLE IF NOT EXISTS session_work_events_fts USING fts5(
    event_id UNINDEXED,
    session_id UNINDEXED,
    work_event_type UNINDEXED,
    text,
    tokenize='{FTS_UNICODE_TOKENIZER}'
);

-- FTS triggers for session_work_events_fts table are now dynamically composed from sql.py
-- (polylogue-a7xr.5: consolidate FTS trigger DDL to single source)

CREATE TABLE IF NOT EXISTS session_phases (
    {TABLE_SPECS["session_phases"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_phases_session
ON session_phases(session_id, position);

CREATE TABLE IF NOT EXISTS session_latency_profiles (
    {TABLE_SPECS["session_latency_profiles"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_latency_profiles_provider
ON session_latency_profiles(source_name);

CREATE INDEX IF NOT EXISTS idx_session_latency_profiles_date
ON session_latency_profiles(canonical_session_date DESC);

CREATE INDEX IF NOT EXISTS idx_session_latency_profiles_stuck
ON session_latency_profiles(stuck_tool_count DESC, canonical_session_date DESC);

CREATE TABLE IF NOT EXISTS session_profiles (
    {TABLE_SPECS["session_profiles"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_profiles_provider
ON session_profiles(source_name);

CREATE INDEX IF NOT EXISTS idx_session_profiles_logical_session
ON session_profiles(logical_session_id);

CREATE INDEX IF NOT EXISTS idx_session_profiles_sort
ON session_profiles(source_sort_key DESC);

CREATE INDEX IF NOT EXISTS idx_session_profiles_first_message
ON session_profiles(first_message_at DESC);

CREATE INDEX IF NOT EXISTS idx_session_profiles_canonical_date
ON session_profiles(canonical_session_date DESC);

-- Session-profile fields supply query-unit session predicates such as repo
-- and tags; a materializer rewrite must invalidate an offset continuation.
CREATE TRIGGER IF NOT EXISTS query_unit_frame_session_profiles_insert
AFTER INSERT ON session_profiles BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;
CREATE TRIGGER IF NOT EXISTS query_unit_frame_session_profiles_update
AFTER UPDATE ON session_profiles BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;
CREATE TRIGGER IF NOT EXISTS query_unit_frame_session_profiles_delete
AFTER DELETE ON session_profiles BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;

-- polylogue-y964: delegations is a versioned, recomputable read model spined
-- on the PARENT's own dispatch actions, not on `session_links` plumbing. Two
-- upstream facts drove a full rebuild rather than a column-alias fix:
--
--   1. `session_links` stores the CHILD in `src_session_id` and the PARENT in
--      `resolved_dst_session_id` (see resolve_session_links_for_session,
--      storage/sqlite/queries/session_links.py: `child_id =
--      row["src_session_id"]`, and the resolved session is written into
--      `sessions.parent_session_id`) -- never the reverse. The prior shipped
--      view aliased these backwards, so every `orchestrator_*` column
--      actually described the CHILD session and vice versa.
--   2. `branch_point_message_id` is the last message the CHILD inherited from
--      the PARENT's prefix for lineage composition (see the `session_links`
--      DDL comment above) -- it is not the message that issued the Task
--      dispatch, and for a spawned-fresh child it is NULL entirely. The
--      prior view joined it against the (mislabeled) parent session as a
--      dispatch pointer, which does not hold in general.
--
-- The spine is therefore every parent-side dispatch action (an `actions` row
-- with semantic_type='subagent' -- a Task tool_use, optionally paired with
-- its tool_result). Stable action-observed identity is
-- (parent_session_id, instruction_tool_use_block_id): two Task calls in one
-- assistant message keep two distinct rows, never a fanout. Every dispatch
-- action gets exactly one row even when no child ever resolves
-- (mapping_state='unresolved' -- a dispatch error, or a resolution still
-- pending). A resolved session_links(subagent) edge with no discoverable
-- parent-side dispatch action (e.g. a Codex async subagent) surfaces as
-- mapping_state='edge_only', counted but never given a fabricated
-- instruction. Quarantined edges (cycle-break, #866/#1260) surface
-- explicitly as mapping_state='quarantined' rather than silently vanishing.
--
-- polylogue-1vpm.7: a dispatch and a resolved child were previously
-- corroborated by RANK-PAIRING IN TRANSCRIPT ORDER (the Nth dispatch <-> the
-- Nth resolved child by timestamp) whenever a parent's dispatch count and
-- resolved-child count happened to agree; a mismatch made every dispatch in
-- that parent surface as mapping_state='ambiguous'. Corpus verification
-- (2026-07-29, full scan) found that heuristic silently WRONG far more often
-- than right: of 1,493 rows it called 'resolved', only 146 (9.8%) actually
-- named the correct child once checked against provider-asserted content --
-- the rest paired siblings dispatched in the same turn to each other's
-- children. Ordinal position across two independently-ordered lists is not
-- identity, and a cardinality mismatch is not a per-SESSION verdict that
-- should poison every dispatch in the cohort.
--
-- The fix joins on identity instead of position. Two cases:
--   1. TRIVIAL cohort (exactly one dispatch action and exactly one resolved
--      child for a parent): there is only one possible pairing -- not a
--      guess among candidates, the only interpretation the observed facts
--      admit -- so it resolves without needing corroborating content.
--   2. NON-TRIVIAL cohort (more than one dispatch and/or more than one
--      child): Claude Code's subagent transcript's first user turn IS --
--      byte for byte -- the dispatching Task tool_use's own `prompt` field
--      (Codex: `message`); this is provider-asserted content, not a
--      heuristic, and it is already fully persisted (`actions.tool_input`,
--      `messages`/`blocks` for the child's first turn) -- no parser change
--      needed. A dispatch and a child pair only when this content matches
--      UNIQUELY in both directions (one dispatch <-> one child); a
--      duplicate-prompt collision on either side is excluded rather than
--      guessed. Corpus-wide collision check (required before relying on any
--      secondary key, polylogue-1vpm.7 AC5): of 3,534 dispatches with at
--      least one same-parent resolved-child candidate, 1,933 (54.7%) match
--      exactly one child's text and vice versa; only 14 dispatches (0.4%)
--      match more than one child's text and 22 children (0.6%) match more
--    than one dispatch's text -- both stay unresolved/edge_only rather than
--      fabricating a winner.
--
-- 'ambiguous' is retired from the mapping_state vocabulary entirely (AC2):
-- with the key, it is not a reachable state. A parent with N dispatches and
-- M<N captured children now yields exactly M resolved and N-M unresolved,
-- never N ambiguous (tests/unit/storage/test_delegations_view.py::
-- test_delegation_dispatch_without_matching_content_stays_unresolved).
--
-- Model identity is deliberately NOT collapsed into one "orchestrator model"
-- column (polylogue-4c27): `dispatch_turn_model` is the model that authored
-- the specific message containing the dispatch action (messages.model_name);
-- `requested_model` is an explicit routing override read from the dispatch
-- tool_input, honestly NULL when the provider recorded no such field;
-- `parent_session_dominant_model`/`child_session_dominant_model` are the
-- session-level dominant-model fallback (by assistant output-token share,
-- see archive/session/runtime.py:_primary_model) -- explicitly named as a
-- session-wide aggregate, excluded from turn-level dispatch claims. Use
-- `archive.semantic.pricing.resolve_model_identity()` to derive
-- vendor/model-line/pricing-source/confidence from any of these raw columns.
--
-- 100% derivable from existing tables -- VIEW, not a table, matching the
-- `actions` precedent (derived tier, no convergence stage needed).
CREATE TABLE IF NOT EXISTS delegation_facts (
    {TABLE_SPECS["delegation_facts"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_delegation_facts_parent_order
ON delegation_facts(parent_session_id, instruction_message_id, delegation_id);
CREATE INDEX IF NOT EXISTS idx_delegation_facts_state
ON delegation_facts(mapping_state, parent_session_id);
CREATE INDEX IF NOT EXISTS idx_delegation_facts_model
ON delegation_facts(requested_model, dispatch_turn_model, child_session_dominant_model);

-- The terminal ``delegations`` adapter reads this materialized relation
-- directly.  Count its maintenance writes even when their source update was
-- outside the terminal transcript tables above.
CREATE TRIGGER IF NOT EXISTS query_unit_frame_delegation_facts_insert
AFTER INSERT ON delegation_facts BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;
CREATE TRIGGER IF NOT EXISTS query_unit_frame_delegation_facts_update
AFTER UPDATE ON delegation_facts BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;
CREATE TRIGGER IF NOT EXISTS query_unit_frame_delegation_facts_delete
AFTER DELETE ON delegation_facts BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;

CREATE TABLE IF NOT EXISTS delegation_refresh_scope (
    {TABLE_SPECS["delegation_refresh_scope"].ddl_body}
) STRICT;

CREATE TABLE IF NOT EXISTS derived_refresh_guard (
    {TABLE_SPECS["derived_refresh_guard"].ddl_body}
) STRICT;

-- Provider-neutral topology and claims. This remains a derived tier: adapters
-- materialize admitted source facts here, while this slice deliberately does
-- not model repository effects or evaluated satisfaction.
CREATE TABLE IF NOT EXISTS work_evidence_graphs (
    {TABLE_SPECS["work_evidence_graphs"].ddl_body}
) STRICT;

CREATE TABLE IF NOT EXISTS work_evidence_nodes (
    {TABLE_SPECS["work_evidence_nodes"].ddl_body}
) STRICT;

CREATE TABLE IF NOT EXISTS work_evidence_edges (
    {TABLE_SPECS["work_evidence_edges"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_work_evidence_edges_source
ON work_evidence_edges(graph_id, source_ref, edge_kind);
CREATE INDEX IF NOT EXISTS idx_work_evidence_edges_target
ON work_evidence_edges(graph_id, target_ref, edge_kind);

CREATE VIEW IF NOT EXISTS delegation_facts_source AS
WITH dispatch_actions AS (
    SELECT
        a.session_id                           AS parent_session_id,
        a.message_id                           AS instruction_message_id,
        a.tool_use_block_id                    AS instruction_tool_use_block_id,
        a.tool_input                           AS instruction_payload,
        a.tool_result_block_id                 AS artifact_block_id,
        a.output_text                          AS artifact_text,
        a.is_error                             AS result_is_error,
        a.exit_code                            AS result_exit_code,
        m.model_name                           AS dispatch_turn_model,
        json_extract(a.tool_input, '$.model')  AS requested_model,
        -- Provider-asserted dispatch text: the exact field the child's own
        -- first turn reproduces verbatim (Claude Code: 'prompt'; Codex:
        -- 'message'; 'instruction'/'task' as generic fallbacks for other
        -- providers). NULLIF/json_type guards a non-text or absent field
        -- rather than coercing it into a spurious match.
        COALESCE(
            NULLIF(CASE WHEN json_type(a.tool_input, '$.prompt') = 'text'
                        THEN json_extract(a.tool_input, '$.prompt') END, ''),
            NULLIF(CASE WHEN json_type(a.tool_input, '$.message') = 'text'
                        THEN json_extract(a.tool_input, '$.message') END, ''),
            NULLIF(CASE WHEN json_type(a.tool_input, '$.instruction') = 'text'
                        THEN json_extract(a.tool_input, '$.instruction') END, ''),
            NULLIF(CASE WHEN json_type(a.tool_input, '$.task') = 'text'
                        THEN json_extract(a.tool_input, '$.task') END, '')
        )                                       AS dispatch_identity_text
    FROM actions a
    JOIN messages m ON m.message_id = a.message_id
    WHERE a.semantic_type = 'subagent'
      AND EXISTS (
          SELECT 1 FROM delegation_refresh_scope scope
          WHERE scope.parent_session_id = a.session_id
      )
),
resolved_children AS (
    SELECT
        l.resolved_dst_session_id              AS parent_session_id,
        l.src_session_id                       AS child_session_id,
        l.branch_point_message_id              AS branch_point_message_id,
        l.confidence                           AS link_confidence,
        l.method                               AS link_method,
        l.inheritance                          AS inheritance
    FROM session_links l
    WHERE l.link_type = 'subagent'
      AND l.resolved_dst_session_id IS NOT NULL
      AND {topology_status_composes_sql("l.status")}
      AND EXISTS (
          SELECT 1 FROM delegation_refresh_scope scope
          WHERE scope.parent_session_id = l.resolved_dst_session_id
      )
),
-- The child's own first user turn: for a subagent this literally IS the
-- text Claude Code/Codex injected as the dispatch's prompt, not a human
-- turn (material_origin is 'generated_context_pack', never
-- 'human_authored' -- see polylogue-1vpm.7 corpus audit), so this
-- deliberately does not filter on material_origin.
child_identity_text AS (
    SELECT
        m.session_id AS child_session_id,
        (
            SELECT b.text FROM blocks b
            WHERE b.message_id = m.message_id AND b.block_type = 'text'
            ORDER BY b.position LIMIT 1
        ) AS first_text
    FROM messages m
    WHERE m.role = 'user'
      AND m.message_type = 'message'
      AND m.position = (
          SELECT MIN(m2.position) FROM messages m2
          WHERE m2.session_id = m.session_id
            AND m2.role = 'user'
            AND m2.message_type = 'message'
      )
),
dispatch_counts AS (
    SELECT parent_session_id, COUNT(*) AS n FROM dispatch_actions GROUP BY parent_session_id
),
child_counts AS (
    SELECT parent_session_id, COUNT(*) AS n FROM resolved_children GROUP BY parent_session_id
),
-- Case 1: trivial cohort. Exactly one dispatch and exactly one resolved
-- child for this parent -- there is no second candidate on either side, so
-- pairing them is not a guess, it is the only interpretation the observed
-- facts admit. No corroborating content required.
trivial_pairs AS (
    SELECT
        d.parent_session_id                    AS parent_session_id,
        d.instruction_tool_use_block_id        AS instruction_tool_use_block_id,
        c.child_session_id                     AS child_session_id
    FROM dispatch_actions d
    JOIN resolved_children c ON c.parent_session_id = d.parent_session_id
    JOIN dispatch_counts dc ON dc.parent_session_id = d.parent_session_id AND dc.n = 1
    JOIN child_counts cc ON cc.parent_session_id = d.parent_session_id AND cc.n = 1
),
-- Case 2: non-trivial cohort, disambiguated by provider-asserted content
-- identity (see the delegation_facts table comment above for the corpus
-- verification numbers).
content_pairs AS (
    SELECT
        d.parent_session_id                    AS parent_session_id,
        d.instruction_tool_use_block_id        AS instruction_tool_use_block_id,
        c.child_session_id                     AS child_session_id
    FROM dispatch_actions d
    JOIN resolved_children c ON c.parent_session_id = d.parent_session_id
    JOIN child_identity_text t ON t.child_session_id = c.child_session_id
    WHERE d.dispatch_identity_text IS NOT NULL
      AND d.dispatch_identity_text = t.first_text
),
candidate_pairs AS (
    SELECT * FROM trivial_pairs
    UNION
    SELECT * FROM content_pairs
),
dispatch_match_counts AS (
    SELECT parent_session_id, instruction_tool_use_block_id, COUNT(DISTINCT child_session_id) AS n
    FROM candidate_pairs GROUP BY 1, 2
),
child_match_counts AS (
    SELECT parent_session_id, child_session_id, COUNT(DISTINCT instruction_tool_use_block_id) AS n
    FROM candidate_pairs GROUP BY 1, 2
),
-- A candidate pair survives only when it is the UNIQUE candidate in both
-- directions: this dispatch matches exactly one child AND this child
-- matches exactly one dispatch. A collision on either side (duplicate
-- prompts, or genuinely ambiguous evidence) is excluded here and falls
-- through to unresolved/edge_only rather than fabricating a winner.
identity_pairs AS (
    SELECT DISTINCT
        cp.parent_session_id                   AS parent_session_id,
        cp.instruction_tool_use_block_id        AS instruction_tool_use_block_id,
        cp.child_session_id                    AS child_session_id
    FROM candidate_pairs cp
    JOIN dispatch_match_counts dmc
      ON dmc.parent_session_id = cp.parent_session_id
     AND dmc.instruction_tool_use_block_id = cp.instruction_tool_use_block_id
     AND dmc.n = 1
    JOIN child_match_counts cmc
      ON cmc.parent_session_id = cp.parent_session_id
     AND cmc.child_session_id = cp.child_session_id
     AND cmc.n = 1
),
resolved_rows AS (
    SELECT
        d.parent_session_id                    AS parent_session_id,
        ip.child_session_id                    AS child_session_id,
        'resolved'                              AS mapping_state,
        c.link_confidence                      AS link_confidence,
        'content-identity-match'                AS link_method,
        c.inheritance                          AS inheritance,
        c.branch_point_message_id              AS branch_point_message_id,
        d.instruction_message_id               AS instruction_message_id,
        d.instruction_tool_use_block_id        AS instruction_tool_use_block_id,
        d.instruction_payload                  AS instruction_payload,
        d.dispatch_turn_model                  AS dispatch_turn_model,
        d.requested_model                      AS requested_model,
        d.artifact_block_id                    AS artifact_block_id,
        d.artifact_text                        AS artifact_text,
        d.result_is_error                      AS result_is_error,
        d.result_exit_code                     AS result_exit_code
    FROM dispatch_actions d
    JOIN identity_pairs ip
      ON ip.parent_session_id = d.parent_session_id
     AND ip.instruction_tool_use_block_id = d.instruction_tool_use_block_id
    JOIN resolved_children c
      ON c.parent_session_id = ip.parent_session_id
     AND c.child_session_id = ip.child_session_id
),
unresolved_rows AS (
    SELECT
        d.parent_session_id                    AS parent_session_id,
        NULL                                    AS child_session_id,
        'unresolved'                             AS mapping_state,
        NULL AS link_confidence, NULL AS link_method, NULL AS inheritance, NULL AS branch_point_message_id,
        d.instruction_message_id               AS instruction_message_id,
        d.instruction_tool_use_block_id        AS instruction_tool_use_block_id,
        d.instruction_payload                  AS instruction_payload,
        d.dispatch_turn_model                  AS dispatch_turn_model,
        d.requested_model                      AS requested_model,
        d.artifact_block_id                    AS artifact_block_id,
        d.artifact_text                        AS artifact_text,
        d.result_is_error                      AS result_is_error,
        d.result_exit_code                     AS result_exit_code
    FROM dispatch_actions d
    WHERE NOT EXISTS (
        SELECT 1 FROM identity_pairs ip
        WHERE ip.parent_session_id = d.parent_session_id
          AND ip.instruction_tool_use_block_id = d.instruction_tool_use_block_id
    )
),
edge_only_rows AS (
    SELECT
        c.parent_session_id                    AS parent_session_id,
        c.child_session_id                     AS child_session_id,
        'edge_only'                              AS mapping_state,
        c.link_confidence                      AS link_confidence,
        c.link_method                          AS link_method,
        c.inheritance                          AS inheritance,
        c.branch_point_message_id              AS branch_point_message_id,
        NULL AS instruction_message_id, NULL AS instruction_tool_use_block_id, NULL AS instruction_payload,
        NULL AS dispatch_turn_model, NULL AS requested_model,
        NULL AS artifact_block_id, NULL AS artifact_text, NULL AS result_is_error, NULL AS result_exit_code
    FROM resolved_children c
    WHERE NOT EXISTS (
        SELECT 1 FROM identity_pairs ip
        WHERE ip.parent_session_id = c.parent_session_id
          AND ip.child_session_id = c.child_session_id
    )
),
quarantined_rows AS (
    SELECT
        l.resolved_dst_session_id              AS parent_session_id,
        l.src_session_id                       AS child_session_id,
        l.status                               AS mapping_state,
        l.confidence                           AS link_confidence,
        l.method                               AS link_method,
        l.inheritance                          AS inheritance,
        l.branch_point_message_id              AS branch_point_message_id,
        NULL AS instruction_message_id, NULL AS instruction_tool_use_block_id, NULL AS instruction_payload,
        NULL AS dispatch_turn_model, NULL AS requested_model,
        NULL AS artifact_block_id, NULL AS artifact_text, NULL AS result_is_error, NULL AS result_exit_code
    FROM session_links l
    WHERE l.link_type = 'subagent'
      AND {topology_status_excluded_sql("l.status")}
      AND l.resolved_dst_session_id IS NOT NULL
      AND EXISTS (
          SELECT 1 FROM delegation_refresh_scope scope
          WHERE scope.parent_session_id = l.resolved_dst_session_id
      )
),
attempts AS (
    SELECT * FROM resolved_rows
    UNION ALL SELECT * FROM unresolved_rows
    UNION ALL SELECT * FROM edge_only_rows
    UNION ALL SELECT * FROM quarantined_rows
)
SELECT
    att.parent_session_id                      AS parent_session_id,
    att.child_session_id                       AS child_session_id,
    att.mapping_state                          AS mapping_state,
    att.link_confidence                        AS link_confidence,
    att.link_method                            AS link_method,
    att.inheritance                            AS inheritance,
    att.branch_point_message_id                AS branch_point_message_id,
    att.instruction_message_id                 AS instruction_message_id,
    att.instruction_tool_use_block_id          AS instruction_tool_use_block_id,
    att.instruction_payload                    AS instruction_payload,
    att.dispatch_turn_model                    AS dispatch_turn_model,
    att.requested_model                        AS requested_model,
    att.artifact_block_id                      AS artifact_block_id,
    att.artifact_text                          AS artifact_text,
    att.result_is_error                        AS result_is_error,
    att.result_exit_code                       AS result_exit_code,
    CASE
        WHEN att.instruction_tool_use_block_id IS NULL THEN 'unknown'
        WHEN att.result_is_error IS NULL               THEN 'unknown'
        WHEN att.result_is_error = 1                   THEN 'error'
        ELSE 'ok'
    END                                          AS result_status,
    p.origin                                    AS parent_origin,
    pp.primary_model_name                       AS parent_session_dominant_model,
    pp.primary_model_family                     AS parent_session_dominant_model_family,
    pp.terminal_state                           AS parent_terminal_state,
    cp.primary_model_name                       AS child_session_dominant_model,
    cp.primary_model_family                     AS child_session_dominant_model_family,
    (SELECT COALESCE(SUM(u.cost_usd),
                     (SELECT reported_cost_usd FROM sessions WHERE session_id = att.child_session_id))
       FROM session_model_usage u
       WHERE u.session_id = att.child_session_id) AS child_cost_usd,
    (SELECT CASE WHEN COUNT(u.model_name) = 0 THEN NULL
                 WHEN COUNT(u.cost_usd) = COUNT(u.model_name) THEN 0 ELSE 1 END
       FROM session_model_usage u
       WHERE u.session_id = att.child_session_id) AS child_cost_is_estimated,
    (SELECT SUM(COALESCE(u.input_tokens, 0) + COALESCE(u.output_tokens, 0)
       + COALESCE(u.cache_read_tokens, 0) + COALESCE(u.cache_write_tokens, 0))
       FROM session_model_usage u
       WHERE u.session_id = att.child_session_id) AS child_tokens,
    cp.wall_duration_ms                         AS child_wall_ms,
    cp.terminal_state                           AS child_terminal_state
FROM attempts att
JOIN sessions p ON p.session_id = att.parent_session_id
LEFT JOIN session_profiles pp ON pp.session_id = att.parent_session_id
LEFT JOIN session_profiles cp ON cp.session_id = att.child_session_id;

CREATE TRIGGER IF NOT EXISTS blocks_action_pairs_ai
AFTER INSERT ON blocks
WHEN NOT EXISTS (SELECT 1 FROM derived_refresh_guard WHERE guard_name = 'session-write') BEGIN
    DELETE FROM action_pairs WHERE session_id = NEW.session_id;
    {action_pairs_refresh_sql("NEW.session_id")};
    DELETE FROM delegation_facts WHERE parent_session_id = NEW.session_id;
    INSERT OR REPLACE INTO delegation_refresh_scope(parent_session_id) VALUES (NEW.session_id);
    {delegation_facts_insert_sql("NEW.session_id")};
    DELETE FROM delegation_refresh_scope;
END;

CREATE TRIGGER IF NOT EXISTS blocks_action_pairs_ad
AFTER DELETE ON blocks
WHEN NOT EXISTS (SELECT 1 FROM derived_refresh_guard WHERE guard_name = 'session-write') BEGIN
    DELETE FROM action_pairs WHERE session_id = OLD.session_id;
    {action_pairs_refresh_sql("OLD.session_id")};
    DELETE FROM delegation_facts WHERE parent_session_id = OLD.session_id;
    INSERT OR REPLACE INTO delegation_refresh_scope(parent_session_id) VALUES (OLD.session_id);
    {delegation_facts_insert_sql("OLD.session_id")};
    DELETE FROM delegation_refresh_scope;
END;

CREATE TRIGGER IF NOT EXISTS blocks_action_pairs_au
AFTER UPDATE ON blocks
WHEN NOT EXISTS (SELECT 1 FROM derived_refresh_guard WHERE guard_name = 'session-write') BEGIN
    DELETE FROM action_pairs WHERE session_id = NEW.session_id;
    {action_pairs_refresh_sql("NEW.session_id")};
    DELETE FROM delegation_facts WHERE parent_session_id = NEW.session_id;
    INSERT OR REPLACE INTO delegation_refresh_scope(parent_session_id) VALUES (NEW.session_id);
    {delegation_facts_insert_sql("NEW.session_id")};
    DELETE FROM delegation_refresh_scope;
END;

CREATE TRIGGER IF NOT EXISTS session_links_delegation_facts_ai
AFTER INSERT ON session_links
WHEN NEW.resolved_dst_session_id IS NOT NULL
 AND NOT EXISTS (SELECT 1 FROM derived_refresh_guard WHERE guard_name = 'session-write') BEGIN
    DELETE FROM delegation_facts WHERE parent_session_id = NEW.resolved_dst_session_id;
    INSERT OR REPLACE INTO delegation_refresh_scope(parent_session_id) VALUES (NEW.resolved_dst_session_id);
    {delegation_facts_insert_sql("NEW.resolved_dst_session_id")};
    DELETE FROM delegation_refresh_scope;
END;

CREATE TRIGGER IF NOT EXISTS session_profiles_delegation_facts_ai
AFTER INSERT ON session_profiles
WHEN NOT EXISTS (SELECT 1 FROM derived_refresh_guard WHERE guard_name = 'session-write') BEGIN
    DELETE FROM delegation_facts WHERE parent_session_id = NEW.session_id;
    INSERT OR REPLACE INTO delegation_refresh_scope(parent_session_id) VALUES (NEW.session_id);
    {delegation_facts_insert_sql("NEW.session_id")};
    DELETE FROM delegation_refresh_scope;
END;

CREATE TRIGGER IF NOT EXISTS session_links_delegation_facts_au
AFTER UPDATE ON session_links
WHEN NOT EXISTS (SELECT 1 FROM derived_refresh_guard WHERE guard_name = 'session-write') BEGIN
    DELETE FROM delegation_facts
    WHERE parent_session_id IN (OLD.resolved_dst_session_id, NEW.resolved_dst_session_id);
    INSERT OR REPLACE INTO delegation_refresh_scope(parent_session_id)
    SELECT parent_session_id
    FROM (
        SELECT OLD.resolved_dst_session_id AS parent_session_id
        UNION
        SELECT NEW.resolved_dst_session_id
    )
    WHERE parent_session_id IS NOT NULL;
    {delegation_facts_insert_sql("NEW.resolved_dst_session_id")};
    DELETE FROM delegation_refresh_scope;
END;

CREATE TRIGGER IF NOT EXISTS session_profiles_delegation_facts_au
AFTER UPDATE ON session_profiles
WHEN NOT EXISTS (SELECT 1 FROM derived_refresh_guard WHERE guard_name = 'session-write') BEGIN
    -- A profile row may be the child-side evidence of many parent
    -- delegations. Preserve those parent cohorts before replacing the stale
    -- derived rows; refreshing only NEW.session_id leaves child model/cost
    -- projections stale.
    INSERT OR REPLACE INTO delegation_refresh_scope(parent_session_id)
    SELECT parent_session_id
    FROM delegation_facts
    WHERE child_session_id = NEW.session_id;
    DELETE FROM delegation_facts
    WHERE parent_session_id = NEW.session_id OR child_session_id = NEW.session_id;
    INSERT OR REPLACE INTO delegation_refresh_scope(parent_session_id) VALUES (NEW.session_id);
    {delegation_facts_insert_sql("NEW.session_id")};
    DELETE FROM delegation_refresh_scope;
END;

CREATE VIEW IF NOT EXISTS delegations AS
SELECT
    parent_session_id, child_session_id, mapping_state, link_confidence, link_method, inheritance,
    branch_point_message_id, instruction_message_id, instruction_tool_use_block_id, instruction_payload,
    dispatch_turn_model, requested_model, artifact_block_id, artifact_text, result_is_error, result_exit_code,
    result_status, parent_origin, parent_session_dominant_model, parent_session_dominant_model_family,
    parent_terminal_state, child_session_dominant_model, child_session_dominant_model_family, child_cost_usd,
    child_cost_is_estimated, child_tokens, child_wall_ms, child_terminal_state
FROM delegation_facts;

CREATE TABLE IF NOT EXISTS session_tag_rollups (
    {TABLE_SPECS["session_tag_rollups"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_tag_rollups_day
ON session_tag_rollups(bucket_day DESC, source_name, tag);

CREATE INDEX IF NOT EXISTS idx_session_tag_rollups_provider
ON session_tag_rollups(source_name, tag);

-- v57 (polylogue-ioz7): one immutable receipt per `sessions` row deleted by
-- the agent-meta-sidecar purge actuator. No FK to `sessions` -- the whole
-- point of the row is that the session it describes no longer exists.
-- `raw_id` is likewise unconstrained here (source.db is a separate
-- database file; the raw row and its blob are deliberately retained there).
CREATE TABLE IF NOT EXISTS agent_meta_sidecar_purge_receipts (
    {TABLE_SPECS["agent_meta_sidecar_purge_receipts"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_agent_meta_sidecar_purge_receipts_purged_at
ON agent_meta_sidecar_purge_receipts(purged_at_ms);
"""

# polylogue-a7xr.5 consolidated the FTS trigger CREATE statements into
# storage/fts/sql.py as the single canonical source (also used by the
# repair-path lifecycle in fts_lifecycle.py), but a fresh-database bootstrap
# still needs them appended to the script it executescript()s -- without
# this, a freshly created index.db has the messages_fts/
# session_work_events_fts virtual tables but no triggers populating them,
# so every insert silently produces an empty search index. Regression:
# devtools render demo-corpus-datasheet started failing with "Search index
# is incomplete" right after #2893 landed.
INDEX_DDL = INDEX_DDL + "\n\n" + ";\n\n".join(FTS_TRIGGER_DDL) + ";\n"

__all__ = ["FTS_FRESHNESS_STATE_DDL", "INDEX_DDL", "INDEX_SCHEMA_VERSION"]
