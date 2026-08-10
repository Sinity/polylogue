"""Source-tier raw-capture DDL fragment for archive.

source.db stores acquired bytes and source evidence. Parsed/read-model tables
live in index.db and are rebuilt from this tier.
"""

from __future__ import annotations

from typing import get_args

from polylogue.archive.revision_authority import RawRevisionAuthority
from polylogue.archive.session_revision_membership import MembershipDecision
from polylogue.core.enums import (
    ArtifactSupportStatus,
    Origin,
    Provider,
    RawAuthorityVerdict,
    ValidationMode,
    ValidationStatus,
)
from polylogue.storage.sqlite.archive_tiers.common import check, literal_check, nullable_check
from polylogue.storage.sqlite.archive_tiers.types import ProvenRevisionAuthority

SOURCE_SCHEMA_VERSION = 31

SOURCE_DDL = f"""
CREATE TABLE IF NOT EXISTS raw_sessions (
    raw_id                  TEXT PRIMARY KEY,
    origin                  TEXT NOT NULL CHECK ({check("origin", Origin)}),
    capture_mode            TEXT CHECK ({nullable_check("capture_mode", Provider)}),
    native_id               TEXT,
    source_path             TEXT NOT NULL,
    source_index            INTEGER NOT NULL DEFAULT 0,
    blob_hash               BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size               INTEGER NOT NULL CHECK(blob_size >= 0),
    acquired_at_ms          INTEGER NOT NULL,
    file_mtime_ms           INTEGER,
    parsed_at_ms            INTEGER,
    parse_error             TEXT,
    validated_at_ms         INTEGER,
    validation_status       TEXT CHECK ({nullable_check("validation_status", ValidationStatus)}),
    validation_error        TEXT,
    validation_drift_count  INTEGER NOT NULL DEFAULT 0 CHECK(validation_drift_count >= 0),
    validation_mode         TEXT CHECK ({nullable_check("validation_mode", ValidationMode)}),
    detection_warnings_json TEXT NOT NULL DEFAULT '[]'
    ,logical_source_key      TEXT
    ,revision_kind           TEXT NOT NULL DEFAULT 'unknown'
        CHECK(revision_kind IN ('full', 'append', 'unknown'))
    ,source_revision         TEXT
    ,predecessor_source_revision TEXT
    ,predecessor_raw_id      TEXT
    ,baseline_raw_id         TEXT
    ,append_start_offset     INTEGER CHECK(append_start_offset >= 0)
    ,append_end_offset       INTEGER CHECK(append_end_offset > append_start_offset)
    ,acquisition_generation  INTEGER CHECK(acquisition_generation >= 0)
    ,revision_authority      TEXT NOT NULL DEFAULT 'quarantined'
        CHECK ({check("revision_authority", RawRevisionAuthority)})
    ,revision_authority_evidence TEXT
        CHECK(revision_authority_evidence IS NULL OR revision_authority_evidence IN ('live_source_verification_v1'))
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_sessions_origin
ON raw_sessions(origin);

CREATE INDEX IF NOT EXISTS idx_raw_sessions_origin_native
ON raw_sessions(origin, native_id)
WHERE native_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_raw_sessions_source_path
ON raw_sessions(source_path, source_index);

CREATE INDEX IF NOT EXISTS idx_raw_sessions_parse_ready
ON raw_sessions(raw_id)
WHERE parsed_at_ms IS NULL
  AND validated_at_ms IS NOT NULL
  AND (validation_status IS NULL OR validation_status != 'failed');

CREATE INDEX IF NOT EXISTS idx_raw_sessions_logical_revision
ON raw_sessions(logical_source_key, acquisition_generation, raw_id)
WHERE logical_source_key IS NOT NULL;

-- v14 (polylogue blob-store audit): blob GC's per-candidate reference check
-- (storage/blob_gc.py:_archive_reference_surfaces) queries
-- "raw_sessions WHERE blob_hash = ?" for every candidate blob on disk, every
-- periodic pass. Without this index, that lookup is a full table scan and
-- the (mostly-empty, since dedup + attachments-tier retention keep almost
-- everything referenced) candidate scan takes minutes per pass -- the whole
-- daemon write pipeline stalls that long since blob-gc holds the process-wide
-- write-coordinator gate for the entire call, not only its bounded final
-- delete phase.
CREATE INDEX IF NOT EXISTS idx_raw_sessions_blob_hash
ON raw_sessions(blob_hash);

-- v15 (polylogue-hord): backs IndexGenerationStore.next_raw_page's content-order
-- rebuild paging (ORDER BY blob_hash, raw_id / keyset WHERE on the same pair)
-- so a full rebuild's per-page scheduling query stays an index scan instead of
-- re-sorting the remaining table on every bounded pass as the archive grows.
CREATE INDEX IF NOT EXISTS idx_raw_sessions_blob_hash_raw_id
ON raw_sessions(blob_hash, raw_id);

-- v16 (polylogue-buns): durable multimap of every acquisition mode ever
-- observed for one raw_id. raw_sessions.capture_mode above caches only the
-- FIRST known mode (write-time UPDATE is gated `WHERE capture_mode IS
-- NULL`), which silently discards any later, different capture_mode for a
-- raw_id that collides on content hash -- e.g. a GEMINI export and a live
-- DRIVE acquisition of byte-identical bytes. This table keeps one row per
-- distinct (raw_id, capture_mode) ever observed so that fact is never lost,
-- and lets a caller read an explicit unambiguous/ambiguous resolution
-- instead of only ever seeing the first-cached value.
CREATE TABLE IF NOT EXISTS raw_capture_observations (
    raw_id               TEXT NOT NULL REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    capture_mode         TEXT NOT NULL CHECK ({check("capture_mode", Provider)}),
    first_observed_at_ms INTEGER NOT NULL CHECK(first_observed_at_ms >= 0),
    PRIMARY KEY (raw_id, capture_mode)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_capture_observations_raw_id
ON raw_capture_observations(raw_id);

CREATE TABLE IF NOT EXISTS raw_session_memberships (
    raw_id                  TEXT NOT NULL REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    logical_source_key      TEXT NOT NULL,
    provider_session_id     TEXT NOT NULL,
    source_revision         TEXT NOT NULL,
    normalized_content_hash BLOB NOT NULL CHECK(length(normalized_content_hash) = 32),
    message_count           INTEGER NOT NULL CHECK(message_count >= 0),
    predecessor_raw_id      TEXT,
    acquisition_generation  INTEGER NOT NULL DEFAULT 0 CHECK(acquisition_generation >= 0),
    -- Narrower than every other revision_authority/previous_revision_authority
    -- column: this value is always an OUTPUT of membership classification
    -- (revision_governance.py's writeback), which never emits ASSERTED --
    -- see ProvenRevisionAuthority in archive_tiers/types.py (polylogue-h57ic).
    revision_authority      TEXT NOT NULL DEFAULT 'quarantined'
        CHECK ({literal_check("revision_authority", *get_args(ProvenRevisionAuthority))}),
    decision                TEXT CHECK ({nullable_check("decision", MembershipDecision)}),
    decided_at_ms           INTEGER CHECK(decided_at_ms >= 0),
    PRIMARY KEY(raw_id, logical_source_key),
    CHECK((decision IS NULL) = (decided_at_ms IS NULL))
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_session_memberships_logical
ON raw_session_memberships(logical_source_key, acquisition_generation, raw_id);

CREATE INDEX IF NOT EXISTS idx_raw_session_memberships_pending
ON raw_session_memberships(raw_id)
WHERE decision IS NULL OR decision IN ('ambiguous', 'deferred');

CREATE TABLE IF NOT EXISTS raw_membership_census (
    raw_id             TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    parser_fingerprint TEXT NOT NULL,
    status             TEXT NOT NULL CHECK(status IN ('complete', 'failed', 'non_session')),
    member_count       INTEGER NOT NULL CHECK(member_count >= 0),
    censused_at_ms     INTEGER NOT NULL CHECK(censused_at_ms >= 0),
    detail             TEXT NOT NULL DEFAULT ''
) STRICT;

-- v17 (polylogue-u19l): one immutable receipt per raw_sessions row promoted
-- out of quarantine by live-source-verification (see
-- polylogue.storage.live_source_reconciliation +
-- polylogue.maintenance.raw_live_source_reconciliation_apply). Records what
-- was compared, what matched, when, by which tool version, and against
-- which verified backup manifest -- never edited after being written.
CREATE TABLE IF NOT EXISTS raw_live_source_reconciliation_receipts (
    raw_id                      TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    verdict                     TEXT NOT NULL CHECK(verdict IN ('exact_match', 'codex_header_strip_match')),
    previous_revision_authority TEXT NOT NULL CHECK ({check("previous_revision_authority", RawRevisionAuthority)}),
    source_path                 TEXT NOT NULL,
    blob_hash                   BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size                   INTEGER NOT NULL CHECK(blob_size >= 0),
    compared_at_ms              INTEGER NOT NULL CHECK(compared_at_ms >= 0),
    tool_version                TEXT NOT NULL,
    backup_manifest_path        TEXT NOT NULL,
    detail                      TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_live_source_reconciliation_receipts_compared_at
ON raw_live_source_reconciliation_receipts(compared_at_ms);

-- v19 (polylogue-lb39z, Phase 1 item 2): one immutable receipt per
-- raw_sessions row promoted out of quarantine because its membership
-- pipeline verdict (raw_session_memberships.decision) was already decided
-- but never written back to revision_authority. See
-- polylogue.storage.raw_membership_writeback +
-- polylogue.maintenance.raw_membership_writeback_apply.
CREATE TABLE IF NOT EXISTS raw_membership_writeback_receipts (
    raw_id                      TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    logical_source_key          TEXT NOT NULL,
    provider_session_id         TEXT NOT NULL,
    membership_decision         TEXT NOT NULL CHECK ({
    literal_check(
        "membership_decision",
        MembershipDecision.APPLIED,
        MembershipDecision.SUPERSEDED_EQUIVALENT,
        MembershipDecision.SUPERSEDED_PREFIX,
    )
}),
    previous_revision_authority TEXT NOT NULL CHECK ({check("previous_revision_authority", RawRevisionAuthority)}),
    promoted_at_ms              INTEGER NOT NULL CHECK(promoted_at_ms >= 0),
    tool_version                TEXT NOT NULL,
    backup_manifest_path        TEXT NOT NULL,
    detail                      TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_membership_writeback_receipts_promoted_at
ON raw_membership_writeback_receipts(promoted_at_ms);

-- v20 (polylogue-lb39z, Phase 1 item 3): one immutable receipt per
-- raw_sessions row promoted out of quarantine because its own claimed
-- [append_start_offset:append_end_offset) byte range was proven directly
-- against its live source file's current bytes -- the membershipless
-- append-chain-backfill population (a row stuck quarantined with no
-- raw_session_memberships row at all because its predecessor is itself
-- unresolved, so the normal _promote_contiguous_append_evidence cascade can
-- never reach it). Reuses revision_authority_evidence=
-- 'live_source_verification_v1' (the proof mechanism is identical to
-- polylogue-u19l's); this table's own existence records the distinct target
-- population. See polylogue.storage.raw_append_chain_backfill +
-- polylogue.maintenance.raw_append_chain_backfill_apply.
CREATE TABLE IF NOT EXISTS raw_append_chain_backfill_receipts (
    raw_id                           TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    logical_source_key                TEXT,
    source_path                      TEXT NOT NULL,
    blob_hash                        BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size                        INTEGER NOT NULL CHECK(blob_size >= 0),
    append_start_offset              INTEGER NOT NULL CHECK(append_start_offset >= 0),
    append_end_offset                INTEGER NOT NULL CHECK(append_end_offset > append_start_offset),
    matched_after_codex_header_strip INTEGER NOT NULL CHECK(matched_after_codex_header_strip IN (0, 1)),
    previous_revision_authority      TEXT NOT NULL CHECK ({check("previous_revision_authority", RawRevisionAuthority)}),
    compared_at_ms                    INTEGER NOT NULL CHECK(compared_at_ms >= 0),
    tool_version                      TEXT NOT NULL,
    backup_manifest_path              TEXT NOT NULL,
    detail                            TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_append_chain_backfill_receipts_compared_at
ON raw_append_chain_backfill_receipts(compared_at_ms);

-- v22 (polylogue-6753s): one immutable receipt per raw_sessions row promoted
-- out of quarantine because it is byte-identical (same blob_hash) to some
-- OTHER raw_sessions row that already has a materialized session in
-- index.db -- a pure re-acquisition/re-sync of already-archived content, not
-- missing content. Deliberately does not widen revision_authority_evidence's
-- closed CHECK vocabulary (would require a full raw_sessions table rebuild,
-- see migration 021); this distinct, independently-verifiable evidence
-- mechanism gets its own receipt table instead, exactly like
-- raw_membership_writeback_receipts (v19) already does for its own distinct
-- mechanism. See polylogue.storage.raw_byte_duplicate_supersession +
-- polylogue.maintenance.raw_byte_duplicate_supersession_apply.
CREATE TABLE IF NOT EXISTS raw_byte_duplicate_supersession_receipts (
    raw_id                      TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    blob_hash                   BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size                   INTEGER NOT NULL CHECK(blob_size >= 0),
    duplicate_of_raw_id         TEXT NOT NULL,
    duplicate_of_session_id     TEXT NOT NULL,
    previous_revision_authority TEXT NOT NULL CHECK(previous_revision_authority IN ('asserted', 'byte_proven', 'quarantined')),
    promoted_at_ms              INTEGER NOT NULL CHECK(promoted_at_ms >= 0),
    tool_version                TEXT NOT NULL,
    backup_manifest_path        TEXT NOT NULL,
    detail                      TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_byte_duplicate_supersession_receipts_promoted_at
ON raw_byte_duplicate_supersession_receipts(promoted_at_ms);

CREATE INDEX IF NOT EXISTS idx_raw_byte_duplicate_supersession_receipts_duplicate_of
ON raw_byte_duplicate_supersession_receipts(duplicate_of_raw_id);

-- v29 (polylogue-dyica): a stopped daemon can leave historical parse failures
-- whose retained bytes are known terminal inputs, but which predate the live
-- writer's typed raw_artifacts outcome.  This immutable receipt records the
-- reviewed manifest and source-tier backup that authorized replacing the
-- stale artifact classification with a terminal lifecycle classification.
-- The raw bytes and original parse diagnostic remain untouched.
CREATE TABLE IF NOT EXISTS raw_failure_disposition_receipts (
    raw_id                     TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    artifact_id                TEXT NOT NULL UNIQUE REFERENCES raw_artifacts(artifact_id),
    origin                     TEXT NOT NULL CHECK ({check("origin", Origin)}),
    source_path                TEXT NOT NULL,
    source_index               INTEGER NOT NULL,
    blob_hash                  BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size                  INTEGER NOT NULL CHECK(blob_size >= 0),
    previous_parse_error       TEXT NOT NULL,
    previous_validation_status TEXT,
    previous_artifact_kind     TEXT NOT NULL,
    previous_support_status    TEXT NOT NULL,
    previous_classification_reason TEXT NOT NULL,
    disposition_kind           TEXT NOT NULL CHECK(disposition_kind IN (
        'terminal_corrupt_input',
        'terminal_unsupported_shape'
    )),
    manifest_sha256            TEXT NOT NULL CHECK(length(manifest_sha256) = 64),
    disposed_at_ms             INTEGER NOT NULL CHECK(disposed_at_ms >= 0),
    tool_version               TEXT NOT NULL,
    backup_manifest_path       TEXT NOT NULL,
    detail                     TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_failure_disposition_receipts_disposed_at
ON raw_failure_disposition_receipts(disposed_at_ms);

-- v27 (polylogue-r9xsj): a non-session classification alone is not a
-- duplicate disposition. This immutable receipt binds an excluded raw blob to
-- the indexed twin whose bytes make the exclusion safe.
CREATE TABLE IF NOT EXISTS raw_non_session_duplicate_exclusion_receipts (
    raw_id                     TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    blob_hash                  BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size                  INTEGER NOT NULL CHECK(blob_size >= 0),
    indexed_twin_raw_id        TEXT NOT NULL,
    indexed_twin_session_id    TEXT NOT NULL,
    parser_fingerprint         TEXT NOT NULL,
    excluded_at_ms             INTEGER NOT NULL CHECK(excluded_at_ms >= 0),
    tool_version               TEXT NOT NULL,
    detail                     TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_non_session_duplicate_exclusion_receipts_twin
ON raw_non_session_duplicate_exclusion_receipts(indexed_twin_raw_id);

-- v25 (polylogue-zm4w8): one immutable receipt per raw_sessions row marked a
-- proven duplicate within a fully-quarantined (source_path, blob_hash) group
-- -- a group where EVERY member starts out quarantined (unlike v22's
-- raw_byte_duplicate_supersession_receipts above, which requires an already-
-- INDEXED twin). A distinct actuator (devtools workspace
-- raw-quarantine-group-dedup-apply) promotes exactly one representative raw
-- per group through the real ingest/materialization path so it becomes a
-- genuine indexed session, then marks the rest of the group 'byte_proven'
-- (reusing the existing closed revision_authority vocabulary -- there is no
-- 'superseded' member and widening it needs a full raw_sessions table
-- rebuild, migration 021's own precedent) with a receipt here pointing at
-- the representative raw and its newly materialized session. See
-- polylogue.storage.raw_quarantine_group_dedup +
-- polylogue.maintenance.raw_quarantine_group_dedup_apply.
CREATE TABLE IF NOT EXISTS raw_quarantine_group_dedup_receipts (
    raw_id                     TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    source_path                TEXT NOT NULL,
    blob_hash                  BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size                  INTEGER NOT NULL CHECK(blob_size >= 0),
    representative_raw_id      TEXT NOT NULL,
    representative_session_id  TEXT NOT NULL,
    promoted_at_ms              INTEGER NOT NULL CHECK(promoted_at_ms >= 0),
    tool_version                TEXT NOT NULL,
    backup_manifest_path        TEXT NOT NULL,
    detail                      TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_quarantine_group_dedup_receipts_promoted_at
ON raw_quarantine_group_dedup_receipts(promoted_at_ms);

CREATE INDEX IF NOT EXISTS idx_raw_quarantine_group_dedup_receipts_representative
ON raw_quarantine_group_dedup_receipts(representative_raw_id);

-- v26 (polylogue-s8s54): durable receipt for the narrow browser-capture
-- unknown-export repair. The actuator changes only source.db's origin and
-- capture_mode. The index.db generated session identity is intentionally
-- repaired later by the normal reparse route, never by this source-tier pass.
CREATE TABLE IF NOT EXISTS raw_unknown_export_reclassification_receipts (
    raw_id                  TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    previous_origin         TEXT NOT NULL CHECK(previous_origin = 'unknown-export'),
    new_origin              TEXT NOT NULL CHECK(new_origin = 'chatgpt-export'),
    previous_capture_mode   TEXT,
    new_capture_mode        TEXT NOT NULL CHECK(new_capture_mode = 'chatgpt'),
    embedded_provider       TEXT NOT NULL CHECK(embedded_provider = 'chatgpt'),
    source_path             TEXT NOT NULL,
    blob_hash               BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size               INTEGER NOT NULL CHECK(blob_size >= 0),
    reclassified_at_ms      INTEGER NOT NULL CHECK(reclassified_at_ms >= 0),
    tool_version            TEXT NOT NULL,
    backup_manifest_path    TEXT NOT NULL,
    index_reparse_required  INTEGER NOT NULL CHECK(index_reparse_required = 1),
    detail                  TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_unknown_export_reclassification_receipts_reclassified_at
ON raw_unknown_export_reclassification_receipts(reclassified_at_ms);

-- Durable authority reconciliation ledger.  The source tier owns this
-- evidence because index.db and ops.db are rebuildable/disposable: neither
-- can be the authority for whether an accepted replay plan was conserved.
CREATE TABLE IF NOT EXISTS raw_authority_parser_census (
    raw_id                  TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    parser_fingerprint      TEXT NOT NULL,
    status                  TEXT NOT NULL CHECK(status IN ('complete', 'failed')),
    logical_keys_json       TEXT NOT NULL CHECK(json_valid(logical_keys_json)),
    detail                  TEXT NOT NULL DEFAULT '',
    censused_at_ms          INTEGER NOT NULL CHECK(censused_at_ms >= 0)
) STRICT;

CREATE TABLE IF NOT EXISTS raw_authority_censuses (
    census_id               TEXT PRIMARY KEY,
    sequence_no             INTEGER NOT NULL UNIQUE CHECK(sequence_no > 0),
    scope_json              TEXT NOT NULL CHECK(json_valid(scope_json)),
    residual_json           TEXT NOT NULL CHECK(json_valid(residual_json)),
    parser_fingerprint      TEXT NOT NULL,
    mode                    TEXT NOT NULL CHECK(mode IN ('census', 'dry_run', 'apply')),
    lifecycle_status        TEXT NOT NULL CHECK(lifecycle_status IN ('planned', 'completed', 'interrupted')),
    quiescent               INTEGER NOT NULL CHECK(quiescent IN (0, 1)),
    inventory_digest        TEXT NOT NULL CHECK(length(inventory_digest) = 64),
    residual_digest         TEXT NOT NULL CHECK(length(residual_digest) = 64),
    plan_count              INTEGER NOT NULL CHECK(plan_count >= 0),
    post_inventory_digest   TEXT CHECK(post_inventory_digest IS NULL OR length(post_inventory_digest) = 64),
    post_residual_json      TEXT CHECK(post_residual_json IS NULL OR json_valid(post_residual_json)),
    post_residual_digest    TEXT CHECK(post_residual_digest IS NULL OR length(post_residual_digest) = 64),
    post_plan_count         INTEGER CHECK(post_plan_count IS NULL OR post_plan_count >= 0),
    postflight_at_ms        INTEGER CHECK(postflight_at_ms IS NULL OR postflight_at_ms >= created_at_ms),
    executable_plan_count   INTEGER NOT NULL CHECK(executable_plan_count >= 0),
    residual_plan_count     INTEGER NOT NULL CHECK(residual_plan_count >= 0),
    predecessor_census_id   TEXT REFERENCES raw_authority_censuses(census_id),
    fixed_point             INTEGER NOT NULL DEFAULT 0 CHECK(fixed_point IN (0, 1)),
    created_at_ms           INTEGER NOT NULL CHECK(created_at_ms >= 0),
    completed_at_ms         INTEGER CHECK(completed_at_ms IS NULL OR completed_at_ms >= created_at_ms),
    CHECK(plan_count >= executable_plan_count),
    CHECK(plan_count >= residual_plan_count),
    CHECK(plan_count = executable_plan_count + residual_plan_count),
    CHECK(
        (lifecycle_status = 'planned' AND completed_at_ms IS NULL)
        OR (lifecycle_status IN ('completed', 'interrupted') AND completed_at_ms IS NOT NULL)
    ),
    CHECK(
        (lifecycle_status = 'planned' AND post_inventory_digest IS NULL
            AND post_residual_json IS NULL AND post_residual_digest IS NULL
            AND post_plan_count IS NULL AND postflight_at_ms IS NULL)
        OR (lifecycle_status IN ('completed', 'interrupted')
            AND post_inventory_digest IS NOT NULL AND post_residual_json IS NOT NULL
            AND post_residual_digest IS NOT NULL AND post_plan_count IS NOT NULL
            AND postflight_at_ms IS NOT NULL)
    )
) STRICT;

CREATE TABLE IF NOT EXISTS raw_authority_plans (
    plan_id                  TEXT PRIMARY KEY,
    input_digest             TEXT NOT NULL CHECK(length(input_digest) = 64),
    input_raw_ids_json       TEXT NOT NULL CHECK(json_valid(input_raw_ids_json)),
    logical_keys_json        TEXT NOT NULL CHECK(json_valid(logical_keys_json)),
    authority_witness_json   TEXT NOT NULL CHECK(json_valid(authority_witness_json)),
    source_preconditions_json TEXT NOT NULL CHECK(json_valid(source_preconditions_json)),
    index_preconditions_json TEXT NOT NULL CHECK(json_valid(index_preconditions_json)),
    created_at_ms            INTEGER NOT NULL CHECK(created_at_ms >= 0)
) STRICT;

CREATE TABLE IF NOT EXISTS raw_authority_census_plans (
    census_id          TEXT NOT NULL REFERENCES raw_authority_censuses(census_id) ON DELETE CASCADE,
    plan_id            TEXT NOT NULL REFERENCES raw_authority_plans(plan_id),
    ordinal            INTEGER NOT NULL CHECK(ordinal >= 0),
    selected           INTEGER NOT NULL CHECK(selected IN (0, 1)),
    outcome_status     TEXT NOT NULL CHECK(outcome_status IN (
                           'executed', 'retryable', 'deferred', 'terminal',
                           'rejected_stale', 'carried_forward'
                       )),
    reason             TEXT NOT NULL,
    next_action        TEXT NOT NULL,
    application_receipt_json TEXT NOT NULL DEFAULT '{{}}' CHECK(json_valid(application_receipt_json)),
    outcome_recorded   INTEGER NOT NULL DEFAULT 0 CHECK(outcome_recorded IN (0, 1)),
    recorded_at_ms     INTEGER NOT NULL CHECK(recorded_at_ms >= 0),
    PRIMARY KEY(census_id, plan_id),
    UNIQUE(census_id, ordinal)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_authority_census_plans_status
ON raw_authority_census_plans(census_id, outcome_status, ordinal);

CREATE INDEX IF NOT EXISTS idx_raw_authority_census_plans_attempts
ON raw_authority_census_plans(plan_id, recorded_at_ms DESC)
WHERE selected = 1;

CREATE TABLE IF NOT EXISTS raw_authority_census_post_plans (
    census_id          TEXT NOT NULL REFERENCES raw_authority_censuses(census_id) ON DELETE CASCADE,
    plan_id            TEXT NOT NULL REFERENCES raw_authority_plans(plan_id),
    ordinal            INTEGER NOT NULL CHECK(ordinal >= 0),
    PRIMARY KEY(census_id, plan_id),
    UNIQUE(census_id, ordinal)
) STRICT;

CREATE TABLE IF NOT EXISTS raw_authority_blockers (
    blocker_id          TEXT PRIMARY KEY,
    plan_id             TEXT NOT NULL REFERENCES raw_authority_plans(plan_id),
    census_id           TEXT NOT NULL REFERENCES raw_authority_censuses(census_id),
    reason              TEXT NOT NULL,
    expected_json       TEXT NOT NULL CHECK(json_valid(expected_json)),
    observed_json       TEXT NOT NULL CHECK(json_valid(observed_json)),
    created_at_ms       INTEGER NOT NULL CHECK(created_at_ms >= 0),
    resolved_at_ms      INTEGER CHECK(resolved_at_ms IS NULL OR resolved_at_ms >= created_at_ms),
    resolution          TEXT,
    CHECK((resolved_at_ms IS NULL) = (resolution IS NULL))
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_authority_blockers_open_plan
ON raw_authority_blockers(plan_id)
WHERE resolved_at_ms IS NULL;

-- v22 (polylogue-tfzw0): 'hook_payload' is a distinct ref_type from
-- 'raw_payload' so blob GC's liveness join (storage/blob_gc.py) can tell a
-- hook event's own durable blob ref apart from a raw_sessions payload ref --
-- write_source_hook_event never mints a raw_sessions row (polylogue-31r1), so
-- a hook-event blob ref keyed as 'raw_payload' with a synthetic ref_id could
-- never join to any raw_sessions row and was therefore permanently
-- GC-invisible under a naive membership check.
-- Persisted cache of the closed RawAuthorityVerdict vocabulary (polylogue-w6hql
-- Phase 2, polylogue-tw4ar). Not a new source of truth: every row here is a
-- read-through cache of what polylogue.storage.raw_authority_verdict_projection
-- .project_raw_authority_verdicts would recompute from raw_sessions + blob
-- storage, kept here only so repeated reads (e.g. blob-GC invariant checks at
-- scale) don't re-run the full byte-proof classifier every time. Invalidated
-- by content, not by elapsed time: cohort_fingerprint is a SHA-256 over the
-- cohort's own (raw_id, revision_kind, blob_hash) rows, so any membership or
-- content change to the logical_source_key cohort changes the fingerprint and
-- the cached rows read as stale on the next lookup (see
-- polylogue.storage.raw_authority_verdict_cache). The daemon's bounded
-- raw_authority_verdict_cache convergence stage warms eligible full/unknown
-- cohorts proactively; get_or_compute_raw_authority_verdicts() remains the
-- read-through fallback for cold or invalidated cohorts. Cohorts containing
-- append revisions are explicitly skipped because their authority uses a
-- separate proof shape, so this cache is never the retention or rebuild
-- authority for those source rows.
CREATE TABLE IF NOT EXISTS raw_authority_verdicts (
    raw_id                TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    logical_source_key    TEXT NOT NULL,
    verdict                TEXT NOT NULL CHECK ({check("verdict", RawAuthorityVerdict)}),
    cohort_member_count   INTEGER NOT NULL CHECK(cohort_member_count >= 1),
    cohort_fingerprint    BLOB NOT NULL CHECK(length(cohort_fingerprint) = 32),
    computed_at_ms        INTEGER NOT NULL CHECK(computed_at_ms >= 0)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_authority_verdicts_logical_source
ON raw_authority_verdicts(logical_source_key);

CREATE TABLE IF NOT EXISTS blob_refs (
    blob_hash       BLOB NOT NULL CHECK(length(blob_hash) = 32),
    ref_id          TEXT NOT NULL,
    ref_type        TEXT NOT NULL CHECK(ref_type IN ('raw_payload', 'attachment', 'sidecar', 'hook_payload')),
    source_path     TEXT,
    size_bytes      INTEGER NOT NULL CHECK(size_bytes >= 0),
    acquired_at_ms  INTEGER NOT NULL,
    PRIMARY KEY(blob_hash, ref_type, ref_id)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_blob_refs_ref_id
ON blob_refs(ref_id);

CREATE TABLE IF NOT EXISTS blob_publication_reservations (
    publication_id   TEXT PRIMARY KEY,
    blob_hash        BLOB NOT NULL CHECK(length(blob_hash) = 32),
    size_bytes       INTEGER NOT NULL CHECK(size_bytes >= 0),
    publisher_id     TEXT NOT NULL,
    reserved_at_ms   INTEGER NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_blob_publication_reservations_hash
ON blob_publication_reservations(blob_hash);

CREATE TABLE IF NOT EXISTS gc_generations (
    generation_id    TEXT PRIMARY KEY,
    started_at_ms    INTEGER NOT NULL,
    completed_at_ms  INTEGER,
    reclaimed_count  INTEGER NOT NULL DEFAULT 0 CHECK(reclaimed_count >= 0),
    reclaimed_bytes  INTEGER NOT NULL DEFAULT 0 CHECK(reclaimed_bytes >= 0)
) STRICT;

-- v31 (feature/fix/raw-authority-census): each bounded artifact-census apply
-- records its canonical receipt in the source tier in the same transaction as
-- its selected raw_artifacts upserts. The receipt is evidence only: it never
-- changes raw authority and its immutable digest identifies the exact page.
CREATE TABLE IF NOT EXISTS raw_authority_artifact_census_receipts (
    receipt_id              TEXT PRIMARY KEY,
    receipt_sha256          TEXT NOT NULL UNIQUE CHECK(length(receipt_sha256) = 64),
    receipt_json            TEXT NOT NULL,
    backup_manifest_path    TEXT NOT NULL,
    applied_at_ms           INTEGER NOT NULL CHECK(applied_at_ms >= 0),
    tool_version            TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_authority_artifact_census_receipts_applied_at
ON raw_authority_artifact_census_receipts(applied_at_ms);

CREATE TABLE IF NOT EXISTS raw_authority_artifact_census_checkpoints (
    census_id               TEXT PRIMARY KEY,
    universe_sha256         TEXT NOT NULL CHECK(length(universe_sha256) = 64),
    candidate_count         INTEGER NOT NULL CHECK(candidate_count >= 0),
    next_after_raw_id       TEXT,
    last_receipt_id         TEXT REFERENCES raw_authority_artifact_census_receipts(receipt_id),
    completed_at_ms         INTEGER,
    created_at_ms           INTEGER NOT NULL CHECK(created_at_ms >= 0)
) STRICT;

CREATE TABLE IF NOT EXISTS raw_authority_artifact_census_checkpoint_members (
    census_id               TEXT NOT NULL REFERENCES raw_authority_artifact_census_checkpoints(census_id) ON DELETE CASCADE,
    ordinal                 INTEGER NOT NULL CHECK(ordinal >= 0),
    raw_id                  TEXT NOT NULL REFERENCES raw_sessions(raw_id),
    PRIMARY KEY (census_id, raw_id),
    UNIQUE (census_id, ordinal)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_authority_artifact_census_checkpoint_members_page
ON raw_authority_artifact_census_checkpoint_members(census_id, ordinal);

CREATE TABLE IF NOT EXISTS raw_artifacts (
    artifact_id              TEXT PRIMARY KEY,
    raw_id                   TEXT NOT NULL REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    origin                   TEXT NOT NULL CHECK ({check("origin", Origin)}),
    source_path              TEXT NOT NULL,
    source_index             INTEGER NOT NULL DEFAULT 0,
    artifact_kind            TEXT NOT NULL,
    support_status           TEXT NOT NULL CHECK ({check("support_status", ArtifactSupportStatus)}),
    classification_reason    TEXT NOT NULL,
    parse_as_session         INTEGER NOT NULL DEFAULT 0 CHECK(parse_as_session IN (0, 1)),
    schema_eligible          INTEGER NOT NULL DEFAULT 0 CHECK(schema_eligible IN (0, 1)),
    malformed_jsonl_lines    INTEGER NOT NULL DEFAULT 0 CHECK(malformed_jsonl_lines >= 0),
    decode_error             TEXT,
    cohort_id                TEXT,
    link_group_key           TEXT,
    sidecar_agent_type       TEXT,
    first_observed_at_ms     INTEGER NOT NULL,
    last_observed_at_ms      INTEGER NOT NULL
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_artifacts_source_identity
ON raw_artifacts(origin, source_path, source_index)
WHERE artifact_kind NOT IN (
    'deferred_hot_jsonl_capture',
    'deferred_claude_code_partial_jsonl',
    'deferred_cas_frontier',
    'deferred_codex_cas_frontier',
    'terminal_corrupt_input',
    'terminal_superseded_deferred_cas_frontier',
    'terminal_unknown_json_decode',
    'terminal_unknown_export_no_session',
    'terminal_unsupported_shape'
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_artifacts_failure_identity
ON raw_artifacts(raw_id, origin, source_path, source_index)
WHERE artifact_kind IN (
    'deferred_hot_jsonl_capture',
    'deferred_claude_code_partial_jsonl',
    'deferred_cas_frontier',
    'deferred_codex_cas_frontier',
    'terminal_corrupt_input',
    'terminal_superseded_deferred_cas_frontier',
    'terminal_unknown_json_decode',
    'terminal_unknown_export_no_session',
    'terminal_unsupported_shape'
);

CREATE INDEX IF NOT EXISTS idx_raw_artifacts_raw_id
ON raw_artifacts(raw_id);

CREATE TABLE IF NOT EXISTS raw_hook_events (
    hook_event_id   TEXT PRIMARY KEY,
    origin          TEXT NOT NULL CHECK ({check("origin", Origin)}),
    native_id       TEXT,
    session_native_id TEXT,
    source_path     TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    observed_at_ms  INTEGER NOT NULL
    -- v22 (polylogue-tfzw0): the SHA-256 of this hook event's own durable
    -- raw_payload blob (see blob_refs above). Populated at write time by
    -- write_source_hook_event; NULL for rows written before v22 until a
    -- one-shot reconciliation pass backfills them (see
    -- polylogue.storage.hook_payload_ref_reconciliation).
    ,blob_hash       BLOB CHECK(blob_hash IS NULL OR length(blob_hash) = 32)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_hook_events_session
ON raw_hook_events(origin, session_native_id, observed_at_ms);

CREATE INDEX IF NOT EXISTS idx_raw_hook_events_source_hash
ON raw_hook_events(source_path, blob_hash);

CREATE TABLE IF NOT EXISTS otlp_spans (
    span_id           TEXT PRIMARY KEY,
    trace_id          TEXT NOT NULL,
    parent_span_id    TEXT,
    origin            TEXT CHECK ({nullable_check("origin", Origin)}),
    session_native_id TEXT,
    name              TEXT NOT NULL,
    kind              TEXT,
    attributes_json   TEXT NOT NULL DEFAULT '{{}}',
    events_json       TEXT NOT NULL DEFAULT '[]',
    started_at_ms     INTEGER,
    ended_at_ms       INTEGER,
    received_at_ms    INTEGER NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_otlp_spans_trace
ON otlp_spans(trace_id, started_at_ms DESC);

CREATE INDEX IF NOT EXISTS idx_otlp_spans_session
ON otlp_spans(origin, session_native_id, started_at_ms DESC)
WHERE session_native_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS history_sidecars (
    sidecar_id      TEXT PRIMARY KEY,
    origin          TEXT NOT NULL CHECK ({check("origin", Origin)}),
    source_path     TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    observed_at_ms  INTEGER NOT NULL,
    content_hash    BLOB NOT NULL CHECK(length(content_hash) = 32)
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_history_sidecars_path_hash
ON history_sidecars(origin, source_path, content_hash);

CREATE TABLE IF NOT EXISTS sinex_publication_obligations (
    object_id           TEXT NOT NULL,
    protocol_version     TEXT NOT NULL CHECK(protocol_version != ''),
    revision_id          TEXT NOT NULL,
    manifest_digest      TEXT NOT NULL,
    mode                 TEXT NOT NULL CHECK(mode IN ('mirror', 'primary')),
    status               TEXT NOT NULL DEFAULT 'pending'
                             CHECK(status IN ('pending', 'publishing', 'confirmed', 'durable_debt', 'rejected')),
    attempt_count        INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count >= 0),
    last_attempt_at_ms   INTEGER,
    last_receipt_state   TEXT,
    last_error           TEXT,
    created_at_ms        INTEGER NOT NULL,
    updated_at_ms        INTEGER NOT NULL,
    retired_at_ms        INTEGER,
    next_attempt_at_ms   INTEGER,
    PRIMARY KEY(object_id, protocol_version, revision_id, manifest_digest)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_sinex_publication_obligations_pending
ON sinex_publication_obligations(status, next_attempt_at_ms, created_at_ms)
WHERE status IN ('pending', 'publishing', 'durable_debt');

CREATE INDEX IF NOT EXISTS idx_sinex_publication_obligations_object
ON sinex_publication_obligations(object_id, created_at_ms DESC);

CREATE TABLE IF NOT EXISTS sinex_publication_payloads (
    object_id            TEXT NOT NULL,
    protocol_version     TEXT NOT NULL,
    revision_id          TEXT NOT NULL,
    manifest_digest      TEXT NOT NULL,
    manifest_bytes       BLOB NOT NULL,
    manifest_sha256      TEXT NOT NULL CHECK(length(manifest_sha256) = 64),
    manifest_size_bytes  INTEGER NOT NULL CHECK(manifest_size_bytes >= 0),
    segment_count        INTEGER NOT NULL CHECK(segment_count >= 0),
    total_size_bytes     INTEGER NOT NULL CHECK(total_size_bytes >= manifest_size_bytes),
    staged_at_ms         INTEGER NOT NULL CHECK(staged_at_ms >= 0),
    PRIMARY KEY(object_id, protocol_version, revision_id, manifest_digest),
    FOREIGN KEY(object_id, protocol_version, revision_id, manifest_digest)
        REFERENCES sinex_publication_obligations(
            object_id, protocol_version, revision_id, manifest_digest
        ) ON DELETE CASCADE
) STRICT;

CREATE TABLE IF NOT EXISTS sinex_publication_segments (
    object_id          TEXT NOT NULL,
    protocol_version   TEXT NOT NULL,
    revision_id        TEXT NOT NULL,
    manifest_digest    TEXT NOT NULL,
    position           INTEGER NOT NULL CHECK(position >= 0),
    segment_name       TEXT NOT NULL CHECK(segment_name != ''),
    segment_bytes      BLOB NOT NULL,
    segment_sha256     TEXT NOT NULL CHECK(length(segment_sha256) = 64),
    size_bytes         INTEGER NOT NULL CHECK(size_bytes >= 0),
    PRIMARY KEY(object_id, protocol_version, revision_id, manifest_digest, position),
    UNIQUE(object_id, protocol_version, revision_id, manifest_digest, segment_name),
    FOREIGN KEY(object_id, protocol_version, revision_id, manifest_digest)
        REFERENCES sinex_publication_payloads(
            object_id, protocol_version, revision_id, manifest_digest
        ) ON DELETE CASCADE
) STRICT;

CREATE TABLE IF NOT EXISTS sinex_publication_receipts (
    object_id          TEXT NOT NULL,
    protocol_version   TEXT NOT NULL,
    revision_id        TEXT NOT NULL,
    manifest_digest    TEXT NOT NULL,
    attempt_number     INTEGER NOT NULL CHECK(attempt_number > 0),
    request_id         TEXT NOT NULL,
    receipt_state      TEXT CHECK(receipt_state IS NULL OR receipt_state IN (
                            'raw_accepted', 'persisted_confirmed', 'durable_debt',
                            'spool_accepted_lossless', 'rejected'
                        )),
    receipt_detail     TEXT NOT NULL DEFAULT '',
    error_code         TEXT,
    received_at_ms     INTEGER NOT NULL CHECK(received_at_ms >= 0),
    PRIMARY KEY(object_id, protocol_version, revision_id, manifest_digest, attempt_number),
    FOREIGN KEY(object_id, protocol_version, revision_id, manifest_digest)
        REFERENCES sinex_publication_obligations(
            object_id, protocol_version, revision_id, manifest_digest
        ) ON DELETE CASCADE
) STRICT;

CREATE INDEX IF NOT EXISTS idx_sinex_publication_receipts_recent
ON sinex_publication_receipts(received_at_ms DESC);

-- Durable removed-content ledger (polylogue-27m). A row here is the
-- authoritative "this content is forgotten on purpose" marker for
-- standalone/off-mode excision: the acquire-time write chokepoint
-- (``write_source_raw_session``/``write_source_raw_session_blob_ref``)
-- refuses to re-store a raw payload whose ``blob_hash`` matches
-- ``removed_hash``, so an ordinary re-ingest of unmodified source files
-- cannot resurrect excised content even after an index.db rebuild.
-- ``span_start``/``span_end`` are populated only for a sub-payload
-- excision (e.g. a detected secret candidate span); both are NULL for a
-- whole-raw-session excision. This table is never queried for its own
-- sake by a reader -- it exists purely as a write-time gate plus forensic
-- trail, so no secret span coordinates ever carry the removed literal,
-- only byte offsets into the (now-deleted) payload.
CREATE TABLE IF NOT EXISTS excised_content (
    removed_hash    BLOB NOT NULL CHECK(length(removed_hash) = 32),
    hash_kind       TEXT NOT NULL DEFAULT 'blob_hash' CHECK(hash_kind IN ('blob_hash')),
    reason          TEXT NOT NULL,
    actor           TEXT NOT NULL,
    prior_revision  TEXT,
    span_start      INTEGER CHECK(span_start IS NULL OR span_start >= 0),
    span_end        INTEGER CHECK(span_end IS NULL OR span_end > span_start),
    excised_at_ms   INTEGER NOT NULL,
    PRIMARY KEY(removed_hash, hash_kind)
) STRICT;

-- polylogue-byw3y: durable verification-receipt cache so a census does not
-- re-hash every accepted frontier blob from scratch on every daemon
-- restart. See migrations/source/017_verified_blob_receipts.sql for the
-- full rationale and the safety invariant (a receipt is trusted only when
-- its fingerprint matches the blob's CURRENT stat() exactly).
CREATE TABLE IF NOT EXISTS verified_blob_receipts (
    blob_hash        BLOB NOT NULL CHECK(length(blob_hash) = 32),
    st_dev           INTEGER NOT NULL,
    st_ino           INTEGER NOT NULL,
    st_size          INTEGER NOT NULL CHECK(st_size >= 0),
    st_mtime_ns      INTEGER NOT NULL,
    st_ctime_ns      INTEGER NOT NULL,
    verified_at_ms   INTEGER NOT NULL CHECK(verified_at_ms >= 0),
    PRIMARY KEY(blob_hash)
) STRICT;

"""

__all__ = ["SOURCE_DDL", "SOURCE_SCHEMA_VERSION"]
