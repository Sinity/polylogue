"""Disposable ops-tier DDL for the archive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from polylogue.core.enums import IngestOutcome, Origin
from polylogue.core.types import (
    ContextInjectionDecision,
    ConvergenceDebtStatus,
    CursorLagSeverity,
    JudgmentSchedulerStatus,
    OperationRunStatus,
    RouteDaemonPath,
)
from polylogue.schemas.drift_sentinel import DriftClassification
from polylogue.storage.sqlite.archive_tiers.common import check, literal_check, nullable_check
from polylogue.storage.sqlite.archive_tiers.schema_identity import DERIVED_SCHEMA_META_DDL

OPS_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class OpsTableDisposition:
    """Recorded owner and lifecycle decision for one canonical OPS table.

    This is deliberately metadata, not a second schema description.  The
    table names remain derived from :data:`OPS_DDL`; tests cross-check that
    every declared table has a disposition before a retirement is proposed.
    """

    owner: str
    grain: str
    restart_required: bool
    replacement: str


# Wave-4 decision record (polylogue-pnxl6): the current OPS population is not
# a four-table telemetry cache.  Keep restart-required state and independently
# shaped attempt records until a replacement and its reader map have landed.
# ``schema_identity`` is bootstrap metadata and is included so the map covers
# every CREATE TABLE in the canonical DDL.
OPS_TABLE_DISPOSITIONS: dict[str, OpsTableDisposition] = {
    "ingest_cursor": OpsTableDisposition("live ingest", "one cursor per source path", True, "retain"),
    "ingest_attempts": OpsTableDisposition("ingest", "one row per ingest attempt", True, "retain"),
    "convergence_debt": OpsTableDisposition("daemon converger", "one retryable debt row per target", True, "retain"),
    "attachment_convergence_cursor": OpsTableDisposition(
        "attachment convergence", "one fair archive sweep position", True, "retain"
    ),
    "cursor_lag_samples": OpsTableDisposition(
        "daemon diagnostics", "one bounded lag sample", False, "retain pending map"
    ),
    "daemon_stage_events": OpsTableDisposition(
        "daemon", "one row per stage transition", False, "retain pending event map"
    ),
    "capture_health_history": OpsTableDisposition(
        "browser capture", "one row per health report", True, "retain independently"
    ),
    "daemon_events": OpsTableDisposition("daemon", "one row per SSE/event-log event", True, "retain"),
    "judgment_scheduler_receipts": OpsTableDisposition(
        "judgment scheduler", "one typed receipt per operation", True, "retain independently"
    ),
    "daemon_lifecycle": OpsTableDisposition("daemon", "one row per daemon run", True, "retain"),
    "daemon_termination_receipts": OpsTableDisposition(
        "daemon", "one reconciled termination receipt per ended daemon run", True, "retain"
    ),
    "daemon_event_retention": OpsTableDisposition("daemon", "one prune watermark per event ledger", True, "retain"),
    "embedding_catchup_runs": OpsTableDisposition(
        "embedding owner", "one row per catch-up run with counters", True, "retain independently"
    ),
    "secret_scan_status": OpsTableDisposition(
        "secret scanner", "one coverage cursor per session and scanner version", True, "retain"
    ),
    "mcp_call_log": OpsTableDisposition("MCP", "one row per tool call", True, "retain pending audit migration"),
    "mcp_call_session_refs": OpsTableDisposition(
        "MCP", "session references per tool call", True, "retain pending audit migration"
    ),
    "fts_drift_samples": OpsTableDisposition(
        "FTS diagnostics", "one bounded drift sample", False, "retain pending map"
    ),
    "schema_drift_samples": OpsTableDisposition(
        "schema sentinel", "one bounded drift sample", False, "retain pending map"
    ),
    "context_injection_ledger": OpsTableDisposition(
        "context scheduler", "one admission decision per candidate item", True, "retain"
    ),
    "schema_identity": OpsTableDisposition("schema bootstrap", "one derived-schema identity", True, "retain"),
    # Frontier inspection: a lost journal or certificate only forces the next
    # inspection to run in full mode; no accepted state is lost.
    "raw_frontier_cursor_changes": OpsTableDisposition(
        "frontier inspection", "one journal row per changed cursor path", False, "retain"
    ),
    "raw_frontier_cursor_journal_control": OpsTableDisposition(
        "frontier inspection", "one retained-floor watermark for the cursor journal", False, "retain"
    ),
    "raw_frontier_inspection": OpsTableDisposition(
        "frontier inspection", "one coverage certificate for the accepted frontier", False, "retain"
    ),
}
# Batch aggregation is a terminal run state distinct from both success and
# failure: completed siblings and retryable failed siblings remain visible.
_OPS_RUN_STATUS_CHECK = literal_check("status", *get_args(OperationRunStatus))
McpCallSessionRelation = Literal["primary", "member"]
"""Storage-local relation labels for this disposable MCP call-reference table."""

_CONVERGENCE_DEBT_STATUS_CHECK = literal_check("status", *get_args(ConvergenceDebtStatus))
_JUDGMENT_SCHEDULER_STATUS_CHECK = literal_check("status", *get_args(JudgmentSchedulerStatus))
_CURSOR_LAG_SEVERITY_CHECK = literal_check("severity", *get_args(CursorLagSeverity))
_MCP_SESSION_RELATION_CHECK = literal_check("relation", *get_args(McpCallSessionRelation))
_ROUTE_DAEMON_PATH_CHECK = literal_check("daemon_path", *get_args(RouteDaemonPath))
_CONTEXT_INJECTION_DECISION_CHECK = literal_check("decision", *get_args(ContextInjectionDecision))
SCHEMA_DRIFT_SAMPLES_DDL = f"""
CREATE TABLE IF NOT EXISTS schema_drift_samples (
    sample_id             TEXT PRIMARY KEY,
    origin                TEXT NOT NULL CHECK ({check("origin", Origin)}),
    element_kind          TEXT NOT NULL,
    classification        TEXT NOT NULL CHECK ({literal_check("classification", *get_args(DriftClassification))}),
    unseen_key_signature  TEXT NOT NULL DEFAULT '',
    native_id_example     TEXT NOT NULL,
    raw_id                TEXT NOT NULL,
    observed_at_ms        INTEGER NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_schema_drift_samples_origin_time
ON schema_drift_samples(origin, observed_at_ms DESC);

CREATE INDEX IF NOT EXISTS idx_schema_drift_samples_time
ON schema_drift_samples(observed_at_ms DESC);
"""

_OPS_INGEST_ATTEMPTS_DDL = f"""
CREATE TABLE IF NOT EXISTS ingest_attempts (
    attempt_id             TEXT PRIMARY KEY,
    source_path            TEXT,
    origin                 TEXT CHECK ({check("origin", Origin)} OR origin IS NULL),
    status                 TEXT NOT NULL CHECK({_OPS_RUN_STATUS_CHECK}),
    phase                  TEXT,
    storage_route          TEXT,
    started_at_ms          INTEGER NOT NULL,
    heartbeat_at_ms        INTEGER,
    finished_at_ms         INTEGER,
    parsed_raw_count       INTEGER NOT NULL DEFAULT 0 CHECK(parsed_raw_count >= 0),
    materialized_count     INTEGER NOT NULL DEFAULT 0 CHECK(materialized_count >= 0),
    error_message          TEXT,
    source_paths_json      TEXT NOT NULL DEFAULT '[]',
    -- polylogue-cnu3: typed, structurally-classified disposition -- never
    -- guessed from ``error_message`` text. ``outcome_code`` defaults to
    -- ``legacy_unknown`` so every pre-existing row (written before this
    -- vocabulary existed) stays honestly queryable as unclassified rather
    -- than silently guessed into a real class (AC4).
    outcome_code           TEXT NOT NULL DEFAULT 'legacy_unknown' CHECK ({check("outcome_code", IngestOutcome)}),
    retryable              INTEGER CHECK(retryable IN (0, 1)),
    evidence_ref           TEXT,
    diagnostic             TEXT,
    remediation            TEXT
) STRICT;
"""

_OPS_EMBEDDING_CATCHUP_RUNS_DDL = f"""
CREATE TABLE IF NOT EXISTS embedding_catchup_runs (
    run_id              TEXT PRIMARY KEY,
    started_at_ms       INTEGER NOT NULL,
    finished_at_ms      INTEGER,
    status              TEXT NOT NULL CHECK({_OPS_RUN_STATUS_CHECK}),
    origin              TEXT CHECK ({nullable_check("origin", Origin)}),
    scanned_sessions    INTEGER NOT NULL DEFAULT 0 CHECK(scanned_sessions >= 0),
    embedded_sessions   INTEGER NOT NULL DEFAULT 0 CHECK(embedded_sessions >= 0),
    skipped_sessions    INTEGER NOT NULL DEFAULT 0 CHECK(skipped_sessions >= 0),
    error_count         INTEGER NOT NULL DEFAULT 0 CHECK(error_count >= 0),
    embedded_messages   INTEGER NOT NULL DEFAULT 0 CHECK(embedded_messages >= 0),
    estimated_cost_usd  REAL,
    error_message       TEXT
) STRICT;
"""

OPS_DDL = f"""
CREATE TABLE IF NOT EXISTS ingest_cursor (
    source_path          TEXT PRIMARY KEY,
    canonical_source_path TEXT,
    captured_profile_key TEXT,
    origin               TEXT CHECK ({check("origin", Origin)} OR origin IS NULL),
    stat_size            INTEGER,
    byte_offset          INTEGER,
    last_complete_newline INTEGER,
    record_count         INTEGER NOT NULL DEFAULT 0 CHECK(record_count >= 0),
    last_record_ts_ms    INTEGER,
    parser_fingerprint   TEXT,
    content_fingerprint  TEXT,
    tail_hash            TEXT,
    st_dev               INTEGER,
    st_ino               INTEGER,
    mtime_ns             INTEGER,
    failure_count        INTEGER NOT NULL DEFAULT 0 CHECK(failure_count >= 0),
    next_retry_at        TEXT,
    excluded             INTEGER NOT NULL DEFAULT 0 CHECK(excluded IN (0, 1)),
    -- polylogue-hat0: the end offset of an append byte range that was
    -- already durably captured (raw written + revision bound in source.db)
    -- but is still awaiting authority resolution (quarantined/ambiguous
    -- parent). NULL means there is no pending deferred capture for this
    -- path. Distinct from byte_offset, which only advances once a plan is
    -- actually applied -- a deferred plan never advances byte_offset, so
    -- without this marker a re-observation of an unchanged file cannot be
    -- told apart from genuine new content and re-mints an identical raw row
    -- forever.
    deferred_end_offset  INTEGER,
    updated_at_ms        INTEGER NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_ingest_cursor_attention
ON ingest_cursor(failure_count, excluded, source_path);

CREATE INDEX IF NOT EXISTS idx_ingest_cursor_canonical_path
ON ingest_cursor(canonical_source_path)
WHERE canonical_source_path IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_ingest_cursor_missing_canonical_path
ON ingest_cursor(source_path)
WHERE canonical_source_path IS NULL AND byte_offset IS NOT NULL AND excluded = 0;

{_OPS_INGEST_ATTEMPTS_DDL}

CREATE INDEX IF NOT EXISTS idx_ingest_attempts_status
ON ingest_attempts(status, heartbeat_at_ms);

CREATE INDEX IF NOT EXISTS idx_ingest_attempts_storage_route
ON ingest_attempts(storage_route);

CREATE INDEX IF NOT EXISTS idx_ingest_attempts_outcome_code
ON ingest_attempts(outcome_code, started_at_ms);

-- polylogue-0glm0: the exact-source attempt lookup behind
-- ``ops status --source <path>`` keys on source_path alone. None of the
-- indexes above can serve it -- SQLite reads an index left to right, so
-- ``(status, heartbeat_at_ms)``, ``(storage_route)`` and
-- ``(outcome_code, started_at_ms)`` are unusable for an unconstrained
-- source_path, and widening one of them would not change that. The read
-- boundary in ``archive/query/source_freshness.py`` rejects a query whose
-- plan scans a protected table, so without this index the named-source
-- status of every archive is an unsafe-plan refusal (exit 3) rather than an
-- answer. Single-column deliberately: the reader's ORDER BY is a COALESCE
-- over three timestamps, which no index can order, so trailing columns would
-- widen every attempt write without removing the temp b-tree.
CREATE INDEX IF NOT EXISTS idx_ingest_attempts_source_path
ON ingest_attempts(source_path);

CREATE TABLE IF NOT EXISTS convergence_debt (
    debt_id        TEXT PRIMARY KEY,
    stage          TEXT NOT NULL,
    target_type    TEXT NOT NULL,
    target_id      TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'failed' CHECK({_CONVERGENCE_DEBT_STATUS_CHECK}),
    priority       INTEGER NOT NULL DEFAULT 0,
    attempts       INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
    last_error     TEXT,
    next_retry_at  TEXT,
    materializer_version TEXT,
    created_at_ms  INTEGER NOT NULL,
    updated_at_ms  INTEGER NOT NULL,
    UNIQUE(stage, target_type, target_id)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_convergence_debt_stage
ON convergence_debt(stage, priority DESC, updated_at_ms);

CREATE TABLE IF NOT EXISTS attachment_convergence_cursor (
    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
    attachment_id TEXT NOT NULL,
    ref_id TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS cursor_lag_samples (
    sample_id        TEXT PRIMARY KEY,
    family           TEXT NOT NULL,
    source_path      TEXT,
    lag_ms           INTEGER NOT NULL CHECK(lag_ms >= 0),
    stuck_file_count INTEGER NOT NULL DEFAULT 1 CHECK(stuck_file_count >= 0),
    p50_lag_ms       INTEGER NOT NULL DEFAULT 0 CHECK(p50_lag_ms >= 0),
    p95_lag_ms       INTEGER NOT NULL DEFAULT 0 CHECK(p95_lag_ms >= 0),
    severity         TEXT NOT NULL CHECK({_CURSOR_LAG_SEVERITY_CHECK}),
    sampled_at_ms    INTEGER NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_cursor_lag_samples_family_time
ON cursor_lag_samples(family, sampled_at_ms DESC);

CREATE TABLE IF NOT EXISTS daemon_stage_events (
    event_id       TEXT PRIMARY KEY,
    attempt_id     TEXT,
    stage          TEXT NOT NULL,
    status         TEXT NOT NULL,
    observed_at_ms INTEGER NOT NULL,
    payload_json   TEXT NOT NULL DEFAULT '{{}}'
) STRICT;

CREATE INDEX IF NOT EXISTS idx_daemon_stage_events_attempt_observed
ON daemon_stage_events(attempt_id, observed_at_ms DESC);

CREATE INDEX IF NOT EXISTS idx_daemon_stage_events_stage_observed
ON daemon_stage_events(stage, observed_at_ms DESC);

CREATE TABLE IF NOT EXISTS capture_health_history (
    id INTEGER PRIMARY KEY,
    report_key TEXT NOT NULL UNIQUE,
    ts_ms INTEGER NOT NULL,
    operation_id TEXT,
    idempotency_key TEXT,
    payload_json TEXT NOT NULL
) STRICT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_capture_health_history_idempotency
ON capture_health_history(idempotency_key) WHERE idempotency_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS daemon_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms          INTEGER NOT NULL,
    kind           TEXT NOT NULL,
    operation_id   TEXT,
    idempotency_key TEXT,
    payload_json   TEXT NOT NULL DEFAULT '{{}}'
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_daemon_events_idempotency
ON daemon_events(kind, idempotency_key) WHERE idempotency_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_daemon_events_kind ON daemon_events(kind);
CREATE INDEX IF NOT EXISTS idx_daemon_events_ts ON daemon_events(ts_ms);
CREATE INDEX IF NOT EXISTS idx_daemon_events_kind_id ON daemon_events(kind, id DESC);
CREATE INDEX IF NOT EXISTS idx_daemon_events_lifecycle ON daemon_events(kind, operation_id, id DESC);

-- polylogue-20d.13.6: the highest daemon_events id retention has removed.
-- Retention keeps interior rows (the newest row of each record kind and
-- object), so MIN(id) no longer says where complete history starts; a resume
-- cursor below this watermark may have missed a pruned row and is refused
-- with the typed aged_out resync.
CREATE TABLE IF NOT EXISTS daemon_event_retention (
    ledger             TEXT PRIMARY KEY,
    pruned_through_id  INTEGER NOT NULL CHECK(pruned_through_id >= 0)
) STRICT;

-- Judgment automation receipts are the typed authority for scheduler health.
-- Keep the legacy daemon_events row as a compatibility/event-stream record,
-- but do not require queue-health readers to decode its JSON payload.
CREATE TABLE IF NOT EXISTS judgment_scheduler_receipts (
    operation_id                    TEXT PRIMARY KEY,
    observed_at_ms                  INTEGER NOT NULL,
    status                          TEXT NOT NULL CHECK({_JUDGMENT_SCHEDULER_STATUS_CHECK}),
    reason                          TEXT NOT NULL,
    retryable                       INTEGER NOT NULL CHECK(retryable IN (0, 1)),
    retry_route                     TEXT NOT NULL,
    batch_limit                     INTEGER NOT NULL CHECK(batch_limit > 0),
    considered                      INTEGER NOT NULL DEFAULT 0 CHECK(considered >= 0),
    accepted                        INTEGER NOT NULL DEFAULT 0 CHECK(accepted >= 0),
    rejected                        INTEGER NOT NULL DEFAULT 0 CHECK(rejected >= 0),
    escalated                       INTEGER NOT NULL DEFAULT 0 CHECK(escalated >= 0),
    idempotent                      INTEGER NOT NULL DEFAULT 0 CHECK(idempotent >= 0),
    failed                          INTEGER NOT NULL DEFAULT 0 CHECK(failed >= 0),
    receipt_persistence_degraded    INTEGER NOT NULL DEFAULT 0 CHECK(receipt_persistence_degraded IN (0, 1)),
    receipt_persistence_recovered   INTEGER NOT NULL DEFAULT 0 CHECK(receipt_persistence_recovered IN (0, 1))
) STRICT;

CREATE INDEX IF NOT EXISTS idx_judgment_scheduler_receipts_observed
ON judgment_scheduler_receipts(observed_at_ms DESC, operation_id);

CREATE TABLE IF NOT EXISTS daemon_lifecycle (
    run_id               TEXT PRIMARY KEY,
    started_at_ms        INTEGER NOT NULL,
    stopped_at_ms        INTEGER,
    last_heartbeat_at_ms INTEGER NOT NULL,
    signal               TEXT,
    exit_kind            TEXT,
    details_json         TEXT NOT NULL DEFAULT '{{}}'
) STRICT;

CREATE INDEX IF NOT EXISTS idx_daemon_lifecycle_latest
ON daemon_lifecycle(started_at_ms DESC);

-- polylogue-peo: how each ended daemon run terminated, reconciled on the next
-- start from service-manager, journal, kernel, cgroup, heartbeat and workload
-- evidence. One row per run; a second reconciliation of the same run is a
-- no-op. ``classification`` vocabulary is validated at the write boundary.
CREATE TABLE IF NOT EXISTS daemon_termination_receipts (
    run_id                TEXT PRIMARY KEY,
    classification        TEXT NOT NULL,
    reconciled_at_ms      INTEGER NOT NULL,
    reconciled_by_run_id  TEXT NOT NULL,
    receipt_json          TEXT NOT NULL CHECK(json_valid(receipt_json))
) STRICT;

CREATE INDEX IF NOT EXISTS idx_daemon_termination_receipts_reconciled
ON daemon_termination_receipts(reconciled_at_ms DESC);

-- Sole live embedding_catchup_runs table (writer: ops_write.upsert_embedding_catchup_run).
-- Status vocabulary is generated from the canonical operation lifecycle enum.
-- The public CLI payload retains its 'stopped'/'complete' display vocabulary;
-- cli/commands/embed.py maps those values to typed operation statuses at this
-- write boundary. A pre-split monolith table of the same name (different
-- shape, statuses incl. 'stopped'/'interrupted') survives read-only via
-- storage/embeddings/progress.py.
{_OPS_EMBEDDING_CATCHUP_RUNS_DDL}

-- Bulk/archive-wide secret-candidate scan coverage (polylogue-layg.1).
-- Sole live table for the bounded, resumable ``scan_archive_for_secret_candidates``
-- sweep (``polylogue/security/secret_scan.py``): one row per session naming the
-- scanner version that last covered it. A session missing here, or present at
-- an older ``scanner_version`` than the current build, is pending work; a
-- scanner-version bump (new pattern rules) makes every existing row stale and
-- schedules an intentional rescan without touching any other tier. Disposable:
-- losing this table only means re-scanning (idempotent by construction --
-- ``record_secret_candidates`` writes deterministic assertion ids), never lost
-- candidates, since the actual findings live durably in user.db.
CREATE TABLE IF NOT EXISTS secret_scan_status (
    session_id       TEXT PRIMARY KEY,
    scanner_version  INTEGER NOT NULL,
    scanned_at_ms    INTEGER NOT NULL,
    blocks_scanned   INTEGER NOT NULL DEFAULT 0 CHECK(blocks_scanned >= 0),
    candidates_found INTEGER NOT NULL DEFAULT 0 CHECK(candidates_found >= 0)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_secret_scan_status_version
ON secret_scan_status(scanner_version);

CREATE TABLE IF NOT EXISTS mcp_call_log (
    call_id         TEXT PRIMARY KEY,
    tool_name       TEXT NOT NULL,
    session_id      TEXT,
    started_at_ms   INTEGER NOT NULL,
    finished_at_ms  INTEGER NOT NULL,
    duration_ms     INTEGER NOT NULL CHECK(duration_ms >= 0),
    success         INTEGER NOT NULL CHECK(success IN (0, 1)),
    error_detail    TEXT
) STRICT;

CREATE INDEX IF NOT EXISTS idx_ops_mcp_call_log_session
ON mcp_call_log(session_id, started_at_ms DESC);

CREATE INDEX IF NOT EXISTS idx_ops_mcp_call_log_tool
ON mcp_call_log(tool_name, started_at_ms DESC);

CREATE INDEX IF NOT EXISTS idx_ops_mcp_call_log_started
ON mcp_call_log(started_at_ms);

CREATE TABLE IF NOT EXISTS mcp_call_session_refs (
    call_id       TEXT NOT NULL REFERENCES mcp_call_log(call_id) ON DELETE CASCADE,
    session_id    TEXT NOT NULL,
    relation      TEXT NOT NULL CHECK({_MCP_SESSION_RELATION_CHECK}),
    PRIMARY KEY (call_id, session_id)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_ops_mcp_call_session_refs_session
ON mcp_call_session_refs(session_id, call_id);

-- polylogue-1xc.12: bounded drift-magnitude history for the fts_freshness_state
-- ledger (index.db). ops.db is disposable, so this is a plain freeform-
-- additive table, pruned by time and row count the same shape as
-- the MCP call log -- a snapshot of the SAME per-surface counters
-- fts_freshness_state already carries (source/indexed/missing/excess/
-- duplicate/identity_mismatch rows), sampled across time so an operator can
-- see drift MAGNITUDE trend, not just the current boolean ready/stale state.
CREATE TABLE IF NOT EXISTS fts_drift_samples (
    sample_id               TEXT PRIMARY KEY,
    surface                 TEXT NOT NULL,
    state                   TEXT NOT NULL,
    source_rows             INTEGER NOT NULL DEFAULT 0 CHECK(source_rows >= 0),
    indexed_rows            INTEGER NOT NULL DEFAULT 0 CHECK(indexed_rows >= 0),
    missing_rows            INTEGER NOT NULL DEFAULT 0 CHECK(missing_rows >= 0),
    excess_rows             INTEGER NOT NULL DEFAULT 0 CHECK(excess_rows >= 0),
    duplicate_rows          INTEGER NOT NULL DEFAULT 0 CHECK(duplicate_rows >= 0),
    identity_mismatch_rows  INTEGER NOT NULL DEFAULT 0 CHECK(identity_mismatch_rows >= 0),
    sampled_at_ms           INTEGER NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_fts_drift_samples_surface_time
ON fts_drift_samples(surface, sampled_at_ms DESC);

-- polylogue-da1: format-drift sentinel. Every ingested record whose shape
-- did not exactly match a committed provider schema package records one
-- bounded sample here, keyed by (origin, element_kind, unseen_key_signature),
-- so "origin X: N% of records since <date> carry unseen shapes" can be
-- read back as a windowed rate instead of discovered manually. ops.db is
-- disposable, so this is a plain freeform-additive table (no migration),
-- pruned by time and row count like fts_drift_samples.
--
-- polylogue-u6tl: `classification` previously hand-listed only 3 of
-- DriftClassification's 4 values (schemas/drift_sentinel.py), silently
-- omitting 'known_field_unread'. record_schema_drift_observations_to_ops_sync
-- (schemas/drift_sentinel_sampling.py) writes classification=observation.
-- classification -- a real DriftClassification value -- inside a bare
-- `except sqlite3.Error: return 0` best-effort guard, so every
-- 'known_field_unread' observation was silently dropped by the CHECK
-- instead of raising. Confirmed live (ops.db, read-only, 2026-07-31):
-- schema_drift_samples has 313 'unseen_shape' rows and exactly 0
-- 'known_field_unread' rows despite the classifier actively producing that
-- label. Generating the CHECK from DriftClassification via literal_check
-- closes the gap; ops.db is disposable so this needs no migration/version
-- bump, just the corrected DDL for the next bootstrap.
{SCHEMA_DRIFT_SAMPLES_DDL}

CREATE TABLE IF NOT EXISTS context_injection_ledger (
    ledger_id TEXT PRIMARY KEY, build_ref TEXT NOT NULL, observed_at_ms INTEGER NOT NULL,
    decision TEXT NOT NULL CHECK({_CONTEXT_INJECTION_DECISION_CHECK}),
    source TEXT NOT NULL, item_ref TEXT NOT NULL, token_cost INTEGER NOT NULL CHECK(token_cost >= 0),
    source_local_rank INTEGER NOT NULL CHECK(source_local_rank > 0),
    budget_before INTEGER NOT NULL CHECK(budget_before >= 0), budget_after INTEGER NOT NULL CHECK(budget_after >= 0),
    disclosure_verdict TEXT NOT NULL, authority_verdict TEXT NOT NULL, authority_reason TEXT NOT NULL,
    policy_refs_json TEXT NOT NULL, target_session TEXT, execution_context_ref TEXT NOT NULL
) STRICT;
CREATE INDEX IF NOT EXISTS idx_context_injection_ledger_build
ON context_injection_ledger(build_ref, observed_at_ms);
CREATE TABLE IF NOT EXISTS raw_frontier_cursor_changes (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path TEXT NOT NULL
) STRICT;
CREATE TABLE IF NOT EXISTS raw_frontier_cursor_journal_control (
    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
    retained_floor INTEGER NOT NULL DEFAULT 0 CHECK(retained_floor>=0)
) STRICT;
INSERT OR IGNORE INTO raw_frontier_cursor_journal_control(singleton) VALUES (1);
CREATE TABLE IF NOT EXISTS raw_frontier_inspection (
    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
    authority_identity TEXT NOT NULL,
    source_watermark INTEGER NOT NULL CHECK(source_watermark>=0),
    index_watermark INTEGER NOT NULL CHECK(index_watermark>=0),
    cursor_watermark INTEGER NOT NULL CHECK(cursor_watermark>=0),
    state TEXT NOT NULL,
    inspected_at_ms INTEGER NOT NULL CHECK(inspected_at_ms>=0),
    detail TEXT
) STRICT;
CREATE TRIGGER IF NOT EXISTS raw_frontier_cursor_insert AFTER INSERT ON ingest_cursor
BEGIN INSERT INTO raw_frontier_cursor_changes(source_path) VALUES (NEW.source_path);
INSERT INTO raw_frontier_cursor_changes(source_path) SELECT NEW.canonical_source_path WHERE NEW.canonical_source_path IS NOT NULL; END;
CREATE TRIGGER IF NOT EXISTS raw_frontier_cursor_update AFTER UPDATE ON ingest_cursor
BEGIN INSERT INTO raw_frontier_cursor_changes(source_path) VALUES (NEW.source_path);
INSERT INTO raw_frontier_cursor_changes(source_path) SELECT NEW.canonical_source_path WHERE NEW.canonical_source_path IS NOT NULL; END;
CREATE TRIGGER IF NOT EXISTS raw_frontier_cursor_old_path AFTER UPDATE OF source_path,canonical_source_path ON ingest_cursor
BEGIN INSERT INTO raw_frontier_cursor_changes(source_path) VALUES (OLD.source_path);
INSERT INTO raw_frontier_cursor_changes(source_path) SELECT OLD.canonical_source_path WHERE OLD.canonical_source_path IS NOT NULL; END;
CREATE TRIGGER IF NOT EXISTS raw_frontier_cursor_delete AFTER DELETE ON ingest_cursor
BEGIN INSERT INTO raw_frontier_cursor_changes(source_path) VALUES (OLD.source_path);
INSERT INTO raw_frontier_cursor_changes(source_path) SELECT OLD.canonical_source_path WHERE OLD.canonical_source_path IS NOT NULL; END;
CREATE TRIGGER IF NOT EXISTS raw_frontier_cursor_prune AFTER DELETE ON raw_frontier_cursor_changes
BEGIN UPDATE raw_frontier_cursor_journal_control SET retained_floor=max(retained_floor,OLD.sequence) WHERE singleton=1; END;
"""

# CREATE TABLE schema_identity; ddl-lifecycle-waiver: derived schema_identity is existing bootstrap metadata; declaring it in canonical DDL changes fresh-bootstrap completeness, not the ops data shape.
OPS_DDL += DERIVED_SCHEMA_META_DDL

__all__ = [
    "OPS_DDL",
    "OPS_SCHEMA_VERSION",
    "OPS_TABLE_DISPOSITIONS",
    "OpsTableDisposition",
    "SCHEMA_DRIFT_SAMPLES_DDL",
]
