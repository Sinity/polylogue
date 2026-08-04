"""Shared archive readiness helpers."""

from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from polylogue.archive.raw_materialization import (
    parsed_non_session_artifact_reason,
    source_path_native_id_candidates,
)
from polylogue.archive.revision_authority import BYTE_AUTHORITY_CENSUS_DETAIL
from polylogue.core.payload_coercion import row_int as _row_int
from polylogue.logging import get_logger
from polylogue.storage.insights.session.status import session_insight_status_sync
from polylogue.storage.introspection import column_exists as _column_exists
from polylogue.storage.introspection import table_exists as _table_exists
from polylogue.storage.raw_authority import raw_authority_detail_query_handle
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

logger = get_logger(__name__)

ArchiveTierVersionStatus = Literal["ok", "missing", "mismatch", "invalid"]


@dataclass(frozen=True, slots=True)
class ArchiveTierProbe:
    """One archive tier file's existence/size/schema-version facts.

    The sole, shared read-only probe of a tier file's ``PRAGMA user_version``
    against ``ARCHIVE_VERSION_BY_TIER`` (polylogue-703 -- ONE status
    assembly). ``polylogue.daemon.status`` (daemon HTTP/TUI/web status) and
    ``polylogue.cli.commands.status`` (CLI direct-fallback status, used when
    the daemon is unreachable) both build their per-tier status payload from
    this probe instead of each reimplementing the PRAGMA read independently
    -- the prior independent implementations were the mechanism behind a
    real production disagreement between bare-CLI and daemon-backed status
    (2026-07-03). Do not add a third independent tier-version probe; import
    this one.
    """

    tier: ArchiveTier
    path: str
    exists: bool
    size_bytes: int
    wal_size_bytes: int
    user_version: int | None
    expected_user_version: int
    version_status: ArchiveTierVersionStatus


def probe_archive_tier(tier: ArchiveTier, path: Path) -> ArchiveTierProbe:
    """Read-only probe of one archive tier file's existence/size/version."""
    expected_user_version = ARCHIVE_VERSION_BY_TIER[tier]
    if not path.exists():
        return ArchiveTierProbe(
            tier=tier,
            path=str(path),
            exists=False,
            size_bytes=0,
            wal_size_bytes=0,
            user_version=None,
            expected_user_version=expected_user_version,
            version_status="missing",
        )
    wal_path = Path(f"{path}-wal")
    user_version: int | None = None
    version_status: ArchiveTierVersionStatus = "invalid"
    try:
        conn = open_readonly_connection(path)
        try:
            user_version = _row_int(conn.execute("PRAGMA user_version").fetchone()[0])
            version_status = "ok" if user_version == expected_user_version else "mismatch"
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("archive tier version probe failed for %s (%s): %s", path, tier.value, exc, exc_info=True)
    return ArchiveTierProbe(
        tier=tier,
        path=str(path),
        exists=True,
        size_bytes=path.stat().st_size,
        wal_size_bytes=wal_path.stat().st_size if wal_path.exists() else 0,
        user_version=user_version,
        expected_user_version=expected_user_version,
        version_status=version_status,
    )


CLAUDE_WORKFLOW_STAGE_NAME = "claude_workflow"
"""daemon_stage_events ``stage`` value written by the claude_workflow
convergence stage (daemon/convergence_stages.py); imported from there so the
writer and this reader cannot drift apart."""

ACTIVE_REBUILD_STALE_AFTER_S = 180.0
"""Maximum heartbeat/start age for a rebuild-index row to count as active."""


def active_rebuild_index_attempts(ops_db: Path) -> list[dict[str, object]]:
    """Return active index-rebuild attempts recorded in the ops tier."""
    if not ops_db.exists():
        return []
    cutoff_ms = int((time.time() - ACTIVE_REBUILD_STALE_AFTER_S) * 1000)
    try:
        with closing(sqlite3.connect(f"file:{ops_db}?mode=ro", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT attempt_id, phase, started_at_ms, heartbeat_at_ms, parsed_raw_count, materialized_count
                FROM ingest_attempts
                WHERE status = 'running'
                  AND phase = 'rebuild-index'
                  AND COALESCE(heartbeat_at_ms, started_at_ms) >= ?
                ORDER BY started_at_ms DESC
                LIMIT 8
                """,
                (cutoff_ms,),
            ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("active rebuild-index attempts query failed for %s: %s", ops_db, exc, exc_info=True)
        return []
    return [
        {
            "attempt_id": str(row["attempt_id"]),
            "phase": str(row["phase"]),
            "started_at_ms": int(row["started_at_ms"]),
            "heartbeat_at_ms": int(row["heartbeat_at_ms"]) if row["heartbeat_at_ms"] is not None else None,
            "parsed_raw_count": int(row["parsed_raw_count"] or 0),
            "materialized_count": int(row["materialized_count"] or 0),
        }
        for row in rows
    ]


def claude_workflow_materialization_status(ops_db: Path) -> dict[str, object] | None:
    """Return the most recently recorded Claude Workflow materialization summary.

    Reads the latest ``daemon_stage_events`` row written by
    ``daemon.convergence_stages``'s claude_workflow stage each time it
    materializes evidence graphs. Returns ``None`` when the stage has never
    run against this archive (ops.db missing, table missing, or no rows).
    """
    if not ops_db.exists():
        return None
    try:
        with closing(sqlite3.connect(f"file:{ops_db}?mode=ro", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'daemon_stage_events'"
            ).fetchone()
            if has_table is None:
                return None
            row = conn.execute(
                """
                SELECT status, observed_at_ms, payload_json
                FROM daemon_stage_events
                WHERE stage = ?
                ORDER BY observed_at_ms DESC, rowid DESC
                LIMIT 1
                """,
                (CLAUDE_WORKFLOW_STAGE_NAME,),
            ).fetchone()
    except sqlite3.Error as exc:
        logger.warning("claude workflow materialization status query failed for %s: %s", ops_db, exc, exc_info=True)
        return None
    if row is None:
        return None
    try:
        payload = json.loads(row["payload_json"] or "{}")
    except (TypeError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload["status"] = str(row["status"])
    payload["observed_at_ms"] = int(row["observed_at_ms"])
    return payload


def _read_int(readiness: Mapping[str, Any], key: str) -> int:
    try:
        return int(readiness.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def raw_materialization_ready(readiness: Mapping[str, Any] | object | None) -> bool:
    """Return whether raw acquisition and index materialization are converged.

    Classified alias/non-session join gaps are acceptable: they mean the raw
    row has been explained. Actionable/open/blocking debt is not acceptable for
    product archive readiness.
    """
    if readiness is None:
        return False
    if not isinstance(readiness, Mapping):
        model_dump = getattr(readiness, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump()
            if not isinstance(dumped, Mapping):
                return False
            readiness = dumped
        else:
            return False
    if not bool(readiness.get("available", False)):
        return False
    # A surface that composes the archive-debt classifier records a failure to
    # run it here (see paths._merge_raw_materialization_debt). Readiness that
    # required the classifier cannot be claimed when the classifier failed.
    if readiness.get("debt_classifier_error"):
        return False
    parser_census = readiness.get("raw_authority_parser_census")
    if isinstance(parser_census, Mapping) and not bool(parser_census.get("available", False)):
        return False
    frontier = readiness.get("raw_authority_frontier")
    if not isinstance(frontier, Mapping) or frontier.get("lifecycle_status") != "completed":
        return False
    blocking_keys = (
        "critical",
        "warning",
        "actionable",
        "blocked",
        "affected_actionable",
        "affected_blocked",
        "affected_open",
        "lost_source_evidence_count",
        "unchecked",
        "affected_unchecked",
        "raw_authority_frontier_blocking_count",
        "raw_authority_blocker_count",
        "raw_authority_pending_census_count",
        "raw_authority_parser_census_incomplete_count",
    )
    return all(_read_int(readiness, key) == 0 for key in blocking_keys)


def raw_materialization_readiness_snapshot(
    active_archive: Path,
    *,
    classify_gaps: bool = True,
) -> dict[str, object]:
    """Return compact raw→index materialization readiness for an archive root.

    Exact classification may inspect every raw-id gap and is therefore reserved
    for explicit diagnostic reads. ``classify_gaps=False`` keeps the aggregate
    counters and durable authority state but marks all unclassified gaps as
    unchecked, which is the bounded periodic-status contract.
    """
    source_db = active_archive / "source.db"
    index_db = active_archive / "index.db"
    if not source_db.exists() or not index_db.exists():
        return {"available": False, "error": "source.db or index.db missing"}
    try:
        with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("ATTACH DATABASE ? AS source", (str(source_db),))
            raw_columns = _table_columns(conn, "source", "raw_sessions")
            session_columns = _table_columns(conn, "main", "sessions")
            row = conn.execute(
                """
                WITH raw_rows AS (
                    SELECT
                        r.raw_id,
                        r.origin,
                        r.validation_status,
                        r.parse_error,
                        r.parsed_at_ms,
                        EXISTS (
                            SELECT 1
                            FROM main.sessions s
                            WHERE s.raw_id = r.raw_id
                        ) AS is_materialized
                    FROM source.raw_sessions r
                    WHERE COALESCE(r.validation_status, '') != 'skipped'
                ),
                materialization AS (
                    SELECT
                        COUNT(*) AS raw_artifact_count,
                        COALESCE(SUM(CASE WHEN is_materialized THEN 1 ELSE 0 END), 0)
                            AS materialized_raw_artifact_count
                    FROM raw_rows
                ),
                session_count AS (
                    SELECT COUNT(*) AS archive_session_count FROM main.sessions
                ),
                gaps AS (
                    SELECT raw_id, origin, validation_status, parse_error, parsed_at_ms
                    FROM raw_rows
                    WHERE NOT is_materialized
                )
                SELECT
                    materialization.raw_artifact_count,
                    materialization.materialized_raw_artifact_count,
                    session_count.archive_session_count,
                    (materialization.raw_artifact_count - materialization.materialized_raw_artifact_count)
                        AS join_gap_count,
                    COUNT(gaps.raw_id) AS total,
                    COALESCE(SUM(CASE WHEN validation_status = 'skipped' THEN 1 ELSE 0 END), 0) AS skipped,
                    COALESCE(SUM(CASE WHEN parse_error IS NOT NULL THEN 1 ELSE 0 END), 0) AS parse_failed,
                    COALESCE(SUM(CASE WHEN parsed_at_ms IS NOT NULL AND parse_error IS NULL THEN 1 ELSE 0 END), 0)
                        AS parsed_without_index_session
                FROM gaps
                CROSS JOIN materialization
                CROSS JOIN session_count
                """
            ).fetchone()
            family_rows = conn.execute(
                """
                SELECT r.origin, COUNT(*) AS count
                FROM source.raw_sessions r
                LEFT JOIN main.sessions s ON s.raw_id = r.raw_id
                WHERE s.raw_id IS NULL
                  AND COALESCE(r.validation_status, '') != 'skipped'
                GROUP BY r.origin
                ORDER BY count DESC, r.origin
                LIMIT 16
                """
            ).fetchall()
            classified_counts: Counter[str] = Counter()
            parse_failed_origins: set[str] = set()
            if classify_gaps:
                raw_select_columns = _raw_gap_select_columns(raw_columns)
                gap_rows = conn.execute(
                    f"""
                    WITH raw_rows AS (
                        SELECT
                            {raw_select_columns},
                            EXISTS (
                                SELECT 1
                                FROM main.sessions s
                                WHERE s.raw_id = r.raw_id
                            ) AS is_materialized
                        FROM source.raw_sessions r
                        WHERE COALESCE(r.validation_status, '') != 'skipped'
                    )
                    SELECT *
                    FROM raw_rows
                    WHERE NOT is_materialized
                    """,
                ).fetchall()
                classified_counts, parse_failed_origins = _classify_raw_gap_rows(
                    conn,
                    active_archive,
                    gap_rows,
                    raw_columns=raw_columns,
                    session_columns=session_columns,
                    has_revision_applications=bool(_table_columns(conn, "main", "raw_revision_applications")),
                    has_membership_census=bool(_table_columns(conn, "source", "raw_membership_census")),
                    has_session_memberships=bool(_table_columns(conn, "source", "raw_session_memberships")),
                )
            adoption_deferred_count = 0
            if _table_columns(conn, "main", "raw_revision_applications"):
                adoption_deferred_count = int(
                    conn.execute(
                        """
                        SELECT COUNT(DISTINCT r.raw_id)
                        FROM source.raw_sessions AS r
                        JOIN main.raw_revision_applications AS a ON a.raw_id = r.raw_id
                        WHERE a.decision = 'deferred'
                          AND a.detail = 'ordinary_replay:incomparable_existing_index_state'
                          AND NOT EXISTS (
                              SELECT 1 FROM main.sessions AS s WHERE s.raw_id = r.raw_id
                          )
                        """
                    ).fetchone()[0]
                    or 0
                )
            lost_source_evidence_count = _missing_source_raw_session_count(conn)
            lost_source_evidence_samples = _missing_source_raw_session_samples(conn)
            authority_census: dict[str, object] | None = None
            authority_frontier: dict[str, object] | None = None
            authority_frontier_blocking_count = 0
            authority_frontier_remediation_refs: list[dict[str, object]] = []
            authority_pending_census_count = 0
            parser_census_available = False
            parser_census_complete_count = 0
            parser_census_incomplete_count = 0
            parser_census_incomplete_blob_bytes = 0
            parser_census_missing_receipt_count = 0
            parser_census_non_complete_receipt_count = 0
            parser_census_origin_summary: list[dict[str, object]] = []
            if _table_columns(conn, "source", "raw_authority_parser_census"):
                from polylogue.storage.raw_authority import (
                    RAW_AUTHORITY_PARSER_FINGERPRINT,
                    SUPERSEDED_MEMBERSHIP_FINGERPRINTS,
                )

                parser_census_available = True
                known_fingerprints = [RAW_AUTHORITY_PARSER_FINGERPRINT, *sorted(SUPERSEDED_MEMBERSHIP_FINGERPRINTS)]
                fingerprint_placeholders = ",".join("?" for _ in known_fingerprints)
                blob_size_expression = "COALESCE(r.blob_size, 0)" if "blob_size" in raw_columns else "0"
                census_rows = conn.execute(
                    f"""
                    WITH parser_census AS (
                        SELECT r.raw_id, r.origin, {blob_size_expression} AS blob_size,
                            CASE WHEN c.status = 'complete' AND c.parser_fingerprint IN ({fingerprint_placeholders}) THEN 1 ELSE 0 END AS is_complete,
                            CASE WHEN c.raw_id IS NULL THEN 1 ELSE 0 END AS is_missing_receipt
                        FROM source.raw_sessions AS r
                        LEFT JOIN source.raw_authority_parser_census AS c ON c.raw_id = r.raw_id
                        WHERE COALESCE(r.validation_status, '') != 'skipped'
                    )
                    SELECT COALESCE(SUM(is_complete), 0), COALESCE(SUM(CASE WHEN is_complete = 0 THEN 1 ELSE 0 END), 0),
                        COALESCE(SUM(CASE WHEN is_complete = 0 THEN blob_size ELSE 0 END), 0), COALESCE(SUM(is_missing_receipt), 0),
                        COALESCE(SUM(CASE WHEN is_complete = 0 AND is_missing_receipt = 0 THEN 1 ELSE 0 END), 0)
                    FROM parser_census
                    """,
                    known_fingerprints,
                ).fetchone()
                parser_census_complete_count = int(census_rows[0] or 0)
                parser_census_incomplete_count = int(census_rows[1] or 0)
                parser_census_incomplete_blob_bytes = int(census_rows[2] or 0)
                parser_census_missing_receipt_count = int(census_rows[3] or 0)
                parser_census_non_complete_receipt_count = int(census_rows[4] or 0)
                parser_census_origin_summary = [
                    {"origin": str(origin), "count": int(count or 0), "blob_bytes": int(blob_bytes or 0)}
                    for origin, count, blob_bytes in conn.execute(
                        f"""
                    SELECT r.origin, COUNT(*), COALESCE(SUM({blob_size_expression}), 0)
                    FROM source.raw_sessions AS r LEFT JOIN source.raw_authority_parser_census AS c ON c.raw_id = r.raw_id
                    WHERE COALESCE(r.validation_status, '') != 'skipped'
                      AND NOT COALESCE(c.status = 'complete' AND c.parser_fingerprint IN ({fingerprint_placeholders}), 0)
                    GROUP BY r.origin ORDER BY 3 DESC, 2 DESC, r.origin LIMIT 16
                    """,
                        known_fingerprints,
                    )
                ]
            if _table_columns(conn, "source", "raw_authority_censuses"):
                authority_pending_census_count = int(
                    conn.execute(
                        """
                        SELECT COUNT(*) FROM source.raw_authority_censuses
                        WHERE lifecycle_status = 'planned'
                        """
                    ).fetchone()[0]
                )
                census_row = conn.execute(
                    """
                    SELECT census_id, sequence_no, inventory_digest, residual_digest,
                           plan_count, post_inventory_digest, post_residual_digest,
                           post_plan_count, executable_plan_count, residual_plan_count,
                           predecessor_census_id, mode, lifecycle_status, quiescent,
                           fixed_point, completed_at_ms
                    FROM source.raw_authority_censuses
                    WHERE lifecycle_status IN ('completed', 'interrupted')
                    ORDER BY sequence_no DESC LIMIT 1
                    """
                ).fetchone()
                if census_row is not None:
                    authority_census = {
                        "census_id": str(census_row["census_id"]),
                        "sequence_no": int(census_row["sequence_no"]),
                        "inventory_digest": str(census_row["inventory_digest"]),
                        "residual_digest": str(census_row["residual_digest"]),
                        "plan_count": int(census_row["plan_count"]),
                        "post_inventory_digest": str(census_row["post_inventory_digest"]),
                        "post_residual_digest": str(census_row["post_residual_digest"]),
                        "post_plan_count": int(census_row["post_plan_count"]),
                        "executable_plan_count": int(census_row["executable_plan_count"]),
                        "residual_plan_count": int(census_row["residual_plan_count"]),
                        "predecessor_census_id": census_row["predecessor_census_id"],
                        "mode": str(census_row["mode"]),
                        "lifecycle_status": str(census_row["lifecycle_status"]),
                        "quiescent": bool(census_row["quiescent"]),
                        "fixed_point": bool(census_row["fixed_point"]),
                        "completed_at_ms": int(census_row["completed_at_ms"]),
                        "pending_census_count": authority_pending_census_count,
                        "query_handle": (f"polylogue://raw-authority-census/{census_row['census_id']}/0"),
                    }
                frontier_row = conn.execute(
                    """
                    SELECT census_id, sequence_no, inventory_digest, residual_digest,
                           plan_count, executable_plan_count, residual_plan_count,
                           lifecycle_status, completed_at_ms, scope_json,
                           post_residual_json
                    FROM source.raw_authority_censuses
                    WHERE lifecycle_status IN ('completed', 'interrupted')
                      AND json_extract(scope_json, '$.schema') =
                          'polylogue.raw-authority-frontier-scope.v1'
                    ORDER BY sequence_no DESC LIMIT 1
                    """
                ).fetchone()
                if frontier_row is not None:
                    import json

                    frontier_scope = json.loads(str(frontier_row["scope_json"]))
                    # An apply census records its pre-application scope for
                    # auditability, then publishes the actual frontier in the
                    # postflight residual.  Readiness must reflect that
                    # terminal state rather than keep an already repaired plan
                    # blocking until some later inspection happens to run.
                    frontier_post_residual = json.loads(str(frontier_row["post_residual_json"] or "{}"))
                    postflight_state_counts = frontier_post_residual.get("frontier_state_counts")
                    postflight_residual_state_counts = frontier_post_residual.get("state_counts")
                    scope_state_counts = frontier_scope.get("state_counts")
                    frontier_state_counts_source = (
                        postflight_state_counts if isinstance(postflight_state_counts, Mapping) else scope_state_counts
                    )
                    frontier_state_counts = {
                        str(key): int(value) for key, value in dict(frontier_state_counts_source or {}).items()
                    }
                    blocking_state_counts = {
                        str(key): int(value)
                        for key, value in dict(
                            postflight_residual_state_counts
                            if isinstance(postflight_residual_state_counts, Mapping)
                            else frontier_state_counts
                        ).items()
                    }
                    nonblocking_states = {"proven_current", "superseded"}
                    authority_frontier_blocking_count = sum(
                        count for state, count in blocking_state_counts.items() if state not in nonblocking_states
                    )
                    authority_frontier = {
                        "census_id": str(frontier_row["census_id"]),
                        "sequence_no": int(frontier_row["sequence_no"]),
                        "inventory_digest": str(frontier_scope.get("inventory_digest") or ""),
                        "plan_inventory_digest": str(frontier_row["inventory_digest"]),
                        "residual_digest": str(frontier_row["residual_digest"]),
                        "plan_count": int(frontier_row["plan_count"]),
                        "executable_plan_count": int(frontier_row["executable_plan_count"]),
                        "residual_plan_count": int(frontier_row["residual_plan_count"]),
                        "state_counts": frontier_state_counts,
                        "blocking_count": authority_frontier_blocking_count,
                        "lifecycle_status": str(frontier_row["lifecycle_status"]),
                        "completed_at_ms": int(frontier_row["completed_at_ms"]),
                        "query_handle": (f"polylogue://raw-authority-census/{frontier_row['census_id']}/0"),
                    }
            authority_blocker_count = 0
            if _table_columns(conn, "source", "raw_authority_blockers"):
                authority_blocker_count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM source.raw_authority_blockers WHERE resolved_at_ms IS NULL"
                    ).fetchone()[0]
                )
                authority_frontier_remediation_refs = [
                    {
                        "blocker_id": str(blocker_id),
                        "plan_id": str(plan_id),
                        "detail_query_handle": raw_authority_detail_query_handle(str(census_id), str(plan_id)),
                    }
                    for blocker_id, plan_id, census_id in conn.execute(
                        """
                        SELECT b.blocker_id, b.plan_id, b.census_id
                        FROM source.raw_authority_blockers AS b
                        JOIN source.raw_authority_plans AS p ON p.plan_id = b.plan_id
                        WHERE b.resolved_at_ms IS NULL
                          AND json_extract(p.authority_witness_json, '$.schema') =
                              'polylogue.raw-authority-frontier-plan.v1'
                        ORDER BY b.created_at_ms, b.blocker_id
                        LIMIT 16
                        """
                    )
                ]
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
        }
    total = int(row["total"] or 0)
    raw_artifact_count = int(row["raw_artifact_count"] or 0)
    materialized_raw_artifact_count = int(row["materialized_raw_artifact_count"] or 0)
    archive_session_count = int(row["archive_session_count"] or 0)
    join_gap_count = int(row["join_gap_count"] or total)
    skipped = int(row["skipped"] or 0)
    raw_parse_failed = int(row["parse_failed"] or 0)
    parsed_without_index_session = int(row["parsed_without_index_session"] or 0)
    parse_failed = classified_counts.get("parse-failed", 0)
    classified = sum(count for category, count in classified_counts.items() if category != "parse-failed")
    actionable = len(parse_failed_origins)
    critical = actionable
    affected_actionable = parse_failed
    unchecked = max(total - classified - affected_actionable - adoption_deferred_count, 0)
    classification = "cheap_projection" if classify_gaps and (classified or adoption_deferred_count) else "not_run"
    raw_id_join_gap_count = unchecked
    category_counts: dict[str, int] = {
        "raw_id_join_gap": raw_id_join_gap_count,
        "skipped": skipped,
        "parse_failed": parse_failed,
        "raw_parse_failed": raw_parse_failed,
        "parsed_without_index_session": parsed_without_index_session,
    }
    if adoption_deferred_count:
        category_counts["adoption_deferred"] = adoption_deferred_count
    category_counts.update(
        {category: count for category, count in classified_counts.items() if category != "parse-failed"}
    )
    return {
        "available": True,
        "classification": classification,
        "precision": "raw_id_join_gap",
        "raw_artifact_count": raw_artifact_count,
        "materialized_raw_artifact_count": materialized_raw_artifact_count,
        "archive_session_count": archive_session_count,
        "join_gap_count": join_gap_count,
        "total": total,
        "critical": critical,
        "warning": 0,
        "actionable": actionable,
        "blocked": adoption_deferred_count,
        "classified": classified,
        "unchecked": unchecked,
        "affected_total": total,
        "affected_actionable": affected_actionable,
        "affected_blocked": adoption_deferred_count,
        "affected_open": 0,
        "affected_classified": classified,
        "affected_unchecked": unchecked,
        "lost_source_evidence_count": lost_source_evidence_count,
        "lost_source_evidence_samples": lost_source_evidence_samples,
        "category_counts": category_counts,
        "source_family_counts": {str(item["origin"]): int(item["count"] or 0) for item in family_rows},
        "raw_authority_census": authority_census,
        "raw_authority_frontier": authority_frontier,
        "raw_authority_frontier_blocking_count": authority_frontier_blocking_count,
        "raw_authority_frontier_remediation_refs": authority_frontier_remediation_refs,
        "raw_authority_blocker_count": authority_blocker_count,
        "raw_authority_pending_census_count": authority_pending_census_count,
        "raw_authority_parser_census": {
            "available": parser_census_available,
            "complete_count": parser_census_complete_count,
            "incomplete_count": parser_census_incomplete_count,
            "incomplete_blob_bytes": parser_census_incomplete_blob_bytes,
            "missing_receipt_count": parser_census_missing_receipt_count,
            "non_complete_receipt_count": parser_census_non_complete_receipt_count,
            "incomplete_origin_summary": parser_census_origin_summary,
        },
        "raw_authority_parser_census_incomplete_count": parser_census_incomplete_count,
        "raw_authority_parser_census_incomplete_blob_bytes": parser_census_incomplete_blob_bytes,
        "raw_authority_ledger_counts": {
            "unresolved_blockers": authority_blocker_count,
            "pending_censuses": authority_pending_census_count,
            "parser_census_incomplete": parser_census_incomplete_count,
        },
    }


def missing_source_raw_session_evidence(active_archive: Path, *, limit: int = 10) -> dict[str, object]:
    """Return indexed sessions whose source raw evidence is no longer present.

    This is the reverse of raw materialization debt. Raw materialization asks
    whether source rows have reached the index. This helper asks whether an
    indexed session still has the source row named by ``sessions.raw_id``. A
    missing row is lost source evidence until the exact raw artifact is
    recovered; it must not be repaired by relinking to a same-native but
    different source row.
    """

    source_db = active_archive / "source.db"
    index_db = active_archive / "index.db"
    if not source_db.exists() or not index_db.exists():
        return {
            "available": False,
            "reason": "source.db or index.db missing",
            "missing_raw_session_count": 0,
            "missing_raw_session_samples": [],
            "lost_source_evidence_count": 0,
            "lost_source_evidence_samples": [],
        }
    try:
        with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("ATTACH DATABASE ? AS source", (str(source_db),))
            if not _table_columns(conn, "main", "sessions") or not _table_columns(conn, "source", "raw_sessions"):
                return {
                    "available": False,
                    "reason": "sessions or raw_sessions table missing",
                    "missing_raw_session_count": 0,
                    "missing_raw_session_samples": [],
                    "lost_source_evidence_count": 0,
                    "lost_source_evidence_samples": [],
                }
            count = _missing_source_raw_session_count(conn)
            samples = _missing_source_raw_session_samples(conn, limit=limit)
    except sqlite3.Error as exc:
        return {
            "available": False,
            "reason": str(exc),
            "missing_raw_session_count": 0,
            "missing_raw_session_samples": [],
            "lost_source_evidence_count": 0,
            "lost_source_evidence_samples": [],
        }
    return {
        "available": True,
        "reason": None,
        "missing_raw_session_count": count,
        "missing_raw_session_samples": samples,
        "lost_source_evidence_count": count,
        "lost_source_evidence_samples": samples,
    }


def _missing_source_raw_session_count(conn: sqlite3.Connection) -> int:
    session_columns = _table_columns(conn, "main", "sessions")
    if "raw_id" not in session_columns:
        return 0
    return _readiness_scalar_int(
        conn,
        """
        SELECT COUNT(*)
        FROM sessions AS s
        WHERE s.raw_id IS NOT NULL
          AND NOT EXISTS (
            SELECT 1 FROM source.raw_sessions AS r WHERE r.raw_id = s.raw_id
          )
        """,
    )


def _missing_source_raw_session_samples(conn: sqlite3.Connection, *, limit: int = 10) -> list[dict[str, object]]:
    session_columns = _table_columns(conn, "main", "sessions")
    if not {"session_id", "raw_id"} <= session_columns:
        return []
    origin_expr = "s.origin" if "origin" in session_columns else "NULL"
    native_id_expr = "s.native_id" if "native_id" in session_columns else "NULL"
    message_count_expr = "s.message_count" if "message_count" in session_columns else "NULL"
    updated_at_expr = "s.updated_at_ms" if "updated_at_ms" in session_columns else "NULL"
    order_expr = "s.updated_at_ms DESC, s.session_id" if "updated_at_ms" in session_columns else "s.session_id"
    rows = conn.execute(
        f"""
        SELECT s.session_id,
               {origin_expr} AS origin,
               {native_id_expr} AS native_id,
               s.raw_id,
               {message_count_expr} AS message_count,
               {updated_at_expr} AS updated_at_ms
        FROM sessions AS s
        WHERE s.raw_id IS NOT NULL
          AND NOT EXISTS (
            SELECT 1 FROM source.raw_sessions AS r WHERE r.raw_id = s.raw_id
          )
        ORDER BY {order_expr}
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [
        {
            "session_id": str(row["session_id"]),
            "origin": str(row["origin"]),
            "native_id": str(row["native_id"]),
            "missing_raw_id": str(row["raw_id"]),
            "message_count": int(row["message_count"] or 0),
            "updated_at_ms": None if row["updated_at_ms"] is None else int(row["updated_at_ms"]),
            "evidence_status": "lost_source_evidence",
            "loss_reason": "index_raw_id_missing_from_source_tier",
            "recovery_requirement": "restore_exact_raw_artifact_or_keep_blocked",
        }
        for row in rows
    ]


def _readiness_scalar_int(conn: sqlite3.Connection, sql: str) -> int:
    row = conn.execute(sql).fetchone()
    return int(row[0] or 0) if row is not None else 0


def _table_columns(conn: sqlite3.Connection, schema: str, table: str) -> frozenset[str]:
    try:
        # ``sessions.session_id`` and other identity columns are generated.
        # table_info omits generated/hidden columns, which made exact lost-raw
        # counts pair with empty samples on the canonical archive schema.
        rows = conn.execute(f"PRAGMA {schema}.table_xinfo({table})").fetchall()
    except sqlite3.Error as exc:
        logger.warning("archive readiness table-columns probe failed for %s.%s: %s", schema, table, exc, exc_info=True)
        return frozenset()
    return frozenset(str(row["name"] if isinstance(row, sqlite3.Row) else row[1]) for row in rows)


def _raw_gap_select_columns(raw_columns: frozenset[str]) -> str:
    def column(name: str) -> str:
        return f"r.{name}" if name in raw_columns else f"NULL AS {name}"

    names = (
        "raw_id",
        "origin",
        "native_id",
        "source_path",
        "blob_hash",
        "source_index",
        "revision_authority",
        "validation_status",
        "parse_error",
        "parsed_at_ms",
    )
    return ",\n                        ".join(column(name) for name in names)


def _classify_raw_gap_rows(
    conn: sqlite3.Connection,
    archive_root: Path,
    rows: list[sqlite3.Row],
    *,
    raw_columns: frozenset[str],
    session_columns: frozenset[str],
    has_revision_applications: bool,
    has_membership_census: bool,
    has_session_memberships: bool,
) -> tuple[Counter[str], set[str]]:
    if not rows:
        return Counter(), set()
    counts: Counter[str] = Counter()
    parse_failed_origins: set[str] = set()
    for row in rows:
        category = _raw_gap_category(
            conn,
            archive_root,
            row,
            raw_columns=raw_columns,
            session_columns=session_columns,
            has_revision_applications=has_revision_applications,
            has_membership_census=has_membership_census,
            has_session_memberships=has_session_memberships,
        )
        if category is not None:
            counts[category] += 1
            if category == "parse-failed":
                parse_failed_origins.add(str(row["origin"] or "unknown"))
    return counts, parse_failed_origins


def _raw_gap_category(
    conn: sqlite3.Connection,
    archive_root: Path,
    row: sqlite3.Row,
    *,
    raw_columns: frozenset[str],
    session_columns: frozenset[str],
    has_revision_applications: bool,
    has_membership_census: bool,
    has_session_memberships: bool,
) -> str | None:
    can_reconcile_alias = not row["parse_error"] or _retryable_decode_missing_blob_error(row["parse_error"])
    if can_reconcile_alias and _raw_gap_materialized_by_alias(conn, row, session_columns=session_columns):
        return "materialized-alias"
    if can_reconcile_alias and _raw_gap_matches_missing_index_raw_link(conn, row, session_columns=session_columns):
        return "lost-source-evidence-alias"
    if _raw_gap_parsed_non_session_artifact(archive_root, row, raw_columns=raw_columns):
        return "parsed-non-session-artifact"
    if row["parse_error"]:
        return "parse-failed"
    authority_category = _raw_gap_authority_category(
        conn,
        row,
        has_revision_applications=has_revision_applications,
        has_membership_census=has_membership_census,
        has_session_memberships=has_session_memberships,
    )
    if authority_category is not None:
        return authority_category
    return None


def _raw_gap_authority_category(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    has_revision_applications: bool,
    has_membership_census: bool,
    has_session_memberships: bool,
) -> str | None:
    raw_id = str(row["raw_id"])
    if has_revision_applications:
        terminal = conn.execute(
            """
            SELECT 1 FROM main.raw_revision_applications
            WHERE raw_id = ?
              AND decision IN ('selected_baseline', 'applied_append', 'superseded', 'ambiguous')
            LIMIT 1
            """,
            (raw_id,),
        ).fetchone()
        if terminal is not None:
            return "revision-application-terminal"

    if has_membership_census and has_session_memberships:
        membership = conn.execute(
            """
            SELECT 1
            FROM source.raw_membership_census AS c
            WHERE c.raw_id = ?
              AND c.status = 'complete'
              AND c.member_count > 0
              AND c.member_count = (
                SELECT COUNT(*) FROM source.raw_session_memberships AS counted
                WHERE counted.raw_id = c.raw_id
              )
              AND NOT EXISTS (
                SELECT 1 FROM source.raw_session_memberships AS m
                WHERE m.raw_id = c.raw_id
                  AND (m.decision IS NULL OR m.decision = 'deferred')
              )
            LIMIT 1
            """,
            (raw_id,),
        ).fetchone()
        if membership is not None:
            return "membership-authority-classified"

    if row["source_index"] == -1:
        if row["revision_authority"] == "byte_proven":
            return "append-authority-proven"
        if has_membership_census:
            quarantined = conn.execute(
                """
                SELECT 1 FROM source.raw_membership_census
                WHERE raw_id = ? AND status = 'failed' AND detail = ?
                LIMIT 1
                """,
                (raw_id, BYTE_AUTHORITY_CENSUS_DETAIL),
            ).fetchone()
            if quarantined is not None:
                return "append-authority-quarantined"
    return None


def _retryable_decode_missing_blob_error(parse_error: object) -> bool:
    if not isinstance(parse_error, str):
        return False
    return parse_error.startswith("decode:") and "No such file or directory" in parse_error


def _raw_gap_materialized_by_alias(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    session_columns: frozenset[str],
) -> bool:
    if not {"origin", "native_id"} <= session_columns:
        return False
    origin = str(row["origin"] or "")
    if not origin:
        return False
    native_ids: list[str] = []

    def add(value: object) -> None:
        if isinstance(value, str) and value and value not in native_ids:
            native_ids.append(value)

    add(row["native_id"])
    for candidate in source_path_native_id_candidates(str(row["source_path"] or "")):
        add(candidate)
    if not native_ids:
        return False
    for native_id in native_ids:
        existing = conn.execute(
            """
            SELECT 1
            FROM main.sessions AS s
            JOIN source.raw_sessions AS existing_raw ON existing_raw.raw_id = s.raw_id
            WHERE s.origin = ?
              AND s.native_id = ?
            LIMIT 1
            """,
            (origin, native_id),
        ).fetchone()
        if existing is not None:
            return True
    return False


def _raw_gap_matches_missing_index_raw_link(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    session_columns: frozenset[str],
) -> bool:
    if not {"origin", "native_id", "raw_id"} <= session_columns:
        return False
    origin = str(row["origin"] or "")
    if not origin:
        return False
    native_ids: list[str] = []

    def add(value: object) -> None:
        if isinstance(value, str) and value and value not in native_ids:
            native_ids.append(value)

    add(row["native_id"])
    for candidate in source_path_native_id_candidates(str(row["source_path"] or "")):
        add(candidate)
    if not native_ids:
        return False
    for native_id in native_ids:
        existing = conn.execute(
            """
            SELECT 1
            FROM main.sessions AS s
            WHERE s.origin = ?
              AND s.native_id = ?
              AND s.raw_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM source.raw_sessions AS existing_raw WHERE existing_raw.raw_id = s.raw_id
              )
            LIMIT 1
            """,
            (origin, native_id),
        ).fetchone()
        if existing is not None:
            return True
    return False


def _raw_gap_parsed_non_session_artifact(
    archive_root: Path,
    row: sqlite3.Row,
    *,
    raw_columns: frozenset[str],
) -> bool:
    if "blob_hash" not in raw_columns:
        return False
    if row["parse_error"] or row["parsed_at_ms"] is None:
        return False
    return (
        parsed_non_session_artifact_reason(
            archive_root=archive_root,
            origin=str(row["origin"] or ""),
            source_path=str(row["source_path"] or ""),
            blob_hash=row["blob_hash"],
        )
        is not None
    )


# ---------------------------------------------------------------------------
# Archive readiness surfaces (polylogue-ogn1)
#
# Extracted from ``polylogue/cli/commands/status.py``: the substrate module
# ``polylogue/maintenance/rebuild_index.py`` was importing a private
# CLI-surface helper (``_archive_readiness_status``) to check whether a freshly
# rebuilt generation is exact-ready before promotion. That is the inverse of
# this repo's documented layering rule ("surfaces may not import substrate
# internals directly", ``docs/plans/layering.yaml``) — here the substrate was
# reaching *up* into a CLI leaf adapter. This block gives both the CLI
# (`status.py`, human-facing readiness reporting) and the substrate
# (`rebuild_index.py`, promotion gating) a single shared home for the
# computation; the CLI now delegates to ``archive_readiness_status`` below
# instead of owning the only copy. The handful of tiny SQLite-introspection
# one-liners below (``_fast_count``/``_safe_int``/``_table_exists``/etc.) are
# intentionally duplicated from ``status.py``'s own private copies rather than
# migrated wholesale: those are used throughout the rest of ``status.py`` for
# unrelated status surfaces outside this cluster's scope, and a bulk
# utility-relocation refactor was not part of the layering fix being made.
# ---------------------------------------------------------------------------


def _fast_count(conn: sqlite3.Connection, sql: str, params: tuple[object, ...] = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0] or 0) if row is not None else 0


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _schema_object_exists(conn: sqlite3.Connection, name: str, *, types: tuple[str, ...]) -> bool:
    placeholders = ", ".join("?" for _ in types)
    row = conn.execute(
        f"SELECT 1 FROM sqlite_master WHERE type IN ({placeholders}) AND name = ? LIMIT 1",
        (*types, name),
    ).fetchone()
    return row is not None


def _view_exists(conn: sqlite3.Connection, view_name: str) -> bool:
    return _schema_object_exists(conn, view_name, types=("view",))


def _action_readiness_counts(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return exact, non-vacuous evidence for the derived ``actions`` view."""
    tool_use_block_count = (
        _fast_count(conn, "SELECT COUNT(*) FROM blocks WHERE block_type = 'tool_use'")
        if _table_exists(conn, "blocks")
        else 0
    )
    actions_view_present = _view_exists(conn, "actions")
    action_count = 0
    actions_view_error: str | None = None
    if actions_view_present:
        try:
            action_count = _fast_count(conn, "SELECT COUNT(*) FROM actions")
        except sqlite3.Error as exc:
            actions_view_error = str(exc)
    return {
        "action_count": action_count,
        "tool_use_block_count": tool_use_block_count,
        "actions_view_present": actions_view_present,
        "actions_view_error": actions_view_error,
    }


def _archive_readiness_counts(
    conn: sqlite3.Connection,
    *,
    source_conn: sqlite3.Connection | None,
    source_check_available: bool,
) -> dict[str, Any]:
    session_count = _fast_count(conn, "SELECT COUNT(*) FROM sessions")
    raw_link_count = (
        _fast_count(conn, "SELECT COUNT(*) FROM sessions WHERE raw_id IS NOT NULL")
        if _column_exists(conn, "sessions", "raw_id")
        else 0
    )
    missing_raw_session_count = 0
    missing_raw_session_samples: list[dict[str, Any]] = []
    if source_check_available and source_conn is not None and _column_exists(conn, "sessions", "raw_id"):
        raw_ids = {
            str(row[0])
            for row in source_conn.execute("SELECT raw_id FROM raw_sessions").fetchall()
            if row[0] is not None
        }
        missing_rows = [
            row
            for row in conn.execute(
                """
                SELECT session_id, origin, native_id, raw_id, message_count, updated_at_ms
                FROM sessions
                WHERE raw_id IS NOT NULL
                ORDER BY updated_at_ms DESC, session_id
                """
            ).fetchall()
            if str(row[3]) not in raw_ids
        ]
        missing_raw_session_count = len(missing_rows)
        missing_raw_session_samples = [
            {
                "session_id": str(row[0]),
                "origin": str(row[1]),
                "native_id": str(row[2]),
                "missing_raw_id": str(row[3]),
                "message_count": int(row[4] or 0),
                "updated_at_ms": None if row[5] is None else int(row[5]),
                "evidence_status": "lost_source_evidence",
                "loss_reason": "index_raw_id_missing_from_source_tier",
                "recovery_requirement": "restore_exact_raw_artifact_or_keep_blocked",
            }
            for row in missing_rows[:10]
        ]
    insight_status = session_insight_status_sync(conn, verify_freshness=True)
    return {
        "session_count": session_count,
        "raw_link_count": raw_link_count,
        "missing_raw_session_count": missing_raw_session_count,
        "missing_raw_session_samples": missing_raw_session_samples,
        "lost_source_evidence_count": missing_raw_session_count,
        "lost_source_evidence_samples": missing_raw_session_samples,
        "message_count": _fast_count(conn, "SELECT COUNT(*) FROM messages") if _table_exists(conn, "messages") else 0,
        "text_block_count": _fast_count(conn, "SELECT COUNT(*) FROM blocks WHERE search_text != ''")
        if _table_exists(conn, "blocks")
        else 0,
        "messages_fts_count": _fast_count(conn, "SELECT COUNT(*) FROM messages_fts")
        if _table_exists(conn, "messages_fts")
        else 0,
        "profile_row_count": insight_status.profile_row_count,
        "missing_profile_row_count": insight_status.missing_profile_row_count,
        "stale_profile_row_count": insight_status.stale_profile_row_count,
        "orphan_profile_row_count": insight_status.orphan_profile_row_count,
        "work_event_row_count": insight_status.work_event_inference_count,
        "expected_work_event_row_count": insight_status.expected_work_event_inference_count,
        "stale_work_event_row_count": insight_status.stale_work_event_inference_count,
        "orphan_work_event_row_count": insight_status.orphan_work_event_inference_count,
        "phase_row_count": insight_status.phase_inference_count,
        "expected_phase_row_count": insight_status.expected_phase_inference_count,
        "stale_phase_row_count": insight_status.stale_phase_inference_count,
        "orphan_phase_row_count": insight_status.orphan_phase_inference_count,
        "thread_count": insight_status.thread_count,
        "root_thread_count": insight_status.root_threads,
        "stale_thread_count": insight_status.stale_thread_count,
        "orphan_thread_count": insight_status.orphan_thread_count,
        **_action_readiness_counts(conn),
        "missing_session_profile_materialization": insight_status.missing_session_profile_materialization_count,
        "missing_work_events_materialization": insight_status.missing_work_event_materialization_count,
        "missing_phases_materialization": insight_status.missing_phase_materialization_count,
        "missing_thread_materialization": insight_status.missing_thread_materialization_count,
        "missing_latency_materialization": insight_status.missing_latency_materialization_count,
    }


def _archive_status_surfaces(counts: dict[str, Any], *, source_check_available: bool) -> dict[str, dict[str, Any]]:
    def surface(*, ready: bool | None, blockers: list[str], evidence: dict[str, Any]) -> dict[str, Any]:
        return {"ready": ready, "blockers": blockers, "evidence": evidence}

    def count(key: str, default: int = 0) -> int:
        return int(counts.get(key, default))

    def present_blockers(*keys: str) -> list[str]:
        return [key for key in keys if count(key) != 0]

    def mismatch_blocker(actual_key: str, expected_key: str, blocker: str) -> list[str]:
        expected = count(expected_key, count(actual_key))
        return [blocker] if count(actual_key) != expected else []

    raw_blockers: list[str] = []
    raw_ready: bool | None
    if not source_check_available:
        raw_ready = None
        raw_blockers.append("source_tier_unavailable")
    elif count("missing_raw_session_count"):
        raw_ready = False
        raw_blockers.append("missing_source_raw_sessions")
    else:
        raw_ready = True

    search_blockers = ["messages_fts_row_mismatch"] if count("text_block_count") != count("messages_fts_count") else []
    profile_blockers: list[str] = []
    if count("missing_profile_row_count"):
        profile_blockers.append("missing_profile_rows")
    profile_blockers.extend(
        present_blockers(
            "missing_session_profile_materialization",
            "stale_profile_row_count",
            "orphan_profile_row_count",
        )
    )

    def materialized(name: str) -> tuple[bool, list[str]]:
        key = f"missing_{name}_materialization"
        missing = count(key)
        return (missing == 0, [] if missing == 0 else [key])

    work_blockers = present_blockers(
        "missing_work_events_materialization",
        "stale_work_event_row_count",
        "orphan_work_event_row_count",
    )
    work_blockers.extend(
        mismatch_blocker("work_event_row_count", "expected_work_event_row_count", "work_event_row_mismatch")
    )
    phase_blockers = present_blockers(
        "missing_phases_materialization",
        "stale_phase_row_count",
        "orphan_phase_row_count",
    )
    phase_blockers.extend(mismatch_blocker("phase_row_count", "expected_phase_row_count", "phase_row_mismatch"))
    thread_blockers = present_blockers(
        "missing_thread_materialization",
        "stale_thread_count",
        "orphan_thread_count",
    )
    thread_blockers.extend(mismatch_blocker("thread_count", "root_thread_count", "thread_root_mismatch"))
    latency_ready, latency_blockers = materialized("latency")
    tool_usage_blockers: list[str] = []
    if not bool(counts.get("actions_view_present", False)):
        tool_usage_blockers.append("actions_view_missing")
    elif counts.get("actions_view_error"):
        tool_usage_blockers.append("actions_view_unreadable")
    elif count("tool_use_block_count") != count("action_count"):
        tool_usage_blockers.append("actions_tool_use_count_mismatch")

    return {
        "archive_sessions": surface(
            ready=True,
            blockers=[],
            evidence={"session_count": count("session_count"), "message_count": count("message_count")},
        ),
        "raw_artifacts": surface(
            ready=raw_ready,
            blockers=raw_blockers,
            evidence={
                "source_check_available": source_check_available,
                "raw_link_count": count("raw_link_count"),
                "missing_raw_session_count": count("missing_raw_session_count"),
                "missing_raw_session_samples": list(counts.get("missing_raw_session_samples") or []),
                "lost_source_evidence_count": count("lost_source_evidence_count"),
                "lost_source_evidence_samples": list(counts.get("lost_source_evidence_samples") or []),
            },
        ),
        "search": surface(
            ready=not search_blockers,
            blockers=search_blockers,
            evidence={
                "text_block_count": count("text_block_count"),
                "messages_fts_count": count("messages_fts_count"),
            },
        ),
        "session_profiles": surface(
            ready=not profile_blockers,
            blockers=profile_blockers,
            evidence={
                "profile_row_count": count("profile_row_count"),
                "missing_profile_row_count": count("missing_profile_row_count"),
                "missing_materialization_count": count("missing_session_profile_materialization"),
                "stale_profile_row_count": count("stale_profile_row_count"),
                "orphan_profile_row_count": count("orphan_profile_row_count"),
            },
        ),
        "timeline_work_events": surface(
            ready=not work_blockers,
            blockers=work_blockers,
            evidence={
                "work_event_row_count": count("work_event_row_count"),
                "expected_work_event_row_count": count("expected_work_event_row_count", count("work_event_row_count")),
                "missing_materialization_count": count("missing_work_events_materialization"),
                "stale_work_event_row_count": count("stale_work_event_row_count"),
                "orphan_work_event_row_count": count("orphan_work_event_row_count"),
            },
        ),
        "timeline_phases": surface(
            ready=not phase_blockers,
            blockers=phase_blockers,
            evidence={
                "phase_row_count": count("phase_row_count"),
                "expected_phase_row_count": count("expected_phase_row_count", count("phase_row_count")),
                "missing_materialization_count": count("missing_phases_materialization"),
                "stale_phase_row_count": count("stale_phase_row_count"),
                "orphan_phase_row_count": count("orphan_phase_row_count"),
            },
        ),
        "threads": surface(
            ready=not thread_blockers,
            blockers=thread_blockers,
            evidence={
                "thread_count": count("thread_count"),
                "root_thread_count": count("root_thread_count", count("thread_count")),
                "missing_materialization_count": count("missing_thread_materialization"),
                "stale_thread_count": count("stale_thread_count"),
                "orphan_thread_count": count("orphan_thread_count"),
            },
        ),
        "tool_usage": surface(
            ready=not tool_usage_blockers,
            blockers=tool_usage_blockers,
            evidence={
                "action_count": count("action_count"),
                "tool_use_block_count": count("tool_use_block_count"),
                "actions_view_present": bool(counts.get("actions_view_present", False)),
                "actions_view_error": counts.get("actions_view_error"),
            },
        ),
        "latency_profiles": surface(
            ready=latency_ready,
            blockers=latency_blockers,
            evidence={"missing_materialization_count": counts["missing_latency_materialization"]},
        ),
    }


def archive_readiness_status(root: Path) -> dict[str, Any]:
    """Return the exact-readiness surface report for one archive root.

    Shared by the CLI's ``status``/``rebuild-index --plan`` reporting and the
    substrate's ``rebuild_index_from_source`` promotion gate: a freshly
    rebuilt generation is only promoted once every surface here reports
    ``ready``.
    """
    index_db = root / "index.db"
    source_db = root / "source.db"
    if not index_db.exists():
        return {"checked": False, "reason": "missing_index_tier", "surfaces": {}}

    missing_source_evidence = missing_source_raw_session_evidence(root)
    try:
        conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
        try:
            if not _table_exists(conn, "sessions"):
                return {"checked": False, "reason": "missing_sessions_table", "surfaces": {}}
            source_check_available = source_db.exists()
            source_conn: sqlite3.Connection | None = None
            try:
                if source_check_available:
                    source_conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
                    source_check_available = _table_exists(source_conn, "raw_sessions")
                counts = _archive_readiness_counts(
                    conn,
                    source_conn=source_conn,
                    source_check_available=source_check_available,
                )
                if missing_source_evidence.get("available"):
                    missing_raw_count = _safe_int(missing_source_evidence.get("missing_raw_session_count"))
                    missing_raw_samples = _safe_list(missing_source_evidence.get("missing_raw_session_samples"))
                    lost_source_count = _safe_int(missing_source_evidence.get("lost_source_evidence_count"))
                    lost_source_samples = _safe_list(missing_source_evidence.get("lost_source_evidence_samples"))
                    counts.update(
                        {
                            "missing_raw_session_count": missing_raw_count,
                            "missing_raw_session_samples": missing_raw_samples
                            or _safe_list(counts.get("missing_raw_session_samples")),
                            "lost_source_evidence_count": lost_source_count,
                            "lost_source_evidence_samples": lost_source_samples
                            or _safe_list(counts.get("lost_source_evidence_samples")),
                        }
                    )
            finally:
                if source_conn is not None:
                    source_conn.close()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return {"checked": False, "reason": str(exc), "surfaces": {}}

    surfaces = _archive_status_surfaces(counts, source_check_available=source_check_available)
    ready_count = sum(1 for info in surfaces.values() if info["ready"] is True)
    blocked_count = sum(1 for info in surfaces.values() if info["ready"] is not True)
    return {
        "checked": True,
        "reason": None,
        "source_check_available": source_check_available,
        "ready_surface_count": ready_count,
        "blocked_surface_count": blocked_count,
        "total_surface_count": len(surfaces),
        "counts": counts,
        "surfaces": surfaces,
    }


__all__ = [
    "ACTIVE_REBUILD_STALE_AFTER_S",
    "active_rebuild_index_attempts",
    "archive_readiness_status",
    "missing_source_raw_session_evidence",
    "raw_materialization_readiness_snapshot",
    "raw_materialization_ready",
]
