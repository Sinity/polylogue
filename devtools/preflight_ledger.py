"""Read-only live-archive preflight ledger.

This module is deliberately a projection, not a repair command. It reads the
durable relations used by deployed status and keeps terminal census verdicts,
actionable parser failures, and absent census coverage distinct.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from polylogue.config import Config
from polylogue.storage.archive_readiness import probe_archive_tier, raw_materialization_readiness_snapshot
from polylogue.storage.fts.fts_lifecycle import fts_invariant_snapshot_sync
from polylogue.storage.raw_failure_lifecycle import read_raw_failure_lifecycle
from polylogue.storage.raw_retention import raw_frontier_integrity_projection
from polylogue.storage.repair import raw_materialization_replay_backlog
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

PREFLIGHT_REPORT_VERSION = 1
_REQUIRED_RAW_COLUMNS = frozenset(
    {"raw_id", "origin", "blob_size", "parse_error", "validation_status", "revision_authority"}
)
_REQUIRED_CENSUS_COLUMNS = frozenset({"raw_id", "status", "member_count"})
_REQUIRED_CURSOR_COLUMNS = frozenset(
    {"source_path", "origin", "stat_size", "byte_offset", "failure_count", "next_retry_at", "excluded"}
)
_VALID_DEBT_STATUSES = frozenset({"failed", "deferred"})


def _gib(size_bytes: int) -> float:
    return round(size_bytes / (1024**3), 3)


def _count(value: object) -> int:
    return 0 if value is None else int(str(value))


def _size_evidence(size_bytes: int) -> dict[str, object]:
    gib = _gib(size_bytes)
    return {"bytes": size_bytes, "gib": gib, "display": f"{gib:.1f}GiB"}


def _status(*, state: str, reason: str | None = None, **evidence: object) -> dict[str, object]:
    payload: dict[str, object] = {"state": state}
    if reason is not None:
        payload["reason"] = reason
    payload.update(evidence)
    return payload


def _table_columns(conn: sqlite3.Connection, table: str) -> frozenset[str]:
    rows = conn.execute("SELECT name FROM pragma_table_info(?)", (table,)).fetchall()
    return frozenset(str(row[0]) for row in rows)


def _relation_error(relation: str, exc: Exception) -> dict[str, object]:
    return _status(state="unknown", reason=f"{relation} unavailable: {exc}", available=False)


def _schema_preflight(root: Path) -> dict[str, object]:
    tiers: dict[str, object] = {}
    blocking: list[str] = []
    for tier in ArchiveTier:
        probe = probe_archive_tier(tier, root / f"{tier.value}.db")
        tiers[tier.value] = {
            "exists": probe.exists,
            "user_version": probe.user_version,
            "expected_user_version": probe.expected_user_version,
            "version_status": probe.version_status,
        }
        if not probe.exists or probe.version_status != "ok":
            blocking.append(tier.value)
    return _status(
        state="pass" if not blocking else "fail",
        reason=None if not blocking else "archive tier schema is missing or mismatched",
        available=True,
        schema_mismatches=blocking,
        tiers=tiers,
    )


def _source_distribution(root: Path) -> dict[str, object]:
    source_db = root / "source.db"
    if not source_db.exists():
        return _relation_error("source.db", FileNotFoundError(source_db))
    try:
        conn = open_readonly_connection(source_db)
        try:
            raw_columns = _table_columns(conn, "raw_sessions")
            census_columns = _table_columns(conn, "raw_membership_census")
            missing_raw = sorted(_REQUIRED_RAW_COLUMNS - raw_columns)
            missing_census_columns = sorted(_REQUIRED_CENSUS_COLUMNS - census_columns)
            if missing_raw or missing_census_columns:
                missing = ", ".join(
                    [
                        *(f"raw_sessions.{column}" for column in missing_raw),
                        *(f"raw_membership_census.{column}" for column in missing_census_columns),
                    ]
                )
                return _status(
                    state="unknown",
                    reason=f"required source relation columns missing: {missing}",
                    available=False,
                )
            rows = conn.execute(
                """
                WITH classified AS (
                    SELECT r.origin, r.blob_size, r.revision_authority, r.parse_error,
                           LOWER(COALESCE(r.validation_status, '')) AS validation_status,
                           c.status AS census_status,
                           CASE WHEN c.raw_id IS NULL THEN 1 ELSE 0 END AS coverage_unknown,
                           CASE WHEN c.status IN ('failed', 'non_session') THEN 1 ELSE 0 END AS terminal,
                           CASE WHEN r.parse_error IS NOT NULL
                                  OR LOWER(COALESCE(r.validation_status, '')) = 'failed'
                                THEN 1 ELSE 0 END AS failure,
                           CASE WHEN c.status = 'complete' AND c.member_count > 0 THEN 1 ELSE 0 END AS census_eligible
                    FROM raw_sessions AS r
                    LEFT JOIN raw_membership_census AS c ON c.raw_id = r.raw_id
                )
                SELECT origin, COUNT(*), COALESCE(SUM(blob_size), 0),
                       COALESCE(SUM(parse_error IS NOT NULL), 0),
                       COALESCE(SUM(validation_status = 'failed'), 0),
                       COALESCE(SUM(revision_authority = 'quarantined'), 0),
                       COALESCE(SUM(CASE WHEN revision_authority = 'quarantined' THEN blob_size ELSE 0 END), 0),
                       COALESCE(SUM(coverage_unknown), 0),
                       COALESCE(SUM(CASE WHEN coverage_unknown = 1 THEN blob_size ELSE 0 END), 0),
                       COALESCE(SUM(CASE WHEN coverage_unknown = 0 AND terminal = 1 THEN 1 ELSE 0 END), 0),
                       COALESCE(SUM(CASE WHEN coverage_unknown = 0 AND terminal = 1 THEN blob_size ELSE 0 END), 0),
                       COALESCE(SUM(CASE WHEN coverage_unknown = 0 AND terminal = 0 AND failure = 1 THEN 1 ELSE 0 END), 0),
                       COALESCE(SUM(CASE WHEN coverage_unknown = 0 AND terminal = 0 AND failure = 1 THEN blob_size ELSE 0 END), 0),
                       COALESCE(SUM(CASE WHEN coverage_unknown = 0 AND terminal = 0 AND failure = 0
                                          AND revision_authority = 'quarantined' THEN 1 ELSE 0 END), 0),
                       COALESCE(SUM(CASE WHEN coverage_unknown = 0 AND terminal = 0 AND failure = 0
                                          AND revision_authority = 'quarantined' THEN blob_size ELSE 0 END), 0),
                       COALESCE(SUM(CASE WHEN coverage_unknown = 0 AND terminal = 0 AND failure = 0
                                          AND revision_authority != 'quarantined' AND census_eligible = 1 THEN 1 ELSE 0 END), 0),
                       COALESCE(SUM(CASE WHEN coverage_unknown = 0 AND terminal = 0 AND failure = 0
                                          AND revision_authority != 'quarantined' AND census_eligible = 1
                                         THEN blob_size ELSE 0 END), 0)
                FROM classified
                GROUP BY origin
                ORDER BY COUNT(*) DESC, origin
                """
            ).fetchall()
            totals = conn.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(blob_size), 0),
                       COALESCE(SUM(parse_error IS NOT NULL), 0),
                       COALESCE(SUM(LOWER(COALESCE(validation_status, '')) = 'failed'), 0),
                       COALESCE(SUM(revision_authority = 'quarantined'), 0),
                       COALESCE(SUM(CASE WHEN revision_authority = 'quarantined' THEN blob_size ELSE 0 END), 0),
                       COALESCE(SUM(CASE WHEN c.raw_id IS NULL THEN 1 ELSE 0 END), 0),
                       COALESCE(SUM(CASE WHEN c.raw_id IS NULL THEN r.blob_size ELSE 0 END), 0)
                FROM raw_sessions AS r
                LEFT JOIN raw_membership_census AS c ON c.raw_id = r.raw_id
                """
            ).fetchone()
        finally:
            conn.close()
    except Exception as exc:
        return _relation_error("source raw/census relations", exc)

    distribution: list[dict[str, object]] = []
    for row in rows:
        distribution.append(
            {
                "origin": str(row[0]),
                "raw_count": int(row[1] or 0),
                "blob": _size_evidence(int(row[2] or 0)),
                "parse_failures": int(row[3] or 0),
                "validation_failures": int(row[4] or 0),
                "quarantine": {"count": int(row[5] or 0), "size": _size_evidence(int(row[6] or 0))},
                "census_coverage": {
                    "status": "missing" if int(row[7] or 0) else "present",
                    "missing_count": int(row[7] or 0),
                    "missing_size": _size_evidence(int(row[8] or 0)),
                },
                "eligibility": {
                    "terminal_count": int(row[9] or 0),
                    "terminal_size": _size_evidence(int(row[10] or 0)),
                    "actionable_count": int(row[11] or 0),
                    "actionable_size": _size_evidence(int(row[12] or 0)),
                    "authority_pending_count": int(row[13] or 0),
                    "authority_pending_size": _size_evidence(int(row[14] or 0)),
                    "eligible_count": int(row[15] or 0),
                    "eligible_size": _size_evidence(int(row[16] or 0)),
                },
            }
        )
    (
        raw_count,
        raw_bytes,
        parse_failures,
        validation_failures,
        quarantined,
        quarantined_bytes,
        missing_census,
        missing_census_bytes,
    ) = tuple(_count(value) for value in totals)
    eligibility_rows = [item["eligibility"] for item in distribution]
    actionable = sum(int(item["actionable_count"]) for item in eligibility_rows if isinstance(item, dict))
    terminal = sum(int(item["terminal_count"]) for item in eligibility_rows if isinstance(item, dict))
    authority_pending = sum(int(item["authority_pending_count"]) for item in eligibility_rows if isinstance(item, dict))
    state = (
        "fail" if actionable else "unknown" if missing_census else "warn" if terminal or authority_pending else "pass"
    )
    return _status(
        state=state,
        reason=(
            "source coverage is incomplete"
            if missing_census
            else "actionable parse/validation failures remain"
            if actionable
            else "known terminal or authority-pending raw evidence"
            if terminal or authority_pending
            else None
        ),
        available=True,
        totals={
            "raw_count": raw_count,
            "raw_size": _size_evidence(raw_bytes),
            "parse_failures": parse_failures,
            "validation_failures": validation_failures,
            "quarantined_count": quarantined,
            "quarantined_size": _size_evidence(quarantined_bytes),
            "missing_census_count": missing_census,
            "missing_census_size": _size_evidence(missing_census_bytes),
            "terminal_count": terminal,
            "actionable_count": actionable,
            "authority_pending_count": authority_pending,
        },
        by_origin=distribution,
        semantics={
            "quarantine": "authority_pending, not automatically bad",
            "missing_census": "coverage_unknown, never terminal or actionable without a census verdict",
            "terminal": "census status failed or non_session",
            "actionable": "parse/validation failure with present non-terminal census evidence",
        },
    )


def _index_profiles(root: Path) -> dict[str, object]:
    index_db = root / "index.db"
    if not index_db.exists():
        return _relation_error("index.db", FileNotFoundError(index_db))
    try:
        conn = open_readonly_connection(index_db)
        try:
            required = {"sessions", "session_profiles"}
            missing = sorted(
                table
                for table in required
                if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
            )
            if missing:
                return _status(
                    state="unknown", reason=f"required index relation(s) missing: {', '.join(missing)}", available=False
                )
            sessions = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] or 0)
            missing_profiles = int(
                conn.execute(
                    "SELECT COUNT(*) FROM sessions AS s WHERE NOT EXISTS (SELECT 1 FROM session_profiles AS p WHERE p.session_id = s.session_id)"
                ).fetchone()[0]
                or 0
            )
        finally:
            conn.close()
    except Exception as exc:
        return _relation_error("index sessions/profile relations", exc)
    return _status(
        state="fail" if missing_profiles else "pass",
        reason="session profile rows are missing" if missing_profiles else None,
        available=True,
        sessions_count=sessions,
        missing_profile_count=missing_profiles,
    )


def _fts_preflight(root: Path) -> dict[str, object]:
    index_db = root / "index.db"
    if not index_db.exists():
        return _relation_error("index.db FTS relations", FileNotFoundError(index_db))
    try:
        conn = open_readonly_connection(index_db)
        try:
            snapshot = fts_invariant_snapshot_sync(conn)
        finally:
            conn.close()
    except Exception as exc:
        return _relation_error("FTS readiness", exc)
    surfaces = {
        surface.name: {
            "source_exists": surface.source_exists,
            "exists": surface.exists,
            "source_rows": surface.source_rows,
            "indexed_rows": surface.indexed_rows,
            "triggers_present": surface.triggers_present,
            "missing_rows": surface.missing_rows,
            "excess_rows": surface.excess_rows,
            "duplicate_rows": surface.duplicate_rows,
            "identity_mismatch_rows": surface.identity_mismatch_rows,
            "ready": surface.ready,
        }
        for surface in snapshot.surfaces
    }
    debt = 0
    for surface in surfaces.values():
        debt += sum(
            int(surface.get(key) or 0)
            for key in ("missing_rows", "excess_rows", "duplicate_rows", "identity_mismatch_rows")
        )
    ready = snapshot.ready
    return _status(
        state="pass" if ready and debt == 0 else "fail",
        reason=None if ready and debt == 0 else "FTS debt or invariant failure is present",
        available=True,
        debt_count=debt,
        coverage_pct=(
            round(snapshot.messages.indexed_rows / snapshot.messages.source_rows * 100, 1)
            if snapshot.messages.source_rows
            else None
        ),
        coverage_exact=True,
        surfaces=surfaces,
    )


def _frontier_preflight(root: Path) -> dict[str, object]:
    try:
        readiness = raw_materialization_readiness_snapshot(root, classify_gaps=True)
        projection = raw_frontier_integrity_projection(root, readiness)
        payload = projection.to_dict()
    except Exception as exc:
        return _relation_error("raw frontier relations", exc)
    overall = str(payload.get("overall_status") or "unknown")
    state = "pass" if overall == "healthy" else "fail" if overall == "violated" else "unknown"
    return _status(
        state=state,
        reason=None
        if state == "pass"
        else str(payload.get("missing_source_raw_reason") or "raw frontier is not healthy"),
        available=bool(payload.get("available")),
        evidence=payload,
    )


def _replay_preflight(root: Path, *, limit: int) -> dict[str, object]:
    try:
        payload = raw_materialization_replay_backlog(
            Config(archive_root=root, render_root=root / "render", sources=[], db_path=root / "index.db"),
            limit=limit,
        )
    except Exception as exc:
        return _relation_error("raw replay relations", exc)
    if payload.get("available") is not True:
        return _status(
            state="unknown",
            reason=str(payload.get("reason") or "raw replay backlog unavailable"),
            available=False,
            evidence=payload,
        )
    candidate_count = _count(payload.get("candidate_count"))
    blocked_count = _count(payload.get("blocked_candidate_count"))
    # ``blocked_candidate_count`` includes authority/resource debt that is
    # intentionally excluded from ``candidate_count``.  Comparing the two
    # numbers can therefore hide one executable row behind one unrelated
    # blocked row.  Any executable candidate is a hard preflight failure;
    # blocked-only work remains a visible warning.
    state = "fail" if candidate_count else "warn" if blocked_count else "pass"
    return _status(
        state=state,
        reason=(
            "executable raw replay candidates remain"
            if state == "fail"
            else "raw replay candidates are authority/resource blocked"
            if state == "warn"
            else None
        ),
        available=True,
        candidate_count=candidate_count,
        blocked_candidate_count=blocked_count,
        authority_quarantined_count=_count(payload.get("authority_quarantined_count")),
        evidence=payload,
    )


def _cursor_preflight(root: Path, *, now: datetime | None = None, limit: int) -> dict[str, object]:
    ops_db = root / "ops.db"
    if not ops_db.exists():
        return _relation_error("ops.db cursor relation", FileNotFoundError(ops_db))
    try:
        conn = open_readonly_connection(ops_db)
        try:
            columns = _table_columns(conn, "ingest_cursor")
            missing = sorted(_REQUIRED_CURSOR_COLUMNS - columns)
            if missing:
                return _status(
                    state="unknown", reason=f"ingest_cursor columns missing: {', '.join(missing)}", available=False
                )
            rows = conn.execute(
                "SELECT source_path, origin, failure_count, next_retry_at, excluded FROM ingest_cursor ORDER BY source_path"
            ).fetchall()
            totals = conn.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(failure_count > 0), 0),
                       COALESCE(SUM(excluded = 1), 0),
                       COALESCE(SUM(failure_count > 0 AND excluded = 0), 0)
                FROM ingest_cursor
                """
            ).fetchone()
        finally:
            conn.close()
    except Exception as exc:
        return _relation_error("ops.db ingest_cursor", exc)
    resolved_now = now or datetime.now(UTC)
    failed = int(totals[1] or 0)
    excluded = int(totals[2] or 0)
    retry_due = 0
    for row in rows:
        if int(row[2] or 0) <= 0 or bool(row[4]):
            continue
        retry_at = row[3]
        if retry_at is None:
            retry_due += 1
            continue
        try:
            if datetime.fromisoformat(str(retry_at)) <= resolved_now:
                retry_due += 1
        except ValueError:
            return _status(
                state="unknown", reason=f"ingest_cursor has malformed retry timestamp: {retry_at!r}", available=False
            )
    state = "fail" if failed else "warn" if excluded else "pass"
    return _status(
        state=state,
        reason="cursor failures remain"
        if failed
        else "excluded cursors are parked until source identity changes"
        if excluded
        else None,
        available=True,
        tracked_count=int(totals[0] or 0),
        failed_count=failed,
        excluded_count=excluded,
        retry_due_count=retry_due,
        sample=[
            {"source_path": str(row[0]), "origin": row[1], "failure_count": int(row[2] or 0), "excluded": bool(row[4])}
            for row in rows[:limit]
        ],
    )


def _convergence_preflight(root: Path) -> dict[str, object]:
    ops_db = root / "ops.db"
    if not ops_db.exists():
        return _relation_error("ops.db convergence_debt relation", FileNotFoundError(ops_db))
    try:
        conn = open_readonly_connection(ops_db)
        try:
            columns = _table_columns(conn, "convergence_debt")
            missing = sorted({"stage", "status", "target_type", "target_id", "updated_at_ms"} - columns)
            if missing:
                return _status(
                    state="unknown", reason=f"convergence_debt columns missing: {', '.join(missing)}", available=False
                )
            rows = conn.execute(
                "SELECT stage, status, target_type, target_id FROM convergence_debt ORDER BY updated_at_ms DESC LIMIT 16"
            ).fetchall()
            counts = conn.execute("SELECT status, COUNT(*) FROM convergence_debt GROUP BY status").fetchall()
        finally:
            conn.close()
    except Exception as exc:
        return _relation_error("ops.db convergence_debt", exc)
    unknown = sorted(str(row[0]) for row in counts if str(row[0]) not in _VALID_DEBT_STATUSES)
    if unknown:
        return _status(
            state="unknown",
            reason=f"convergence_debt has unknown status value(s): {', '.join(unknown)}",
            available=False,
        )
    failed = sum(int(row[1] or 0) for row in counts if str(row[0]) == "failed")
    deferred = sum(int(row[1] or 0) for row in counts if str(row[0]) == "deferred")
    return _status(
        state="fail" if failed else "warn" if deferred else "pass",
        reason="failed convergence debt rows remain"
        if failed
        else "deferred convergence debt remains"
        if deferred
        else None,
        available=True,
        failed_count=failed,
        deferred_count=deferred,
        row_count=failed + deferred,
        sample=[
            {"stage": str(row[0]), "status": str(row[1]), "target_type": str(row[2]), "target_id": str(row[3])}
            for row in rows
        ],
    )


def _raw_failure_preflight(root: Path, *, limit: int) -> dict[str, object]:
    """Project typed lifecycle evidence into the stopped-daemon ledger."""
    snapshot = read_raw_failure_lifecycle(root / "source.db", sample_limit=limit)
    if not snapshot.available:
        return _status(
            state="unknown",
            reason=snapshot.reason or "raw failure lifecycle unavailable",
            available=False,
            evidence=snapshot.to_dict(),
        )
    state = "fail" if snapshot.unexplained else "warn" if snapshot.deferred or snapshot.terminal else "pass"
    return _status(
        state=state,
        reason=(
            "raw failures lack typed lifecycle evidence"
            if snapshot.unexplained
            else "raw failures are classified as deferred or terminal"
            if snapshot.deferred or snapshot.terminal
            else None
        ),
        available=True,
        evidence=snapshot.to_dict(),
    )


def build_preflight_ledger(root: Path, *, limit: int = 10, now: datetime | None = None) -> dict[str, object]:
    """Build a read-only preflight ledger from the deployed archive relations."""
    checks = {
        "schema": _schema_preflight(root),
        "source": _source_distribution(root),
        "index_profiles": _index_profiles(root),
        "fts": _fts_preflight(root),
        "source_frontier": _frontier_preflight(root),
        "replay_backlog": _replay_preflight(root, limit=limit),
        "cursor_failures": _cursor_preflight(root, now=now, limit=limit),
        "raw_failure_lifecycle": _raw_failure_preflight(root, limit=limit),
        "convergence_debt": _convergence_preflight(root),
    }
    states = [str(value.get("state")) for value in checks.values()]
    blocking = [name for name, value in checks.items() if value.get("state") in {"fail", "unknown"}]
    warnings = [name for name, value in checks.items() if value.get("state") == "warn"]
    gate = "blocked" if blocking else "ready_with_warnings" if warnings else "ready"
    return {
        "report_version": PREFLIGHT_REPORT_VERSION,
        "read_only": True,
        "mutation_operations": [],
        "state": gate,
        "ok": not blocking,
        "blocking_checks": blocking,
        "warning_checks": warnings,
        "checks": checks,
        "evidence": {
            "source": "deployed archive status relations",
            "states_observed": states,
            "denominator_policy": "counts and bytes are exact SQL aggregates; cursor samples are bounded",
        },
    }


__all__ = ["PREFLIGHT_REPORT_VERSION", "build_preflight_ledger"]
