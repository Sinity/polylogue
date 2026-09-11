"""Pinned status projections for ops and source tier evidence.

These functions take the operation reader selected by the daemon runtime.
They intentionally do not open a tier, resolve a root, or inspect configured
watch sources.  The latter means convergence family attribution is honestly
``unknown`` rather than reading ambient source configuration.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from datetime import UTC, datetime

from polylogue.core.raw_failure_evidence import raw_failure_outcome_code, validated_raw_failure_evidence_kind
from polylogue.storage.raw_failure_lifecycle import read_raw_failure_lifecycle_from_connection

_WORKLOAD_THROUGHPUT_WINDOW_MS = 5 * 60 * 1000
_WORKLOAD_HEARTBEAT_STALE_MS = 90 * 1000
_CONVERGENCE_DEBT_STATUSES = frozenset(("failed", "deferred"))
_REQUIRED_CONVERGENCE_DEBT_COLUMNS = frozenset(
    (
        "debt_id",
        "stage",
        "target_type",
        "target_id",
        "status",
        "priority",
        "attempts",
        "last_error",
        "next_retry_at",
        "materializer_version",
        "created_at_ms",
        "updated_at_ms",
    )
)
_PATH_REDACTION_RE = re.compile(r"/(?:[a-zA-Z0-9._\-]+/)*[a-zA-Z0-9._\-]+")


def _table_exists(conn: sqlite3.Connection, schema: str, table: str) -> bool:
    return (
        conn.execute(
            f"SELECT 1 FROM {schema}.sqlite_schema WHERE type IN ('table', 'view') AND name = ?", (table,)
        ).fetchone()
        is not None
    )


def _require_reader_schema(schema: str) -> None:
    if schema not in {"main", "ops_tier"}:
        raise ValueError(f"unsupported status reader schema: {schema!r}")


def _count(conn: sqlite3.Connection, sql: str, params: tuple[object, ...] = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0] or 0) if row is not None else 0


def ops_workload_status_from_connection(
    conn: sqlite3.Connection | None,
    *,
    now_ms: int,
    schema: str = "ops_tier",
) -> dict[str, object]:
    """Return the legacy ingest-workload payload from attached ops evidence."""

    _require_reader_schema(schema)
    if conn is None:
        return {"available": False, "reason": "missing_ops_tier"}
    if not _table_exists(conn, schema, "ingest_attempts"):
        return {"available": False, "reason": "missing_ingest_attempts"}
    if not _table_exists(conn, schema, "convergence_debt"):
        return {"available": False, "reason": "missing_convergence_debt"}
    running_rows = conn.execute(
        f"""
        SELECT phase, origin, started_at_ms, heartbeat_at_ms, parsed_raw_count, materialized_count
        FROM {schema}.ingest_attempts WHERE status = 'running' ORDER BY started_at_ms DESC
        """
    ).fetchall()
    running: list[dict[str, object]] = []
    actively_ingesting = False
    for row in running_rows:
        heartbeat = int(row[3] or 0)
        heartbeat_age_ms = now_ms - heartbeat if heartbeat else None
        fresh = heartbeat_age_ms is not None and heartbeat_age_ms <= _WORKLOAD_HEARTBEAT_STALE_MS
        actively_ingesting = actively_ingesting or fresh
        running.append(
            {
                "phase": row[0],
                "origin": row[1],
                "age_ms": now_ms - int(row[2] or now_ms),
                "heartbeat_age_ms": heartbeat_age_ms,
                "heartbeat_fresh": fresh,
            }
        )
    throughput_row = conn.execute(
        f"""
        SELECT COUNT(*), COALESCE(SUM(parsed_raw_count), 0), COALESCE(SUM(materialized_count), 0),
               COALESCE(SUM(finished_at_ms - started_at_ms), 0)
        FROM {schema}.ingest_attempts
        WHERE status = 'completed' AND finished_at_ms >= ?
        """,
        (now_ms - _WORKLOAD_THROUGHPUT_WINDOW_MS,),
    ).fetchone()
    batches = int(throughput_row[0] or 0) if throughput_row is not None else 0
    files = int(throughput_row[1] or 0) if throughput_row is not None else 0
    materialized = int(throughput_row[2] or 0) if throughput_row is not None else 0
    busy_ms = int(throughput_row[3] or 0) if throughput_row is not None else 0
    cursor: dict[str, int] = {}
    if _table_exists(conn, schema, "ingest_cursor"):
        cursor = {
            "tracked": _count(conn, f"SELECT COUNT(*) FROM {schema}.ingest_cursor"),
            "excluded": _count(conn, f"SELECT COUNT(*) FROM {schema}.ingest_cursor WHERE excluded = 1"),
            "retry_pending": _count(
                conn, f"SELECT COUNT(*) FROM {schema}.ingest_cursor WHERE failure_count > 0 AND excluded = 0"
            ),
        }
    debt_rows = conn.execute(f"SELECT status, COUNT(*) FROM {schema}.convergence_debt GROUP BY status").fetchall()
    unknown_statuses = sorted(
        {repr(status) for status, _count in debt_rows if status not in _CONVERGENCE_DEBT_STATUSES}
    )
    if unknown_statuses:
        return {
            "available": False,
            "reason": "convergence debt status unavailable: unknown status value(s): " + ", ".join(unknown_statuses),
        }
    debt = {str(status): int(count or 0) for status, count in debt_rows}
    return {
        "available": True,
        "actively_ingesting": actively_ingesting,
        "running_count": len(running),
        "running": running,
        "throughput": {
            "window_minutes": _WORKLOAD_THROUGHPUT_WINDOW_MS // 60_000,
            "batches": batches,
            "files": files,
            "materialized": materialized,
            "files_per_second": round(files / (busy_ms / 1000.0), 2) if busy_ms else 0.0,
        },
        "cursor": cursor,
        "debt": {"total": sum(debt.values()), "by_status": debt},
        "lifetime_attempts": {
            str(status): int(count or 0)
            for status, count in conn.execute(
                f"SELECT status, COUNT(*) FROM {schema}.ingest_attempts GROUP BY status"
            ).fetchall()
        },
    }


def convergence_status_from_connection(
    conn: sqlite3.Connection | None,
    *,
    now_ms: int,
    schema: str = "ops_tier",
) -> dict[str, object]:
    """Return the canonical convergence-debt shape from a pinned ops reader."""

    _require_reader_schema(schema)
    unavailable = {
        "available": False,
        "error": "convergence debt table is unavailable",
        "failed_count": 0,
        "deferred_count": 0,
        "retry_due_count": 0,
        "stage_summaries": [],
        "family_summaries": [],
        "recent": [],
    }
    if conn is None or not _table_exists(conn, schema, "convergence_debt"):
        return unavailable
    columns = {str(row[1]) for row in conn.execute(f"PRAGMA {schema}.table_info(convergence_debt)").fetchall()}
    missing = sorted(_REQUIRED_CONVERGENCE_DEBT_COLUMNS - columns)
    if missing:
        return {**unavailable, "error": "convergence_debt is missing required column(s): " + ", ".join(missing)}
    rows = conn.execute(
        f"""
        SELECT stage, target_type, target_id, status, attempts, updated_at_ms, last_error, next_retry_at
        FROM {schema}.convergence_debt
        ORDER BY updated_at_ms DESC, priority DESC, debt_id DESC
        """
    ).fetchall()
    if not rows:
        return {**unavailable, "available": True, "error": None}
    statuses = {str(row[3]) for row in rows}
    unknown = sorted(repr(status) for status in statuses - _CONVERGENCE_DEBT_STATUSES)
    if unknown:
        return {**unavailable, "error": "convergence_debt contains unknown status value(s): " + ", ".join(unknown)}
    now = datetime.fromtimestamp(now_ms / 1000, tz=UTC)
    recent: list[dict[str, object]] = []
    failed_by_stage: Counter[str] = Counter()
    deferred_by_stage: Counter[str] = Counter()
    retry_due_by_stage: Counter[str] = Counter()
    for row in rows:
        stage, target_type, target_id, status = (str(row[0]), str(row[1]), str(row[2]), str(row[3]))
        retry_due = status == "failed" and _retry_due(row[7], now=now)
        if status == "failed":
            failed_by_stage[stage] += 1
            retry_due_by_stage[stage] += int(retry_due)
        else:
            deferred_by_stage[stage] += 1
        if len(recent) < 10:
            updated_at_ms = int(row[5] or 0)
            recent.append(
                {
                    "stage": stage,
                    "subject_type": target_type,
                    "subject_id": target_id,
                    "status": status,
                    "failure_count": int(row[4] or 0),
                    "last_failed_at": datetime.fromtimestamp(updated_at_ms / 1000, tz=UTC).isoformat(),
                    "next_retry_at": row[7],
                    "retry_due": retry_due,
                    "last_error": row[6],
                }
            )
    stage_summaries = [
        {
            "stage": stage,
            "failed_count": failed_by_stage[stage],
            "deferred_count": deferred_by_stage[stage],
            "retry_due_count": retry_due_by_stage[stage],
        }
        for stage in sorted(
            set(failed_by_stage) | set(deferred_by_stage),
            key=lambda value: (-(failed_by_stage[value] + deferred_by_stage[value]), value),
        )
    ]
    # Legacy family classification consults configured watch roots.  No such
    # runtime input belongs to a pinned archive read, so preserve its safe
    # fallback rather than resolving configuration here.
    return {
        "available": True,
        "error": None,
        "failed_count": sum(failed_by_stage.values()),
        "deferred_count": sum(deferred_by_stage.values()),
        "retry_due_count": sum(retry_due_by_stage.values()),
        "stage_summaries": stage_summaries,
        "family_summaries": [
            {
                "family": "unknown",
                "failed_count": sum(failed_by_stage.values()),
                "deferred_count": sum(deferred_by_stage.values()),
            }
        ],
        "recent": recent,
        "family_attribution": {"state": "not_observed", "reason": "configured watch roots not supplied"},
    }


def raw_failure_status_from_connection(
    conn: sqlite3.Connection | None,
    *,
    schema: str = "main",
    sample_limit: int = 50,
) -> dict[str, object]:
    """Project raw-failure lifecycle using the canonical supplied-reader classifier."""

    _require_reader_schema(schema)
    unavailable = {
        "raw_parse_failures": 0,
        "raw_validation_failures": 0,
        "raw_quarantined": 0,
        "raw_deferred_failures": 0,
        "raw_terminal_rejections": 0,
        "raw_unexplained_failures": 0,
        "raw_failure_lifecycle_available": False,
        "raw_failure_lifecycle_state": "unavailable",
        "raw_failure_lifecycle_reason": "source tier unavailable",
        "raw_failure_samples": [],
    }
    if conn is None or not _table_exists(conn, schema, "raw_sessions"):
        return unavailable
    if schema != "main":
        # The source reader is opened as its own pinned handle.  The existing
        # classifier deliberately works only against its main schema.
        return {**unavailable, "raw_failure_lifecycle_reason": "source reader must use its main schema"}
    lifecycle = read_raw_failure_lifecycle_from_connection(conn, sample_limit=sample_limit)
    if not lifecycle.available:
        return {**unavailable, "raw_failure_lifecycle_reason": lifecycle.reason}
    sample_ids = [str(sample["raw_id"]) for sample in lifecycle.samples if sample.get("raw_id") is not None]
    rows_by_id: dict[str, sqlite3.Row | tuple[object, ...]] = {}
    if sample_ids:
        placeholders = ", ".join("?" for _ in sample_ids)
        rows_by_id = {
            str(row[0]): row
            for row in conn.execute(
                f"""
                SELECT raw_id, origin, parse_error, validation_status, validation_error
                FROM raw_sessions
                WHERE raw_id IN ({placeholders})
                  AND ((parse_error IS NOT NULL AND TRIM(parse_error) != '') OR validation_status = 'failed')
                """,
                tuple(sample_ids),
            ).fetchall()
        }
    samples: list[dict[str, object]] = []
    for sample in lifecycle.samples:
        raw_id = str(sample.get("raw_id") or "")
        row = rows_by_id.get(raw_id)
        if row is None:
            continue
        parse_error = str(row[2] or "")
        validation_status = str(row[3] or "")
        validation_error = str(row[4] or "")
        evidence = validated_raw_failure_evidence_kind(
            sample.get("artifact_kind"),
            sample.get("support_status"),
            validation_failed=validation_status == "failed",
            classification_reason=sample.get("classification_reason"),
            outcome_code=raw_failure_outcome_code(sample.get("classification_reason")),
        )
        kind = (
            evidence.value
            if evidence is not None and sample.get("lifecycle") == evidence.lifecycle
            else (
                "decode_error"
                if "JSONDecodeError" in parse_error or "decode error" in parse_error.lower()
                else "schema_violation"
                if validation_status == "failed"
                else "parse_error"
                if parse_error
                else "unknown"
            )
        )
        samples.append(
            {
                "failure_kind": kind,
                "provider_hint": None if row[1] is None else str(row[1]),
                "redacted_error": _redact_file_paths(parse_error or validation_error),
                "lifecycle": sample.get("lifecycle"),
            }
        )
    unexplained = lifecycle.unexplained
    return {
        "raw_parse_failures": lifecycle.parse_failures,
        "raw_validation_failures": lifecycle.validation_failures,
        "raw_quarantined": _count(
            conn,
            f"""SELECT COUNT(*) FROM {schema}.raw_sessions
                WHERE parsed_at_ms IS NULL AND ((parse_error IS NOT NULL AND TRIM(parse_error) != '') OR validation_status = 'failed')""",
        ),
        "raw_deferred_failures": lifecycle.deferred,
        "raw_terminal_rejections": lifecycle.terminal,
        "raw_unexplained_failures": unexplained,
        "raw_failure_lifecycle_available": True,
        "raw_failure_lifecycle_state": lifecycle.state,
        "raw_failure_lifecycle_reason": None,
        "raw_failure_samples": samples,
    }


def _retry_due(value: object, *, now: datetime) -> bool:
    if not value:
        return True
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return True
    return parsed.replace(tzinfo=UTC) <= now if parsed.tzinfo is None else parsed <= now


def _redact_file_paths(value: str) -> str:
    """Preserve the public raw-failure sample privacy rule without daemon imports."""

    def replace(match: re.Match[str]) -> str:
        start = match.start()
        if start == 0:
            return "[redacted]"
        previous = value[start - 1]
        if previous.isalnum() or previous in (".", ":"):
            return match.group(0)
        if "://" in value[max(0, start - 16) : start + 1]:
            return match.group(0)
        return "[redacted]"

    return _PATH_REDACTION_RE.sub(replace, value)
