"""Read-only daemon workload probe for live ingest and convergence hot paths.

The probe captures a stable, JSON-serializable snapshot of daemon-relevant
state that can be diffed across convergence cycles.  Operators run::

    devtools daemon-workload-probe --json > before.json
    # ...run convergence work...
    devtools daemon-workload-probe --json > after.json
    devtools daemon-workload-probe --compare before.json after.json

to produce structured before/after evidence (see issue #845).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from polylogue.paths import db_path as default_db_path
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

# Bumped when the JSON shape gains new top-level keys or changes a field type.
# The compare path uses this to refuse incompatible inputs loudly.
REPORT_VERSION = 3  # v3 adds top-level `topology_quarantine_state` (#1260)

# Tables whose row counts form the convergence "boundary" — daemon work moves
# rows into and out of these tables, so the deltas describe convergence shape.
_BOUNDARY_TABLES: tuple[str, ...] = (
    "raw_conversations",
    "conversations",
    "messages",
    "content_blocks",
    "artifact_observations",
    "messages_fts_docsize",
    "action_events",
    "action_events_fts_docsize",
    "message_embeddings",
    "session_profile",
    "live_ingest_attempt",
    "live_convergence_debt",
    "pending_blob_refs",
    "topology_edges",
)

# Expected FTS-sync triggers.  A missing trigger means the FTS index can drift
# silently (suspended for bulk operations and never restored, for example).
_EXPECTED_FTS_TRIGGERS: tuple[str, ...] = (
    "messages_fts_ai",
    "messages_fts_ad",
    "messages_fts_au",
    "action_events_fts_ai",
    "action_events_fts_ad",
    "action_events_fts_au",
)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(conn, table):
        return set()
    try:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def _scalar_int(conn: sqlite3.Connection, sql: str, params: tuple[object, ...] = ()) -> int:
    try:
        row = conn.execute(sql, params).fetchone()
    except sqlite3.Error:
        return 0
    return int(row[0] or 0) if row is not None else 0


def _recent_attempts(conn: sqlite3.Connection, *, limit: int) -> list[dict[str, Any]]:
    if not _table_exists(conn, "live_ingest_attempt"):
        return []
    columns = _columns(conn, "live_ingest_attempt")
    stale_expr = "stale_cursor_write_count" if "stale_cursor_write_count" in columns else "0"
    source_paths_expr = "source_paths_json" if "source_paths_json" in columns else "'[]'"
    rows = conn.execute(
        f"""
        SELECT
            attempt_id,
            started_at,
            updated_at,
            completed_at,
            status,
            phase,
            queued_file_count,
            needed_file_count,
            succeeded_file_count,
            failed_file_count,
            input_bytes,
            source_payload_read_bytes,
            cursor_fingerprint_read_bytes,
            parse_time_s,
            convergence_time_s,
            {stale_expr},
            {source_paths_expr}
        FROM live_ingest_attempt
        ORDER BY updated_at DESC, started_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    attempts: list[dict[str, Any]] = []
    for row in rows:
        input_bytes = int(row[10] or 0)
        read_bytes = int(row[11] or 0) + int(row[12] or 0)
        attempts.append(
            {
                "attempt_id": row[0],
                "started_at": row[1],
                "updated_at": row[2],
                "completed_at": row[3],
                "status": row[4],
                "phase": row[5],
                "queued_file_count": int(row[6] or 0),
                "needed_file_count": int(row[7] or 0),
                "succeeded_file_count": int(row[8] or 0),
                "failed_file_count": int(row[9] or 0),
                "input_bytes": input_bytes,
                "source_payload_read_bytes": int(row[11] or 0),
                "cursor_fingerprint_read_bytes": int(row[12] or 0),
                "total_read_bytes": read_bytes,
                "read_amplification": round(read_bytes / input_bytes, 3) if input_bytes > 0 else 0.0,
                "parse_time_s": float(row[13] or 0.0),
                "convergence_time_s": float(row[14] or 0.0),
                "stale_cursor_write_count": int(row[15] or 0),
                "source_paths": _json_list(row[16]),
            }
        )
    return attempts


def _attempt_counts(conn: sqlite3.Connection) -> dict[str, Any]:
    if not _table_exists(conn, "live_ingest_attempt"):
        return {
            "total": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "stale_cursor_writes": 0,
            "overlapping_source_paths": [],
        }
    running_rows = conn.execute(
        """
        SELECT source_paths_json
        FROM live_ingest_attempt
        WHERE status = 'running'
        """
    ).fetchall()
    path_counts: dict[str, int] = {}
    for row in running_rows:
        for path in _json_list(row[0]):
            path_counts[path] = path_counts.get(path, 0) + 1
    overlapping = [
        {"source_path": path, "running_attempt_count": count}
        for path, count in sorted(path_counts.items())
        if count > 1
    ]
    has_stale_col = "stale_cursor_write_count" in _columns(conn, "live_ingest_attempt")
    return {
        "total": _scalar_int(conn, "SELECT COUNT(*) FROM live_ingest_attempt"),
        "running": _scalar_int(conn, "SELECT COUNT(*) FROM live_ingest_attempt WHERE status = 'running'"),
        "completed": _scalar_int(conn, "SELECT COUNT(*) FROM live_ingest_attempt WHERE status = 'completed'"),
        "failed": _scalar_int(conn, "SELECT COUNT(*) FROM live_ingest_attempt WHERE status = 'failed'"),
        "stale_cursor_writes": _scalar_int(
            conn,
            "SELECT COALESCE(SUM(stale_cursor_write_count), 0) FROM live_ingest_attempt"
            if has_stale_col
            else "SELECT 0",
        ),
        "overlapping_source_paths": overlapping[:20],
    }


def _source_path_churn(conn: sqlite3.Connection, *, attempts: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if not _table_exists(conn, "raw_conversations"):
        return []
    source_paths = sorted({path for attempt in attempts for path in attempt.get("source_paths", []) if path})
    if not source_paths:
        return []
    has_conversations = _table_exists(conn, "conversations")
    join = "LEFT JOIN conversations AS c ON c.raw_id = r.raw_id" if has_conversations else ""
    conversation_count = "COUNT(DISTINCT c.conversation_id)" if has_conversations else "0"
    placeholders = ",".join("?" for _ in source_paths)
    rows = conn.execute(
        f"""
        SELECT
            r.source_path,
            COUNT(*) AS raw_count,
            SUM(CASE WHEN COALESCE(r.source_index, 0) >= 0 THEN 1 ELSE 0 END) AS full_raw_count,
            SUM(CASE WHEN r.source_index = -1 THEN 1 ELSE 0 END) AS append_raw_count,
            {conversation_count} AS conversation_count,
            SUM(COALESCE(r.blob_size, 0)) AS total_blob_bytes,
            MAX(r.acquired_at) AS latest_acquired_at
        FROM raw_conversations AS r
        {join}
        WHERE r.source_path IN ({placeholders})
        GROUP BY r.source_path
        HAVING raw_count > 1
        ORDER BY raw_count DESC, total_blob_bytes DESC, r.source_path
        LIMIT ?
        """,
        (*source_paths, limit),
    ).fetchall()
    items: list[dict[str, Any]] = []
    for row in rows:
        raw_count = int(row[1] or 0)
        conversation_count_value = int(row[4] or 0)
        items.append(
            {
                "source_path": row[0],
                "raw_count": raw_count,
                "full_raw_count": int(row[2] or 0),
                "append_raw_count": int(row[3] or 0),
                "conversation_count": conversation_count_value,
                "orphan_raw_count": max(0, raw_count - conversation_count_value),
                "total_blob_bytes": int(row[5] or 0),
                "latest_acquired_at": row[6],
            }
        )
    return items


def _cursor_lag_baselines(conn: sqlite3.Connection) -> dict[str, Any]:
    """Rolling-window cursor-lag baseline state per source family (#1349).

    Reads ``live_cursor_lag_sample`` and returns the same per-family
    snapshot the anomaly check uses: sample-count, time range, p50, p95.
    Stable JSON shape so before/after probes can detect baseline drift in
    convergence proofs (e.g. *"after this run, claude-code-session's
    rolling p95 dropped from 600s to 12s — convergence is healthy"*).
    """
    if not _table_exists(conn, "live_cursor_lag_sample"):
        return {"table_present": False, "family_count": 0, "total_sample_count": 0, "families": []}
    rows = conn.execute(
        """
        SELECT family,
               COUNT(*),
               MIN(observed_at),
               MAX(observed_at),
               MAX(max_lag_s),
               AVG(max_lag_s),
               SUM(stuck_file_count)
        FROM live_cursor_lag_sample
        GROUP BY family
        ORDER BY COUNT(*) DESC, family
        """
    ).fetchall()
    families: list[dict[str, Any]] = []
    for row in rows:
        family = str(row[0])
        sample_count = int(row[1] or 0)
        # Compute p50/p95 per family with a small follow-up read; bounded
        # at ~200K rows total across families this stays under a second.
        per_family = [
            float(r[0])
            for r in conn.execute(
                "SELECT max_lag_s FROM live_cursor_lag_sample WHERE family = ? ORDER BY max_lag_s",
                (family,),
            ).fetchall()
        ]
        p50 = _percentile_from_sorted(per_family, 0.5)
        p95 = _percentile_from_sorted(per_family, 0.95)
        families.append(
            {
                "family": family,
                "sample_count": sample_count,
                "first_observed_at": row[2],
                "last_observed_at": row[3],
                "max_lag_s_seen": round(float(row[4] or 0.0), 3),
                "mean_lag_s": round(float(row[5] or 0.0), 3),
                "stuck_file_total": int(row[6] or 0),
                "p50_lag_s": round(p50, 3),
                "p95_lag_s": round(p95, 3),
            }
        )
    return {
        "table_present": True,
        "family_count": len(families),
        "total_sample_count": sum(f["sample_count"] for f in families),
        "families": families,
    }


def _percentile_from_sorted(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])
    position = q * (len(sorted_values) - 1)
    lo = int(position)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = position - lo
    return float(sorted_values[lo]) * (1.0 - frac) + float(sorted_values[hi]) * frac


def _convergence_debt(conn: sqlite3.Connection) -> dict[str, Any]:
    if not _table_exists(conn, "live_convergence_debt"):
        return {"failed_count": 0, "by_stage": []}
    rows = conn.execute(
        """
        SELECT stage, COUNT(*) AS failed_count
        FROM live_convergence_debt
        WHERE status != 'resolved'
        GROUP BY stage
        ORDER BY failed_count DESC, stage
        """
    ).fetchall()
    return {
        "failed_count": sum(int(row[1] or 0) for row in rows),
        "by_stage": [{"stage": row[0], "failed_count": int(row[1] or 0)} for row in rows],
    }


def _boundary_table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in _BOUNDARY_TABLES:
        if _table_exists(conn, table):
            counts[table] = _scalar_int(conn, f"SELECT COUNT(*) FROM {table}")
        else:
            counts[table] = -1
    return counts


def _topology_quarantine_state(conn: sqlite3.Connection) -> dict[str, Any]:
    """Report the state of the topology-edge cycle quarantine (#1260).

    Surfaces:
    - ``table_present`` — is ``topology_edges`` materialized in the database
    - ``unresolved_count`` — edges still awaiting their parent
    - ``resolved_count`` — edges whose parent has been ingested
    - ``quarantined_count`` — edges rejected because they would create a
      cycle in ``conversations.parent_conversation_id``
    - ``oldest_quarantined_at`` — oldest ``resolved_at`` timestamp on a
      quarantined edge (the field is repurposed as the
      "decision-recorded-at" timestamp for non-resolved terminal states)

    Operators monitor ``quarantined_count`` as a health indicator — a
    non-zero value means at least one source emitted a cycle and the
    fast-path graph was prevented from entering it.
    """

    if not _table_exists(conn, "topology_edges"):
        return {
            "table_present": False,
            "unresolved_count": 0,
            "resolved_count": 0,
            "quarantined_count": 0,
            "oldest_quarantined_at": None,
        }
    rows = conn.execute("SELECT status, COUNT(*) AS n FROM topology_edges GROUP BY status").fetchall()
    status_counts = {str(row[0]): int(row[1]) for row in rows}
    oldest_row = conn.execute("SELECT MIN(resolved_at) FROM topology_edges WHERE status = 'quarantined'").fetchone()
    return {
        "table_present": True,
        "unresolved_count": int(status_counts.get("unresolved", 0)),
        "resolved_count": int(status_counts.get("resolved", 0)),
        "quarantined_count": int(status_counts.get("quarantined", 0)),
        "oldest_quarantined_at": oldest_row[0] if oldest_row else None,
    }


def _blob_lease_state(conn: sqlite3.Connection) -> dict[str, Any]:
    if not _table_exists(conn, "pending_blob_refs"):
        return {
            "table_present": False,
            "pending_lease_count": 0,
            "distinct_operations": 0,
            "oldest_acquired_at": None,
        }
    pending = _scalar_int(conn, "SELECT COUNT(*) FROM pending_blob_refs")
    operations = _scalar_int(conn, "SELECT COUNT(DISTINCT operation_id) FROM pending_blob_refs")
    oldest_row = conn.execute("SELECT MIN(acquired_at) FROM pending_blob_refs").fetchone()
    oldest = int(oldest_row[0]) if oldest_row is not None and oldest_row[0] is not None else None
    return {
        "table_present": True,
        "pending_lease_count": pending,
        "distinct_operations": operations,
        "oldest_acquired_at": oldest,
    }


def _gc_state(conn: sqlite3.Connection) -> dict[str, Any]:
    if not _table_exists(conn, "gc_generations"):
        return {
            "table_present": False,
            "high_water_generation": 0,
            "last_completed_at": None,
            "generation_count": 0,
        }
    row = conn.execute(
        """
        SELECT
            COALESCE(MAX(generation), 0) AS high_water,
            COUNT(*) AS generation_count
        FROM gc_generations
        """
    ).fetchone()
    high_water = int(row[0] or 0) if row is not None else 0
    generation_count = int(row[1] or 0) if row is not None else 0
    completed_row = conn.execute(
        "SELECT completed_at FROM gc_generations WHERE generation = ? LIMIT 1",
        (high_water,),
    ).fetchone()
    completed_at = int(completed_row[0]) if completed_row is not None and completed_row[0] is not None else None
    return {
        "table_present": True,
        "high_water_generation": high_water,
        "last_completed_at": completed_at,
        "generation_count": generation_count,
    }


def _fts_trigger_state(conn: sqlite3.Connection) -> dict[str, Any]:
    present = sorted(
        str(row[0])
        for row in conn.execute(
            f"""
            SELECT name FROM sqlite_master
            WHERE type='trigger' AND name IN ({",".join("?" for _ in _EXPECTED_FTS_TRIGGERS)})
            """,
            _EXPECTED_FTS_TRIGGERS,
        ).fetchall()
    )
    expected = set(_EXPECTED_FTS_TRIGGERS)
    missing = sorted(expected - set(present))
    return {
        "expected": list(_EXPECTED_FTS_TRIGGERS),
        "present": present,
        "missing": missing,
        "all_present": not missing,
    }


def _convergence_stage_timings(
    attempts: list[dict[str, Any]],
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    completed = [a for a in attempts if a.get("status") == "completed"]
    if not completed:
        return {
            "sample_size": 0,
            "parse_time_s": _summary_stat([]),
            "convergence_time_s": _summary_stat([]),
            "read_amplification": _summary_stat([]),
            "per_stage_s": {},
        }
    parse_times = [float(a.get("parse_time_s") or 0.0) for a in completed]
    convergence_times = [float(a.get("convergence_time_s") or 0.0) for a in completed]
    amps = [float(a.get("read_amplification") or 0.0) for a in completed]
    return {
        "sample_size": len(completed),
        "parse_time_s": _summary_stat(parse_times),
        "convergence_time_s": _summary_stat(convergence_times),
        "read_amplification": _summary_stat(amps),
        "per_stage_s": _per_stage_timings(
            [str(a.get("attempt_id")) for a in completed if a.get("attempt_id")],
            conn,
        ),
    }


def _per_stage_timings(
    attempt_ids: list[str],
    conn: sqlite3.Connection | None,
) -> dict[str, dict[str, float]]:
    """Aggregate per-stage timings from ``live_ingest_stage_event``.

    Returns ``{stage_name: summary_stat}`` over the latest ``stage_timings_json``
    payload for each attempt.  Returns ``{}`` when the table does not exist or
    no completed attempts carry timings.
    """
    if conn is None or not attempt_ids:
        return {}
    if not _table_exists(conn, "live_ingest_stage_event"):
        return {}
    placeholders = ",".join("?" for _ in attempt_ids)
    rows = conn.execute(
        f"""
        SELECT attempt_id, stage_timings_json
        FROM live_ingest_stage_event
        WHERE attempt_id IN ({placeholders})
          AND stage_timings_json IS NOT NULL
        ORDER BY attempt_id, sequence DESC
        """,
        tuple(attempt_ids),
    ).fetchall()
    latest_by_attempt: dict[str, str] = {}
    for row in rows:
        attempt_id = str(row[0])
        if attempt_id in latest_by_attempt:
            continue
        if isinstance(row[1], str) and row[1]:
            latest_by_attempt[attempt_id] = row[1]
    per_stage: dict[str, list[float]] = {}
    for payload in latest_by_attempt.values():
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(decoded, dict):
            continue
        for stage_name, value in decoded.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            per_stage.setdefault(str(stage_name), []).append(float(value))
    return {stage: _summary_stat(values) for stage, values in sorted(per_stage.items())}


def _summary_stat(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "sum": 0.0, "mean": 0.0}
    total = sum(values)
    return {
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "sum": round(total, 6),
        "mean": round(total / len(values), 6),
    }


def _daemon_resource_signal(attempts: list[dict[str, Any]], conn: sqlite3.Connection) -> dict[str, Any]:
    """Read the most recent RSS / cgroup snapshot from `live_ingest_attempt`.

    These are the only daemon-RSS signals the probe can read without IPC.
    The probe is intentionally read-only.
    """

    if not _table_exists(conn, "live_ingest_attempt"):
        return {"available": False}
    columns = _columns(conn, "live_ingest_attempt")
    optional = (
        "rss_current_mb",
        "rss_peak_self_mb",
        "rss_peak_children_mb",
        "cgroup_memory_current_mb",
        "cgroup_memory_peak_mb",
        "cgroup_memory_swap_current_mb",
        "worker_in_flight_count",
        "worker_completed_count",
        "worker_total_count",
    )
    available_columns = [name for name in optional if name in columns]
    if not available_columns:
        return {"available": False}
    select_list = ", ".join(available_columns)
    row = conn.execute(
        f"""
        SELECT {select_list}
        FROM live_ingest_attempt
        ORDER BY updated_at DESC, started_at DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return {"available": False}
    return {"available": True, **{col: row[idx] for idx, col in enumerate(available_columns)}}


def _explain(conn: sqlite3.Connection, sql: str, params: tuple[object, ...]) -> list[str]:
    try:
        return [str(row[3]) for row in conn.execute(f"EXPLAIN QUERY PLAN {sql}", params).fetchall()]
    except sqlite3.Error as exc:
        return [f"unavailable: {exc}"]


def _query_plans(conn: sqlite3.Connection) -> dict[str, Any]:
    plans: dict[str, Any] = {}
    if _table_exists(conn, "raw_conversations") and _table_exists(conn, "conversations"):
        source_row = conn.execute(
            "SELECT source_path FROM raw_conversations WHERE source_path IS NOT NULL LIMIT 1"
        ).fetchone()
        if source_row is not None:
            plan = _explain(
                conn,
                """
                SELECT DISTINCT r.source_path, c.conversation_id
                FROM raw_conversations AS r
                JOIN conversations AS c ON c.raw_id = r.raw_id
                WHERE r.source_path IN (?)
                ORDER BY r.source_path, c.conversation_id
                """,
                (source_row[0],),
            )
            plans["source_path_lookup"] = {
                "plan": plan,
                "hazards": [item for item in plan if "SCAN c" in item],
            }
    if _table_exists(conn, "messages") and _table_exists(conn, "messages_fts_docsize"):
        conv_row = conn.execute("SELECT conversation_id FROM messages LIMIT 1").fetchone()
        if conv_row is not None:
            plan = _explain(
                conn,
                """
                SELECT COUNT(*)
                FROM messages AS m
                LEFT JOIN messages_fts_docsize AS d ON d.id = m.rowid
                WHERE m.text IS NOT NULL
                  AND m.conversation_id IN (?)
                  AND d.id IS NULL
                """,
                (conv_row[0],),
            )
            plans["message_fts_gap_probe"] = {
                "plan": plan,
                "hazards": [item for item in plan if "messages_fts " in item],
            }
    if _table_exists(conn, "action_events") and _table_exists(conn, "action_events_fts_docsize"):
        conv_row = conn.execute("SELECT conversation_id FROM action_events LIMIT 1").fetchone()
        if conv_row is not None:
            plan = _explain(
                conn,
                """
                SELECT COUNT(*)
                FROM action_events AS ae
                LEFT JOIN action_events_fts_docsize AS d ON d.id = ae.rowid
                WHERE ae.conversation_id IN (?)
                  AND d.id IS NULL
                """,
                (conv_row[0],),
            )
            plans["action_fts_gap_probe"] = {
                "plan": plan,
                "hazards": [item for item in plan if "action_events_fts " in item],
            }
    return plans


def probe(db: Path, *, limit: int = 5) -> dict[str, Any]:
    if not db.exists():
        return {
            "ok": False,
            "report_version": REPORT_VERSION,
            "captured_at": _now_iso(),
            "db_path": str(db),
            "error": "database does not exist",
        }
    conn = open_readonly_connection(db)
    try:
        recent_attempts = _recent_attempts(conn, limit=limit)
        return {
            "ok": True,
            "report_version": REPORT_VERSION,
            "captured_at": _now_iso(),
            "db_path": str(db),
            "attempt_counts": _attempt_counts(conn),
            "recent_attempts": recent_attempts,
            "convergence_stage_timings": _convergence_stage_timings(recent_attempts, conn),
            "boundary_table_counts": _boundary_table_counts(conn),
            "topology_quarantine_state": _topology_quarantine_state(conn),
            "blob_lease_state": _blob_lease_state(conn),
            "gc_state": _gc_state(conn),
            "fts_trigger_state": _fts_trigger_state(conn),
            "daemon_resource_signal": _daemon_resource_signal(recent_attempts, conn),
            "source_path_churn": _source_path_churn(conn, attempts=recent_attempts, limit=limit),
            "convergence_debt": _convergence_debt(conn),
            "cursor_lag_baselines": _cursor_lag_baselines(conn),
            "query_plans": _query_plans(conn),
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Produce a structured before/after diff of two probe reports.

    The diff is keyed by canonical sections.  Each section's value contains
    enough state to be read on its own (no implicit ordering across sections).
    """

    before_version = int(before.get("report_version", 0))
    after_version = int(after.get("report_version", 0))
    if before_version != after_version:
        return {
            "ok": False,
            "error": (
                f"report_version mismatch: before={before_version} after={after_version}; "
                "regenerate both reports with the same devtools build"
            ),
            "before_version": before_version,
            "after_version": after_version,
        }
    if not before.get("ok") or not after.get("ok"):
        return {
            "ok": False,
            "error": "compare requires two successful probe reports",
            "before_ok": bool(before.get("ok")),
            "after_ok": bool(after.get("ok")),
        }

    before_counts = _coerce_int_map(before.get("attempt_counts") or {})
    after_counts = _coerce_int_map(after.get("attempt_counts") or {})
    attempt_delta = {
        key: {
            "before": before_counts.get(key, 0),
            "after": after_counts.get(key, 0),
            "delta": after_counts.get(key, 0) - before_counts.get(key, 0),
        }
        for key in sorted(set(before_counts) | set(after_counts))
    }

    before_tables = _coerce_int_map(before.get("boundary_table_counts") or {})
    after_tables = _coerce_int_map(after.get("boundary_table_counts") or {})
    table_delta = {
        key: {
            "before": before_tables.get(key, 0),
            "after": after_tables.get(key, 0),
            "delta": after_tables.get(key, 0) - before_tables.get(key, 0),
        }
        for key in sorted(set(before_tables) | set(after_tables))
    }

    before_leases = before.get("blob_lease_state") or {}
    after_leases = after.get("blob_lease_state") or {}
    lease_delta = {
        "pending_lease_count": {
            "before": int(before_leases.get("pending_lease_count") or 0),
            "after": int(after_leases.get("pending_lease_count") or 0),
            "delta": int(after_leases.get("pending_lease_count") or 0)
            - int(before_leases.get("pending_lease_count") or 0),
        },
        "distinct_operations": {
            "before": int(before_leases.get("distinct_operations") or 0),
            "after": int(after_leases.get("distinct_operations") or 0),
            "delta": int(after_leases.get("distinct_operations") or 0)
            - int(before_leases.get("distinct_operations") or 0),
        },
    }

    before_gc = before.get("gc_state") or {}
    after_gc = after.get("gc_state") or {}
    gc_delta = {
        "high_water_generation": {
            "before": int(before_gc.get("high_water_generation") or 0),
            "after": int(after_gc.get("high_water_generation") or 0),
            "delta": int(after_gc.get("high_water_generation") or 0) - int(before_gc.get("high_water_generation") or 0),
        },
        "generation_count": {
            "before": int(before_gc.get("generation_count") or 0),
            "after": int(after_gc.get("generation_count") or 0),
            "delta": int(after_gc.get("generation_count") or 0) - int(before_gc.get("generation_count") or 0),
        },
    }

    before_fts = before.get("fts_trigger_state") or {}
    after_fts = after.get("fts_trigger_state") or {}
    fts_delta = {
        "all_present_before": bool(before_fts.get("all_present")),
        "all_present_after": bool(after_fts.get("all_present")),
        "missing_before": list(before_fts.get("missing") or []),
        "missing_after": list(after_fts.get("missing") or []),
        "regressed": sorted(set(after_fts.get("missing") or []) - set(before_fts.get("missing") or [])),
        "restored": sorted(set(before_fts.get("missing") or []) - set(after_fts.get("missing") or [])),
    }

    before_debt = before.get("convergence_debt") or {}
    after_debt = after.get("convergence_debt") or {}
    debt_delta = {
        "failed_count": {
            "before": int(before_debt.get("failed_count") or 0),
            "after": int(after_debt.get("failed_count") or 0),
            "delta": int(after_debt.get("failed_count") or 0) - int(before_debt.get("failed_count") or 0),
        }
    }

    before_timings = before.get("convergence_stage_timings") or {}
    after_timings = after.get("convergence_stage_timings") or {}
    timing_delta: dict[str, Any] = {
        "sample_size_before": int(before_timings.get("sample_size") or 0),
        "sample_size_after": int(after_timings.get("sample_size") or 0),
    }
    for field in ("parse_time_s", "convergence_time_s", "read_amplification"):
        b = before_timings.get(field) or {}
        a = after_timings.get(field) or {}
        timing_delta[field] = {
            "mean": {
                "before": float(b.get("mean") or 0.0),
                "after": float(a.get("mean") or 0.0),
                "delta": float(a.get("mean") or 0.0) - float(b.get("mean") or 0.0),
            },
            "max": {
                "before": float(b.get("max") or 0.0),
                "after": float(a.get("max") or 0.0),
                "delta": float(a.get("max") or 0.0) - float(b.get("max") or 0.0),
            },
        }

    return {
        "ok": True,
        "report_version": REPORT_VERSION,
        "before_captured_at": before.get("captured_at"),
        "after_captured_at": after.get("captured_at"),
        "before_db_path": before.get("db_path"),
        "after_db_path": after.get("db_path"),
        "attempt_counts": attempt_delta,
        "boundary_table_counts": table_delta,
        "blob_lease_state": lease_delta,
        "gc_state": gc_delta,
        "fts_trigger_state": fts_delta,
        "convergence_debt": debt_delta,
        "convergence_stage_timings": timing_delta,
    }


def _coerce_int_map(source: dict[str, Any]) -> dict[str, int]:
    """Pull out ``int``-valued entries from a mixed-shape dict.

    The probe stores integer counts alongside list/dict context (for example
    ``attempt_counts.overlapping_source_paths``).  Compare only diffs the
    scalar integer fields; non-int values are skipped so the diff stays
    arithmetic.
    """

    out: dict[str, int] = {}
    for key, value in source.items():
        if isinstance(value, bool):  # bool is int-subclass; reject explicitly
            continue
        if isinstance(value, int):
            out[key] = value
    return out


def _format_compare_human(diff: dict[str, Any]) -> str:
    if not diff.get("ok"):
        return f"compare failed: {diff.get('error')}"
    lines: list[str] = []
    lines.append("Daemon workload probe — before/after diff")
    lines.append(f"  before: {diff.get('before_captured_at')} ({diff.get('before_db_path')})")
    lines.append(f"  after:  {diff.get('after_captured_at')} ({diff.get('after_db_path')})")
    lines.append("")
    lines.append("Attempt counts:")
    for key, entry in diff["attempt_counts"].items():
        lines.append(f"  {key}: {entry['before']} -> {entry['after']} (Δ {entry['delta']:+d})")
    lines.append("")
    lines.append("Boundary table counts:")
    for key, entry in diff["boundary_table_counts"].items():
        lines.append(f"  {key}: {entry['before']} -> {entry['after']} (Δ {entry['delta']:+d})")
    leases = diff["blob_lease_state"]
    lines.append("")
    lines.append("Blob lease state:")
    for key, entry in leases.items():
        lines.append(f"  {key}: {entry['before']} -> {entry['after']} (Δ {entry['delta']:+d})")
    gc = diff["gc_state"]
    lines.append("")
    lines.append("GC state:")
    for key, entry in gc.items():
        lines.append(f"  {key}: {entry['before']} -> {entry['after']} (Δ {entry['delta']:+d})")
    fts = diff["fts_trigger_state"]
    lines.append("")
    lines.append("FTS trigger state:")
    lines.append(f"  all_present: {fts['all_present_before']} -> {fts['all_present_after']}")
    if fts["regressed"]:
        lines.append(f"  regressed (newly missing): {', '.join(fts['regressed'])}")
    if fts["restored"]:
        lines.append(f"  restored: {', '.join(fts['restored'])}")
    if not fts["regressed"] and not fts["restored"]:
        lines.append("  no trigger drift")
    debt = diff["convergence_debt"]["failed_count"]
    lines.append("")
    lines.append(f"Convergence debt failed_count: {debt['before']} -> {debt['after']} (Δ {debt['delta']:+d})")
    timings = diff["convergence_stage_timings"]
    lines.append("")
    lines.append(
        f"Convergence stage timings (completed-attempt sample {timings['sample_size_before']} -> "
        f"{timings['sample_size_after']}):"
    )
    for field in ("parse_time_s", "convergence_time_s", "read_amplification"):
        entry = timings[field]
        mean = entry["mean"]
        max_ = entry["max"]
        lines.append(f"  {field}.mean: {mean['before']:.3f} -> {mean['after']:.3f} (Δ {mean['delta']:+.3f})")
        lines.append(f"  {field}.max:  {max_['before']:.3f} -> {max_['after']:.3f} (Δ {max_['delta']:+.3f})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=None, help="Archive SQLite database path")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--limit", type=int, default=5, help="Recent attempt limit")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        type=Path,
        default=None,
        help="Compare two saved probe reports and report structured deltas",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.compare is not None:
        before_path, after_path = args.compare
        try:
            before_payload = json.loads(Path(before_path).read_text())
            after_payload = json.loads(Path(after_path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"failed to read compare inputs: {exc}")
            return 2
        diff = compare(before_payload, after_payload)
        if args.json:
            print(json.dumps(diff, indent=2, sort_keys=True))
        else:
            print(_format_compare_human(diff))
        return 0 if diff.get("ok") else 1

    payload = probe(args.db or default_db_path(), limit=max(1, args.limit))
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload.get("ok") else 1

    print(f"Daemon workload probe: {payload.get('db_path')}")
    print(f"  captured_at: {payload.get('captured_at')}")
    if not payload.get("ok"):
        print(f"  error: {payload.get('error')}")
        return 1
    counts = payload["attempt_counts"]
    print(f"  attempts: {counts['total']} total, {counts['running']} running, {counts['failed']} failed")
    print(f"  stale cursor writes: {counts.get('stale_cursor_writes', 0)}")
    overlaps = counts.get("overlapping_source_paths") or []
    print(f"  overlapping source paths: {len(overlaps)}")
    table_counts = payload.get("boundary_table_counts") or {}
    if table_counts:
        print("  boundary table counts:")
        for table, count in table_counts.items():
            shown = "missing" if count < 0 else str(count)
            print(f"    {table}: {shown}")
    leases = payload.get("blob_lease_state") or {}
    print(
        "  blob leases: "
        f"{leases.get('pending_lease_count', 0)} pending across "
        f"{leases.get('distinct_operations', 0)} operations"
    )
    gc = payload.get("gc_state") or {}
    print(f"  gc: high-water generation {gc.get('high_water_generation', 0)} (count {gc.get('generation_count', 0)})")
    fts = payload.get("fts_trigger_state") or {}
    if fts.get("all_present"):
        print(f"  fts triggers: all {len(fts.get('expected') or [])} present")
    else:
        missing = ", ".join(fts.get("missing") or [])
        print(f"  fts triggers: missing {missing or '(unknown)'}")
    recent = payload["recent_attempts"]
    if recent:
        latest = recent[0]
        print(
            "  latest: "
            f"{latest['status']} {latest['phase']} "
            f"{latest['succeeded_file_count']}/{latest['needed_file_count']} files, "
            f"read amp {latest['read_amplification']:.2f}x"
        )
    timings = payload.get("convergence_stage_timings") or {}
    print(
        "  completed-attempt timings (n="
        f"{timings.get('sample_size', 0)}): "
        f"parse mean {timings.get('parse_time_s', {}).get('mean', 0.0):.3f}s, "
        f"convergence mean {timings.get('convergence_time_s', {}).get('mean', 0.0):.3f}s"
    )
    debt = payload["convergence_debt"]
    print(f"  convergence debt: {debt['failed_count']} unresolved")
    churn = payload.get("source_path_churn") or []
    if churn:
        worst = churn[0]
        print(
            "  hottest source path: "
            f"{worst['raw_count']} raw rows, {worst['full_raw_count']} full, "
            f"{worst['append_raw_count']} append, {worst['orphan_raw_count']} orphan"
        )
    plan_hazards = [
        f"{name}: {hazard}" for name, info in payload["query_plans"].items() for hazard in info.get("hazards", [])
    ]
    print(f"  query plan hazards: {len(plan_hazards)}")
    for hazard in plan_hazards:
        print(f"    {hazard}")
    return 0


def _json_list(value: object) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if isinstance(item, str)]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
