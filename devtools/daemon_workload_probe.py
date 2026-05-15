"""Read-only daemon workload probe for live ingest and convergence hot paths."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from polylogue.paths import db_path as default_db_path
from polylogue.storage.sqlite.connection_profile import open_readonly_connection


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _scalar_int(conn: sqlite3.Connection, sql: str, params: tuple[object, ...] = ()) -> int:
    try:
        row = conn.execute(sql, params).fetchone()
    except sqlite3.Error:
        return 0
    return int(row[0] or 0) if row is not None else 0


def _recent_attempts(conn: sqlite3.Connection, *, limit: int) -> list[dict[str, Any]]:
    if not _table_exists(conn, "live_ingest_attempt"):
        return []
    rows = conn.execute(
        """
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
            convergence_time_s
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
            }
        )
    return attempts


def _convergence_debt(conn: sqlite3.Connection) -> dict[str, Any]:
    if not _table_exists(conn, "live_convergence_debt"):
        return {"failed_count": 0, "retry_due_count": 0, "by_stage": []}
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
        return {"ok": False, "db_path": str(db), "error": "database does not exist"}
    conn = open_readonly_connection(db)
    try:
        return {
            "ok": True,
            "db_path": str(db),
            "attempt_counts": {
                "total": _scalar_int(conn, "SELECT COUNT(*) FROM live_ingest_attempt")
                if _table_exists(conn, "live_ingest_attempt")
                else 0,
                "running": _scalar_int(conn, "SELECT COUNT(*) FROM live_ingest_attempt WHERE status = 'running'")
                if _table_exists(conn, "live_ingest_attempt")
                else 0,
                "failed": _scalar_int(conn, "SELECT COUNT(*) FROM live_ingest_attempt WHERE status = 'failed'")
                if _table_exists(conn, "live_ingest_attempt")
                else 0,
            },
            "recent_attempts": _recent_attempts(conn, limit=limit),
            "convergence_debt": _convergence_debt(conn),
            "query_plans": _query_plans(conn),
        }
    finally:
        conn.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=None, help="Archive SQLite database path")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--limit", type=int, default=5, help="Recent attempt limit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = probe(args.db or default_db_path(), limit=max(1, args.limit))
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload.get("ok") else 1

    print(f"Daemon workload probe: {payload.get('db_path')}")
    if not payload.get("ok"):
        print(f"  error: {payload.get('error')}")
        return 1
    counts = payload["attempt_counts"]
    print(f"  attempts: {counts['total']} total, {counts['running']} running, {counts['failed']} failed")
    recent = payload["recent_attempts"]
    if recent:
        latest = recent[0]
        print(
            "  latest: "
            f"{latest['status']} {latest['phase']} "
            f"{latest['succeeded_file_count']}/{latest['needed_file_count']} files, "
            f"read amp {latest['read_amplification']:.2f}x"
        )
    debt = payload["convergence_debt"]
    print(f"  convergence debt: {debt['failed_count']} unresolved")
    plan_hazards = [
        f"{name}: {hazard}" for name, info in payload["query_plans"].items() for hazard in info.get("hazards", [])
    ]
    print(f"  query plan hazards: {len(plan_hazards)}")
    for hazard in plan_hazards:
        print(f"    {hazard}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
