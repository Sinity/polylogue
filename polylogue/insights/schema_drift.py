"""Windowed format-drift status derivation (light import path).

This lives outside ``polylogue.cli.commands.status`` deliberately: the root
CLI callback's drift marker (``click_app._emit_schema_drift_marker``) and the
daemon health check both need only this derivation, while ``status`` pulls the
readiness/repair stack (~1s of imports). Keeping this module stdlib-light means
every ``polylogue`` invocation that never touches the status surface does not
pay that import tax just to print the drift sentinel.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

# polylogue-da1: format-drift sentinel window. Windowed since a date, not
# lifetime, so an archive with years of clean history does not permanently
# dilute a recent provider-shape regression signal.
SCHEMA_DRIFT_WINDOW_MS = 30 * 24 * 60 * 60 * 1000
SCHEMA_DRIFT_RISKY_WARN_RATE = 0.05
SCHEMA_DRIFT_RISKY_ERROR_RATE = 0.20
SCHEMA_DRIFT_MIN_SAMPLE = 5


def _drift_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table') AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def schema_drift_status(active_root: Path, *, now_ms: int, window_ms: int = SCHEMA_DRIFT_WINDOW_MS) -> dict[str, Any]:
    """Derive windowed format-drift rates per origin from ops.db (read-only).

    Reads ``schema_drift_samples`` (populated at ingest time -- see
    ``polylogue.schemas.drift_sentinel``) and returns one entry per origin
    with a sample in the window. Returns ``available: False`` when the ops
    tier or table is absent, matching ``_ops_workload_status``'s contract
    so a synthetic/mid-bootstrap archive degrades quietly.
    """
    ops_db = active_root / "ops.db"
    if not ops_db.exists():
        return {"available": False, "reason": "missing_ops_tier"}
    try:
        conn = sqlite3.connect(f"file:{ops_db}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return {"available": False, "reason": str(exc)}
    try:
        if not _drift_table_exists(conn, "schema_drift_samples"):
            return {"available": False, "reason": "missing_schema_drift_samples"}
        from polylogue.storage.sqlite.archive_tiers.ops_write import summarize_schema_drift_since

        since_ms = now_ms - window_ms
        summaries = summarize_schema_drift_since(conn, since_ms=since_ms)
    except sqlite3.Error as exc:
        return {"available": False, "reason": str(exc)}
    finally:
        conn.close()

    origins = []
    for summary in summaries:
        risky_rate = summary.risky_rate
        if summary.total < SCHEMA_DRIFT_MIN_SAMPLE:
            severity = "ok"
        elif risky_rate >= SCHEMA_DRIFT_RISKY_ERROR_RATE:
            severity = "error"
        elif risky_rate >= SCHEMA_DRIFT_RISKY_WARN_RATE:
            severity = "warning"
        else:
            severity = "ok"
        origins.append(
            {
                "origin": summary.origin,
                "total": summary.total,
                "risky": summary.risky,
                "benign": summary.benign,
                "risky_rate": round(risky_rate, 4),
                "severity": severity,
                "example_native_ids": list(summary.example_native_ids),
            }
        )
    return {
        "available": True,
        "since_ms": since_ms,
        "window_days": window_ms // (24 * 60 * 60 * 1000),
        "origins": origins,
    }
