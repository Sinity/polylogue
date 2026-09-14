"""Windowed format-drift status derivation (light import path).

This lives outside ``polylogue.cli.commands.status`` deliberately: the status
surface and daemon health check both need this derivation, while ``status``
pulls the readiness/repair stack (~1s of imports). Keeping this module
stdlib-light keeps the daemon health path independent from the CLI surface.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.ops_write import SchemaDriftOriginSummary

# polylogue-da1: format-drift sentinel window. Windowed since a date, not
# lifetime, so an archive with years of clean history does not permanently
# dilute a recent provider-shape regression signal.
SCHEMA_DRIFT_WINDOW_MS = 30 * 24 * 60 * 60 * 1000
SCHEMA_DRIFT_RISKY_WARN_RATE = 0.05
SCHEMA_DRIFT_RISKY_ERROR_RATE = 0.20
SCHEMA_DRIFT_MIN_SAMPLE = 5


def _schema_drift_status_from_summaries(
    summaries: Iterable[SchemaDriftOriginSummary],
    *,
    since_ms: int,
    window_ms: int,
) -> dict[str, Any]:
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


def schema_drift_status_from_connection(
    conn: sqlite3.Connection | None,
    *,
    now_ms: int,
    window_ms: int = SCHEMA_DRIFT_WINDOW_MS,
    schema: str = "ops_tier",
) -> dict[str, Any]:
    """Project format drift from a supplied, already-pinned ops reader.

    Shares :func:`schema_drift_status`'s failure contract: a degraded-but-
    readable ops tier (SQLITE_BUSY on the pinned reader, a malformed page, a
    column the pinned snapshot predates) resolves to
    ``{"available": False, "reason": ...}`` rather than raising. The only
    caller -- ``_schema_drift_status`` in
    :mod:`polylogue.operations.daemon_status` -- invokes this unguarded, so a
    bare driver error here fails the whole status operation instead of
    reporting one unavailable component.

    ``ValueError`` for an unsupported schema stays a raise: that is a caller
    bug, not a tier degradation.
    """

    if conn is None:
        return {"available": False, "reason": "missing_ops_tier"}
    if schema not in {"main", "ops_tier"}:
        raise ValueError(f"unsupported schema-drift reader schema: {schema!r}")
    try:
        return _schema_drift_status_from_connection(conn, now_ms=now_ms, window_ms=window_ms, schema=schema)
    except (OSError, sqlite3.Error) as exc:
        return {"available": False, "reason": str(exc)}


def _schema_drift_status_from_connection(
    conn: sqlite3.Connection,
    *,
    now_ms: int,
    window_ms: int,
    schema: str,
) -> dict[str, Any]:
    """Unguarded projection body; see the wrapper for the failure contract."""

    if (
        conn.execute(
            f"SELECT 1 FROM {schema}.sqlite_schema WHERE type = 'table' AND name = 'schema_drift_samples'"
        ).fetchone()
        is None
    ):
        return {"available": False, "reason": "missing_schema_drift_samples"}
    from polylogue.storage.sqlite.archive_tiers.ops_write import summarize_schema_drift_since

    since_ms = now_ms - window_ms
    return _schema_drift_status_from_summaries(
        summarize_schema_drift_since(conn, since_ms=since_ms, schema=schema),
        since_ms=since_ms,
        window_ms=window_ms,
    )


def schema_drift_status(active_root: Path, *, now_ms: int, window_ms: int = SCHEMA_DRIFT_WINDOW_MS) -> dict[str, Any]:
    """Derive windowed format-drift rates per origin from ops.db (read-only).

    Reads ``schema_drift_samples`` (populated at ingest time -- see
    ``polylogue.schemas.drift_sentinel``) and returns one entry per origin
    with a sample in the window. Returns ``available: False`` when the ops
    tier or table is absent, matching ``_ops_workload_status``'s contract
    so a synthetic/mid-bootstrap archive degrades quietly.

    This owns tier resolution only; the projection itself is
    :func:`schema_drift_status_from_connection`, which also owns the shared
    read-failure contract. The two used to carry duplicate bodies, and the
    pinned one was landed without this one's ``sqlite3.Error`` handling --
    delegating is what keeps them from diverging again.
    """
    ops_db = active_root / "ops.db"
    if not ops_db.exists():
        return {"available": False, "reason": "missing_ops_tier"}
    try:
        conn = sqlite3.connect(f"file:{ops_db}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return {"available": False, "reason": str(exc)}
    try:
        return schema_drift_status_from_connection(conn, now_ms=now_ms, window_ms=window_ms, schema="main")
    finally:
        conn.close()
