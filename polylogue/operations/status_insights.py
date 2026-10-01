"""Standalone insight-status acquisition for daemon diagnostics.

Pinned status keeps its supplied connections; this path owner acquires and
closes the standalone reader and projects failures as unavailable evidence.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.core.status_error_privacy import redact_status_error
from polylogue.logging import WARNING, emit
from polylogue.operations.daemon_status import insight_freshness_from_connection
from polylogue.storage.sqlite.connection_profile import open_readonly_connection


def insight_freshness_for_path(dbf: Path) -> dict[str, object]:
    """Inspect profile outputs, retaining unavailable SQLite evidence explicitly."""
    from polylogue.core.evidence import Measured, Unavailable
    from polylogue.storage.tier_access import capture_sqlite_read

    if not dbf.exists():
        return {
            "checked": False,
            "reason": "index tier is unavailable",
            "sessions_with_profiles": None,
            "total_sessions": None,
        }

    def read() -> dict[str, object]:
        try:
            conn = open_readonly_connection(dbf, validate_schema=False)
            try:
                return insight_freshness_from_connection(conn)
            finally:
                conn.close()
        except sqlite3.Error as exc:
            emit(
                "daemon.status.query_failed",
                level=WARNING,
                outcome="degraded",
                reason="insight_freshness_unreadable",
                path=dbf,
                error_type=type(exc).__name__,
                error_detail=redact_status_error(str(exc)),
            )
            raise

    evidence = capture_sqlite_read(read)
    if isinstance(evidence, Measured):
        return evidence.value
    if not isinstance(evidence, Unavailable):
        raise AssertionError("insight freshness read produced unsupported evidence")
    reason = redact_status_error(evidence.detail or evidence.reason)
    return {
        "checked": False,
        "reason": reason,
        "sessions_with_profiles": None,
        "total_sessions": None,
    }
