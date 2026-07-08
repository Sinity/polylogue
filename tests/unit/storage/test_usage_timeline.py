"""usage_timeline must not silently drop timeless-session usage/cost (polylogue-rvtu).

Regression coverage for the sort_key_ms COALESCE audit
(.agent/reports/sort-key-ms-coalesce-audit-2026-07-08.md): the base
filters in ``list_usage_timeline_insights`` used to require
``COALESCE(occurred_at_ms, sort_key_ms, 0) > 0`` (event scan) /
``sort_key_ms > 0`` (cost scan), which unconditionally excluded any
session with neither a reliable event timestamp nor a session
timestamp -- not just under a since/until window. Real usage from such a
session vanished from every monthly bucket forever with no signal.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _insert_timeless_session(conn: sqlite3.Connection, *, native_id: str, origin: str = "codex-session") -> str:
    conn.execute(
        "INSERT INTO sessions (native_id, origin, content_hash) VALUES (?, ?, ?)",
        (native_id, origin, bytes(32)),
    )
    return f"{origin}:{native_id}"


def test_usage_timeline_includes_timeless_session_event_in_unknown_bucket(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path / "archive") as facade:
        conn = facade._conn
        session_id = _insert_timeless_session(conn, native_id="timeless-usage-event")
        conn.execute(
            """
            INSERT INTO session_provider_usage_events (
                session_id, position, provider_event_type, model_name,
                last_input_tokens, last_output_tokens, last_total_tokens
            ) VALUES (?, 0, 'token_count', 'gpt-5', 10, 5, 15)
            """,
            (session_id,),
        )
        conn.commit()

        rows = facade.list_usage_timeline_insights()

    unknown = [row for row in rows if row.bucket == "unknown"]
    assert len(unknown) == 1
    row = unknown[0]
    assert row.event_count == 1
    assert row.usage.input_tokens == 10
    assert row.usage.output_tokens == 5


def test_usage_timeline_includes_timeless_session_cost_in_unknown_bucket(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path / "archive") as facade:
        conn = facade._conn
        session_id = _insert_timeless_session(conn, native_id="timeless-usage-cost")
        conn.execute(
            "INSERT INTO session_model_usage (session_id, model_name, input_tokens, output_tokens, cost_usd) "
            "VALUES (?, 'gpt-5', 10, 5, 0.02)",
            (session_id,),
        )
        conn.commit()

        rows = facade.list_usage_timeline_insights()

    unknown = [row for row in rows if row.bucket == "unknown"]
    assert len(unknown) == 1
    assert unknown[0].stored_cost_usd == pytest.approx(0.02)


def test_usage_timeline_still_buckets_timestamped_sessions_normally(tmp_path: Path) -> None:
    """Sanity check the fix does not disturb ordinary timestamped rows."""
    with ArchiveStore(tmp_path / "archive") as facade:
        conn = facade._conn
        conn.execute(
            "INSERT INTO sessions (native_id, origin, content_hash, updated_at_ms) VALUES (?, ?, ?, ?)",
            ("timestamped-usage", "codex-session", bytes(32), 1_700_000_000_000),
        )
        session_id = "codex-session:timestamped-usage"
        conn.execute(
            """
            INSERT INTO session_provider_usage_events (
                session_id, position, provider_event_type, model_name, last_input_tokens
            ) VALUES (?, 0, 'token_count', 'gpt-5', 7)
            """,
            (session_id,),
        )
        conn.commit()

        rows = facade.list_usage_timeline_insights()

    assert [row.bucket for row in rows] == ["2023-11"]
    assert rows[0].usage.input_tokens == 7
