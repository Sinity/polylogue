"""list_session_work_event_insights / list_session_phase_insights must not
silently exclude a timeless work-event/phase from a since/until window
(polylogue-2seq, sort_key_ms COALESCE audit,
.agent/reports/sort-key-ms-coalesce-audit-2026-07-08.md).

Both queries filtered on ``COALESCE(row.started_at_ms, session.sort_key_ms)
>= ?`` / ``<= ?`` with no NULL guard. When neither the row nor its session
carries a reliable timestamp, that COALESCE evaluates to NULL, and SQL's
NULL propagation means ``NULL >= ?``/``NULL <= ?`` is never true -- the row
silently vanished from any since/until-windowed query, indistinguishable
from genuinely falling outside the window.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _insert_timeless_session(conn: sqlite3.Connection, *, native_id: str, origin: str = "codex-session") -> str:
    conn.execute(
        "INSERT INTO sessions (native_id, origin, content_hash) VALUES (?, ?, ?)",
        (native_id, origin, bytes(32)),
    )
    return f"{origin}:{native_id}"


def test_work_event_since_until_window_includes_timeless_event(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path / "archive") as facade:
        conn = facade._conn
        session_id = _insert_timeless_session(conn, native_id="timeless-work-event")
        conn.execute(
            """
            INSERT INTO session_work_events (session_id, position, work_event_type, summary)
            VALUES (?, 0, 'implementation', 'built it')
            """,
            (session_id,),
        )
        conn.commit()

        since_rows = facade.list_session_work_event_insights(since_ms=1_700_000_000_000)
        until_rows = facade.list_session_work_event_insights(until_ms=1_700_000_000_000)

    assert {row.session_id for row in since_rows} == {session_id}
    assert {row.session_id for row in until_rows} == {session_id}


def test_phase_since_until_window_includes_timeless_phase(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path / "archive") as facade:
        conn = facade._conn
        session_id = _insert_timeless_session(conn, native_id="timeless-phase")
        conn.execute(
            "INSERT INTO session_phases (session_id, position) VALUES (?, 0)",
            (session_id,),
        )
        conn.commit()

        since_rows = facade.list_session_phase_insights(since_ms=1_700_000_000_000)
        until_rows = facade.list_session_phase_insights(until_ms=1_700_000_000_000)

    assert {row.session_id for row in since_rows} == {session_id}
    assert {row.session_id for row in until_rows} == {session_id}


def test_work_event_since_until_window_still_excludes_out_of_range_timestamped_event(tmp_path: Path) -> None:
    """Sanity check the fix does not disturb ordinary since/until exclusion."""
    with ArchiveStore(tmp_path / "archive") as facade:
        conn = facade._conn
        conn.execute(
            "INSERT INTO sessions (native_id, origin, content_hash, updated_at_ms) VALUES (?, ?, ?, ?)",
            ("old-work-event", "codex-session", bytes(32), 1_600_000_000_000),
        )
        session_id = "codex-session:old-work-event"
        conn.execute(
            """
            INSERT INTO session_work_events (session_id, position, work_event_type, summary)
            VALUES (?, 0, 'implementation', 'built it')
            """,
            (session_id,),
        )
        conn.commit()

        rows = facade.list_session_work_event_insights(since_ms=1_700_000_000_000)

    assert rows == []
