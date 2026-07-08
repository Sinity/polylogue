"""_query_unit_time_expression / _time_predicate_clause must not epoch-pin
or silently exclude timeless rows (polylogue-z29t, sort_key_ms COALESCE
audit, .agent/reports/sort-key-ms-coalesce-audit-2026-07-08.md).

Before this fix, every COALESCE(...) chain backing the ``time`` field
predicate and ``sort=time`` ordering for the message/action/block/file
query units terminated in a literal epoch 0: a row with no reliable
timestamp collapsed to 1970, which always failed a ``>``/``>=`` time
filter (silent exclusion) and always passed a ``<``/``<=`` filter (silent
false-inclusion as "old"). These tests seed a genuinely timeless session
(no created_at_ms/updated_at_ms, so the generated sort_key_ms column is
NULL) with a timeless message/action, and prove the row is no longer
excluded by any time comparison operator and does not crash ordering.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.query.expression import parse_unit_source_expression
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

_TIMELESS_ORIGIN = "codex-session"


def _insert_timeless_session(conn: sqlite3.Connection, *, native_id: str) -> str:
    conn.execute(
        "INSERT INTO sessions (native_id, origin, content_hash) VALUES (?, ?, ?)",
        (native_id, _TIMELESS_ORIGIN, bytes(32)),
    )
    return f"{_TIMELESS_ORIGIN}:{native_id}"


def _insert_message(
    conn: sqlite3.Connection, *, session_id: str, position: int, occurred_at_ms: int | None = None
) -> None:
    conn.execute(
        "INSERT INTO messages (session_id, position, role, content_hash, occurred_at_ms) VALUES (?, ?, 'assistant', ?, ?)",
        (session_id, position, bytes(32), occurred_at_ms),
    )


def test_query_messages_time_filter_includes_timeless_row_for_every_operator(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path / "archive") as facade:
        conn = facade._conn
        timeless = _insert_timeless_session(conn, native_id="timeless-message")
        _insert_message(conn, session_id=timeless, position=0)
        timestamped = _insert_timeless_session(conn, native_id="timestamped-message")
        conn.execute("UPDATE sessions SET updated_at_ms = ? WHERE session_id = ?", (1_700_000_000_000, timestamped))
        _insert_message(conn, session_id=timestamped, position=0, occurred_at_ms=1_700_000_000_000)
        conn.commit()

        for op in (">", ">=", "<", "<="):
            source = parse_unit_source_expression(f"messages where time {op} 2026-01-01T00:00:00+00:00")
            assert source is not None
            rows = facade.query_messages(source.predicate, limit=100)
            session_ids = {row.session_id for row in rows}
            assert timeless in session_ids, f"timeless row missing for op={op!r}"


def test_query_messages_sort_time_does_not_crash_with_timeless_rows(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path / "archive") as facade:
        conn = facade._conn
        timeless = _insert_timeless_session(conn, native_id="timeless-sort")
        _insert_message(conn, session_id=timeless, position=0)
        timestamped = _insert_timeless_session(conn, native_id="timestamped-sort")
        conn.execute("UPDATE sessions SET updated_at_ms = ? WHERE session_id = ?", (1_700_000_000_000, timestamped))
        _insert_message(conn, session_id=timestamped, position=0, occurred_at_ms=1_700_000_000_000)
        conn.commit()

        source = parse_unit_source_expression("messages where role:assistant")
        assert source is not None
        desc_rows = facade.query_messages(source.predicate, sort="time", sort_direction="desc", limit=100)
        asc_rows = facade.query_messages(source.predicate, sort="time", sort_direction="asc", limit=100)

    # NULL (timeless) sorts last in DESC and first in ASC per SQLite's
    # default NULL ordering -- both rows must still be present either way.
    assert {row.session_id for row in desc_rows} == {timeless, timestamped}
    assert {row.session_id for row in asc_rows} == {timeless, timestamped}
    assert desc_rows[-1].session_id == timeless
    assert asc_rows[0].session_id == timeless


def test_get_session_tree_does_not_epoch_collide_timeless_sibling(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path / "archive") as facade:
        conn = facade._conn
        root = _insert_timeless_session(conn, native_id="tree-root")
        conn.execute(
            "UPDATE sessions SET updated_at_ms = ?, root_session_id = ? WHERE session_id = ?",
            (1_700_000_000_000, root, root),
        )
        timeless_sibling = _insert_timeless_session(conn, native_id="tree-timeless-child")
        conn.execute(
            "UPDATE sessions SET parent_session_id = ?, root_session_id = ? WHERE session_id = ?",
            (root, root, timeless_sibling),
        )
        conn.commit()

        tree = facade.get_session_tree(root)

    assert {envelope.session_id for envelope in tree} == {root, timeless_sibling}


def test_query_files_first_last_seen_ms_is_none_not_epoch_for_timeless_action(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path / "archive") as facade:
        conn = facade._conn
        timeless = _insert_timeless_session(conn, native_id="timeless-file")
        _insert_message(conn, session_id=timeless, position=0)
        message_id = f"{timeless}:0.0"
        conn.execute(
            """
            INSERT INTO blocks (message_id, session_id, position, block_type, tool_name, tool_id, tool_input)
            VALUES (?, ?, 0, 'tool_use', 'Edit', 'tool-1', '{"file_path": "src/example.py"}')
            """,
            (message_id, timeless),
        )
        conn.commit()

        source = parse_unit_source_expression("files where path:example.py")
        assert source is not None
        rows = facade.query_files(source.predicate, limit=100)

    assert len(rows) == 1
    assert rows[0].first_seen_ms is None
    assert rows[0].last_seen_ms is None
