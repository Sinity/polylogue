"""Comma-bearing session ids must not corrupt distinct-session aggregates.

``sessions.session_id`` is ``origin || ':' || native_id`` and ``native_id``
comes from provider JSON with no character restriction, so a provider
conversation id may legitimately contain a comma. The cost-rollup and
usage-timeline aggregates used to emit ``GROUP_CONCAT(DISTINCT x)`` and split
the result on ``","`` in Python, which shattered such an id into fragments
and inflated the distinct-session count (polylogue-3sic0).

Anti-vacuity for every test here: restore ``GROUP_CONCAT(DISTINCT ...)`` plus
``str(row["session_ids"]).split(",")`` in
``polylogue/storage/sqlite/archive_tiers/archive.py`` and each observed count
below grows past the number of sessions actually inserted.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

# One provider conversation id carrying two commas: splitting on "," turns
# this single session into three fragments.
COMMA_NATIVE_ID = "conv,with,commas"
ORIGIN = "codex-session"
COMMA_SESSION_ID = f"{ORIGIN}:{COMMA_NATIVE_ID}"


def _insert_session(conn: sqlite3.Connection, native_id: str, *, sort_key_ms: int | None) -> str:
    conn.execute(
        "INSERT INTO sessions (native_id, origin, content_hash, updated_at_ms) VALUES (?, ?, ?, ?)",
        (native_id, ORIGIN, bytes(32), sort_key_ms),
    )
    return f"{ORIGIN}:{native_id}"


def test_cost_rollup_counts_one_comma_bearing_session_once(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path / "archive") as facade:
        conn = facade._conn
        session_id = _insert_session(conn, COMMA_NATIVE_ID, sort_key_ms=1_700_000_000_000)
        assert session_id == COMMA_SESSION_ID
        conn.execute(
            "INSERT INTO session_model_usage (session_id, model_name, input_tokens, output_tokens, catalog_cost_usd) "
            "VALUES (?, 'gpt-5', 10, 5, 0.02)",
            (session_id,),
        )
        conn.commit()

        rows = facade.list_cost_rollup_insights()

    priced = [row for row in rows if row.model_name == "gpt-5"]
    assert len(priced) == 1
    assert priced[0].session_count == 1


def test_cost_rollup_counts_a_usage_free_comma_bearing_session_once(tmp_path: Path) -> None:
    """The no-usage branch aggregates ``sessions`` directly and had the same split."""

    with ArchiveStore(tmp_path / "archive") as facade:
        conn = facade._conn
        conn.execute(
            "INSERT INTO sessions (native_id, origin, content_hash, updated_at_ms, reported_cost_usd) "
            "VALUES (?, ?, ?, ?, ?)",
            (COMMA_NATIVE_ID, ORIGIN, bytes(32), 1_700_000_000_000, 0.5),
        )
        conn.commit()

        rows = facade.list_cost_rollup_insights()

    assert [row.session_count for row in rows] == [1]


def test_usage_timeline_event_scan_counts_a_comma_bearing_session_once(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path / "archive") as facade:
        conn = facade._conn
        session_id = _insert_session(conn, COMMA_NATIVE_ID, sort_key_ms=1_700_000_000_000)
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

    assert len(rows) == 1
    assert rows[0].event_count == 1
    assert rows[0].session_count == 1


def test_usage_timeline_cost_scan_counts_a_comma_bearing_session_once(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path / "archive") as facade:
        conn = facade._conn
        session_id = _insert_session(conn, COMMA_NATIVE_ID, sort_key_ms=1_700_000_000_000)
        conn.execute(
            "INSERT INTO session_model_usage (session_id, model_name, input_tokens, output_tokens, catalog_cost_usd) "
            "VALUES (?, 'gpt-5', 10, 5, 0.02)",
            (session_id,),
        )
        conn.commit()

        rows = facade.list_usage_timeline_insights()

    assert len(rows) == 1
    assert rows[0].session_count == 1


def test_distinct_sessions_are_still_counted_separately(tmp_path: Path) -> None:
    """The fix must not collapse genuinely distinct sessions into one."""

    with ArchiveStore(tmp_path / "archive") as facade:
        conn = facade._conn
        for native_id in (COMMA_NATIVE_ID, "plain-conv"):
            session_id = _insert_session(conn, native_id, sort_key_ms=1_700_000_000_000)
            conn.execute(
                "INSERT INTO session_model_usage "
                "(session_id, model_name, input_tokens, output_tokens, catalog_cost_usd) "
                "VALUES (?, 'gpt-5', 10, 5, 0.02)",
                (session_id,),
            )
        conn.commit()

        timeline = facade.list_usage_timeline_insights()
        rollup = facade.list_cost_rollup_insights()

    assert len(timeline) == 1
    assert timeline[0].session_count == 2
    assert [row.session_count for row in rollup] == [2]
