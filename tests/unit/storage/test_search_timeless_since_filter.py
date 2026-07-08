"""build_ranked_session_search_query / build_ranked_action_search_query /
search_attachment_identity_evidence_hits must not silently exclude a
timeless session from a --since filter (polylogue-s5mm, sort_key_ms
COALESCE audit, .agent/reports/sort-key-ms-coalesce-audit-2026-07-08.md).

All three builders filtered on a plain
``COALESCE(m.occurred_at_ms, s.sort_key_ms, ..., 0) >= ?`` (or without the
trailing ``0``, still NULL under ordinary propagation) with no NULL guard.
A session with no reliable timestamp anywhere in its fallback chain
silently vanished from any --since-filtered search, indistinguishable from
genuinely falling outside the window.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.search.query_builders import (
    build_ranked_action_search_query,
    build_ranked_session_search_query,
)
from polylogue.storage.search.runtime import search_messages_impl
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.queries.attachment_records import search_attachment_identity_evidence_hits
from polylogue.storage.sqlite.schema import SCHEMA_DDL


def _insert_timeless_session_with_text_block(
    conn: sqlite3.Connection, *, native_id: str, text: str, origin: str = "codex-session"
) -> str:
    conn.execute(
        "INSERT INTO sessions (native_id, origin, content_hash) VALUES (?, ?, ?)",
        (native_id, origin, bytes(32)),
    )
    session_id = f"{origin}:{native_id}"
    conn.execute(
        "INSERT INTO messages (session_id, position, role, content_hash) VALUES (?, 0, 'assistant', ?)",
        (session_id, bytes(32)),
    )
    message_id = f"{session_id}:0.0"
    conn.execute(
        "INSERT INTO blocks (message_id, session_id, position, block_type, text) VALUES (?, ?, 0, 'text', ?)",
        (message_id, session_id, text),
    )
    return session_id


def test_ranked_session_search_since_filter_includes_timeless_session(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path / "archive") as facade:
        conn = facade._conn
        session_id = _insert_timeless_session_with_text_block(
            conn, native_id="timeless-search", text="the quick fox jumps"
        )
        conn.commit()

        query_spec = build_ranked_session_search_query(query="quick fox", limit=100, since="2020-01-01")
        assert query_spec is not None
        rows = conn.execute(query_spec.sql, query_spec.params).fetchall()

    assert {row["session_id"] for row in rows} == {session_id}
    assert rows[0]["sort_key"] is None


def test_search_messages_impl_since_filter_includes_timeless_session(tmp_path: Path) -> None:
    """End-to-end through search_messages_impl -> _search_archive_blocks."""
    archive_root = tmp_path / "archive"
    with ArchiveStore(archive_root) as facade:
        conn = facade._conn
        session_id = _insert_timeless_session_with_text_block(
            conn, native_id="timeless-impl-search", text="the quick fox jumps"
        )
        conn.commit()
        db_path = facade.index_db_path

    result = search_messages_impl(
        query="quick fox",
        archive_root=archive_root,
        db_path=db_path,
        limit=100,
        source=None,
        since="2020-01-01",
    )

    assert {hit.session_id for hit in result.hits} == {session_id}
    assert result.hits[0].timestamp is None


def test_ranked_action_search_since_filter_includes_timeless_session(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path / "archive") as facade:
        conn = facade._conn
        conn.execute(
            "INSERT INTO sessions (native_id, origin, content_hash) VALUES (?, ?, ?)",
            ("timeless-action-search", "codex-session", bytes(32)),
        )
        session_id = "codex-session:timeless-action-search"
        conn.execute(
            "INSERT INTO messages (session_id, position, role, content_hash) VALUES (?, 0, 'assistant', ?)",
            (session_id, bytes(32)),
        )
        message_id = f"{session_id}:0.0"
        conn.execute(
            "INSERT INTO blocks (message_id, session_id, position, block_type, tool_name, tool_id, text) "
            "VALUES (?, ?, 0, 'tool_use', 'Bash', 'tool-1', 'run pytest suite')",
            (message_id, session_id),
        )
        conn.commit()

        query_spec = build_ranked_action_search_query(query="pytest suite", limit=100, since="2020-01-01")
        assert query_spec is not None
        rows = conn.execute(query_spec.sql, query_spec.params).fetchall()

    assert {row["session_id"] for row in rows} == {session_id}


def test_ranked_session_search_since_filter_still_excludes_out_of_range_timestamped_session(tmp_path: Path) -> None:
    """Sanity check the fix does not disturb ordinary since exclusion."""
    with ArchiveStore(tmp_path / "archive") as facade:
        conn = facade._conn
        conn.execute(
            "INSERT INTO sessions (native_id, origin, content_hash, updated_at_ms) VALUES (?, ?, ?, ?)",
            ("old-search", "codex-session", bytes(32), 1_500_000_000_000),
        )
        session_id = "codex-session:old-search"
        conn.execute(
            "INSERT INTO messages (session_id, position, role, content_hash) VALUES (?, 0, 'assistant', ?)",
            (session_id, bytes(32)),
        )
        message_id = f"{session_id}:0.0"
        conn.execute(
            "INSERT INTO blocks (message_id, session_id, position, block_type, text) VALUES (?, ?, 0, 'text', ?)",
            (message_id, session_id, "the quick fox jumps"),
        )
        conn.commit()

        query_spec = build_ranked_session_search_query(query="quick fox", limit=100, since="2020-01-01")
        assert query_spec is not None
        rows = conn.execute(query_spec.sql, query_spec.params).fetchall()

    assert rows == []


@pytest.mark.asyncio
async def test_attachment_identity_search_since_filter_includes_timeless_session(tmp_path: Path) -> None:
    import aiosqlite

    db_path = tmp_path / "ident.db"
    bootstrap = sqlite3.connect(db_path)
    try:
        bootstrap.executescript(SCHEMA_DDL)
        bootstrap.execute(
            "INSERT INTO sessions (native_id, origin, content_hash) VALUES (?, ?, zeroblob(32))",
            ("timeless-attachment", "gemini-cli-session"),
        )
        bootstrap.execute(
            """INSERT INTO attachments (
                attachment_id, display_name, media_type, byte_count, blob_hash, ref_count
            ) VALUES (?, ?, ?, ?, zeroblob(32), ?)""",
            ("att-timeless", "drive doc", None, 0, 1),
        )
        bootstrap.execute(
            """
            INSERT INTO messages (session_id, native_id, position, role, message_type, content_hash)
            VALUES ('gemini-cli-session:timeless-attachment', 'msg-1', 0, 'user', 'message', zeroblob(32))
            """
        )
        bootstrap.execute(
            """INSERT INTO attachment_refs (
                attachment_id, session_id, message_id, position, upload_origin
            ) VALUES (?, ?, ?, ?, ?)""",
            (
                "att-timeless",
                "gemini-cli-session:timeless-attachment",
                "gemini-cli-session:timeless-attachment:msg-1",
                0,
                "drive",
            ),
        )
        bootstrap.execute(
            """INSERT INTO attachment_native_ids (ref_id, id_kind, native_id)
            VALUES ('gemini-cli-session:timeless-attachment:msg-1:attachment:0', 'attachment', 'prov-att-timeless')"""
        )
        bootstrap.commit()
    finally:
        bootstrap.close()

    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        hits = await search_attachment_identity_evidence_hits(
            conn, query="prov-att-timeless", limit=10, since="2020-01-01"
        )

    assert len(hits) == 1
    assert hits[0].session_id == "gemini-cli-session:timeless-attachment"
