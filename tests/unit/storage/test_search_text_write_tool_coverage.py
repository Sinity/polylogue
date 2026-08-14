"""Coverage-boundary tests for polylogue-013x.

``blocks.search_text`` (the generated column FTS5 indexes) concatenates only
a fixed subset of block fields -- it deliberately excludes ``tool_input``
keys that carry file bodies an agent authored or edited (``Write``'s
``$.content``, ``Edit``'s ``$.old_string``/``$.new_string``). That boundary
is now documented in ``docs/search.md`` § "Searchable Content Coverage".

This module proves the user-visible FTS boundary:

1. A distinctive string that only appears inside a ``Write``/``Edit`` tool
   body is genuinely NOT reachable through FTS (proves the gap is real, not
   just documented).
2. The documented raw-SQL workaround (``json_extract`` + ``LIKE`` over
   ``tool_input``) DOES find it, so the documented escape hatch actually
   works.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from polylogue.storage.search.query_support import escape_fts5_query
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_WRITE_TOKEN = "quokka-manifesto-9f3c1a"
_EDIT_OLD_TOKEN = "legacy-walrus-descriptor-77b2"
_EDIT_NEW_TOKEN = "renamed-walrus-descriptor-4e91"


def _make_archive(tmp_path: Path) -> Path:
    db_path = tmp_path / "index.db"
    initialize_archive_database(db_path, ArchiveTier.INDEX)
    return db_path


def _insert_session_and_message(conn: sqlite3.Connection, *, native_id: str, message_native_id: str) -> tuple[str, str]:
    session_id = f"unknown-export:{native_id}"
    message_id = f"{session_id}:{message_native_id}"
    conn.execute(
        "INSERT INTO sessions (native_id, origin, content_hash) VALUES (?, ?, ?)",
        (native_id, "unknown-export", bytes(32)),
    )
    conn.execute(
        """
        INSERT INTO messages (
            session_id, native_id, position, role, message_type, content_hash
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (session_id, message_native_id, 0, "assistant", "message", bytes(32)),
    )
    return session_id, message_id


def _insert_tool_block(
    conn: sqlite3.Connection,
    *,
    message_id: str,
    session_id: str,
    position: int,
    tool_name: str,
    tool_id: str,
    tool_input: dict[str, str],
) -> None:
    conn.execute(
        """
        INSERT INTO blocks (
            message_id, session_id, position, block_type, tool_name, tool_id, tool_input
        ) VALUES (?, ?, ?, 'tool_use', ?, ?, ?)
        """,
        (message_id, session_id, position, tool_name, tool_id, json.dumps(tool_input)),
    )


def test_write_tool_body_not_reachable_via_fts(tmp_path: Path) -> None:
    """A distinctive string only present in a Write tool's file body has zero FTS hits."""
    db_path = _make_archive(tmp_path)
    with sqlite3.connect(db_path) as conn:
        session_id, message_id = _insert_session_and_message(
            conn, native_id="conv-write-gap", message_native_id="msg-write-gap"
        )
        _insert_tool_block(
            conn,
            message_id=message_id,
            session_id=session_id,
            position=0,
            tool_name="Write",
            tool_id="tool-write-gap",
            tool_input={"file_path": "/tmp/plan.md", "content": f"# Plan\n\n{_WRITE_TOKEN} is the secret marker."},
        )

        search_text = conn.execute("SELECT search_text FROM blocks WHERE tool_id = ?", ("tool-write-gap",)).fetchone()[
            0
        ]
        assert _WRITE_TOKEN not in search_text, "Write body content leaked into search_text unexpectedly"

        fts_hits = conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?",
            (escape_fts5_query(_WRITE_TOKEN),),
        ).fetchone()[0]
        assert fts_hits == 0, "documented coverage gap regressed: Write body became FTS-searchable"


def test_edit_tool_body_not_reachable_via_fts(tmp_path: Path) -> None:
    """Edit's old_string/new_string are excluded from search_text the same way Write's content is."""
    db_path = _make_archive(tmp_path)
    with sqlite3.connect(db_path) as conn:
        session_id, message_id = _insert_session_and_message(
            conn, native_id="conv-edit-gap", message_native_id="msg-edit-gap"
        )
        _insert_tool_block(
            conn,
            message_id=message_id,
            session_id=session_id,
            position=0,
            tool_name="Edit",
            tool_id="tool-edit-gap",
            tool_input={
                "file_path": "/tmp/module.py",
                "old_string": _EDIT_OLD_TOKEN,
                "new_string": _EDIT_NEW_TOKEN,
            },
        )

        for token in (_EDIT_OLD_TOKEN, _EDIT_NEW_TOKEN):
            fts_hits = conn.execute(
                "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?",
                (escape_fts5_query(token),),
            ).fetchone()[0]
            assert fts_hits == 0, f"documented coverage gap regressed: Edit body token {token!r} became searchable"


def test_documented_workaround_finds_write_and_edit_tool_bodies(tmp_path: Path) -> None:
    """The raw-SQL workaround documented in docs/search.md actually finds excluded tool bodies."""
    db_path = _make_archive(tmp_path)
    with sqlite3.connect(db_path) as conn:
        write_session_id, write_message_id = _insert_session_and_message(
            conn, native_id="conv-write-workaround", message_native_id="msg-write-workaround"
        )
        _insert_tool_block(
            conn,
            message_id=write_message_id,
            session_id=write_session_id,
            position=0,
            tool_name="Write",
            tool_id="tool-write-workaround",
            tool_input={"file_path": "/tmp/plan.md", "content": f"body containing {_WRITE_TOKEN}"},
        )

        edit_session_id, edit_message_id = _insert_session_and_message(
            conn, native_id="conv-edit-workaround", message_native_id="msg-edit-workaround"
        )
        _insert_tool_block(
            conn,
            message_id=edit_message_id,
            session_id=edit_session_id,
            position=0,
            tool_name="Edit",
            tool_id="tool-edit-workaround",
            tool_input={
                "file_path": "/tmp/module.py",
                "old_string": _EDIT_OLD_TOKEN,
                "new_string": _EDIT_NEW_TOKEN,
            },
        )

        # The exact query documented in docs/search.md's "Searchable Content
        # Coverage" workaround.
        workaround_sql = """
            SELECT block_id, session_id, tool_name,
                   json_extract(tool_input, '$.file_path') AS file_path
            FROM blocks
            WHERE tool_name IN ('Write', 'Edit')
              AND (
                json_extract(tool_input, '$.content') LIKE ?
                OR json_extract(tool_input, '$.old_string') LIKE ?
                OR json_extract(tool_input, '$.new_string') LIKE ?
              )
        """

        write_hits = conn.execute(
            workaround_sql, (f"%{_WRITE_TOKEN}%", f"%{_WRITE_TOKEN}%", f"%{_WRITE_TOKEN}%")
        ).fetchall()
        assert [row[0] for row in write_hits] == [write_message_id + ":0"]

        old_hits = conn.execute(
            workaround_sql, (f"%{_EDIT_OLD_TOKEN}%", f"%{_EDIT_OLD_TOKEN}%", f"%{_EDIT_OLD_TOKEN}%")
        ).fetchall()
        assert [row[0] for row in old_hits] == [edit_message_id + ":0"]

        new_hits = conn.execute(
            workaround_sql, (f"%{_EDIT_NEW_TOKEN}%", f"%{_EDIT_NEW_TOKEN}%", f"%{_EDIT_NEW_TOKEN}%")
        ).fetchall()
        assert [row[0] for row in new_hits] == [edit_message_id + ":0"]
