"""Coverage-boundary tests for polylogue-013x.

``blocks.search_text`` (the generated column FTS5 indexes) concatenates only
a fixed subset of block fields -- it deliberately excludes ``tool_input``
keys that carry file bodies an agent authored or edited (``Write``'s
``$.content``, ``Edit``'s ``$.old_string``/``$.new_string``). That boundary
is now documented in ``docs/search.md`` § "Searchable Content Coverage".

This module pins three things:

1. The live ``search_text`` DDL expression matches what the docs claim is
   indexed (drift check) -- if a future change adds/removes a
   ``json_extract`` path from the generated column without updating the
   docs, this test fails.
2. A distinctive string that only appears inside a ``Write``/``Edit`` tool
   body is genuinely NOT reachable through FTS (proves the gap is real, not
   just documented).
3. The documented raw-SQL workaround (``json_extract`` + ``LIKE`` over
   ``tool_input``) DOES find it, so the documented escape hatch actually
   works.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from polylogue.storage.search.query_support import escape_fts5_query
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_REPO_ROOT = Path(__file__).parents[3]
_SEARCH_DOC = _REPO_ROOT / "docs" / "search.md"

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


def test_search_text_ddl_matches_documented_coverage_matrix() -> None:
    """Extract the live search_text generated-column expression and pin its shape.

    This is the drift check required by polylogue-013x's acceptance criteria:
    docs/search.md's coverage matrix must match the live DDL. If someone adds
    a new json_extract path to search_text (e.g. `$.content`) without also
    updating the docs, the negative assertions below fail; if someone removes
    one of the four currently-indexed paths, the positive assertions fail.
    """
    ddl_source = Path(
        Path(__file__).parents[3] / "polylogue" / "storage" / "sqlite" / "archive_tiers" / "index.py"
    ).read_text(encoding="utf-8")

    match = re.search(
        r"search_text\s+TEXT GENERATED ALWAYS AS \((?P<expr>.*?)\)\s*VIRTUAL,",
        ddl_source,
        re.DOTALL,
    )
    assert match is not None, "blocks.search_text generated-column definition not found in index.py DDL"
    expr = match.group("expr")

    # Documented as indexed (docs/search.md "Searchable Content Coverage").
    assert "COALESCE(text, '')" in expr
    assert "COALESCE(tool_name, '')" in expr
    assert "json_extract(tool_input, '$.command')" in expr
    assert "json_extract(tool_input, '$.file_path')" in expr
    assert "json_extract(tool_input, '$.path')" in expr

    # Documented as NOT indexed -- the coverage gap this bead exists for.
    assert "$.content" not in expr
    assert "$.old_string" not in expr
    assert "$.new_string" not in expr

    doc_text = _SEARCH_DOC.read_text(encoding="utf-8")
    assert "Searchable Content Coverage" in doc_text
    assert "tool_input.$.content" in doc_text
    assert "tool_input.$.old_string" in doc_text
    assert "json_extract(tool_input, '$.content') LIKE" in doc_text


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
