"""ohbx: `blocks_command_trigram` is an external-content FTS5 trigram index
over `blocks.tool_detail_text` (lowercased tool_command+tool_path), backing
a fast substring lookup ("did this bash/exec_command block invoke
`polylogue`?") that a plain LIKE scan over generated tool_command/tool_path
columns cannot use an index for. Correctness has two non-obvious pitfalls
verified here: (1) external-content deletes need the special 'delete'
command form with the OLD value supplied, or stale postings later raise
"fts5: missing row N from content table"; (2) callers MUST drive the query
via `blocks.rowid IN (SELECT rowid FROM blocks_command_trigram WHERE ...)`,
not a plain JOIN -- a JOIN lets SQLite pick `blocks` as the outer loop and
probe the trigram table per row, which measured slower than the original
unindexed scan at 300K rows (26s vs 0.15s) despite the trigram index itself
being correctly built.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_HASH = b"x" * 32


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _insert_session(conn: sqlite3.Connection, *, native_id: str = "s1") -> str:
    conn.execute(
        """
        INSERT INTO sessions (native_id, origin, title, content_hash, created_at_ms, updated_at_ms)
        VALUES (?, 'claude-code-session', 'test session', ?, 0, 0)
        """,
        (native_id, _HASH),
    )
    row = conn.execute("SELECT session_id FROM sessions WHERE native_id = ?", (native_id,)).fetchone()
    return str(row["session_id"])


def _insert_message(conn: sqlite3.Connection, *, session_id: str, native_id: str = "m1") -> str:
    conn.execute(
        """
        INSERT INTO messages (session_id, native_id, position, variant_index, role, content_hash)
        VALUES (?, ?, 1, 0, 'assistant', ?)
        """,
        (session_id, native_id, _HASH),
    )
    row = conn.execute(
        "SELECT message_id FROM messages WHERE session_id = ? AND native_id = ?",
        (session_id, native_id),
    ).fetchone()
    return str(row["message_id"])


def _insert_tool_use_block(
    conn: sqlite3.Connection, *, message_id: str, session_id: str, position: int, command: str
) -> int:
    conn.execute(
        """
        INSERT INTO blocks (message_id, session_id, position, block_type, tool_name, tool_input)
        VALUES (?, ?, ?, 'tool_use', 'bash', ?)
        """,
        (message_id, session_id, position, json.dumps({"command": command})),
    )
    row = conn.execute(
        "SELECT rowid FROM blocks WHERE message_id = ? AND position = ?", (message_id, position)
    ).fetchone()
    return int(row["rowid"])


def _matching_rowids(conn: sqlite3.Connection, pattern: str = "%polylogue%") -> list[int]:
    rows = conn.execute(
        """
        SELECT u.rowid FROM blocks AS u
        WHERE u.rowid IN (SELECT rowid FROM blocks_command_trigram WHERE tool_detail_text LIKE ?)
          AND u.block_type = 'tool_use'
        ORDER BY u.rowid
        """,
        (pattern,),
    ).fetchall()
    return [int(row["rowid"]) for row in rows]


def test_insert_populates_trigram_index_for_tool_use_blocks(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session_id = _insert_session(conn)
    message_id = _insert_message(conn, session_id=session_id)
    _insert_tool_use_block(
        conn, message_id=message_id, session_id=session_id, position=0, command="polylogue find repo:foo"
    )
    _insert_tool_use_block(conn, message_id=message_id, session_id=session_id, position=1, command="ls -la")
    conn.commit()

    assert _matching_rowids(conn) == [1]


def test_substring_match_without_token_boundary(tmp_path: Path) -> None:
    """Trigram tokenization matches true substrings, unlike unicode61's
    token-boundary matching -- a path with "polylogue" embedded but not
    surrounded by word boundaries must still match."""
    conn = _connect(tmp_path / "index.db")
    session_id = _insert_session(conn)
    message_id = _insert_message(conn, session_id=session_id)
    _insert_tool_use_block(
        conn, message_id=message_id, session_id=session_id, position=0, command="cat /tmp/notpolyloguefile.txt"
    )
    conn.commit()

    assert _matching_rowids(conn) == [1]


def test_non_tool_use_blocks_are_never_indexed(tmp_path: Path) -> None:
    """A prose text block mentioning "polylogue" must not surface via the
    trigram lookup -- only tool_use blocks' command/path text is indexed."""
    conn = _connect(tmp_path / "index.db")
    session_id = _insert_session(conn)
    message_id = _insert_message(conn, session_id=session_id)
    _insert_tool_use_block(conn, message_id=message_id, session_id=session_id, position=0, command="polylogue find")
    conn.execute(
        """
        INSERT INTO blocks (message_id, session_id, position, block_type, text)
        VALUES (?, ?, 1, 'text', 'mentions polylogue constantly in prose here')
        """,
        (message_id, session_id),
    )
    conn.commit()

    assert _matching_rowids(conn) == [1]


def test_update_repopulates_trigram_index(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session_id = _insert_session(conn)
    message_id = _insert_message(conn, session_id=session_id)
    _insert_tool_use_block(conn, message_id=message_id, session_id=session_id, position=0, command="ls -la")
    conn.commit()
    assert _matching_rowids(conn) == []

    conn.execute(
        "UPDATE blocks SET tool_input = ? WHERE message_id = ? AND position = 0",
        (json.dumps({"command": "echo polylogue"}), message_id),
    )
    conn.commit()

    assert _matching_rowids(conn) == [1]


def test_delete_removes_from_trigram_index_without_integrity_error(tmp_path: Path) -> None:
    """Regression: a plain `DELETE FROM blocks_command_trigram WHERE rowid =
    old.rowid` (instead of the FTS5 'delete' command form) leaves stale
    postings that raise "fts5: missing row N from content table" once the
    real row is gone. This must not happen."""
    conn = _connect(tmp_path / "index.db")
    session_id = _insert_session(conn)
    message_id = _insert_message(conn, session_id=session_id)
    _insert_tool_use_block(conn, message_id=message_id, session_id=session_id, position=0, command="polylogue find")
    _insert_tool_use_block(conn, message_id=message_id, session_id=session_id, position=1, command="ls -la")
    conn.commit()
    assert _matching_rowids(conn) == [1]

    conn.execute("DELETE FROM blocks WHERE message_id = ? AND position = 0", (message_id,))
    conn.commit()

    # Must not raise "fts5: missing row N from content table".
    assert _matching_rowids(conn) == []


def test_in_subquery_shape_matches_raw_scan_at_scale(tmp_path: Path) -> None:
    """Correctness cross-check between the new trigram-driven query and the
    original unindexed LIKE scan, at a scale large enough to exercise the
    query planner's real join-order choice (see module docstring: a naive
    JOIN measured slower than the original scan despite the index being
    correctly populated)."""
    conn = _connect(tmp_path / "index.db")
    session_id = _insert_session(conn)
    message_id = _insert_message(conn, session_id=session_id)

    generic_tools = ("exec_command", "functions", "functions.exec_command", "bash", "shell", "client")
    n = 2000
    rows = [
        (
            message_id,
            session_id,
            i,
            "polylogue find repo:foo" if i % 200 == 0 else f"ls -la /some/path/{i}",
        )
        for i in range(n)
    ]
    conn.executemany(
        """
        INSERT INTO blocks (message_id, session_id, position, block_type, tool_name, tool_input)
        VALUES (?, ?, ?, 'tool_use', 'bash', json_object('command', ?))
        """,
        rows,
    )
    conn.commit()

    tool_expr = "COALESCE(NULLIF(LOWER(u.tool_name), ''), 'unknown')"
    placeholders = ", ".join("?" for _ in generic_tools)

    new_count = conn.execute(
        f"""
        SELECT count(*) FROM blocks AS u
        WHERE u.rowid IN (SELECT rowid FROM blocks_command_trigram WHERE tool_detail_text LIKE '%polylogue%')
          AND u.block_type = 'tool_use'
          AND {tool_expr} IN ({placeholders})
        """,
        generic_tools,
    ).fetchone()[0]
    old_count = conn.execute(
        f"""
        SELECT count(*) FROM blocks AS u
        WHERE u.block_type = 'tool_use'
          AND {tool_expr} IN ({placeholders})
          AND (
              lower(coalesce(u.tool_command, '')) LIKE '%polylogue%'
              OR lower(coalesce(u.tool_path, '')) LIKE '%polylogue%'
          )
        """,
        generic_tools,
    ).fetchone()[0]

    assert new_count == old_count == n // 200
