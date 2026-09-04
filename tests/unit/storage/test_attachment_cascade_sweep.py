"""A session delete must not strand the attachment rows its refs pointed at.

``attachment_refs`` cascades from ``sessions`` and ``messages``, so a session
delete removes the refs with no Python code observing it. The ``attachments``
rows they pointed at then survive with a stale ``ref_count`` and no reachable
ref — unreachable from every read path, which both join through
``attachment_refs`` — while still reporting ``acquisition_status = 'acquired'``
and demanding blob preservation in every ``full_evidence`` backup.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

_SESSION_ID = "gemini-cli-session:s1"
_MESSAGE_ID = "gemini-cli-session:s1:n:m1"


def _seed_session_with_attachment(index_db: Path, *, attachment_id: str = "att-1") -> None:
    conn = sqlite3.connect(index_db)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            "INSERT INTO sessions (native_id, origin, content_hash) VALUES ('s1', 'gemini-cli-session', zeroblob(32))"
        )
        conn.execute(
            "INSERT INTO messages (session_id, native_id, position, role, message_type, content_hash)"
            " VALUES (?, 'm1', 0, 'user', 'message', zeroblob(32))",
            (_SESSION_ID,),
        )
        conn.execute(
            "INSERT INTO attachments (attachment_id, display_name, media_type, byte_count, blob_hash, ref_count)"
            " VALUES (?, NULL, NULL, 7, zeroblob(32), 1)",
            (attachment_id,),
        )
        conn.execute(
            "INSERT INTO attachment_refs (attachment_id, session_id, message_id, position) VALUES (?, ?, ?, 0)",
            (attachment_id, _SESSION_ID, _MESSAGE_ID),
        )
        conn.commit()
    finally:
        conn.close()


def _attachment_rows(index_db: Path) -> list[tuple[str, int]]:
    conn = sqlite3.connect(index_db)
    try:
        return [(str(row[0]), int(row[1])) for row in conn.execute("SELECT attachment_id, ref_count FROM attachments")]
    finally:
        conn.close()


def test_deleting_a_session_retires_the_attachment_rows_its_refs_cascaded_away(tmp_path: Path) -> None:
    """Anti-vacuity: drop the sweep from ``delete_sessions`` and the row survives
    with ``ref_count`` still 1 while its only ref is gone."""
    root = tmp_path / "archive"
    with ArchiveStore(root):
        pass
    index_db = (root / "index.db").resolve()
    _seed_session_with_attachment(index_db)
    assert _attachment_rows(index_db) == [("att-1", 1)]

    with ArchiveStore(root) as archive:
        assert archive.delete_sessions((_SESSION_ID,)) == 1

    conn = sqlite3.connect(index_db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM attachment_refs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    finally:
        conn.close()
    assert _attachment_rows(index_db) == []


def test_a_session_delete_leaves_another_session_s_attachment_alone(tmp_path: Path) -> None:
    """The sweep retires what lost its last ref, never what still has one.

    Anti-vacuity: sweep on session membership instead of recomputed ref_count
    and the shared attachment disappears with the first session deleted.
    """
    root = tmp_path / "archive"
    with ArchiveStore(root):
        pass
    index_db = (root / "index.db").resolve()
    _seed_session_with_attachment(index_db)
    conn = sqlite3.connect(index_db)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            "INSERT INTO sessions (native_id, origin, content_hash) VALUES ('s2', 'gemini-cli-session', zeroblob(32))"
        )
        conn.execute(
            "INSERT INTO messages (session_id, native_id, position, role, message_type, content_hash)"
            " VALUES ('gemini-cli-session:s2', 'm1', 0, 'user', 'message', zeroblob(32))"
        )
        conn.execute(
            "INSERT INTO attachment_refs (attachment_id, session_id, message_id, position)"
            " VALUES ('att-1', 'gemini-cli-session:s2', 'gemini-cli-session:s2:n:m1', 0)"
        )
        conn.execute("UPDATE attachments SET ref_count = 2 WHERE attachment_id = 'att-1'")
        conn.commit()
    finally:
        conn.close()

    with ArchiveStore(root) as archive:
        assert archive.delete_sessions((_SESSION_ID,)) == 1

    assert _attachment_rows(index_db) == [("att-1", 1)]
