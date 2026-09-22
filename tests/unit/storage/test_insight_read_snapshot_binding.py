"""Insight reads must stay inside the snapshot their caller pinned.

``run_archive_read`` / ``ArchiveQueryControl.control_store`` call
``begin_read_snapshot()`` so every statement of one operation answers from one
database generation. ``ArchiveStore._read_insights`` then issued an
unconditional ``commit()`` to refresh a stale long-lived connection, which ended
exactly that transaction: an insight built from several statements could mix
generations if the daemon committed between them, and on a writable store the
same commit published (and made unrollbackable) a caller-owned transaction.

Anti-vacuity, per test:

* ``test_pinned_snapshot_hides_a_later_commit`` is red the moment the refresh
  becomes unconditional again: the second read then answers from the newer
  generation and reports the tool the external writer added.
* ``test_unpinned_reader_sees_a_later_commit`` is the opposite direction. It
  fails if the refresh is simply deleted rather than made conditional, because a
  reader that never pinned a snapshot must still observe externally committed
  derived rows.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.storage_records import SessionBuilder

_SESSION = "claude-code-session:ext-snapshot"


def _seed(root: Path) -> Path:
    initialize_active_archive_root(root)
    index_db = root / "index.db"
    (
        SessionBuilder(index_db, "snapshot")
        .provider("claude-code")
        .title("snapshot")
        .add_message("m-0", role="assistant", text="ran a tool")
        .save()
    )
    conn = sqlite3.connect(index_db)
    try:
        with conn:
            conn.execute("PRAGMA journal_mode=WAL")
            _insert_tool_block(conn, position=11, tool_name="Bash")
    finally:
        conn.close()
    return index_db


def _insert_tool_block(conn: sqlite3.Connection, *, position: int, tool_name: str) -> None:
    conn.execute(
        """
        INSERT INTO blocks (message_id, session_id, position, block_type, tool_name, tool_id, text)
        VALUES (?, ?, ?, 'tool_use', ?, ?, ?)
        """,
        (f"{_SESSION}:n:m-0", _SESSION, position, tool_name, f"tool-{position}", tool_name),
    )


def _tool_names(archive: ArchiveStore) -> set[str]:
    insights = archive.list_tool_usage_insights()
    return {entry.normalized_tool_name for insight in insights for entry in insight.entries}


def test_pinned_snapshot_hides_a_later_commit(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    index_db = _seed(root)
    with ArchiveStore.open_existing(root) as archive:
        archive.begin_read_snapshot()
        try:
            assert _tool_names(archive) == {"bash"}
            writer = sqlite3.connect(index_db)
            try:
                with writer:
                    _insert_tool_block(writer, position=12, tool_name="Grep")
            finally:
                writer.close()
            assert _tool_names(archive) == {"bash"}
        finally:
            archive.end_read_snapshot()


def test_unpinned_reader_sees_a_later_commit(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    index_db = _seed(root)
    with ArchiveStore.open_existing(root) as archive:
        assert _tool_names(archive) == {"bash"}
        writer = sqlite3.connect(index_db)
        try:
            with writer:
                _insert_tool_block(writer, position=12, tool_name="Grep")
        finally:
            writer.close()
        assert _tool_names(archive) == {"bash", "grep"}
