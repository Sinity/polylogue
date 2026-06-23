"""Regression tests pinning the messages_fts no-raw-body invariants (#817/#1486).

These guard the conversion of ``messages_fts`` to contentless-delete FTS5.
The table keeps only tokens/docsize and supports targeted rowid deletes; full
message bodies live in canonical archive tables instead of inside FTS shadow
storage.

* the DDL still carries the external-content options,
* on synthetic data the index pages stay well under the source table size,
* DELETE on ``messages`` removes the row from FTS via the AD trigger.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession


def _seed_session(conn: sqlite3.Connection, native_id: str = "c1") -> str:
    conn.execute(
        "INSERT INTO sessions(native_id, origin, content_hash) VALUES (?, ?, ?)",
        (native_id, "codex-session", bytes(32)),
    )
    return f"codex-session:{native_id}"


def _seed_message(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    native_id: str,
    position: int,
    text: str,
) -> str:
    conn.execute(
        """
        INSERT INTO messages(session_id, native_id, position, role, message_type, content_hash)
        VALUES (?, ?, ?, 'user', 'message', ?)
        """,
        (session_id, native_id, position, bytes(32)),
    )
    message_id = f"{session_id}:{native_id}"
    conn.execute(
        """
        INSERT INTO blocks(message_id, session_id, position, block_type, text)
        VALUES (?, ?, 0, 'text', ?)
        """,
        (message_id, session_id, text),
    )
    return message_id


def test_messages_fts_is_contentless_delete(tmp_path: Path) -> None:
    """``messages_fts`` must not store raw message bodies (#817/#1486)."""
    from polylogue.storage.fts.fts_lifecycle import ensure_fts_index_sync
    from polylogue.storage.sqlite.schema import SCHEMA_DDL

    conn = sqlite3.connect(str(tmp_path / "fts_contentless.db"))
    try:
        conn.executescript(SCHEMA_DDL)
        ensure_fts_index_sync(conn)
        row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='messages_fts'").fetchone()
        assert row is not None
        ddl = row[0].lower()
        assert "content=''" in ddl or "content=" in ddl, ddl
        assert "contentless_delete=1" in ddl, ddl
    finally:
        conn.close()


def test_messages_fts_storage_does_not_duplicate_message_bodies(tmp_path: Path) -> None:
    """External-content FTS must not store its own copy of message text (#817).

    Synthesises 2000 large messages (~5 KB each, ~10 MB total in ``messages``)
    and asserts ``messages_fts*`` pages combined stay well under the message
    table size. With external-content FTS the index keeps only a docsize
    table plus tokenized posting lists — typically <25% of the source. A
    regression to standalone FTS5 would push the ratio above 1.0 (full
    duplicate copy) and the assertion would fail.
    """
    from polylogue.storage.sqlite.schema import SCHEMA_DDL

    db = tmp_path / "fts_bloat.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.executescript(SCHEMA_DDL)
        session_id = _seed_session(conn)
        body = "lorem ipsum dolor sit amet consectetur adipiscing elit " * 100
        for i in range(2000):
            _seed_message(conn, session_id=session_id, native_id=f"m{i}", position=i, text=body)
        conn.commit()
        conn.execute("VACUUM")

        def _table_bytes(name_prefix: str) -> int:
            cursor = conn.execute(
                "SELECT COALESCE(SUM(pgsize),0) FROM dbstat WHERE name LIKE ?",
                (f"{name_prefix}%",),
            )
            row = cursor.fetchone()
            return int(row[0]) if row and row[0] is not None else 0

        blocks_bytes = _table_bytes("blocks")
        fts_bytes = _table_bytes("messages_fts")
        source_bytes = blocks_bytes
        assert source_bytes > 0, "blocks table should have measurable storage"
        ratio = fts_bytes / source_bytes
        # External-content FTS keeps tokens + docsize, not full text. The
        # observed ratio on this fixture is ~0.2; we allow generous headroom
        # so unrelated SQLite version changes don't false-trip, while
        # catching any regression to a full standalone FTS5 (ratio >= 1).
        assert ratio < 0.5, (
            f"messages_fts pages ({fts_bytes}) should be well under blocks pages "
            f"({source_bytes}); ratio={ratio:.2f} suggests FTS bloat regression"
        )
    finally:
        conn.close()


def test_messages_fts_deletion_consistency_with_external_content(tmp_path: Path) -> None:
    """DELETE on ``messages`` must remove the row from FTS via the AD trigger (#817).

    An earlier attempt to convert ``messages_fts`` to external content was
    reverted because rows kept matching after their source was deleted. The
    fix is the ``messages_fts_ad`` trigger that issues the FTS5 ``'delete'``
    command with the original ``text`` payload. This test guards against
    regressing that trigger (or losing the rowid-aligned source-table linkage).
    """
    from polylogue.storage.sqlite.schema import SCHEMA_DDL

    conn = sqlite3.connect(str(tmp_path / "fts_delete.db"))
    try:
        conn.executescript(SCHEMA_DDL)
        session_id = _seed_session(conn)
        m1 = _seed_message(
            conn,
            session_id=session_id,
            native_id="m1",
            position=0,
            text="unique_token_alpha here",
        )
        _seed_message(
            conn,
            session_id=session_id,
            native_id="m2",
            position=1,
            text="unique_token_beta here",
        )
        conn.commit()
        assert (
            conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'unique_token_alpha'").fetchone()[
                0
            ]
            == 1
        )

        conn.execute("DELETE FROM blocks WHERE message_id = ?", (m1,))
        conn.commit()

        # m1 must no longer be findable.
        alpha_hits = conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'unique_token_alpha'"
        ).fetchone()[0]
        assert alpha_hits == 0, "FTS still finds deleted message (external-content delete regression)"
        # m2 must still be findable.
        beta_hits = conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'unique_token_beta'"
        ).fetchone()[0]
        assert beta_hits == 1
    finally:
        conn.close()


def test_session_replacement_purges_fts_when_delete_triggers_missing(tmp_path: Path) -> None:
    """Replacement must not orphan FTS rows if bulk ingest has suspended triggers."""
    from polylogue.storage.session_replacement import replace_session_runtime_state_sync
    from polylogue.storage.sqlite.schema import SCHEMA_DDL

    conn = sqlite3.connect(str(tmp_path / "fts_replace_missing_triggers.db"))
    try:
        conn.executescript(SCHEMA_DDL)
        session_id = _seed_session(conn)
        _seed_message(conn, session_id=session_id, native_id="m1", position=0, text="replace orphan needle")
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 1

        conn.execute("DROP TRIGGER messages_fts_ad")
        replace_session_runtime_state_sync(conn, session_id)
        conn.commit()

        message_orphans = conn.execute(
            """
            SELECT COUNT(*)
            FROM messages_fts_docsize AS d
            LEFT JOIN blocks AS b ON b.rowid = d.id
            WHERE b.rowid IS NULL
            """
        ).fetchone()[0]
        assert message_orphans == 0
    finally:
        conn.close()


def test_parsed_session_rewrite_purges_fts_when_bulk_triggers_suspended(tmp_path: Path) -> None:
    """Full session rewrite must not orphan old FTS rowids in dropped-trigger bulk mode."""
    from polylogue.storage.fts.fts_lifecycle import repair_message_fts_index_sync, restore_fts_triggers_sync
    from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
    from polylogue.storage.sqlite.schema import SCHEMA_DDL

    def parsed_session(*texts: str) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="bulk-rewrite-orphan",
            title="Bulk rewrite orphan",
            messages=[
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.USER,
                    text="\n".join(texts),
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text) for text in texts],
                )
            ],
        )

    conn = sqlite3.connect(str(tmp_path / "fts_bulk_rewrite.db"))
    try:
        conn.executescript(SCHEMA_DDL)
        write_parsed_session_to_archive(conn, parsed_session("old orphan needle", "old removed block"))
        old_block_rowids = {row[0] for row in conn.execute("SELECT rowid FROM blocks").fetchall()}
        assert conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 2

        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("DROP TRIGGER messages_fts_ad")
        conn.execute("DROP TRIGGER messages_fts_ai")
        conn.execute("DROP TRIGGER messages_fts_au")
        write_parsed_session_to_archive(conn, parsed_session("new orphan needle"))
        restore_fts_triggers_sync(conn)
        repair_message_fts_index_sync(conn, ["codex-session:bulk-rewrite-orphan"])
        conn.commit()

        new_block_rowids = {row[0] for row in conn.execute("SELECT rowid FROM blocks").fetchall()}
        assert old_block_rowids - new_block_rowids
        message_orphans = conn.execute(
            """
            SELECT COUNT(*)
            FROM messages_fts_docsize AS d
            LEFT JOIN blocks AS b ON b.rowid = d.id
            WHERE b.rowid IS NULL
            """
        ).fetchone()[0]
        assert message_orphans == 0
        assert conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'old'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'new'").fetchone()[0] == 1
    finally:
        conn.close()
