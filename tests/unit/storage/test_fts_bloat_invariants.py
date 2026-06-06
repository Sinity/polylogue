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


def test_messages_fts_is_contentless_delete(tmp_path: Path) -> None:
    """``messages_fts`` must not store raw message bodies (#817/#1486)."""
    from polylogue.storage.fts.fts_lifecycle import ensure_fts_index_sync
    from polylogue.storage.sqlite.schema_ddl import SCHEMA_DDL

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
    from polylogue.storage.sqlite.schema_ddl import SCHEMA_DDL

    db = tmp_path / "fts_bloat.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.executescript(SCHEMA_DDL)
        conn.execute(
            "INSERT INTO sessions(session_id, source_name, provider_session_id, version) VALUES('c1','test','pc1',1)"
        )
        body = "lorem ipsum dolor sit amet consectetur adipiscing elit " * 100
        for i in range(2000):
            conn.execute(
                "INSERT INTO messages(message_id, session_id, role, text, source_name, version) VALUES(?,?,?,?,?,1)",
                (f"m{i}", "c1", "user", body, "test"),
            )
        conn.commit()
        conn.execute("VACUUM")

        def _table_bytes(name_prefix: str) -> int:
            cursor = conn.execute(
                "SELECT COALESCE(SUM(pgsize),0) FROM dbstat WHERE name LIKE ?",
                (f"{name_prefix}%",),
            )
            row = cursor.fetchone()
            return int(row[0]) if row and row[0] is not None else 0

        messages_bytes = _table_bytes("messages")
        # Subtract messages_fts* contribution captured by the prefix wildcard.
        fts_bytes = _table_bytes("messages_fts")
        source_bytes = messages_bytes - fts_bytes
        assert source_bytes > 0, "messages table should have measurable storage"
        ratio = fts_bytes / source_bytes
        # External-content FTS keeps tokens + docsize, not full text. The
        # observed ratio on this fixture is ~0.2; we allow generous headroom
        # so unrelated SQLite version changes don't false-trip, while
        # catching any regression to a full standalone FTS5 (ratio >= 1).
        assert ratio < 0.5, (
            f"messages_fts pages ({fts_bytes}) should be well under messages pages "
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
    from polylogue.storage.sqlite.schema_ddl import SCHEMA_DDL

    conn = sqlite3.connect(str(tmp_path / "fts_delete.db"))
    try:
        conn.executescript(SCHEMA_DDL)
        conn.execute(
            "INSERT INTO sessions(session_id, source_name, provider_session_id, version) VALUES('c1','test','pc1',1)"
        )
        conn.execute(
            "INSERT INTO messages(message_id, session_id, role, text, source_name, version) "
            "VALUES('m1','c1','user','unique_token_alpha here','test',1)"
        )
        conn.execute(
            "INSERT INTO messages(message_id, session_id, role, text, source_name, version) "
            "VALUES('m2','c1','user','unique_token_beta here','test',1)"
        )
        conn.commit()
        assert (
            conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'unique_token_alpha'").fetchone()[
                0
            ]
            == 1
        )

        conn.execute("DELETE FROM messages WHERE message_id='m1'")
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
    from polylogue.storage.sqlite.schema_ddl import SCHEMA_DDL

    conn = sqlite3.connect(str(tmp_path / "fts_replace_missing_triggers.db"))
    try:
        conn.executescript(SCHEMA_DDL)
        conn.execute(
            "INSERT INTO sessions(session_id, source_name, provider_session_id, version) VALUES('c1','test','pc1',1)"
        )
        conn.execute(
            "INSERT INTO messages(message_id, session_id, role, text, source_name, version) "
            "VALUES('m1','c1','user','replace orphan needle','test',1)"
        )
        conn.execute(
            """
            INSERT INTO action_events (
                event_id, session_id, message_id, sequence_index,
                action_kind, normalized_tool_name, search_text
            ) VALUES ('a1', 'c1', 'm1', 0, 'shell', 'bash', 'action orphan needle')
            """
        )
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM action_events_fts_docsize").fetchone()[0] == 1

        conn.execute("DROP TRIGGER messages_fts_ad")
        conn.execute("DROP TRIGGER action_events_fts_ad")
        replace_session_runtime_state_sync(conn, "c1")
        conn.commit()

        message_orphans = conn.execute(
            """
            SELECT COUNT(*)
            FROM messages_fts_docsize AS d
            LEFT JOIN messages AS m ON m.rowid = d.id
            WHERE m.rowid IS NULL
            """
        ).fetchone()[0]
        action_orphans = conn.execute(
            """
            SELECT COUNT(*)
            FROM action_events_fts_docsize AS d
            LEFT JOIN action_events AS ae ON ae.rowid = d.id
            WHERE ae.rowid IS NULL
            """
        ).fetchone()[0]
        assert message_orphans == 0
        assert action_orphans == 0
    finally:
        conn.close()
