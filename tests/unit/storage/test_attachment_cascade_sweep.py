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
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.lifecycle import (
    INDEX_DELTA_DECLARATIONS,
    DerivedDeltaClass,
    FastForwardOperationKind,
    index_delta_declaration_report,
    index_fast_forward_plan,
    resolve_canonical_index_objects,
)

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


def test_the_v77_delta_is_an_index_only_fast_forward_that_resolves(tmp_path: Path) -> None:
    """The declaration must be executable, and reach an existing archive.

    The previous attempt at this fix declared an operation kind the executor's
    canonical lookup did not know, so apply raised and no archive reached the
    new version — while the whole fast-forward suite stayed green.

    Anti-vacuity: remove the v77 declaration and
    `index_delta_declaration_report` reports a missing version; point the
    operation at an object with no canonical DDL and resolution returns
    nothing for it.
    """
    declaration = next(d for d in INDEX_DELTA_DECLARATIONS if d.version == 77)

    assert declaration.classes == (DerivedDeltaClass.INDEX_ONLY,)
    assert declaration.operations[0].kind is FastForwardOperationKind.CREATE_INDEX
    assert declaration.operations[0].objects == (("index", "idx_attachment_refs_attachment"),)

    plan = index_fast_forward_plan(76, 77)
    assert plan is not None
    assert plan.eligible_for_sql_fast_forward is True
    assert plan.requires_semantic_reparse is False

    resolved = resolve_canonical_index_objects(declaration.operations[0].objects)
    assert "idx_attachment_refs_attachment" in resolved[("index", "idx_attachment_refs_attachment")]

    report = index_delta_declaration_report(INDEX_SCHEMA_VERSION)
    assert report["ok"] is True
    assert report["missing_versions"] == ()


def test_an_archive_at_the_previous_version_ends_up_with_the_index(tmp_path: Path) -> None:
    """The end state an operator actually gets when opening a v76 archive."""
    index_db = tmp_path / "index.db"
    conn = sqlite3.connect(index_db)
    try:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == INDEX_SCHEMA_VERSION
        conn.execute("DROP INDEX IF EXISTS idx_attachment_refs_attachment")
        conn.execute("PRAGMA user_version = 76")
        conn.commit()
    finally:
        conn.close()

    conn = sqlite3.connect(index_db)
    try:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 77
        assert (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'idx_attachment_refs_attachment'"
            ).fetchone()
            is not None
        )
    finally:
        conn.close()
