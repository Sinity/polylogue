"""Runtime open-path executor for declared index-tier fast-forward plans.

polylogue-t3gk: the live 2026-07-21 incident this guards against was a fresh
v42 index.db that could not be opened by v43 code, even though ``lifecycle.py``
declares the v43 delta (``messages_fts_identity`` ledger + refreshed trigger
bodies) as clone-safe. These tests build a v42-*shaped* index.db (current
schema, downgraded to the pre-v43 trigger bodies with no identity ledger) and
open it through ``initialize_archive_database`` -- the exact runtime path
``polylogued`` and every CLI/API entry point calls -- asserting the fast-
forward executes instead of raising the rebuild-required error.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.fts.sql import message_identity_mismatch_sql
from polylogue.storage.sqlite.archive_tiers.bootstrap import (
    initialize_archive_database,
    initialize_archive_tier,
)
from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.lifecycle import (
    DerivedDeltaClass,
    FastForwardOperation,
    FastForwardOperationKind,
    IndexDeltaDeclaration,
)

_HASH = b"x" * 32

# The exact messages_fts trigger bodies as they existed BEFORE polylogue-1xc.12
# (v43) added messages_fts_identity maintenance -- see git history of
# polylogue/storage/fts/sql.py's BLOCKS_FTS_TRIGGER_DDL for the pre-v43 shape.
_PRE_V43_BLOCKS_FTS_TRIGGER_DDL = [
    """CREATE TRIGGER messages_fts_ai
       AFTER INSERT ON blocks WHEN new.search_text != '' BEGIN
           INSERT INTO messages_fts(rowid, block_id, message_id, session_id, block_type, text)
           VALUES (new.rowid, new.block_id, new.message_id, new.session_id, new.block_type, new.search_text);
       END""",
    """CREATE TRIGGER messages_fts_ad
       AFTER DELETE ON blocks WHEN old.search_text != '' BEGIN
           DELETE FROM messages_fts WHERE rowid = old.rowid;
       END""",
    """CREATE TRIGGER messages_fts_au
       AFTER UPDATE ON blocks BEGIN
           DELETE FROM messages_fts WHERE rowid = old.rowid;
           INSERT INTO messages_fts(rowid, block_id, message_id, session_id, block_type, text)
           SELECT new.rowid, new.block_id, new.message_id, new.session_id, new.block_type, new.search_text
           WHERE new.search_text != '';
       END""",
]


def _seed_indexable_block(conn: sqlite3.Connection, *, native_suffix: str, text: str) -> None:
    conn.execute(
        """
        INSERT INTO sessions (
            native_id, origin, title, content_hash, created_at_ms, updated_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (f"native-session-{native_suffix}", "codex-session", "fast-forward fixture", _HASH, 1, 1),
    )
    session = conn.execute(
        "SELECT session_id FROM sessions WHERE native_id = ?",
        (f"native-session-{native_suffix}",),
    ).fetchone()
    conn.execute(
        """
        INSERT INTO messages (
            session_id, native_id, position, role, message_type, content_hash, occurred_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (session[0], f"native-message-{native_suffix}", 0, "assistant", "message", _HASH, 1),
    )
    message = conn.execute(
        "SELECT message_id FROM messages WHERE session_id = ?",
        (session[0],),
    ).fetchone()
    conn.execute(
        """
        INSERT INTO blocks (
            message_id, session_id, position, block_type, text
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (message[0], session[0], 0, "text", text),
    )


def _build_v42_shaped_index_db(path: Path) -> None:
    """Build a v43-schema index.db, then downgrade it to the v42 trigger shape."""
    conn = sqlite3.connect(path)
    try:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == INDEX_SCHEMA_VERSION

        _seed_indexable_block(conn, native_suffix="a", text="needle prose one")
        _seed_indexable_block(conn, native_suffix="b", text="needle prose two")
        conn.commit()

        indexable_before = conn.execute("SELECT COUNT(*) FROM blocks WHERE search_text != ''").fetchone()[0]
        assert indexable_before == 2

        # Simulate a v42 generation: drop the ledger this delta introduces and
        # replay the pre-v43 trigger bodies (no identity-ledger maintenance).
        for name in ("messages_fts_ai", "messages_fts_ad", "messages_fts_au"):
            conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        conn.execute("DELETE FROM messages_fts_identity")
        conn.execute("DROP TABLE messages_fts_identity")
        for ddl in _PRE_V43_BLOCKS_FTS_TRIGGER_DDL:
            conn.execute(ddl)
        conn.execute("PRAGMA user_version = 42")
        conn.commit()
    finally:
        conn.close()


def test_v42_index_db_fast_forwards_to_v43_on_open(tmp_path: Path) -> None:
    """polylogue-t3gk: the exact live-incident shape must open without a rebuild error."""
    path = tmp_path / "index.db"
    _build_v42_shaped_index_db(path)

    # ANTI-VACUITY: removing the bootstrap.py wiring to
    # apply_index_fast_forward (or reverting to the pre-t3gk bootstrap.py)
    # makes this call raise RuntimeError("... is not the current index tier
    # version ..."), which pytest.raises would need to wrap -- this bare call
    # is the assertion that no such error is raised.
    initialize_archive_database(path, ArchiveTier.INDEX)

    conn = sqlite3.connect(path)
    try:
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == INDEX_SCHEMA_VERSION

        ledger_rows = conn.execute("SELECT COUNT(*) FROM messages_fts_identity").fetchone()[0]
        indexable_rows = conn.execute("SELECT COUNT(*) FROM blocks WHERE search_text != ''").fetchone()[0]
        assert indexable_rows == 2
        # ANTI-VACUITY: if _apply_rebuild_fts stopped calling
        # insert_all_message_identity_rows_sql() after ensure_fts_index_sync
        # (e.g. only recreating the empty table), this would read 0 instead
        # of 2.
        assert ledger_rows == indexable_rows

        mismatches = conn.execute(message_identity_mismatch_sql()).fetchone()[0]
        # ANTI-VACUITY: if the executor recreated messages_fts_identity from
        # a stale/wrong recipe id or left rowids unbound to their current
        # block_id, this reconciliation query (the same one the daemon's
        # readiness check uses) would report a nonzero mismatch count.
        assert mismatches == 0
    finally:
        conn.close()


def test_v42_index_db_reopen_is_idempotent(tmp_path: Path) -> None:
    """A second open of an already fast-forwarded archive must not re-raise or re-mutate."""
    path = tmp_path / "index.db"
    _build_v42_shaped_index_db(path)

    initialize_archive_database(path, ArchiveTier.INDEX)
    initialize_archive_database(path, ArchiveTier.INDEX)

    conn = sqlite3.connect(path)
    try:
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == INDEX_SCHEMA_VERSION
        assert conn.execute(message_identity_mismatch_sql()).fetchone()[0] == 0
    finally:
        conn.close()


def test_semantic_reparse_gap_still_raises_rebuild_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A SEMANTIC_REPARSE declaration in the gap must still refuse fast-forward.

    ANTI-VACUITY: deleting the ``if plan is not None`` guard in
    ``bootstrap.py`` (always falling through) would make this test pass
    vacuously along with the happy-path test failing instead; deleting the
    ``eligible_for_sql_fast_forward`` check in ``lifecycle.index_fast_forward_plan``
    (so a SEMANTIC_REPARSE span still returns a plan) is what this test
    actually guards against -- it would make ``plan`` non-``None`` here and
    the RuntimeError would no longer be raised.
    """
    import polylogue.storage.sqlite.lifecycle as lifecycle

    path = tmp_path / "index.db"
    conn = sqlite3.connect(path)
    try:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        conn.execute(f"PRAGMA user_version = {INDEX_SCHEMA_VERSION - 1}")
        conn.commit()
    finally:
        conn.close()

    semantic_declaration = IndexDeltaDeclaration(
        version=INDEX_SCHEMA_VERSION,
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    )
    fast_forward_eligible_declaration = next(
        declaration for declaration in lifecycle.INDEX_DELTA_DECLARATIONS if declaration.version == INDEX_SCHEMA_VERSION
    )
    assert fast_forward_eligible_declaration.operations, (
        "fixture assumption: the real current-version declaration is fast-forward eligible; "
        "monkeypatching it out for a SEMANTIC_REPARSE stand-in below is what this test exercises"
    )
    monkeypatch.setattr(
        lifecycle,
        "INDEX_DELTA_DECLARATIONS",
        tuple(
            declaration if declaration.version != INDEX_SCHEMA_VERSION else semantic_declaration
            for declaration in lifecycle.INDEX_DELTA_DECLARATIONS
        ),
    )

    with pytest.raises(RuntimeError, match="move it aside and rebuild the archive root"):
        initialize_archive_database(path, ArchiveTier.INDEX)


def test_apply_index_fast_forward_rejects_an_ineligible_plan() -> None:
    """The executor itself refuses a plan lacking sql-fast-forward eligibility."""
    from polylogue.storage.sqlite.archive_tiers.index_fast_forward_executor import apply_index_fast_forward
    from polylogue.storage.sqlite.lifecycle import IndexFastForwardPlan

    semantic_declaration = IndexDeltaDeclaration(
        version=44,
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    )
    plan = IndexFastForwardPlan(source_version=43, target_version=44, declarations=(semantic_declaration,))
    conn = sqlite3.connect(":memory:")
    try:
        with pytest.raises(RuntimeError, match="not eligible for SQL fast-forward"):
            apply_index_fast_forward(conn, plan)
    finally:
        conn.close()


def test_apply_index_fast_forward_dispatches_unknown_kind_generically() -> None:
    """Dispatch is a registry over FastForwardOperationKind, not per-version code.

    ANTI-VACUITY: this exercises a synthetic, never-declared-in-production
    kind combination (DROP_TABLE on a table that does not exist) purely to
    prove the dispatch path is reached without any v43-specific branch --
    removing the DROP_TABLE arm from ``_apply_operation`` would make this
    raise instead of completing.
    """
    from polylogue.storage.sqlite.archive_tiers.index_fast_forward_executor import apply_index_fast_forward
    from polylogue.storage.sqlite.lifecycle import IndexFastForwardPlan

    declaration = IndexDeltaDeclaration(
        version=44,
        classes=(DerivedDeltaClass.CACHE_REMOVAL,),
        operations=(
            FastForwardOperation(
                name="synthetic-drop",
                kind=FastForwardOperationKind.DROP_TABLE,
                objects=(("table", "does_not_exist"),),
            ),
        ),
    )
    plan = IndexFastForwardPlan(source_version=43, target_version=44, declarations=(declaration,))
    conn = sqlite3.connect(":memory:")
    try:
        apply_index_fast_forward(conn, plan)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 44
    finally:
        conn.close()
