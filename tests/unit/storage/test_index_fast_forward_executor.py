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


def _build_downgradable_index_db(path: Path, downgrade_from_version: int, downgrade_to_version: int) -> None:
    """Build a current-schema index.db, then downgrade it to a specific version shape.

    Used to test fast-forward from a specific version. Assumes the downgrade only
    requires trigger replacement (simple schema surgery without structural changes).
    """
    conn = sqlite3.connect(path)
    try:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == INDEX_SCHEMA_VERSION

        _seed_indexable_block(conn, native_suffix="a", text="needle prose one")
        _seed_indexable_block(conn, native_suffix="b", text="needle prose two")
        conn.commit()

        indexable_before = conn.execute("SELECT COUNT(*) FROM blocks WHERE search_text != ''").fetchone()[0]
        assert indexable_before == 2

        # For downgrading from v43 to v42: drop the ledger and replay pre-v43 trigger bodies.
        if downgrade_to_version == 42 and downgrade_from_version == 43:
            for name in ("messages_fts_ai", "messages_fts_ad", "messages_fts_au"):
                conn.execute(f"DROP TRIGGER IF EXISTS {name}")
            conn.execute("DELETE FROM messages_fts_identity")
            conn.execute("DROP TABLE messages_fts_identity")
            for ddl in _PRE_V43_BLOCKS_FTS_TRIGGER_DDL:
                conn.execute(ddl)

        conn.execute(f"PRAGMA user_version = {downgrade_to_version}")
        conn.commit()
    finally:
        conn.close()


def _build_v42_shaped_index_db(path: Path) -> None:
    """Build a v43-schema index.db, then downgrade it to the v42 trigger shape."""
    _build_downgradable_index_db(path, downgrade_from_version=43, downgrade_to_version=42)


def test_sql_fast_forwardable_index_db_reaches_current_on_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that an index.db can fast-forward using synthetic v42→v43 declarations.

    ANTI-VACUITY: removing the bootstrap.py wiring to apply_index_fast_forward
    (or reverting to the pre-t3gk bootstrap.py) makes this call raise
    RuntimeError("... is not the current index tier version ..."), which
    pytest.raises would need to wrap -- this bare call is the assertion that
    no such error is raised.
    """
    import polylogue.storage.sqlite.lifecycle as lifecycle
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER

    # Create synthetic declarations for v42→v43 fast-forward (FTS_REINDEX only).
    # Use v43's real operations from the canonical schema.
    v43_real = next(d for d in lifecycle.INDEX_DELTA_DECLARATIONS if d.version == 43)
    synthetic_decls = (
        IndexDeltaDeclaration(
            version=42,
            classes=(DerivedDeltaClass.CONSTRAINT_ONLY,),
        ),
        IndexDeltaDeclaration(
            version=43,
            classes=v43_real.classes,
            operations=v43_real.operations,
        ),
    )

    path = tmp_path / "index.db"
    _build_v42_shaped_index_db(path)

    # Monkeypatch to use synthetic declarations so v42→v43 is fast-forwardable.
    # This tests the executor without depending on real v44/v45 SEMANTIC_REPARSE.
    monkeypatch.setattr(
        lifecycle,
        "INDEX_DELTA_DECLARATIONS",
        synthetic_decls,
    )
    # Also patch ARCHIVE_VERSION_BY_TIER so the expected version is 43.
    monkeypatch.setitem(ARCHIVE_VERSION_BY_TIER, ArchiveTier.INDEX, 43)

    # ANTI-VACUITY: this bare call asserts no rebuild error is raised.
    initialize_archive_database(path, ArchiveTier.INDEX)

    conn = sqlite3.connect(path)
    try:
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 43
        # Verify data survived the fast-forward
        indexable_rows = conn.execute("SELECT COUNT(*) FROM blocks WHERE search_text != ''").fetchone()[0]
        assert indexable_rows == 2
    finally:
        conn.close()


def test_sql_fast_forwardable_index_db_reopen_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second open of an already fast-forwarded archive must not re-raise or re-mutate."""
    import polylogue.storage.sqlite.lifecycle as lifecycle
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER

    # Create synthetic declarations for v42→v43 fast-forward.
    v43_real = next(d for d in lifecycle.INDEX_DELTA_DECLARATIONS if d.version == 43)
    synthetic_decls = (
        IndexDeltaDeclaration(
            version=42,
            classes=(DerivedDeltaClass.CONSTRAINT_ONLY,),
        ),
        IndexDeltaDeclaration(
            version=43,
            classes=v43_real.classes,
            operations=v43_real.operations,
        ),
    )

    path = tmp_path / "index.db"
    _build_v42_shaped_index_db(path)

    # Monkeypatch to use synthetic declarations for consistent fast-forward.
    monkeypatch.setattr(
        lifecycle,
        "INDEX_DELTA_DECLARATIONS",
        synthetic_decls,
    )
    monkeypatch.setitem(ARCHIVE_VERSION_BY_TIER, ArchiveTier.INDEX, 43)

    initialize_archive_database(path, ArchiveTier.INDEX)
    initialize_archive_database(path, ArchiveTier.INDEX)

    conn = sqlite3.connect(path)
    try:
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 43
    finally:
        conn.close()


def test_semantic_reparse_gap_still_raises_rebuild_required(
    tmp_path: Path,
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
    # Test with a version where SEMANTIC_REPARSE will block the gap.
    # We use v43 as source (which is FTS_REINDEX-only, fastforwardable)
    # but set v44 as target (which is SEMANTIC_REPARSE, NOT fastforwardable).
    # However, our initialize_archive_database normalizes to current version,
    # so we test that v43 -> current fails due to the SEMANTIC_REPARSE at v44.
    source_version = 43

    path = tmp_path / "index.db"
    conn = sqlite3.connect(path)
    try:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        conn.execute(f"PRAGMA user_version = {source_version}")
        conn.commit()
    finally:
        conn.close()

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


def test_replace_table_sanitizes_poisoned_fts_freshness_ready_row(tmp_path: Path) -> None:
    """A pre-existing poisoned 'ready' row is downgraded, not rejected, by the v51 copy.

    polylogue-rlvj: before this fix, ``_mark_message_fts_ready_after_targeted_repair``
    could write ``state='ready'`` with ``source_rows != indexed_rows`` on the
    single global ``messages_fts`` freshness row -- the live incident this PR
    fixes measured a 12,659-row gap reported as ``ready``. Any archive that
    ran the old code before upgrading carries that exact poisoned row. The
    v51 declaration adds a CHECK enforcing the invariant the poisoned row
    violates; a naive shared-column copy into the new schema would raise
    ``sqlite3.IntegrityError`` and abort the whole fast-forward for those
    archives.

    ANTI-VACUITY: removing ``_REPLACE_TABLE_SANITIZERS``'s ``fts_freshness_state``
    entry (or the ``sanitizer is not None`` guard in ``_apply_replace_table``)
    makes this raise ``sqlite3.IntegrityError`` instead of completing --
    verified by temporarily deleting the entry and re-running, then restoring
    it.
    """
    from polylogue.storage.sqlite.archive_tiers.index_fast_forward_executor import _apply_replace_table
    from polylogue.storage.sqlite.lifecycle import resolve_canonical_index_objects

    # Resolved the same way the real executor does (sqlite_master.sql from a
    # scratch connection executing the canonical DDL), not the raw triple-
    # quoted constant -- the latter carries leading whitespace the
    # replace-table regex does not tolerate.
    canonical_sql = resolve_canonical_index_objects((("table", "fts_freshness_state"),))[
        ("table", "fts_freshness_state")
    ]

    path = tmp_path / "scratch.db"
    conn = sqlite3.connect(path)
    try:
        # Pre-v51 shape: no invariant CHECK, only the state-vocabulary CHECK.
        conn.execute(
            """
            CREATE TABLE fts_freshness_state (
                surface TEXT PRIMARY KEY,
                state TEXT NOT NULL CHECK (state IN ('ready', 'stale', 'unknown')),
                checked_at TEXT NOT NULL,
                source_rows INTEGER NOT NULL DEFAULT 0,
                indexed_rows INTEGER NOT NULL DEFAULT 0,
                missing_rows INTEGER NOT NULL DEFAULT 0,
                excess_rows INTEGER NOT NULL DEFAULT 0,
                duplicate_rows INTEGER NOT NULL DEFAULT 0,
                detail TEXT
            ) STRICT
            """
        )
        conn.execute(
            """
            INSERT INTO fts_freshness_state (
                surface, state, checked_at, source_rows, indexed_rows,
                missing_rows, excess_rows, duplicate_rows, detail
            ) VALUES (
                'messages_fts', 'ready', '2026-07-31T00:00:00+00:00',
                4970352, 4957693, 0, 0, 0, 'targeted changed-session repair complete'
            )
            """
        )
        conn.execute(
            """
            INSERT INTO fts_freshness_state (
                surface, state, checked_at, source_rows, indexed_rows,
                missing_rows, excess_rows, duplicate_rows, detail
            ) VALUES (
                'threads_fts', 'stale', '2026-07-31T00:00:00+00:00',
                15411, 15401, 10, 0, 0, 'exact invariant failed'
            )
            """
        )
        conn.commit()

        _apply_replace_table(conn, "fts_freshness_state", canonical_sql)
        conn.commit()

        rows = {
            str(row[0]): row
            for row in conn.execute(
                "SELECT surface, state, source_rows, indexed_rows, missing_rows FROM fts_freshness_state"
            ).fetchall()
        }
    finally:
        conn.close()

    # The poisoned row is downgraded to an honest 'stale' -- counts survive
    # (nothing measured is lost) but the false 'ready' claim does not.
    assert rows["messages_fts"][1] == "stale"
    assert rows["messages_fts"][2:] == (4970352, 4957693, 0)
    # An already-honest row is untouched.
    assert rows["threads_fts"][1] == "stale"
    assert rows["threads_fts"][2:] == (15411, 15401, 10)


def test_shape_forward_targeted_reprocess_enqueues_bounded_ops_debt(tmp_path: Path) -> None:
    """polylogue-9rw0.1: a SHAPE_FORWARD_TARGETED_REPROCESS fast-forward enqueues real debt.

    Builds a pre-v44 shaped ``sessions`` table (no ``title_ref``/
    ``title_confidence`` columns), seeds two codex-session sessions and one
    other-origin session, then fast-forwards through the production v44
    declaration. The DDL copy-forward alone would silently leave every new
    column NULL with no record that anything is owed; this asserts the
    executor also enqueues one ``convergence_debt`` row per *in-scope*
    session in the sibling ``ops.db`` -- exactly the origin-scoped population
    the declaration states as data, not the whole archive.

    ANTI-VACUITY: deleting the ``_enqueue_targeted_reprocess_debt`` call in
    ``apply_index_fast_forward`` makes the ``ops.db`` file never get created
    (or, if it already existed, its ``convergence_debt`` table stays empty)
    -- this test's final assertion on ``debt_rows`` would fail from 2 to 0.
    """
    from polylogue.storage.sqlite.archive_tiers.index_fast_forward_executor import apply_index_fast_forward
    from polylogue.storage.sqlite.lifecycle import index_fast_forward_plan

    index_path = tmp_path / "index.db"
    conn = sqlite3.connect(index_path)
    try:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        conn.execute('ALTER TABLE sessions DROP COLUMN "title_ref"')
        conn.execute('ALTER TABLE sessions DROP COLUMN "title_confidence"')
        conn.commit()

        _seed_indexable_block(conn, native_suffix="codex-a", text="codex prose one")
        _seed_indexable_block(conn, native_suffix="codex-b", text="codex prose two")
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, title, content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("native-session-other", "claude-code-session", "unrelated origin", _HASH, 1, 1),
        )
        conn.commit()
        codex_session_ids = {
            str(row[0])
            for row in conn.execute("SELECT session_id FROM sessions WHERE origin = 'codex-session'").fetchall()
        }
        assert len(codex_session_ids) == 2

        conn.execute("PRAGMA user_version = 43")
        conn.commit()
    finally:
        conn.close()

    conn = sqlite3.connect(index_path)
    try:
        plan = index_fast_forward_plan(43, 44)
        assert plan is not None
        apply_index_fast_forward(conn, plan)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 44
        # Shape landed; values are NOT backfilled by the DDL copy-forward alone.
        title_refs = {row[0] for row in conn.execute("SELECT title_ref FROM sessions").fetchall()}
        assert title_refs == {None}
    finally:
        conn.close()

    ops_path = tmp_path / "ops.db"
    assert ops_path.exists()
    ops_conn = sqlite3.connect(ops_path)
    try:
        debt_rows = ops_conn.execute(
            "SELECT target_id, status FROM convergence_debt WHERE stage = 'index-v44-targeted-reprocess'"
        ).fetchall()
    finally:
        ops_conn.close()

    assert {row[0] for row in debt_rows} == codex_session_ids
    assert all(row[1] == "deferred" for row in debt_rows)
