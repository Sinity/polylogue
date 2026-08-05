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


def test_v64_to_v65_fast_forward_replaces_actions_view_and_exposes_result_state(tmp_path: Path) -> None:
    """The production archive-open path upgrades v64 action queries in place."""

    path = tmp_path / "index.db"
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        conn.execute(
            """
            INSERT INTO sessions (native_id, origin, title, content_hash, created_at_ms, updated_at_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("v65-action-session", "codex-session", "v65 action fixture", _HASH, 1, 1),
        )
        session_id = conn.execute("SELECT session_id FROM sessions WHERE native_id = 'v65-action-session'").fetchone()[
            "session_id"
        ]
        conn.execute(
            """
            INSERT INTO messages (session_id, native_id, position, role, message_type, content_hash, occurred_at_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, "v65-action-message", 0, "assistant", "message", _HASH, 1),
        )
        message_id = conn.execute("SELECT message_id FROM messages WHERE session_id = ?", (session_id,)).fetchone()[
            "message_id"
        ]
        conn.execute(
            """
            INSERT INTO blocks (message_id, session_id, position, block_type, tool_name, tool_id, tool_input)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (message_id, session_id, 0, "tool_use", "shell", "v65-matched", '{"command": "matched"}'),
        )
        conn.execute(
            """
            INSERT INTO blocks (message_id, session_id, position, block_type, tool_name, tool_id, tool_input)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (message_id, session_id, 1, "tool_use", "shell", "v65-absent", '{"command": "absent"}'),
        )
        conn.execute(
            """
            INSERT INTO blocks (message_id, session_id, position, block_type, text, tool_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (message_id, session_id, 2, "tool_result", "unknown outcome", "v65-matched"),
        )
        conn.execute("DROP VIEW actions")
        conn.execute(
            """
            CREATE VIEW actions AS
            SELECT
                ap.session_id, ap.message_id, ap.tool_use_block_id, ap.tool_name, ap.semantic_type,
                ap.tool_command, ap.tool_path, tu.tool_input AS tool_input, tr.text AS output_text,
                ap.is_error, ap.exit_code, ap.tool_result_block_id
            FROM action_pairs ap
            JOIN blocks tu ON tu.block_id = ap.tool_use_block_id
            LEFT JOIN blocks tr ON tr.block_id = ap.tool_result_block_id
            """
        )
        conn.execute("PRAGMA user_version = 64")
        conn.commit()
    finally:
        conn.close()

    initialize_archive_database(path, ArchiveTier.INDEX)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT tool_command, tool_result_block_id, result_state FROM actions ORDER BY tool_command"
        ).fetchall()
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    finally:
        conn.close()

    assert version == 65
    assert [dict(row) for row in rows] == [
        {"tool_command": "absent", "tool_result_block_id": None, "result_state": "no_result"},
        {
            "tool_command": "matched",
            "tool_result_block_id": "codex-session:v65-action-session:v65-action-message:2",
            "result_state": "outcome_unknown",
        },
    ]


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


def test_v61_replace_table_drops_pricing_columns_and_keeps_the_rest(tmp_path: Path) -> None:
    """polylogue-resk: the v61 REPLACE_TABLE drops priced_with/priced_at_ms.

    Builds a pre-v61 shaped ``session_model_usage`` (the ``priced_with`` FK
    to ``price_catalogs``, ``priced_at_ms``, and the wider two-column CHECK),
    seeds a row that satisfies the OLD, stricter CHECK (``cost_provenance =
    'priced'`` requires both ``cost_usd`` and ``priced_with`` set), then
    replace-tables it onto the v61 canonical shape.

    ANTI-VACUITY: reverting ``INDEX_DDL``'s ``session_model_usage`` back to
    declaring ``priced_with``/``priced_at_ms`` (undoing the polylogue-resk
    DDL edit) makes the ``columns`` assertions below fail (both columns
    would still be present after the copy-forward).
    """
    from polylogue.storage.sqlite.archive_tiers.index_fast_forward_executor import _apply_replace_table
    from polylogue.storage.sqlite.lifecycle import resolve_canonical_index_objects

    canonical_sql = resolve_canonical_index_objects((("table", "session_model_usage"),))[
        ("table", "session_model_usage")
    ]

    path = tmp_path / "scratch.db"
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                native_id TEXT NOT NULL
            ) STRICT
            """
        )
        conn.execute("INSERT INTO sessions (session_id, native_id) VALUES ('codex-session:s1', 's1')")
        conn.execute(
            """
            CREATE TABLE price_catalogs (
                catalog_id TEXT PRIMARY KEY,
                catalog_hash TEXT NOT NULL,
                source_name TEXT NOT NULL,
                effective_at_ms INTEGER,
                loaded_at_ms INTEGER NOT NULL
            ) STRICT
            """
        )
        conn.execute(
            "INSERT INTO price_catalogs (catalog_id, catalog_hash, source_name, loaded_at_ms) "
            "VALUES ('legacy-catalog', 'legacy-hash', 'legacy', 0)"
        )
        # Pre-v61 shape: priced_with FK + priced_at_ms + the wider CHECK.
        conn.execute(
            """
            CREATE TABLE session_model_usage (
                session_id              TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                model_name              TEXT NOT NULL,
                input_tokens            INTEGER NOT NULL DEFAULT 0 CHECK(input_tokens >= 0),
                output_tokens           INTEGER NOT NULL DEFAULT 0 CHECK(output_tokens >= 0),
                cache_read_tokens       INTEGER NOT NULL DEFAULT 0 CHECK(cache_read_tokens >= 0),
                cache_write_tokens      INTEGER NOT NULL DEFAULT 0 CHECK(cache_write_tokens >= 0),
                message_count           INTEGER NOT NULL DEFAULT 0 CHECK(message_count >= 0),
                priced_with             TEXT REFERENCES price_catalogs(catalog_id) ON DELETE SET NULL,
                priced_at_ms            INTEGER,
                cost_usd                REAL,
                cost_credits            REAL,
                cost_provenance         TEXT CHECK(cost_provenance IN ('origin_reported', 'priced', 'estimated') OR cost_provenance IS NULL),
                CHECK (
                    cost_provenance != 'priced' OR (cost_usd IS NOT NULL AND priced_with IS NOT NULL)
                ),
                CHECK (cost_provenance != 'origin_reported' OR cost_usd IS NOT NULL),
                PRIMARY KEY(session_id, model_name)
            ) STRICT
            """
        )
        conn.execute(
            """
            INSERT INTO session_model_usage (
                session_id, model_name, input_tokens, output_tokens,
                message_count, priced_with, priced_at_ms, cost_usd, cost_provenance
            ) VALUES (
                'codex-session:s1', 'gpt-4o', 1000, 500,
                1, 'legacy-catalog', 1700000000000, 0.05, 'priced'
            )
            """
        )
        conn.commit()

        _apply_replace_table(conn, "session_model_usage", canonical_sql)
        conn.commit()

        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(session_model_usage)").fetchall()}
        row = conn.execute(
            "SELECT input_tokens, output_tokens, message_count, cost_usd, cost_provenance "
            "FROM session_model_usage WHERE model_name = 'gpt-4o'"
        ).fetchone()
    finally:
        conn.close()

    assert "priced_with" not in columns
    assert "priced_at_ms" not in columns
    assert row == (1000, 500, 1, 0.05, "priced")


def test_v63_drop_table_removes_threads_fts_and_its_triggers(tmp_path: Path) -> None:
    """polylogue-eizc: the v63 DROP_TABLE declaration drops threads_fts and
    its three triggers via the real ``apply_index_fast_forward`` plan
    machinery, not just the low-level ``_apply_operation`` dispatch.

    Builds a pre-v63 shaped archive (``threads`` + ``threads_fts`` +
    triggers all present, as a v61 archive would have), applies the plan
    covering just the real v63 declaration, and asserts both the table and
    all three triggers are gone afterward while an unrelated sibling
    surface (``blocks_command_trigram``, kept -- not part of this
    declaration) survives untouched.

    ANTI-VACUITY: reverting lifecycle.py's v63 declaration (or the
    executor's trigger-object DROP_TABLE handling in
    ``index_fast_forward_executor.py``) makes the ``threads_fts`` table
    survive `apply_index_fast_forward` -- the ``exists_after`` assertion
    below would fail.
    """
    import polylogue.storage.sqlite.lifecycle as lifecycle
    from polylogue.storage.sqlite.archive_tiers.index_fast_forward_executor import apply_index_fast_forward

    v63_real = next(d for d in lifecycle.INDEX_DELTA_DECLARATIONS if d.version == 63)
    plan = lifecycle.IndexFastForwardPlan(source_version=62, target_version=63, declarations=(v63_real,))

    path = tmp_path / "scratch.db"
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE threads (
                thread_id TEXT PRIMARY KEY,
                search_text TEXT NOT NULL DEFAULT ''
            ) STRICT;
            CREATE VIRTUAL TABLE threads_fts USING fts5(
                thread_id UNINDEXED, root_id UNINDEXED, text, tokenize='unicode61'
            );
            CREATE TRIGGER threads_fts_ai AFTER INSERT ON threads BEGIN
                INSERT INTO threads_fts (thread_id, root_id, text) VALUES (new.thread_id, new.thread_id, new.search_text);
            END;
            CREATE TRIGGER threads_fts_ad AFTER DELETE ON threads BEGIN
                DELETE FROM threads_fts WHERE thread_id = old.thread_id;
            END;
            CREATE TRIGGER threads_fts_au AFTER UPDATE ON threads BEGIN
                DELETE FROM threads_fts WHERE thread_id = old.thread_id;
                INSERT INTO threads_fts (thread_id, root_id, text) VALUES (new.thread_id, new.thread_id, new.search_text);
            END;
            CREATE TABLE blocks (rowid INTEGER PRIMARY KEY, tool_detail_text TEXT, block_type TEXT);
            CREATE VIRTUAL TABLE blocks_command_trigram USING fts5(
                tool_detail_text, tokenize='trigram', content='blocks', content_rowid='rowid'
            );
            """
        )
        conn.execute("INSERT INTO threads (thread_id, search_text) VALUES ('t1', 'needle')")
        conn.execute("PRAGMA user_version = 62")
        conn.commit()

        apply_index_fast_forward(conn, plan)

        surfaces = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'trigger')").fetchall()
        }
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    finally:
        conn.close()

    assert "threads_fts" not in surfaces
    assert "threads_fts_ai" not in surfaces
    assert "threads_fts_ad" not in surfaces
    assert "threads_fts_au" not in surfaces
    assert "threads" in surfaces
    assert "blocks_command_trigram" in surfaces
    assert version == 63


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
