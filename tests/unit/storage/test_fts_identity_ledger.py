"""Exact identity-ledger reconciliation for messages_fts (polylogue-1xc.12).

``messages_fts`` is a CONTENTLESS FTS5 table: its ``block_id`` UNINDEXED
column is write-only and never retrievable by a later ``SELECT`` (see
``storage/fts/sql.py``). SQLite reuses freed rowids -- deleting the
highest-rowid block then inserting a new one commonly gets the SAME rowid
back, exactly what a full-session-replace does -- so a bare rowid comparison
cannot prove which block a ``messages_fts`` row currently represents, and
count-only reconciliation (``source_rows == indexed_rows``) cannot see a
stale rowid that has silently rebound to a different block: both sides still
balance. These tests exercise the real block triggers (never a mock) and
prove the ``messages_fts_identity`` ledger + exact reconciliation actually
catch that class of drift, not merely that happy-path counts agree.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from polylogue.storage.fts.freshness import (
    READY,
    ensure_fts_freshness_table_sync,
    freshness_ready_record_trusted,
    record_fts_surface_state_sync,
)
from polylogue.storage.fts.fts_lifecycle import (
    fts_invariant_snapshot_sync,
    restore_fts_triggers_sync,
)
from polylogue.storage.fts.sql import FTS_MESSAGES_IDENTITY_RECIPE_ID, message_identity_mismatch_sql
from polylogue.storage.sqlite.archive_tiers.ops_write import list_fts_drift_samples, record_fts_drift_sample

if TYPE_CHECKING:
    from polylogue.sources.parsers.base import ParsedSession


def _seed_block(
    conn: sqlite3.Connection,
    *,
    native_session_id: str,
    native_message_id: str,
    text: str,
    content_hash: bytes = b"x" * 32,
    message_position: int = 0,
) -> str:
    """Insert one minimal session/message/block row and return the block_id.

    ``message_position`` must be distinct per message within a session
    (``messages`` is UNIQUE on ``(session_id, position, variant_index)``);
    the block itself is always at block position 0 within its own message.
    """
    origin = "unknown-export"
    session_id = f"{origin}:{native_session_id}"
    message_id = f"{session_id}:{native_message_id}"
    conn.execute(
        "INSERT OR IGNORE INTO sessions (native_id, origin, title, content_hash) VALUES (?, ?, ?, ?)",
        (native_session_id, origin, "Identity ledger test", content_hash),
    )
    conn.execute(
        """
        INSERT INTO messages (session_id, native_id, position, role, message_type, content_hash)
        VALUES (?, ?, ?, 'user', 'message', ?)
        """,
        (session_id, native_message_id, message_position, content_hash),
    )
    conn.execute(
        """
        INSERT INTO blocks (message_id, session_id, position, block_type, text, content_hash)
        VALUES (?, ?, 0, 'text', ?, ?)
        """,
        (message_id, session_id, text, content_hash),
    )
    return f"{message_id}:0"


def _block_rowid(conn: sqlite3.Connection, block_id: str) -> int:
    row = conn.execute("SELECT rowid FROM blocks WHERE block_id = ?", (block_id,)).fetchone()
    assert row is not None
    return int(row[0])


def _identity_row(conn: sqlite3.Connection, rowid: int) -> tuple[str, bytes | None, str] | None:
    row = conn.execute(
        "SELECT block_id, source_hash, recipe_id FROM messages_fts_identity WHERE rowid = ?",
        (rowid,),
    ).fetchone()
    return None if row is None else (str(row[0]), row[1], str(row[2]))


def _identity_mismatch_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute(message_identity_mismatch_sql()).fetchone()[0] or 0)


class TestIdentityLedgerHappyPath:
    """Real triggers correctly maintain the ledger, including rowid reuse."""

    def test_insert_populates_identity_ledger(self, test_conn: sqlite3.Connection) -> None:
        restore_fts_triggers_sync(test_conn)
        content_hash = b"a" * 32
        block_id = _seed_block(
            test_conn,
            native_session_id="conv-identity-insert",
            native_message_id="msg-identity-insert",
            text="hello identity ledger",
            content_hash=content_hash,
        )
        rowid = _block_rowid(test_conn, block_id)
        identity = _identity_row(test_conn, rowid)
        assert identity == (block_id, content_hash, FTS_MESSAGES_IDENTITY_RECIPE_ID)
        assert _identity_mismatch_count(test_conn) == 0

    def test_rowid_reuse_after_delete_rebinds_identity_to_new_block(self, test_conn: sqlite3.Connection) -> None:
        """The keystone case: a freed rowid must never keep the old block's identity.

        Mutation this proves fails without the trigger's identity DELETE+INSERT
        pair: if the ``messages_fts_ad``/``messages_fts_ai`` bodies stopped
        touching ``messages_fts_identity``, the ledger row for the reused
        rowid would still say ``block_id_a`` after block B was inserted at
        the same rowid, and this test's final assertion would fail.
        """
        restore_fts_triggers_sync(test_conn)
        block_id_a = _seed_block(
            test_conn,
            native_session_id="conv-identity-reuse",
            native_message_id="msg-identity-reuse-a",
            text="first block occupying the rowid",
            content_hash=b"a" * 32,
        )
        rowid = _block_rowid(test_conn, block_id_a)
        assert _identity_row(test_conn, rowid) is not None

        # Deleting the only (highest-rowid) block frees that exact rowid --
        # SQLite's default rowid allocator reuses it on the very next insert.
        test_conn.execute("DELETE FROM blocks WHERE block_id = ?", (block_id_a,))
        assert _identity_row(test_conn, rowid) is None

        block_id_b = _seed_block(
            test_conn,
            native_session_id="conv-identity-reuse",
            native_message_id="msg-identity-reuse-b",
            text="second block reusing the same rowid",
            content_hash=b"b" * 32,
            message_position=1,
        )
        reused_rowid = _block_rowid(test_conn, block_id_b)
        assert reused_rowid == rowid, "test setup expected SQLite to reuse the freed rowid"

        identity = _identity_row(test_conn, reused_rowid)
        assert identity is not None
        bound_block_id, bound_source_hash, _recipe = identity
        assert bound_block_id == block_id_b
        assert bound_block_id != block_id_a
        assert bound_source_hash == b"b" * 32
        assert _identity_mismatch_count(test_conn) == 0

        snapshot = fts_invariant_snapshot_sync(test_conn)
        assert snapshot.messages.identity_mismatch_rows == 0
        assert snapshot.messages.ready

    def test_text_change_refreshes_source_hash(self, test_conn: sqlite3.Connection) -> None:
        restore_fts_triggers_sync(test_conn)
        block_id = _seed_block(
            test_conn,
            native_session_id="conv-identity-textchange",
            native_message_id="msg-identity-textchange",
            text="original text",
            content_hash=b"c" * 32,
        )
        rowid = _block_rowid(test_conn, block_id)
        test_conn.execute(
            "UPDATE blocks SET text = ?, content_hash = ? WHERE block_id = ?",
            ("edited text", b"d" * 32, block_id),
        )
        identity = _identity_row(test_conn, rowid)
        assert identity is not None
        assert identity[1] == b"d" * 32
        assert _identity_mismatch_count(test_conn) == 0

    def test_empty_text_transition_removes_identity_row(self, test_conn: sqlite3.Connection) -> None:
        """Text going to empty must drop both messages_fts AND its identity row."""
        restore_fts_triggers_sync(test_conn)
        block_id = _seed_block(
            test_conn,
            native_session_id="conv-identity-emptytransition",
            native_message_id="msg-identity-emptytransition",
            text="will become empty",
        )
        rowid = _block_rowid(test_conn, block_id)
        assert _identity_row(test_conn, rowid) is not None

        test_conn.execute("UPDATE blocks SET text = NULL WHERE block_id = ?", (block_id,))
        assert _identity_row(test_conn, rowid) is None
        docsize_row = test_conn.execute("SELECT 1 FROM messages_fts_docsize WHERE id = ?", (rowid,)).fetchone()
        assert docsize_row is None
        assert _identity_mismatch_count(test_conn) == 0


class TestIdentityLedgerBlockIdCollisionRepro:
    """polylogue-miwv: an orphaned ledger row must never abort a later write.

    ``messages_fts_identity`` carries TWO independent uniqueness constraints
    -- the ``rowid`` INTEGER PRIMARY KEY and the ``block_id TEXT ... UNIQUE``
    column -- but the ``messages_fts_ad``/``au`` trigger arms and every bulk
    delete companion in ``storage/fts/sql.py`` only ever clean up by rowid.
    If ANY path (an interrupted bulk-guarded delete, a future bug, a
    partial/rolled-back attempt) leaves a stale ``(old_rowid, block_id=X)``
    ledger row behind, a later write that computes that same ``block_id`` at
    a *different* rowid used to hit ``UNIQUE constraint failed:
    messages_fts_identity.block_id`` and abort outright -- reported as a
    live pre-existing regression alongside this bead's other work. Since the
    exact write-sequence trigger for that orphan in production code was not
    reproducible against the two originally-flagged test files
    (``tests/unit/storage/test_revision_replay.py``,
    ``tests/unit/sources/test_live_batch_support.py`` -- both pass cleanly,
    repeatedly, serially and under ``-n auto`` xdist, against both this
    branch and a fresh ``origin/master`` checkout), this test proves the
    INVARIANT directly against the real trigger DDL: construct a genuine
    orphaned ledger row (a delete that ran with the AD trigger structurally
    absent, the same shape a bulk-guarded delete leaves if a companion
    cleanup is ever missed), then insert a fresh block that computes the
    same ``block_id`` at a new rowid through the real ``messages_fts_ai``
    trigger. Anti-vacuity: reverting the ``INSERT OR REPLACE`` fix in
    ``BLOCKS_FTS_TRIGGER_DDL`` (back to a bare ``INSERT``) makes this raise
    ``sqlite3.IntegrityError`` again.
    """

    def test_orphaned_ledger_row_does_not_abort_a_colliding_insert(self, test_conn: sqlite3.Connection) -> None:
        restore_fts_triggers_sync(test_conn)
        block_id = _seed_block(
            test_conn,
            native_session_id="conv-identity-collision",
            native_message_id="msg-identity-collision",
            text="original occupant of this block_id",
            content_hash=b"a" * 32,
        )
        rowid = _block_rowid(test_conn, block_id)
        assert _identity_row(test_conn, rowid) == (block_id, b"a" * 32, FTS_MESSAGES_IDENTITY_RECIPE_ID)

        # Construct a genuine orphan: delete the block with the AD trigger
        # structurally absent (the same shape a bulk-guarded delete leaves
        # behind if its companion identity cleanup is ever skipped), so the
        # ledger row survives even though its block is gone.
        test_conn.execute("DROP TRIGGER messages_fts_ad")
        test_conn.execute("DELETE FROM blocks WHERE block_id = ?", (block_id,))
        assert _identity_row(test_conn, rowid) is not None, "test setup must leave a real orphaned ledger row"
        restore_fts_triggers_sync(test_conn)

        # Re-insert a block computing the SAME block_id (same message_id +
        # position) forced to a different rowid -- reproduces the exact
        # bug shape via the real messages_fts_ai trigger, not a toy.
        new_rowid = rowid + 1000
        test_conn.execute(
            """
            INSERT INTO blocks (rowid, message_id, session_id, position, block_type, text, content_hash)
            VALUES (?, ?, ?, 0, 'text', ?, ?)
            """,
            (
                new_rowid,
                f"{block_id.rsplit(':', 1)[0]}",
                "unknown-export:conv-identity-collision",
                "new occupant of this block_id",
                b"b" * 32,
            ),
        )

        identity = _identity_row(test_conn, new_rowid)
        assert identity == (block_id, b"b" * 32, FTS_MESSAGES_IDENTITY_RECIPE_ID)
        assert _identity_row(test_conn, rowid) is None, "the stale orphan holder must be evicted, not duplicated"
        assert _identity_mismatch_count(test_conn) == 0


class TestIdentityMismatchDetection:
    """Anti-vacuity: prove the exact check actually flags corruption, not just 0/0."""

    def test_corrupted_ledger_row_is_detected_as_mismatch(self, test_conn: sqlite3.Connection) -> None:
        """Simulate the historical bug directly: hand-write a stale identity row.

        This bypasses the trigger entirely to reproduce exactly what a
        missing/broken identity trigger arm would leave behind -- a
        ``messages_fts_identity`` row whose ``block_id`` does not match the
        block currently bound to that rowid. Count-only reconciliation
        (source_rows == indexed_rows) would report this archive as fully
        healthy; the identity check must not.
        """
        restore_fts_triggers_sync(test_conn)
        block_id = _seed_block(
            test_conn,
            native_session_id="conv-identity-corrupt",
            native_message_id="msg-identity-corrupt",
            text="genuine current block",
            content_hash=b"e" * 32,
        )
        rowid = _block_rowid(test_conn, block_id)
        assert _identity_mismatch_count(test_conn) == 0

        # Hand-corrupt the ledger to point at a block_id that never existed
        # at this rowid -- the exact rowid-reuse-gone-wrong shape.
        test_conn.execute(
            "UPDATE messages_fts_identity SET block_id = 'stale:ghost:0' WHERE rowid = ?",
            (rowid,),
        )
        assert _identity_mismatch_count(test_conn) == 1

        snapshot = fts_invariant_snapshot_sync(test_conn)
        assert snapshot.messages.identity_mismatch_rows == 1
        assert not snapshot.messages.ready

    def test_stale_source_hash_is_detected_as_mismatch(self, test_conn: sqlite3.Connection) -> None:
        """A block's content_hash moved on but the ledger row didn't -- caught."""
        restore_fts_triggers_sync(test_conn)
        block_id = _seed_block(
            test_conn,
            native_session_id="conv-identity-stalehash",
            native_message_id="msg-identity-stalehash",
            text="content that will diverge from its ledgered hash",
            content_hash=b"f" * 32,
        )
        rowid = _block_rowid(test_conn, block_id)
        assert _identity_mismatch_count(test_conn) == 0

        test_conn.execute(
            "UPDATE messages_fts_identity SET source_hash = ? WHERE rowid = ?",
            (b"0" * 32, rowid),
        )
        assert _identity_mismatch_count(test_conn) == 1

    def test_stale_recipe_id_is_detected_as_mismatch(self, test_conn: sqlite3.Connection) -> None:
        """An archive that never rebuilt after a tokenizer/fold recipe bump."""
        restore_fts_triggers_sync(test_conn)
        block_id = _seed_block(
            test_conn,
            native_session_id="conv-identity-stalerecipe",
            native_message_id="msg-identity-stalerecipe",
            text="indexed under an old recipe",
        )
        rowid = _block_rowid(test_conn, block_id)
        assert _identity_mismatch_count(test_conn) == 0

        test_conn.execute(
            "UPDATE messages_fts_identity SET recipe_id = 'messages_fts.v0:legacy' WHERE rowid = ?",
            (rowid,),
        )
        assert _identity_mismatch_count(test_conn) == 1

    def test_orphan_identity_row_without_docsize_is_detected(self, test_conn: sqlite3.Connection) -> None:
        """An identity row surviving after its messages_fts row vanished."""
        restore_fts_triggers_sync(test_conn)
        _seed_block(
            test_conn,
            native_session_id="conv-identity-orphan",
            native_message_id="msg-identity-orphan",
            text="will be deleted from messages_fts only",
        )
        block_id = "unknown-export:conv-identity-orphan:msg-identity-orphan:0"
        rowid = _block_rowid(test_conn, block_id)
        assert _identity_mismatch_count(test_conn) == 0

        # Remove only the FTS row (as if a partial/interrupted repair left
        # the identity ledger behind) -- never do this outside a test.
        test_conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (rowid,))
        assert _identity_mismatch_count(test_conn) == 1

    def test_missing_identity_row_is_not_counted_as_mismatch(self, test_conn: sqlite3.Connection) -> None:
        """A coverage GAP is not a CONFLICT -- this is the design boundary that
        keeps ``storage/sqlite/archive_tiers/write.py``'s non-bulk
        full-session-replace fast path (which does not populate the identity
        ledger inline, see the polylogue-1xc.12 STOP-and-report note) from
        making ``ready`` permanently false on ordinary archives. Simulates
        that exact gap directly: a docsize-indexed rowid with NO identity
        row at all must not count, while a PRESENT-but-wrong row (proven
        elsewhere in this class) must.
        """
        restore_fts_triggers_sync(test_conn)
        block_id = _seed_block(
            test_conn,
            native_session_id="conv-identity-coverage-gap",
            native_message_id="msg-identity-coverage-gap",
            text="indexed via messages_fts but never ledgered",
        )
        rowid = _block_rowid(test_conn, block_id)
        assert _identity_row(test_conn, rowid) is not None

        test_conn.execute("DELETE FROM messages_fts_identity WHERE rowid = ?", (rowid,))
        assert _identity_row(test_conn, rowid) is None
        docsize_row = test_conn.execute("SELECT 1 FROM messages_fts_docsize WHERE id = ?", (rowid,)).fetchone()
        assert docsize_row is not None, "messages_fts row must survive -- only its ledger entry was removed"

        assert _identity_mismatch_count(test_conn) == 0


class TestIdentityMismatchGatesReadiness:
    """Wired into the same readiness contract missing_rows/excess_rows use."""

    def test_freshness_ready_record_trusted_rejects_nonzero_identity_mismatch(self) -> None:
        assert not freshness_ready_record_trusted(
            state=READY,
            source_rows=10,
            indexed_rows=10,
            missing_rows=0,
            excess_rows=0,
            duplicate_rows=0,
            identity_mismatch_rows=1,
            source_has_rows=True,
        )

    def test_freshness_ready_record_trusted_defaults_identity_mismatch_to_zero(self) -> None:
        """Backward compatibility: callers that never computed it still pass."""
        assert freshness_ready_record_trusted(
            state=READY,
            source_rows=10,
            indexed_rows=10,
            missing_rows=0,
            excess_rows=0,
            duplicate_rows=0,
            source_has_rows=True,
        )

    def test_record_fts_surface_state_round_trips_identity_mismatch_column(self, test_conn: sqlite3.Connection) -> None:
        ensure_fts_freshness_table_sync(test_conn)
        record_fts_surface_state_sync(
            test_conn,
            surface="messages_fts",
            state=READY,
            source_rows=5,
            indexed_rows=5,
            identity_mismatch_rows=2,
        )
        row = test_conn.execute(
            "SELECT identity_mismatch_rows FROM fts_freshness_state WHERE surface = 'messages_fts'"
        ).fetchone()
        assert row is not None
        assert int(row[0]) == 2

    def test_upgraded_freshness_table_defaults_identity_mismatch_to_zero(self, test_conn: sqlite3.Connection) -> None:
        """A row written before this column existed reads back as 0, not NULL."""
        test_conn.execute("DROP TABLE IF EXISTS fts_freshness_state")
        test_conn.execute(
            """
            CREATE TABLE fts_freshness_state (
                surface TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                checked_at TEXT NOT NULL
            )
            """
        )
        test_conn.execute(
            "INSERT INTO fts_freshness_state (surface, state, checked_at) VALUES ('messages_fts', 'ready', 'x')"
        )
        ensure_fts_freshness_table_sync(test_conn)
        row = test_conn.execute(
            "SELECT identity_mismatch_rows FROM fts_freshness_state WHERE surface = 'messages_fts'"
        ).fetchone()
        assert row is not None
        assert int(row[0]) == 0


def _missing_identity_entry_count(conn: sqlite3.Connection) -> int:
    """Count indexed rows with NO ``messages_fts_identity`` entry at all.

    Deliberately separate from :func:`_identity_mismatch_count`
    (``message_identity_mismatch_sql``): that check is documented to NOT
    count a coverage gap (an indexed row with no ledger entry), only a
    PRESENT-but-WRONG one. This helper counts exactly the gap class, which is
    what the write-path companion-call fix (polylogue-miwv) closes for the
    non-bulk full-session-replace fast path.
    """
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM messages_fts_docsize AS d
        LEFT JOIN messages_fts_identity AS i ON i.rowid = d.id
        WHERE i.rowid IS NULL
        """
    ).fetchone()
    return int(row[0] or 0)


class TestWritePathIdentityCompanions:
    """polylogue-miwv: write.py's non-bulk full-session-replace fast path.

    ``_replace_full_session_messages_and_blocks`` bypasses the per-row
    ``messages_fts_{ai,ad,au}`` triggers for a scoped delete+reinsert (see its
    docstring) and previously called only ``delete_session_rows_sql``/
    ``insert_session_rows_sql`` -- the two ``messages_fts`` bulk helpers --
    without their ``messages_fts_identity`` companions. That left every
    session written through the dominant real-world path (``write_parsed_
    session_to_archive`` with ``merge_append=False``, the default) coverage-
    incomplete: ``messages_fts_docsize`` had a row but ``messages_fts_identity``
    never did, until the next repair pass backfilled it.

    These tests exercise the REAL writer entry point end-to-end (never a
    toy/mock of the SQL), proving zero missing ledger entries survive a
    full-session-replace, in addition to the mismatch-only invariant
    ``message_identity_mismatch_sql`` already covers.
    """

    def _parsed_session(self, native_id: str, *texts: str, start_index: int = 0) -> ParsedSession:
        from polylogue.archive.message.roles import Role
        from polylogue.core.enums import BlockType, Provider
        from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession

        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=native_id,
            title="Write-path identity companion test",
            messages=[
                ParsedMessage(
                    provider_message_id=f"m{start_index + i}",
                    role=Role.USER,
                    text=text,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
                )
                for i, text in enumerate(texts)
            ],
        )

    def test_full_session_replace_leaves_zero_missing_identity_entries(self, test_conn: sqlite3.Connection) -> None:
        from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive

        session_v1 = self._parsed_session(
            "write-path-identity-companion",
            "first version needle alpha",
            "second message needle beta",
        )
        write_parsed_session_to_archive(test_conn, session_v1)
        test_conn.commit()

        # Sanity: the first write already indexed both blocks.
        assert test_conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 2
        assert _identity_mismatch_count(test_conn) == 0
        assert _missing_identity_entry_count(test_conn) == 0

        # A genuine full-session-replace: same provider_session_id, entirely
        # different message content, so the content hash differs and this is
        # not skipped as an idempotent re-import. merge_append defaults to
        # False, so this goes through ``_replace_full_session_messages_and_
        # blocks``'s scoped delete+reinsert fast path (triggers are present
        # on a freshly initialized schema, so ``use_scoped_fts_rebuild`` is
        # True -- exactly the site this bead's fix targets).
        session_v2 = self._parsed_session(
            "write-path-identity-companion",
            "replaced content needle gamma",
        )
        write_parsed_session_to_archive(test_conn, session_v2, force_replace=True)
        test_conn.commit()

        assert test_conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 1
        assert _identity_mismatch_count(test_conn) == 0
        assert _missing_identity_entry_count(test_conn) == 0, (
            "full-session-replace left a messages_fts_docsize row with no "
            "messages_fts_identity companion -- the write.py fast-path pairing "
            "(polylogue-miwv) regressed"
        )

    def test_merge_append_path_also_leaves_zero_missing_identity_entries(self, test_conn: sqlite3.Connection) -> None:
        """The merge-append path writes blocks through the ordinary trigger
        path (no scoped delete/reinsert), so it was never the gap this bead
        closes -- pinned here as a control so a future regression in the
        *other* write path is caught too."""
        from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive

        session = self._parsed_session("write-path-identity-merge-append", "base message needle delta")
        write_parsed_session_to_archive(test_conn, session)
        test_conn.commit()

        appended = self._parsed_session(
            "write-path-identity-merge-append", "appended message needle epsilon", start_index=1
        )
        write_parsed_session_to_archive(test_conn, appended, merge_append=True)
        test_conn.commit()

        assert test_conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 2
        assert _identity_mismatch_count(test_conn) == 0
        assert _missing_identity_entry_count(test_conn) == 0


class TestFtsDriftSamples:
    """Bounded ops.db drift-magnitude history (polylogue-1xc.12)."""

    def test_record_and_list_round_trip(self, tmp_path: object) -> None:
        import sqlite3 as _sqlite3

        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        ops_path = tmp_path / "ops.db"  # type: ignore[operator]
        conn = _sqlite3.connect(str(ops_path))
        try:
            initialize_archive_tier(conn, ArchiveTier.OPS)
            record_fts_drift_sample(
                conn,
                surface="messages_fts",
                state="ready",
                source_rows=100,
                indexed_rows=100,
                missing_rows=0,
                excess_rows=0,
                duplicate_rows=0,
                identity_mismatch_rows=0,
                sampled_at_ms=1_000,
            )
            record_fts_drift_sample(
                conn,
                surface="messages_fts",
                state="stale",
                source_rows=100,
                indexed_rows=97,
                missing_rows=3,
                excess_rows=0,
                duplicate_rows=0,
                identity_mismatch_rows=1,
                sampled_at_ms=2_000,
            )
            samples = list_fts_drift_samples(conn, surface="messages_fts")
            assert len(samples) == 2
            newest = samples[0]
            assert newest.sampled_at_ms == 2_000
            assert newest.missing_rows == 3
            assert newest.identity_mismatch_rows == 1
        finally:
            conn.close()

    def test_retention_prunes_old_samples(self, tmp_path: object) -> None:
        import sqlite3 as _sqlite3

        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
        from polylogue.storage.sqlite.archive_tiers.ops_write import FTS_DRIFT_SAMPLE_RETENTION_MS
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        ops_path = tmp_path / "ops.db"  # type: ignore[operator]
        conn = _sqlite3.connect(str(ops_path))
        try:
            initialize_archive_tier(conn, ArchiveTier.OPS)
            record_fts_drift_sample(
                conn,
                surface="messages_fts",
                state="ready",
                source_rows=1,
                indexed_rows=1,
                missing_rows=0,
                excess_rows=0,
                duplicate_rows=0,
                identity_mismatch_rows=0,
                sampled_at_ms=0,
            )
            # A sample recorded well beyond the retention window must prune
            # the ancient row on the next write -- this is the mutation that
            # fails without the DELETE ... WHERE sampled_at_ms < ? pruning
            # statement in record_fts_drift_sample.
            record_fts_drift_sample(
                conn,
                surface="messages_fts",
                state="ready",
                source_rows=1,
                indexed_rows=1,
                missing_rows=0,
                excess_rows=0,
                duplicate_rows=0,
                identity_mismatch_rows=0,
                sampled_at_ms=FTS_DRIFT_SAMPLE_RETENTION_MS * 2,
            )
            samples = list_fts_drift_samples(conn, surface="messages_fts", limit=100)
            assert len(samples) == 1
            assert samples[0].sampled_at_ms == FTS_DRIFT_SAMPLE_RETENTION_MS * 2
        finally:
            conn.close()

    def test_drift_sample_writer_never_stores_negative_counts(self, tmp_path: object) -> None:
        import sqlite3 as _sqlite3

        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        ops_path = tmp_path / "ops.db"  # type: ignore[operator]
        conn = _sqlite3.connect(str(ops_path))
        try:
            initialize_archive_tier(conn, ArchiveTier.OPS)
            record_fts_drift_sample(
                conn,
                surface="messages_fts",
                state="ready",
                source_rows=-5,
                indexed_rows=-1,
                missing_rows=-1,
                excess_rows=-1,
                duplicate_rows=-1,
                identity_mismatch_rows=-1,
                sampled_at_ms=1,
            )
            sample = list_fts_drift_samples(conn, surface="messages_fts")[0]
            assert sample.source_rows == 0
            assert sample.identity_mismatch_rows == 0
        finally:
            conn.close()
