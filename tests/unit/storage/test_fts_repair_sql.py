"""Targeted SQL contracts for incremental FTS repair."""

from __future__ import annotations

import re
import sqlite3
from builtins import BaseExceptionGroup
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from polylogue.storage.fts import fts_lifecycle
from polylogue.storage.fts.derivation import (
    GLOBAL_PARTITION,
    FtsDerivationAdapter,
    converge_fts_partition_sync,
    session_partition_is_valid_sync,
)
from polylogue.storage.fts.fts_lifecycle import (
    FTS_TRIGGER_NAMES,
    insert_missing_message_rows_batched_sync,
    rebuild_fts_index_sync,
    repair_message_fts_index_sync,
    replace_fts_triggers_sync,
    reset_message_fts_index_sync,
    restore_fts_triggers_sync,
)
from polylogue.storage.fts.sql import (
    delete_session_rows_sql,
    insert_missing_message_rows_range_sql,
    insert_session_rows_sql,
)
from polylogue.storage.io_phase_metrics import close_connection_cursor, live_connection_cursors
from tests.infra.identity import archive_message_id, fixture_block_content_identity
from tests.infra.sqlite_cursor_settlement import ControlledCursor


@pytest.fixture
def test_conn(test_db: Path) -> Iterator[sqlite3.Connection]:
    """Settle this module's caller-owned transaction before the write lease closes.

    The cached connection refuses to close an open transaction; these laws
    write through the fixture connection and commit as their caller would.
    """
    from polylogue.storage.sqlite.connection import open_connection

    with open_connection(test_db) as conn:
        yield conn
        conn.commit()


def _seed_text_block(conn: sqlite3.Connection, *, native_session_id: str, native_message_id: str, text: str) -> str:
    origin = "unknown-export"
    session_id = f"{origin}:{native_session_id}"
    message_id = archive_message_id(session_id, native_message_id)
    content_hash = b"x" * 32
    conn.execute(
        "INSERT INTO sessions (native_id, origin, title, content_hash) VALUES (?, ?, ?, ?)",
        (native_session_id, origin, "Message repair", content_hash),
    )
    conn.execute(
        """
        INSERT INTO messages (session_id, native_id, position, role, message_type, content_hash)
        VALUES (?, ?, 0, 'user', 'message', ?)
        """,
        (session_id, native_message_id, content_hash),
    )
    conn.execute(
        "INSERT INTO blocks (message_id, session_id, position, block_type, text, content_identity, content_occurrence) VALUES (?, ?, 0, 'text', ?, ?, 0)",
        (
            message_id,
            session_id,
            text,
            fixture_block_content_identity("text", text),
        ),
    )
    return message_id


def test_incremental_fts_repair_deletes_via_block_rowid(test_conn: sqlite3.Connection) -> None:
    """Incremental FTS repair is keyed by canonical block rowids."""
    message_delete_sql = " ".join(delete_session_rows_sql(1).split())

    assert "DELETE FROM messages_fts WHERE rowid IN" in message_delete_sql
    assert "DELETE FROM messages_fts WHERE session_id" not in message_delete_sql
    assert "FROM blocks" in message_delete_sql
    message_insert_sql = " ".join(insert_session_rows_sql(1).split())
    assert "SELECT DISTINCT session_id FROM raw_target_sessions" in message_insert_sql
    assert "INSERT INTO messages_fts (rowid, text)" in message_insert_sql

    plan = "\n".join(
        row[3]
        for row in test_conn.execute(
            f"EXPLAIN QUERY PLAN {delete_session_rows_sql(1)}",
            ("test:conv1",),
        )
    )
    assert "SEARCH blocks USING" in plan


def test_incremental_fts_repair_uses_direct_fts_rowid_deletes(test_conn: sqlite3.Connection) -> None:
    """A changed session must not make FTS5 scan the whole archive to delete.

    Anti-vacuity: widen any FTS delete back to a ``session_id`` predicate, an
    unfiltered ``DELETE FROM messages_fts``, or a rowid set that reaches past
    the repaired session's own blocks, and this goes red.
    """
    restore_fts_triggers_sync(test_conn)
    message_id = _seed_text_block(
        test_conn,
        native_session_id="conv-message-repair-direct-delete",
        native_message_id="msg-message-repair-direct-delete",
        text="direct FTS rowid delete needle",
    )
    session_id = "unknown-export:conv-message-repair-direct-delete"

    traced: list[str] = []
    test_conn.set_trace_callback(traced.append)
    try:
        repair_message_fts_index_sync(test_conn, [session_id])
    finally:
        test_conn.set_trace_callback(None)

    session_rowids = {
        int(row[0])
        for row in test_conn.execute(
            "SELECT rowid FROM blocks WHERE message_id = ?",
            (message_id,),
        )
    }
    assert session_rowids, "the fixture must seed at least one block to repair"

    delete_statements = [sql for sql in traced if sql.startswith("DELETE FROM messages_fts")]
    assert delete_statements, "targeted repair issued no FTS delete at all"

    # Every FTS delete a session repair issues must name the exact block rowids
    # of that session. A literal rowid predicate is what keeps FTS5 from
    # scanning the whole archive to find the rows it is about to drop.
    for sql in delete_statements:
        predicate = sql.partition(" WHERE ")[2]
        assert predicate, f"unfiltered FTS delete would drop the whole surface: {sql!r}"
        assert predicate.startswith("rowid"), f"FTS delete is not keyed by rowid: {sql!r}"
        named = {int(token) for token in re.findall(r"\d+", predicate)}
        assert named == session_rowids, f"FTS delete reached beyond the repaired session: {sql!r}"


def test_targeted_repair_never_runs_an_archive_wide_exact_snapshot(
    test_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session repair never inspects or certifies an unrelated global surface."""
    import polylogue.storage.fts.fts_lifecycle as lifecycle

    restore_fts_triggers_sync(test_conn)
    _seed_text_block(
        test_conn,
        native_session_id="conv-no-duplicate-exact",
        native_message_id="msg-no-duplicate-exact",
        text="bounded target",
    )
    monkeypatch.setattr(
        lifecycle,
        "fts_invariant_snapshot_sync",
        lambda _conn: (_ for _ in ()).throw(AssertionError("targeted repair attempted an exact snapshot")),
    )

    repair_message_fts_index_sync(test_conn, ["unknown-export:conv-no-duplicate-exact"])


def test_global_missing_fts_sql_is_rowid_bounded() -> None:
    sql = " ".join(insert_missing_message_rows_range_sql().split())

    assert "AND b.rowid > ?" in sql
    assert "AND b.rowid <= ?" in sql
    assert "LEFT JOIN messages_fts_docsize" in sql


def test_missing_fts_repair_commits_and_checkpoints_batches(test_conn: sqlite3.Connection) -> None:
    restore_fts_triggers_sync(test_conn)
    for index in range(3):
        _seed_text_block(
            test_conn,
            native_session_id=f"conv-batched-missing-{index}",
            native_message_id=f"msg-batched-missing-{index}",
            text=f"batched missing needle {index}",
        )
    test_conn.execute("DELETE FROM messages_fts")

    traced: list[str] = []
    test_conn.set_trace_callback(traced.append)
    try:
        inserted = insert_missing_message_rows_batched_sync(test_conn, batch_rows=1)
    finally:
        test_conn.set_trace_callback(None)

    assert inserted == 3
    assert test_conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 3
    # Repair commits per batch and checkpoints never: the daemon's recurring
    # coordinator is the process' only ordinary checkpoint owner, and a
    # checkpoint reintroduced here would run inside this hold.
    assert [sql for sql in traced if "wal_checkpoint" in sql] == []
    assert len([sql for sql in traced if sql.strip().upper() == "COMMIT"]) == 3


def test_bulk_fts_rebuild_resumes_from_committed_missing_rows(test_conn: sqlite3.Connection) -> None:
    """The bulk-generation mode keeps already committed FTS rows on retry."""
    restore_fts_triggers_sync(test_conn)
    for index in range(3):
        _seed_text_block(
            test_conn,
            native_session_id=f"conv-bulk-resume-{index}",
            native_message_id=f"msg-bulk-resume-{index}",
            text=f"bulk resume needle {index}",
        )
    test_conn.execute("DELETE FROM messages_fts")
    test_conn.execute("DELETE FROM messages_fts_identity")

    inserted = insert_missing_message_rows_batched_sync(test_conn, batch_rows=1)
    assert inserted == 3
    identity_before = test_conn.execute("SELECT COUNT(*) FROM messages_fts_identity").fetchone()[0]

    rebuild_fts_index_sync(test_conn, resume_from_empty_message_index=True)

    source_rows = test_conn.execute("SELECT COUNT(*) FROM blocks WHERE search_text != ''").fetchone()[0]
    assert test_conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == source_rows
    assert test_conn.execute("SELECT COUNT(*) FROM messages_fts_identity").fetchone()[0] == source_rows
    assert identity_before == source_rows


def test_message_fts_repair_dedupes_duplicate_session_ids(test_conn: sqlite3.Connection) -> None:
    restore_fts_triggers_sync(test_conn)
    message_id = _seed_text_block(
        test_conn,
        native_session_id="conv-message-repair-dupe",
        native_message_id="msg-message-repair-dupe",
        text="repair duplicate needle",
    )
    session_id = "unknown-export:conv-message-repair-dupe"

    repair_message_fts_index_sync(
        test_conn,
        [
            session_id,
            session_id,
        ],
    )

    row = test_conn.execute(
        """
        SELECT COUNT(*)
        FROM messages_fts_docsize
        WHERE id = (SELECT rowid FROM blocks WHERE message_id = ?)
        """,
        (message_id,),
    ).fetchone()
    assert row[0] == 1


def test_message_fts_repair_leaves_a_directly_valid_partition(test_conn: sqlite3.Connection) -> None:
    restore_fts_triggers_sync(test_conn)
    _seed_text_block(
        test_conn,
        native_session_id="conv-message-repair-freshness",
        native_message_id="msg-message-repair-freshness",
        text="partition repair needle",
    )
    session_id = "unknown-export:conv-message-repair-freshness"
    repair_message_fts_index_sync(test_conn, [session_id])

    assert FtsDerivationAdapter().inspect_partition(test_conn, session_id).valid


def test_message_fts_trigger_rowids_track_block_rowids(test_conn: sqlite3.Connection) -> None:
    """Message FTS triggers use block rowids so rowid deletes are targeted."""
    restore_fts_triggers_sync(test_conn)
    message_id = _seed_text_block(
        test_conn,
        native_session_id="conv-action-rowid",
        native_message_id="msg-action-rowid",
        text="Ran command",
    )

    # ``messages_fts`` is a contentless FTS5 table (content=''), so its stored
    # columns are not retrievable via plain SELECT — only the rowid and MATCH
    # are. The trigger keys each FTS row by the block rowid; prove the tracking
    # by matching the indexed text and comparing the matched FTS rowid to the
    # canonical block rowid.
    block_rowid = test_conn.execute(
        "SELECT rowid FROM blocks WHERE message_id = ?",
        (message_id,),
    ).fetchone()["rowid"]
    fts_rowid = test_conn.execute(
        "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?",
        ("command",),
    ).fetchone()["rowid"]
    assert fts_rowid == block_rowid

    test_conn.execute("DELETE FROM blocks WHERE message_id = ?", (message_id,))
    remaining = test_conn.execute(
        "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?",
        ("command",),
    ).fetchone()[0]
    assert remaining == 0


def test_message_fts_reset_drops_orphan_docsize_rows(test_conn: sqlite3.Connection) -> None:
    restore_fts_triggers_sync(test_conn)
    message_id = _seed_text_block(
        test_conn,
        native_session_id="conv-reset-orphan",
        native_message_id="msg-reset-orphan",
        text="reset orphan needle",
    )
    block_rowid = test_conn.execute(
        "SELECT rowid FROM blocks WHERE message_id = ?",
        (message_id,),
    ).fetchone()["rowid"]
    rebuild_fts_index_sync(test_conn)
    test_conn.execute("DROP TRIGGER messages_fts_ad")
    test_conn.execute("DELETE FROM blocks WHERE rowid = ?", (block_rowid,))

    assert test_conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 1

    reset_message_fts_index_sync(test_conn)

    assert test_conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 0
    assert FtsDerivationAdapter().inspect_partition(test_conn, GLOBAL_PARTITION).valid


def _dropped_trigger_statements(conn: sqlite3.Connection, call: Callable[[sqlite3.Connection], None]) -> list[str]:
    traced: list[str] = []
    conn.set_trace_callback(traced.append)
    try:
        call(conn)
    finally:
        conn.set_trace_callback(None)
    return [stmt for stmt in traced if "DROP TRIGGER" in stmt.upper()]


def test_restore_fts_triggers_never_drops_first(test_conn: sqlite3.Connection) -> None:
    """The recovery path must not open a dropped-trigger window (polylogue-u66s3).

    Trigger DDL runs in autocommit, so a DROP that precedes the CREATEs is
    durable: a process death in between leaves index.db permanently without
    FTS triggers and every later block write unindexed.

    Anti-vacuity: reintroduce ``suspend_fts_triggers_sync(conn)`` (or any other
    ``DROP TRIGGER``) inside ``restore_fts_triggers_sync`` and this goes red.
    """
    restore_fts_triggers_sync(test_conn)
    present_before = {
        row[0] for row in test_conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall()
    }
    assert set(FTS_TRIGGER_NAMES) & present_before

    dropped = _dropped_trigger_statements(test_conn, restore_fts_triggers_sync)

    assert dropped == [], f"restore path issued DROP TRIGGER: {dropped}"
    present_after = {
        row[0] for row in test_conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall()
    }
    assert present_before <= present_after


def test_replace_fts_triggers_still_replaces_definitions(test_conn: sqlite3.Connection) -> None:
    """The explicit rebuild path keeps its drop-and-recreate semantics.

    Anti-vacuity: delete the ``suspend_fts_triggers_sync`` call from
    ``replace_fts_triggers_sync`` and this goes red, proving the split did not
    silently downgrade the rebuild caller.
    """
    restore_fts_triggers_sync(test_conn)

    dropped = _dropped_trigger_statements(test_conn, replace_fts_triggers_sync)

    assert dropped, "replace path issued no DROP TRIGGER"
    assert _triggers_present(test_conn)


def _triggers_present(conn: sqlite3.Connection) -> bool:
    names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall()}
    return bool(names & set(FTS_TRIGGER_NAMES))


def test_write_path_partition_convergence_replaces_identity_drift(test_conn: sqlite3.Connection) -> None:
    """The canonical write path's FTS route detects and repairs identity drift.

    This is the contract that retired ``storage/fts/session_repair.py``: the
    write path asks the FTS domain, not a second staleness rule of its own.

    Anti-vacuity: corrupting ``messages_fts_identity.source_hash`` is exactly
    what the deleted probe detected. If ``session_partition_is_valid_sync``
    stopped consulting the identity relation, the first assertion goes True and
    ``converge_fts_partition_sync`` returns False, failing this test; if the
    republish stopped happening, the final ``valid`` assertion fails.
    """
    restore_fts_triggers_sync(test_conn)
    _seed_text_block(
        test_conn,
        native_session_id="conv-writepath-converge",
        native_message_id="msg-writepath-converge",
        text="write path convergence needle",
    )
    session_id = "unknown-export:conv-writepath-converge"
    repair_message_fts_index_sync(test_conn, [session_id])
    assert session_partition_is_valid_sync(test_conn, session_id)
    # A second pass over an unchanged, valid partition must do nothing.
    assert converge_fts_partition_sync(test_conn, session_id) is False

    test_conn.execute(
        "UPDATE messages_fts_identity SET source_hash = ? WHERE block_id LIKE ?",
        (b"drift" + b"\x00" * 27, f"{session_id}:%"),
    )
    assert not session_partition_is_valid_sync(test_conn, session_id)
    assert converge_fts_partition_sync(test_conn, session_id) is True
    assert session_partition_is_valid_sync(test_conn, session_id)


def _transaction_statements(traced: list[str]) -> tuple[int, int]:
    """Count the ``BEGIN``/``COMMIT`` statements in a trace, in order."""
    begins = sum(1 for statement in traced if statement.lstrip().upper().startswith("BEGIN"))
    commits = sum(1 for statement in traced if statement.lstrip().upper().startswith("COMMIT"))
    return begins, commits


def _seed_repair_batch(conn: sqlite3.Connection, count: int) -> list[str]:
    session_ids = []
    for index in range(count):
        native = f"conv-batch-{index}"
        _seed_text_block(
            conn,
            native_session_id=native,
            native_message_id=f"msg-batch-{index}",
            text=f"batched repair needle {index}",
        )
        session_ids.append(f"unknown-export:{native}")
    return session_ids


def test_repair_message_fts_batch_commits_once(test_conn: sqlite3.Connection) -> None:
    """A multi-session repair is one transaction, not one per session.

    ``repair_message_fts_index_sync`` looped ``replace_fts_partition_sync`` per
    session, and ``publish_partition`` opens its own ``BEGIN IMMEDIATE`` when
    the connection is not already inside one -- so a four-session repair paid
    four commits, and four FTS5 segment flushes, for one logical repair
    (polylogue-av5j1). Measured on a 2000-session x 40-block synthetic index
    tier: min-of-6 3.286 s before, 2.281 s after.

    Anti-vacuity: restore the per-session loop and the counts become 4/4.
    The opposite direction is pinned by
    ``test_repair_message_fts_batch_is_all_or_nothing`` (a batch that commits
    nothing at all fails there) and by the membership assertion below, so
    "never open a transaction" cannot pass either.
    """
    restore_fts_triggers_sync(test_conn)
    session_ids = _seed_repair_batch(test_conn, 4)
    test_conn.commit()
    assert not test_conn.in_transaction

    traced: list[str] = []
    test_conn.set_trace_callback(traced.append)
    try:
        repair_message_fts_index_sync(test_conn, session_ids)
    finally:
        test_conn.set_trace_callback(None)

    begins, commits = _transaction_statements(traced)
    assert (begins, commits) == (1, 1), f"expected one transaction for the batch, saw {begins} BEGIN / {commits} COMMIT"
    for session_id in session_ids:
        assert session_partition_is_valid_sync(test_conn, session_id)


def test_repair_message_fts_batch_is_all_or_nothing(
    test_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repair that fails part-way publishes no partition at all.

    Per-session transactions left the sessions already visited committed and
    the rest unpublished, so an interrupted repair produced a half-published
    batch that no caller could distinguish from a completed one. Batching the
    transaction makes the failure leave the partition set exactly as it was.

    Anti-vacuity: restore the per-session loop and the first two partitions
    stay published, so the "no session became valid" assertion goes red.
    """
    restore_fts_triggers_sync(test_conn)
    session_ids = _seed_repair_batch(test_conn, 4)
    # Drift every partition's recorded identity so each one genuinely needs the
    # repair; the trigger-maintained rows are otherwise already valid.
    for session_id in session_ids:
        test_conn.execute(
            "UPDATE messages_fts_identity SET source_hash = ? WHERE block_id LIKE ?",
            (b"drift" + b"\x00" * 27, f"{session_id}:%"),
        )
    test_conn.commit()
    assert not any(session_partition_is_valid_sync(test_conn, sid) for sid in session_ids)

    from polylogue.storage.fts import derivation as derivation_module

    real_replace = derivation_module.replace_fts_partition_sync
    calls: list[str] = []

    def failing_replace(conn: sqlite3.Connection, session_id: str) -> bool:
        calls.append(session_id)
        if len(calls) == 3:
            raise RuntimeError("interrupted mid-batch")
        return real_replace(conn, session_id)

    monkeypatch.setattr(derivation_module, "replace_fts_partition_sync", failing_replace)
    with pytest.raises(RuntimeError, match="interrupted mid-batch"):
        repair_message_fts_index_sync(test_conn, session_ids)
    monkeypatch.undo()

    assert len(calls) == 3
    assert not test_conn.in_transaction, "a failed batch must not leave its transaction open"
    published = [sid for sid in session_ids if session_partition_is_valid_sync(test_conn, sid)]
    assert published == [], f"an interrupted batch published {published}"


def test_repair_message_fts_defers_to_a_caller_transaction(test_conn: sqlite3.Connection) -> None:
    """A caller that already owns the transaction keeps owning it.

    ``archive/write_effects`` runs this inside the canonical write's own
    transaction. Opening one unconditionally would raise "cannot start a
    transaction within a transaction" there.

    Anti-vacuity: make the ``BEGIN IMMEDIATE`` unconditional and this raises;
    the count assertion additionally refuses a stray COMMIT that would end the
    caller's transaction early.
    """
    restore_fts_triggers_sync(test_conn)
    session_ids = _seed_repair_batch(test_conn, 3)
    assert test_conn.in_transaction, "seeding leaves the caller inside its own transaction"

    traced: list[str] = []
    test_conn.set_trace_callback(traced.append)
    try:
        repair_message_fts_index_sync(test_conn, session_ids)
    finally:
        test_conn.set_trace_callback(None)

    assert _transaction_statements(traced) == (0, 0), f"the repair took over the caller's transaction: {traced[:8]}"
    assert test_conn.in_transaction
    for session_id in session_ids:
        assert session_partition_is_valid_sync(test_conn, session_id)


def test_in_transaction_partition_replace_never_hashes_the_partition_input(
    test_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inside one transaction a replace reads each partition's text zero times.

    ``input_for`` reads and hashes every block's ``search_text``. Its only use
    in a replace is to detect drift between a read outside the transaction and
    the write inside it; a batch repair holds one transaction across both, so
    the rebuild route computed it twice per session for no evidence
    (polylogue-av5j1). The replaced rows are still exactly valid.

    Anti-vacuity: route ``replace_fts_partition_sync`` back through
    ``publish_partition(conn, adapter.input_for(...))`` and the count becomes
    two per session.
    """
    restore_fts_triggers_sync(test_conn)
    session_ids = _seed_repair_batch(test_conn, 3)
    for session_id in session_ids:
        test_conn.execute(
            "UPDATE messages_fts_identity SET source_hash = ? WHERE block_id LIKE ?",
            (b"drift" + b"\x00" * 27, f"{session_id}:%"),
        )
    test_conn.commit()

    calls: list[str] = []
    real_input_for = FtsDerivationAdapter.input_for

    def counting_input_for(self: FtsDerivationAdapter, conn: sqlite3.Connection, key: str):  # type: ignore[no-untyped-def]
        calls.append(key)
        return real_input_for(self, conn, key)

    monkeypatch.setattr(FtsDerivationAdapter, "input_for", counting_input_for)
    repair_message_fts_index_sync(test_conn, session_ids)
    monkeypatch.undo()

    assert calls == []
    for session_id in session_ids:
        assert session_partition_is_valid_sync(test_conn, session_id)


def test_partition_replace_outside_a_transaction_still_revalidates(test_conn: sqlite3.Connection) -> None:
    """A replace computed outside a transaction still refuses a drifted input.

    Anti-vacuity: dropping the ``current != computed`` check from
    ``publish_partition`` publishes the stale projection and returns True.
    """
    restore_fts_triggers_sync(test_conn)
    session_ids = _seed_repair_batch(test_conn, 1)
    test_conn.commit()
    adapter = FtsDerivationAdapter()
    stale = adapter.input_for(test_conn, session_ids[0])
    test_conn.execute("UPDATE blocks SET text = 'drifted after the read' WHERE session_id = ?", (session_ids[0],))
    test_conn.commit()

    assert adapter.publish_partition(test_conn, stale) is False
    assert not test_conn.in_transaction


def test_partition_deletes_page_canonical_and_residue_rows_under_actual_bind_limit(
    test_conn: sqlite3.Connection,
) -> None:
    """Both populations exceed the bind budget, including signed rowid edges.

    Restoring a whole-session rowid set breaches the observed read-page budget;
    omitting residue or starting at rowid zero leaves searchable orphan rows.
    """
    restore_fts_triggers_sync(test_conn)
    message_id = _seed_text_block(
        test_conn, native_session_id="paged", native_message_id="message", text="paged canonical needle"
    )
    session_id = "unknown-export:paged"
    test_conn.executemany(
        "INSERT INTO blocks(message_id, session_id, position, block_type, text, content_identity, content_occurrence) VALUES (?, ?, ?, 'text', ?, ?, ?)",
        (
            (
                message_id,
                session_id,
                position,
                "paged canonical needle",
                fixture_block_content_identity("text", "paged canonical needle"),
                content_occurrence,
            )
            for content_occurrence, position in enumerate(range(1, 17), start=1)
        ),
    )
    test_conn.execute("UPDATE blocks SET rowid = ? WHERE message_id = ? AND position = 0", (-(2**63), message_id))
    for position in range(13):
        rowid = -100 + position
        test_conn.execute("INSERT INTO messages_fts(rowid, text) VALUES (?, 'paged residue needle')", (rowid,))
        test_conn.execute(
            "INSERT INTO messages_fts_identity(rowid, block_id, source_hash, recipe_id) VALUES (?, ?, NULL, ?)",
            (rowid, f"{session_id}:n:removed:{position}", FtsDerivationAdapter.recipe_id),
        )
    test_conn.commit()
    page_sizes: list[int] = []

    class PageCursor:
        def __init__(self, cursor: sqlite3.Cursor) -> None:
            self.cursor = cursor
            self.closed = False

        def fetchall(self):  # type: ignore[no-untyped-def]
            rows = self.cursor.fetchall()
            page_sizes.append(len(rows))
            assert len(rows) <= 4, "a delete read retained more than one connection-bounded page"
            return rows

        def __iter__(self):  # type: ignore[no-untyped-def]
            for count, row in enumerate(self.cursor, start=1):
                assert count <= 4, "a delete read retained the whole partition"
                yield row

        def close(self) -> None:
            self.cursor.close()
            self.closed = True

    class ObservedConnection:
        def __init__(self) -> None:
            self.last_page: PageCursor | None = None

        def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
            return getattr(test_conn, name)

        def execute(self, sql: str, params: tuple[object, ...] = ()):  # type: ignore[no-untyped-def]
            if sql.startswith("DELETE FROM messages_fts") and self.last_page is not None:
                assert self.last_page.closed, "delete began before its read cursor physically closed"
            cursor = test_conn.execute(sql, params)
            if sql.lstrip().startswith(("SELECT rowid", "SELECT i.rowid")):
                self.last_page = PageCursor(cursor)
                return self.last_page
            return cursor

    prior_limit = test_conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 4)
    try:
        repair_message_fts_index_sync(ObservedConnection(), [session_id])  # type: ignore[arg-type]
    finally:
        test_conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, prior_limit)
    assert len(page_sizes) >= 10, page_sizes
    assert max(page_sizes) == 4
    assert test_conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'residue'").fetchone()[0] == 0
    assert (
        test_conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'canonical'").fetchone()[0] == 17
    )
    assert FtsDerivationAdapter().inspect_partition(test_conn, session_id).valid


def test_partition_page_interrupt_rolls_back_the_owned_batch(test_conn: sqlite3.Connection) -> None:
    """An SQL interrupt after a deletion page cannot leave a partial batch."""
    restore_fts_triggers_sync(test_conn)
    session_ids = _seed_repair_batch(test_conn, 3)
    test_conn.commit()
    before = tuple(test_conn.execute("SELECT * FROM messages_fts_identity ORDER BY rowid"))
    interrupted = False

    def interrupt_after_delete(sql: str) -> None:
        nonlocal interrupted
        if sql.startswith("DELETE FROM messages_fts_identity") and not interrupted:
            interrupted = True

            def interrupt() -> int:
                test_conn.set_progress_handler(None, 0)
                return 1

            test_conn.set_progress_handler(interrupt, 1)

    test_conn.set_trace_callback(interrupt_after_delete)
    try:
        with pytest.raises(sqlite3.OperationalError, match="interrupt"):
            repair_message_fts_index_sync(test_conn, session_ids)
    finally:
        test_conn.set_trace_callback(None)
        test_conn.set_progress_handler(None, 0)
    assert interrupted
    assert not test_conn.in_transaction
    assert tuple(test_conn.execute("SELECT * FROM messages_fts_identity ORDER BY rowid")) == before
    assert all(FtsDerivationAdapter().inspect_partition(test_conn, session_id).valid for session_id in session_ids)


@pytest.mark.parametrize("observer_refuses", [False, True])
def test_full_fts_rebuild_pages_preserve_pairing_and_caller_transaction(
    test_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch, observer_refuses: bool
) -> None:
    """A page is complete only after both projections; no page commits caller work."""
    monkeypatch.setattr(fts_lifecycle, "FTS_REBUILD_SESSION_PAGE_SIZE", 1)
    for index in range(3):
        _seed_text_block(test_conn, native_session_id=f"page-{index}", native_message_id="m", text="canonical needle")
    test_conn.execute("INSERT INTO messages_fts(rowid, text) VALUES (-100, 'orphan needle')")
    test_conn.execute(
        "INSERT INTO messages_fts_identity(rowid, block_id, recipe_id) VALUES (-100, 'orphan', ?)",
        (FtsDerivationAdapter.recipe_id,),
    )
    test_conn.commit()
    test_conn.execute("BEGIN")
    test_conn.execute("UPDATE sessions SET title = 'caller pending title'")
    events: list[tuple[int, int, int]] = []

    observer_error = RuntimeError("synthetic observer refusal")

    def observe(amount: int, processed: int, total: int) -> None:
        events.append((amount, processed, total))
        assert test_conn.in_transaction
        assert test_conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == processed
        assert test_conn.execute("SELECT COUNT(*) FROM messages_fts_identity").fetchone()[0] == processed
        if observer_refuses and processed == 1:
            raise observer_error

    if observer_refuses:
        with pytest.raises(RuntimeError) as caught:
            rebuild_fts_index_sync(test_conn, progress_callback=observe)
        assert caught.value is observer_error
        assert events == [(0, 0, 3), (1, 1, 3)]
    else:
        rebuild_fts_index_sync(test_conn, progress_callback=observe)
        assert events == [(0, 0, 3), (1, 1, 3), (1, 2, 3), (1, 3, 3)]
    test_conn.rollback()
    assert test_conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 4
    assert test_conn.execute("SELECT COUNT(*) FROM messages_fts_identity").fetchone()[0] == 4
    assert test_conn.execute("SELECT DISTINCT title FROM sessions").fetchone()[0] == "Message repair"


def test_full_fts_rebuild_observes_cancellation_between_actual_pages(
    test_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from polylogue.core.compute import DaemonOperationCancelled
    from polylogue.core.compute_cancel import compute_cancel

    monkeypatch.setattr(fts_lifecycle, "FTS_REBUILD_SESSION_PAGE_SIZE", 1)
    for index in range(3):
        _seed_text_block(test_conn, native_session_id=f"cancel-{index}", native_message_id="m", text="canonical needle")
    test_conn.commit()
    cancelled = threading.Event()
    events = []

    def cancel_after_first_page(amount: int, processed: int, total: int) -> None:
        events.append((amount, processed, total))
        if processed == 1:
            cancelled.set()

    token = compute_cancel.set(cancelled)
    try:
        with pytest.raises(DaemonOperationCancelled):
            rebuild_fts_index_sync(test_conn, progress_callback=cancel_after_first_page)
    finally:
        compute_cancel.reset(token)
    assert events == [(0, 0, 3), (1, 1, 3)]
    assert test_conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 1
    assert test_conn.execute("SELECT COUNT(*) FROM messages_fts_identity").fetchone()[0] == 1
    assert not live_connection_cursors(test_conn)
    test_conn.rollback()
    assert test_conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 3


def test_empty_full_fts_rebuild_reports_completion_after_both_orphan_resets(test_conn: sqlite3.Connection) -> None:
    test_conn.execute("INSERT INTO messages_fts(rowid, text) VALUES (-100, 'orphan needle')")
    test_conn.execute(
        "INSERT INTO messages_fts_identity(rowid, block_id, recipe_id) VALUES (-100, 'orphan', ?)",
        (FtsDerivationAdapter.recipe_id,),
    )
    events: list[tuple[int, int, int]] = []

    def observe(amount: int, processed: int, total: int) -> None:
        assert test_conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 0
        assert test_conn.execute("SELECT COUNT(*) FROM messages_fts_identity").fetchone()[0] == 0
        events.append((amount, processed, total))

    rebuild_fts_index_sync(test_conn, progress_callback=observe)
    assert events == [(0, 0, 0)]


def test_full_fts_rebuild_preserves_primary_observer_and_cursor_cleanup_failures(
    test_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_text_block(test_conn, native_session_id="failure-pair", native_message_id="m", text="canonical needle")
    test_conn.commit()
    original_execute = test_conn.execute
    opened: list[ControlledCursor] = []
    primary = RuntimeError("synthetic primary observer failure")

    def execute(sql: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
        if sql == "SELECT session_id FROM sessions ORDER BY session_id":
            cursor = test_conn.cursor(factory=ControlledCursor)
            assert isinstance(cursor, ControlledCursor)
            cursor.allow_cleanup.clear()
            opened.append(cursor)
            return cursor.execute(sql, *args, **kwargs)
        return original_execute(sql, *args, **kwargs)

    def observe(amount: int, processed: int, total: int) -> None:
        if processed:
            raise primary

    monkeypatch.setattr(test_conn, "execute", execute)
    try:
        with pytest.raises(BaseExceptionGroup) as caught:
            rebuild_fts_index_sync(test_conn, progress_callback=observe)
        assert len(opened) == 1
        cursor = opened[0]
        assert caught.value.exceptions == (primary, cursor.cleanup_failure)
        assert cursor in live_connection_cursors(test_conn)
    finally:
        for cursor in opened:
            cursor.allow_cleanup.set()
            close_connection_cursor(test_conn, cursor)
        test_conn.rollback()
