"""Targeted SQL contracts for incremental FTS repair."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable

import pytest

from polylogue.storage.fts.derivation import (
    GLOBAL_PARTITION,
    FtsDerivationAdapter,
    converge_fts_partition_sync,
    session_partition_is_valid_sync,
)
from polylogue.storage.fts.fts_lifecycle import (
    FTS_TRIGGER_NAMES,
    delete_excess_message_rows_batched_sync,
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
from tests.infra.identity import archive_message_id


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
        """
        INSERT INTO blocks (message_id, session_id, position, block_type, text)
        VALUES (?, ?, 0, 'text', ?)
        """,
        (message_id, session_id, text),
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
        repair_message_fts_index_sync(test_conn, [session_id], record_exact_snapshot=False)
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
    """A session repair remains bounded even if a legacy caller asks for exact."""
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

    repair_message_fts_index_sync(
        test_conn,
        ["unknown-export:conv-no-duplicate-exact"],
        record_exact_snapshot=True,
    )


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


def test_excess_fts_repair_deletes_orphan_docsize_rows(test_conn: sqlite3.Connection) -> None:
    restore_fts_triggers_sync(test_conn)
    message_id = _seed_text_block(
        test_conn,
        native_session_id="conv-batched-excess",
        native_message_id="msg-batched-excess",
        text="batched excess needle",
    )
    block_rowid = test_conn.execute(
        "SELECT rowid FROM blocks WHERE message_id = ?",
        (message_id,),
    ).fetchone()["rowid"]
    rebuild_fts_index_sync(test_conn)
    test_conn.execute("DROP TRIGGER messages_fts_ad")
    test_conn.execute("DELETE FROM blocks WHERE rowid = ?", (block_rowid,))

    row_before = test_conn.execute(
        "SELECT COUNT(*) FROM messages_fts_docsize WHERE id = ?",
        (block_rowid,),
    ).fetchone()
    assert row_before[0] == 1

    progress: list[int] = []
    deleted = delete_excess_message_rows_batched_sync(
        test_conn,
        batch_rows=1,
        progress_callback=progress.append,
    )

    assert deleted >= 1
    assert sum(progress) == deleted
    row_after = test_conn.execute(
        "SELECT COUNT(*) FROM messages_fts_docsize WHERE id = ?",
        (block_rowid,),
    ).fetchone()
    assert row_after[0] == 0


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
