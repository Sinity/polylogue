"""Semantic laws for the domain-owned message FTS derivation."""

from __future__ import annotations

import sqlite3

from polylogue.storage.fts.derivation import (
    GLOBAL_PARTITION,
    FtsDerivationAdapter,
    FtsKeyStatus,
    FtsOutcome,
)
from polylogue.storage.fts.fts_lifecycle import restore_fts_triggers_sync


def _seed_session(conn: sqlite3.Connection, native_id: str = "derivation") -> tuple[str, int]:
    content_hash = b"a" * 32
    conn.execute(
        "INSERT INTO sessions(native_id, origin, title, content_hash) VALUES (?, 'unknown-export', 'test', ?)",
        (native_id, content_hash),
    )
    session_id = f"unknown-export:{native_id}"
    conn.execute(
        "INSERT INTO messages(session_id, native_id, position, role, message_type, content_hash) "
        "VALUES (?, 'm0', 0, 'user', 'message', ?)",
        (session_id, content_hash),
    )
    message_id = conn.execute("SELECT message_id FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0]
    conn.execute(
        "INSERT INTO blocks(message_id, session_id, position, block_type, text, content_hash) "
        "VALUES (?, ?, 0, 'text', 'derivation input', ?)",
        (message_id, session_id, content_hash),
    )
    conn.commit()
    rowid = conn.execute("SELECT rowid FROM blocks WHERE session_id = ?", (session_id,)).fetchone()[0]
    return session_id, int(rowid)


def test_valid_empty_partition_is_done_without_a_fake_row(test_conn: sqlite3.Connection) -> None:
    """Anti-vacuity: treating an empty output as missing would create a row."""
    adapter = FtsDerivationAdapter()
    conn = test_conn
    conn.execute(
        "INSERT INTO sessions(native_id, origin, title, content_hash) VALUES ('empty', 'unknown-export', 'empty', ?)",
        (b"e" * 32,),
    )
    conn.commit()

    inspection = adapter.inspect(conn, "unknown-export:empty")
    result = adapter.converge(conn, keys=("unknown-export:empty",))

    assert inspection.status is FtsKeyStatus.VALID
    assert inspection.required_rows == 0
    assert result.outcome is FtsOutcome.DONE
    assert result.written_partitions == 0


def test_missing_wrong_identity_and_excess_are_authoritative_membership_failures(
    test_conn: sqlite3.Connection,
) -> None:
    """Anti-vacuity: removing identity/excess inspection must leave this red."""
    adapter = FtsDerivationAdapter()
    session_id, rowid = _seed_session(test_conn)

    test_conn.execute("DELETE FROM messages_fts_identity WHERE rowid = ?", (rowid,))
    test_conn.commit()
    missing_identity = adapter.inspect(test_conn, session_id)
    assert missing_identity.status is FtsKeyStatus.STALE
    assert missing_identity.wrong_identity_rows == 1
    assert adapter.converge(test_conn, keys=(session_id,)).outcome is FtsOutcome.DONE

    test_conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (rowid,))
    test_conn.execute("DELETE FROM messages_fts_identity WHERE rowid = ?", (rowid,))
    test_conn.commit()
    missing = adapter.inspect(test_conn, session_id)
    assert missing.status is FtsKeyStatus.STALE
    assert missing.missing_rows == 1

    assert adapter.converge(test_conn, keys=(session_id,)).outcome is FtsOutcome.DONE
    test_conn.execute(
        "UPDATE messages_fts_identity SET block_id = 'wrong:identity' WHERE rowid = ?",
        (rowid,),
    )
    test_conn.commit()
    wrong = adapter.inspect(test_conn, session_id)
    assert wrong.status is FtsKeyStatus.STALE
    assert wrong.wrong_identity_rows == 1

    assert adapter.converge(test_conn, keys=(session_id,)).outcome is FtsOutcome.DONE
    test_conn.execute(
        "INSERT INTO messages_fts(rowid, block_id, message_id, session_id, block_type, text) "
        "VALUES (999999, 'orphan:block', 'orphan:message', 'orphan:session', 'text', 'orphan')"
    )
    test_conn.execute(
        "INSERT INTO messages_fts_identity(rowid, block_id, source_hash, recipe_id) "
        "VALUES (999999, 'orphan:block', ?, ?)",
        (b"o" * 32, adapter.recipe_id),
    )
    test_conn.commit()
    global_inspection = adapter.inspect(test_conn, GLOBAL_PARTITION)
    assert global_inspection.status is FtsKeyStatus.EXCESS
    assert global_inspection.excess_rows == 1


def test_scoped_convergence_does_not_rewrite_a_healthy_sibling(test_conn: sqlite3.Connection) -> None:
    """Anti-vacuity: a global replacement would alter the healthy sibling."""
    adapter = FtsDerivationAdapter()
    target, target_rowid = _seed_session(test_conn, "target")
    sibling, sibling_rowid = _seed_session(test_conn, "sibling")
    sibling_input = adapter.input_for(test_conn, sibling)
    test_conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (target_rowid,))
    test_conn.commit()

    result = adapter.converge(test_conn, keys=(target,))

    assert result.outcome is FtsOutcome.DONE
    assert adapter.inspect(test_conn, target).valid
    assert adapter.inspect(test_conn, sibling).valid
    assert adapter.input_for(test_conn, sibling) == sibling_input
    assert test_conn.execute("SELECT 1 FROM messages_fts_docsize WHERE id = ?", (sibling_rowid,)).fetchone()


def test_healthy_second_pass_has_no_publication_work(test_conn: sqlite3.Connection) -> None:
    """Anti-vacuity: an unconditional rebuild would report writes on pass two."""
    adapter = FtsDerivationAdapter()
    session_id, _ = _seed_session(test_conn)

    first = adapter.converge(test_conn, keys=(session_id,))
    second = adapter.converge(test_conn, keys=(session_id,))

    assert first.outcome is FtsOutcome.DONE
    assert second.outcome is FtsOutcome.DONE
    assert first.written_partitions == 0
    assert second.written_partitions == 0


def test_publish_revalidates_input_before_writing(test_conn: sqlite3.Connection) -> None:
    """Anti-vacuity: removing input revalidation would publish the stale snapshot."""
    adapter = FtsDerivationAdapter()
    session_id, _ = _seed_session(test_conn)
    computed = adapter.input_for(test_conn, session_id)
    test_conn.execute("UPDATE blocks SET text = 'changed before publish', content_hash = ?", (b"c" * 32,))
    test_conn.commit()

    assert adapter.publish(test_conn, computed) is False
    assert adapter.inspect(test_conn, session_id).status is FtsKeyStatus.VALID


def test_canonical_block_write_keeps_membership_transactional(test_conn: sqlite3.Connection) -> None:
    """Anti-vacuity: bypassing canonical trigger maintenance breaks this rollback law."""
    adapter = FtsDerivationAdapter()
    session_id, rowid = _seed_session(test_conn)
    original = adapter.input_for(test_conn, session_id)

    test_conn.execute(
        "UPDATE blocks SET text = 'transactional update', content_hash = ? WHERE rowid = ?", (b"t" * 32, rowid)
    )
    assert adapter.inspect(test_conn, session_id).status is FtsKeyStatus.VALID
    test_conn.rollback()

    assert adapter.input_for(test_conn, session_id) == original
    assert adapter.inspect(test_conn, session_id).status is FtsKeyStatus.VALID


def test_generation_change_rejects_a_stale_publication(test_conn: sqlite3.Connection) -> None:
    """Anti-vacuity: omitting generation binding would accept the old snapshot."""
    adapter = FtsDerivationAdapter()
    session_id, _ = _seed_session(test_conn)
    computed = adapter.input_for(test_conn, session_id)
    generation = int(test_conn.execute("PRAGMA user_version").fetchone()[0])
    test_conn.execute(f"PRAGMA user_version = {generation + 1}")
    test_conn.commit()

    assert adapter.publish(test_conn, computed) is False
    assert adapter.inspect(test_conn, session_id).status is FtsKeyStatus.VALID


def test_trigger_loss_is_incompatible_and_never_runtime_repaired(test_conn: sqlite3.Connection) -> None:
    """Anti-vacuity: runtime trigger recreation would make this incompatible case green."""
    adapter = FtsDerivationAdapter()
    session_id, _ = _seed_session(test_conn)
    test_conn.execute("DROP TRIGGER messages_fts_ad")
    test_conn.commit()

    inspection = adapter.inspect(test_conn, session_id)
    result = adapter.converge(test_conn, keys=(session_id,))

    assert inspection.triggers_compatible is False
    assert result.outcome is FtsOutcome.PENDING
    assert test_conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'messages_fts_ad'").fetchone() is None

    restore_fts_triggers_sync(test_conn)


def test_partition_inspection_and_publish_search_the_block_id_index(test_conn: sqlite3.Connection) -> None:
    """Anti-vacuity: a ``substr(block_id, ...)`` prefix predicate plans as a table scan.

    Every statement the partition inspection and publish issue against
    ``messages_fts_identity`` must be served by the ``block_id`` index, so a
    partition's cost is bounded by its own rows rather than the archive's.
    The sibling key ``s1`` / ``s10`` proves the range is exact at the ``:``
    boundary.
    """
    adapter = FtsDerivationAdapter()
    short_key, _short_rowid = _seed_session(test_conn, "s1")
    long_key, long_rowid = _seed_session(test_conn, "s10")
    test_conn.execute("DELETE FROM messages_fts_identity WHERE rowid = ?", (long_rowid,))
    test_conn.commit()

    statements: list[str] = []
    test_conn.set_trace_callback(statements.append)
    try:
        short_inspection = adapter.inspect(test_conn, short_key)
        assert adapter.publish(test_conn, adapter.input_for(test_conn, short_key))
    finally:
        test_conn.set_trace_callback(None)

    assert short_inspection.status is FtsKeyStatus.VALID
    assert adapter.inspect(test_conn, long_key).status is FtsKeyStatus.STALE
    identity_statements = [
        sql for sql in statements if "messages_fts_identity" in sql and sql.lstrip().upper().startswith("SELECT")
    ]
    assert identity_statements, "inspection and publish must consult the identity ledger"
    for sql in identity_statements:
        plan = " | ".join(str(row[3]) for row in test_conn.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall())
        assert "SCAN i" not in plan and "SCAN messages_fts_identity" not in plan, (sql, plan)
