"""Semantic laws for message FTS through the common derivation kernel."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tracemalloc
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.daemon.convergence import DaemonConverger
from polylogue.daemon.derivation import Budget, DerivationFrame, DerivationReport, Outcome
from polylogue.storage.fts.derivation import (
    GLOBAL_PARTITION,
    FtsDerivationAdapter,
    FtsKeyStatus,
    FtsOrphanReplacement,
    FtsPartitionReplacement,
)
from polylogue.storage.fts.fts_lifecycle import restore_fts_triggers_sync, suspend_fts_triggers_sync


def _adapter(db_path: Path, *, orphan_interval_s: float | None = None) -> FtsDerivationAdapter:
    """Use independent real SQLite connections, as daemon compute/publish do."""

    def read_connection() -> sqlite3.Connection:
        return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    def write_connection() -> sqlite3.Connection:
        return sqlite3.connect(db_path)

    return FtsDerivationAdapter(
        read_connection,
        write_connection,
        generation_binding=lambda: str(db_path.resolve()),
        orphan_interval_s=orphan_interval_s,
        monotonic=lambda: 100.0,
    )


def _frame(db_path: Path, scope: Sequence[str] | None = None) -> DerivationFrame:
    adapter = FtsDerivationAdapter()
    return DerivationFrame(
        archive_root=str(db_path.parent),
        source_revision=f"index-generation:{db_path.resolve()}",
        recipe_versions={adapter.domain: adapter.recipe_id},
        scope=None if scope is None else tuple(scope),
    )


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


def _converge(
    adapter: FtsDerivationAdapter, frame: DerivationFrame, *, budget: Budget | None = None
) -> DerivationReport:
    return DaemonConverger([], derivations=[adapter]).converge_derivations(frame, budget=budget)


def test_valid_empty_partition_is_done_without_a_fake_row(test_conn: sqlite3.Connection, test_db: Path) -> None:
    """Anti-vacuity: treating a zero-output session as missing would publish a fake row."""
    test_conn.execute(
        "INSERT INTO sessions(native_id, origin, title, content_hash) VALUES ('empty', 'unknown-export', 'empty', ?)",
        (b"e" * 32,),
    )
    test_conn.commit()
    session_id = "unknown-export:empty"
    adapter = _adapter(test_db)

    report = _converge(adapter, _frame(test_db, (session_id,)))

    assert adapter.inspect_partition(test_conn, session_id).status is FtsKeyStatus.VALID
    assert report.done == 0
    assert report.made_no_publication_attempts
    assert test_conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 0


def test_kernel_repairs_one_stale_partition_without_rewriting_a_healthy_sibling(
    test_conn: sqlite3.Connection, test_db: Path
) -> None:
    """Anti-vacuity: a global rebuild would alter the healthy sibling partition."""
    target, target_rowid = _seed_session(test_conn, "target")
    sibling, sibling_rowid = _seed_session(test_conn, "sibling")
    test_conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (target_rowid,))
    test_conn.execute("DELETE FROM messages_fts_identity WHERE rowid = ?", (target_rowid,))
    test_conn.commit()
    adapter = _adapter(test_db)

    report = _converge(adapter, _frame(test_db, (target,)))

    assert report.count(Outcome.DONE) == 1
    assert adapter.inspect_partition(test_conn, target).valid
    assert adapter.inspect_partition(test_conn, sibling).valid
    assert test_conn.execute("SELECT 1 FROM messages_fts_docsize WHERE id = ?", (sibling_rowid,)).fetchone()


def test_parent_partition_repair_preserves_a_colon_prefixed_sibling(
    test_conn: sqlite3.Connection, test_db: Path
) -> None:
    """Anti-vacuity: block-id prefix selection deletes the healthy child session."""
    parent, parent_rowid = _seed_session(test_conn, "a")
    child, child_rowid = _seed_session(test_conn, "a:child")
    test_conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (parent_rowid,))
    test_conn.execute("DELETE FROM messages_fts_identity WHERE rowid = ?", (parent_rowid,))
    test_conn.commit()
    adapter = _adapter(test_db)

    report = _converge(adapter, _frame(test_db, (parent,)))

    assert report.done == 1
    assert adapter.inspect_partition(test_conn, parent).valid
    assert adapter.inspect_partition(test_conn, child).valid
    assert test_conn.execute("SELECT 1 FROM messages_fts_docsize WHERE id = ?", (child_rowid,)).fetchone()
    assert test_conn.execute("SELECT 1 FROM messages_fts_identity WHERE rowid = ?", (child_rowid,)).fetchone()


def test_unchanged_second_pass_publishes_zero_replacements(test_conn: sqlite3.Connection, test_db: Path) -> None:
    """Anti-vacuity: unconditional publication makes the second pass non-empty."""
    session_id, rowid = _seed_session(test_conn)
    test_conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (rowid,))
    test_conn.commit()
    adapter = _adapter(test_db)
    converger = DaemonConverger([], derivations=[adapter])
    frame = _frame(test_db, (session_id,))

    first = converger.converge_derivations(frame)
    second = converger.converge_derivations(frame)

    assert first.done == 1
    assert second.done == 0
    assert second.made_no_publication_attempts


def test_frame_drift_rejects_prepared_partition_before_any_fts_write(
    test_conn: sqlite3.Connection, test_db: Path
) -> None:
    """Anti-vacuity: omitting publish revalidation would install a stale snapshot."""
    session_id, rowid = _seed_session(test_conn)
    test_conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (rowid,))
    test_conn.commit()
    adapter = _adapter(test_db)
    frame = _frame(test_db, (session_id,))
    replacement = adapter.compute(frame, session_id)
    test_conn.execute("UPDATE blocks SET text = 'changed before publish', content_hash = ?", (b"c" * 32,))
    test_conn.commit()

    assert adapter.publish(frame, replacement) is False
    assert adapter.input_for(test_conn, session_id) != replacement.payload
    assert adapter.inspect_partition(test_conn, session_id).status is FtsKeyStatus.VALID


def test_large_partition_binding_streams_text_and_revalidates_changes(
    test_conn: sqlite3.Connection, test_db: Path
) -> None:
    """Retaining all text makes the measured Python peak exceed the bound."""
    session_id, _ = _seed_session(test_conn, "large-binding")
    message_id = test_conn.execute("SELECT message_id FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0]
    for position in range(1, 65):
        test_conn.execute(
            "INSERT INTO blocks(message_id, session_id, position, block_type, text, content_hash) "
            "VALUES (?, ?, ?, 'text', ?, ?)",
            (message_id, session_id, position, f"{position}:" + "x" * 65536, b"a" * 32),
        )
    test_conn.commit()
    adapter = _adapter(test_db)

    tracemalloc.start()
    try:
        binding = adapter.input_for(test_conn, session_id)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert binding.row_count == 65
    assert peak < 2 * 1024 * 1024
    rows = test_conn.execute(
        "SELECT rowid, block_id, message_id, session_id, block_type, search_text, content_hash "
        "FROM blocks WHERE session_id = ? AND search_text != '' ORDER BY rowid",
        (session_id,),
    )
    expected_digest = hashlib.sha256(
        json.dumps(
            [[*row[:6], bytes(row[6]).hex()] for row in rows],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert binding.digest == expected_digest
    test_conn.execute("UPDATE blocks SET text = 'changed' WHERE session_id = ? AND position = 64", (session_id,))
    test_conn.commit()
    assert adapter.publish_partition(test_conn, binding) is False


def test_required_discovery_is_keyset_bounded_and_resumes(test_conn: sqlite3.Connection, test_db: Path) -> None:
    """Anti-vacuity: eager required-key materialization discovers all sessions in pass one."""
    session_ids = tuple(_seed_session(test_conn, f"p{index}")[0] for index in range(5))
    for session_id in session_ids:
        rowid = test_conn.execute("SELECT rowid FROM blocks WHERE session_id = ?", (session_id,)).fetchone()[0]
        test_conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (rowid,))
    test_conn.commit()
    adapter = _adapter(test_db)
    converger = DaemonConverger([], derivations=[adapter])
    frame = _frame(test_db)
    budget = Budget(page=2, discovery=2, inspection=4, compute=2, publication=2)

    first = converger.converge_derivations(frame, budget=budget)
    second = converger.converge_derivations(frame, budget=budget)

    assert first.work.discovered == 2
    assert first.done == 2
    assert first.cursor.position(adapter.domain).page_cursor is not None
    assert second.work.discovered == 2
    assert second.done == 2


def test_global_residue_key_deletes_only_docsize_proven_orphans(test_conn: sqlite3.Connection, test_db: Path) -> None:
    """Anti-vacuity: restoring ``DELETE FROM messages_fts`` removes the live sibling too."""
    _live_session, live_rowid = _seed_session(test_conn, "live")
    adapter = _adapter(test_db)
    test_conn.execute("INSERT INTO messages_fts(rowid, text) VALUES (999999, 'orphan')")
    test_conn.execute(
        "INSERT INTO messages_fts_identity(rowid, block_id, source_hash, recipe_id) VALUES (999999, 'orphan:block', ?, ?)",
        (b"o" * 32, adapter.recipe_id),
    )
    test_conn.commit()

    report = _converge(adapter, _frame(test_db, ()))

    assert report.done == 1
    assert test_conn.execute("SELECT 1 FROM messages_fts_docsize WHERE id = ?", (live_rowid,)).fetchone()
    assert test_conn.execute("SELECT 1 FROM messages_fts_docsize WHERE id = 999999").fetchone() is None
    assert adapter.inspect_partition(test_conn, GLOBAL_PARTITION).status is FtsKeyStatus.VALID


def test_global_orphan_discovery_uses_exists_not_an_unbounded_rowid_collection(
    test_conn: sqlite3.Connection, test_db: Path
) -> None:
    """Anti-vacuity: selecting orphan IDs during discovery grows with all residue."""
    for rowid in range(900_000, 900_256):
        test_conn.execute(
            "INSERT INTO messages_fts(rowid, text) VALUES (?, 'orphan')",
            (rowid,),
        )
    test_conn.commit()
    statements: list[str] = []

    def read_connection() -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{test_db}?mode=ro", uri=True)
        conn.set_trace_callback(statements.append)
        return conn

    adapter = FtsDerivationAdapter(
        read_connection,
        lambda: sqlite3.connect(test_db),
        generation_binding=lambda: str(test_db.resolve()),
        orphan_interval_s=None,
    )

    keys, cursor = adapter.excess_page(_frame(test_db), cursor=None, limit=1)

    assert (keys, cursor) == ((GLOBAL_PARTITION,), None)
    assert any("SELECT EXISTS" in statement for statement in statements)
    assert not any("SELECT d.id" in statement for statement in statements)


def test_trigger_loss_stays_pending_and_is_never_recreated_at_runtime(
    test_conn: sqlite3.Connection, test_db: Path
) -> None:
    """Anti-vacuity: runtime trigger recreation makes a schema failure look converged."""
    session_id, rowid = _seed_session(test_conn)
    test_conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (rowid,))
    test_conn.execute("DROP TRIGGER messages_fts_ad")
    test_conn.commit()
    adapter = _adapter(test_db)

    report = _converge(adapter, _frame(test_db, (session_id,)))

    assert report.pending == 1
    assert test_conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'messages_fts_ad'").fetchone() is None
    restore_fts_triggers_sync(test_conn)


@pytest.mark.parametrize("binding", ["generation", "recipe"])
def test_fts_inspection_refuses_a_retired_frame(
    test_conn: sqlite3.Connection,
    test_db: Path,
    binding: str,
) -> None:
    """Anti-vacuity: current valid rows cannot certify a different input frame."""
    session_id, _ = _seed_session(test_conn)
    adapter = _adapter(test_db)
    frame = _frame(test_db, (session_id,))
    retired = (
        replace(frame, source_revision="index-generation:retired")
        if binding == "generation"
        else replace(frame, recipe_versions={adapter.domain: "old"})
    )
    with pytest.raises(RuntimeError, match="frame"):
        adapter.inspect(retired, (session_id,))


def test_orphan_retirement_succeeds_despite_poisoned_session(
    test_conn: sqlite3.Connection,
    test_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anti-vacuity: a global readiness check misreports successful orphan cleanup."""
    session_id, rowid = _seed_session(test_conn)
    test_conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (rowid,))
    test_conn.execute("INSERT INTO messages_fts(rowid, text) VALUES (999999, 'orphan')")
    test_conn.commit()
    adapter = _adapter(test_db)
    compute = adapter.compute

    def poison(frame: object, key: str) -> FtsPartitionReplacement | FtsOrphanReplacement:
        if key == session_id:
            raise ValueError("poison session")
        return compute(frame, key)

    monkeypatch.setattr(adapter, "compute", poison)
    report = _converge(adapter, _frame(test_db, (session_id,)))
    assert report.failed == 1
    assert report.done == 1
    assert report.by_outcome(Outcome.FAILED)[0].key.key == session_id
    assert report.by_outcome(Outcome.DONE)[0].key.key == GLOBAL_PARTITION
    assert test_conn.execute("SELECT 1 FROM messages_fts_docsize WHERE id = 999999").fetchone() is None


def test_session_partition_retires_fts_rows_whose_block_was_deleted(
    test_conn: sqlite3.Connection, test_db: Path
) -> None:
    """A block deleted behind suspended triggers is session-visible excess.

    Bulk ingest suspends the FTS triggers around block writes, so a deleted
    block leaves an FTS/identity row with no canonical block behind it. That
    residue belongs to its session partition, not only to the global residue
    key: searching must not keep returning a row for a block that is gone.

    Anti-vacuity: resolve the partition's residue through an inner join to
    ``blocks`` — which by construction cannot see a row whose block was
    deleted — and inspection calls this partition VALID while ``messages_fts``
    still matches the deleted block, so this goes red.
    """
    session_id, _ = _seed_session(test_conn, "residue")
    second_rowid = test_conn.execute(
        "SELECT rowid FROM blocks WHERE session_id = ? ORDER BY rowid", (session_id,)
    ).fetchone()[0]

    suspend_fts_triggers_sync(test_conn)
    test_conn.execute("DELETE FROM blocks WHERE rowid = ?", (second_rowid,))
    test_conn.commit()
    restore_fts_triggers_sync(test_conn)

    # The residue is real: FTS still holds a row for the deleted block.
    assert test_conn.execute("SELECT 1 FROM messages_fts_docsize WHERE id = ?", (second_rowid,)).fetchone()

    adapter = _adapter(test_db)
    stale = adapter.inspect_partition(test_conn, session_id)
    assert stale.status is FtsKeyStatus.EXCESS, stale
    assert stale.excess_rows == 1

    report = _converge(adapter, _frame(test_db, (session_id,)))

    assert report.done == 1
    assert test_conn.execute("SELECT 1 FROM messages_fts_docsize WHERE id = ?", (second_rowid,)).fetchone() is None
    assert test_conn.execute("SELECT 1 FROM messages_fts_identity WHERE rowid = ?", (second_rowid,)).fetchone() is None
    assert adapter.inspect_partition(test_conn, session_id).valid


def _seed_blocks(conn: sqlite3.Connection, native_id: str, count: int) -> str:
    """Seed one session carrying ``count`` searchable blocks."""
    conn.execute(
        "INSERT INTO sessions(native_id, origin, title, content_hash) VALUES (?, 'unknown-export', 'test', ?)",
        (native_id, bytes([count % 251]) * 32),
    )
    session_id = f"unknown-export:{native_id}"
    conn.execute(
        "INSERT INTO messages(session_id, native_id, position, role, message_type, content_hash) "
        "VALUES (?, 'm0', 0, 'user', 'message', ?)",
        (session_id, b"m" * 32),
    )
    message_id = conn.execute("SELECT message_id FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0]
    for position in range(count):
        conn.execute(
            "INSERT INTO blocks(message_id, session_id, position, block_type, text, content_hash) "
            "VALUES (?, ?, ?, 'text', ?, ?)",
            (message_id, session_id, position, f"derivation input {position}", bytes([position % 251]) * 32),
        )
    conn.commit()
    return session_id


def test_partition_publish_chunks_rowids_under_the_connection_variable_limit(
    test_conn: sqlite3.Connection, test_db: Path
) -> None:
    """Anti-vacuity: with the connection's bind-variable limit lowered below the
    partition's block count, an unchunked ``rowid IN (...)`` list raises
    sqlite3.OperationalError('too many SQL variables') and this publish fails."""
    session_id = _seed_blocks(test_conn, "over-the-variable-limit", 6)
    adapter = _adapter(test_db)
    computed = adapter.input_for(test_conn, session_id)
    previous_limit = test_conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    test_conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 4)
    try:
        assert adapter.publish_partition(test_conn, computed) is True
    finally:
        test_conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous_limit)

    assert adapter.inspect_partition(test_conn, session_id).valid
    assert test_conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 6
