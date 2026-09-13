"""Semantic laws for message FTS through the common derivation kernel."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

from polylogue.daemon.convergence import DaemonConverger
from polylogue.daemon.derivation import Budget, DerivationFrame, Outcome
from polylogue.storage.fts.derivation import GLOBAL_PARTITION, FtsDerivationAdapter, FtsKeyStatus
from polylogue.storage.fts.fts_lifecycle import restore_fts_triggers_sync


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


def _converge(adapter: FtsDerivationAdapter, frame: DerivationFrame, *, budget: Budget | None = None):
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
    assert report.wrote_nothing
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
    assert second.wrote_nothing


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
    test_conn.execute(
        "INSERT INTO messages_fts(rowid, block_id, message_id, session_id, block_type, text) "
        "VALUES (999999, 'orphan:block', 'orphan:message', 'orphan:session', 'text', 'orphan')"
    )
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
            "INSERT INTO messages_fts(rowid, block_id, message_id, session_id, block_type, text) "
            "VALUES (?, ?, 'orphan:message', 'orphan:session', 'text', 'orphan')",
            (rowid, f"orphan:{rowid}"),
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
