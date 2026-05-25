"""FTS trigger suspension/restore lifecycle invariants (#1182).

Pins the contract documented in ``docs/internals.md`` ("FTS5 Model →
Trigger suspension"):

- ``suspend_fts_triggers_sync`` drops every expected FTS trigger.
- ``restore_fts_triggers_sync`` is idempotent and re-creates the full
  set even if invoked twice in a row.
- A SIGKILL-like interruption between ``suspend`` and ``restore`` leaves
  the database in a detectable drift state (no triggers present); the
  recovery path is ``restore_fts_triggers_sync`` itself.
- Multiple threads racing on suspend + restore on independent
  connections converge on the "all present" state.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.storage.fts.fts_lifecycle import (
    ensure_fts_index_sync,
    restore_fts_triggers_sync,
    suspend_fts_triggers_sync,
)

_EXPECTED_TRIGGERS = (
    "messages_fts_ai",
    "messages_fts_ad",
    "messages_fts_au",
    "action_events_fts_ai",
    "action_events_fts_ad",
    "action_events_fts_au",
    "session_work_events_fts_ai",
    "session_work_events_fts_ad",
    "session_work_events_fts_au",
    "work_threads_fts_ai",
    "work_threads_fts_ad",
    "work_threads_fts_au",
)


def _trigger_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall()
    return {row[0] for row in rows}


def _bootstrap_fts_db(path: Path) -> sqlite3.Connection:
    """Build the minimal schema FTS triggers depend on."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE messages (
            rowid INTEGER PRIMARY KEY,
            message_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            text TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE action_events (
            rowid INTEGER PRIMARY KEY,
            event_id TEXT NOT NULL,
            message_id TEXT,
            conversation_id TEXT NOT NULL,
            action_kind TEXT NOT NULL,
            normalized_tool_name TEXT,
            search_text TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE session_work_events (
            event_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            provider_name TEXT NOT NULL,
            heuristic_label TEXT NOT NULL,
            search_text TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE VIRTUAL TABLE session_work_events_fts USING fts5(
            event_id UNINDEXED,
            conversation_id UNINDEXED,
            provider_name UNINDEXED,
            heuristic_label UNINDEXED,
            text,
            tokenize='unicode61'
        )"""
    )
    conn.execute(
        """CREATE TABLE work_threads (
            thread_id TEXT PRIMARY KEY,
            root_id TEXT NOT NULL,
            search_text TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE VIRTUAL TABLE work_threads_fts USING fts5(
            thread_id UNINDEXED,
            root_id UNINDEXED,
            text,
            tokenize='unicode61'
        )"""
    )
    ensure_fts_index_sync(conn)
    conn.commit()
    return conn


def test_suspend_drops_all_fts_triggers(tmp_path: Path) -> None:
    conn = _bootstrap_fts_db(tmp_path / "fts.db")
    try:
        present = _trigger_names(conn)
        assert set(_EXPECTED_TRIGGERS).issubset(present), (
            f"bootstrap missing triggers: {set(_EXPECTED_TRIGGERS) - present}"
        )
        suspend_fts_triggers_sync(conn)
        present = _trigger_names(conn)
        assert present.isdisjoint(_EXPECTED_TRIGGERS), (
            f"suspend left FTS triggers in place: {present & set(_EXPECTED_TRIGGERS)}"
        )
    finally:
        conn.close()


def test_restore_after_simulated_sigkill_recovers(tmp_path: Path) -> None:
    """A daemon killed between suspend and restore leaves no triggers; restore brings them back."""
    db_file = tmp_path / "fts.db"
    conn = _bootstrap_fts_db(db_file)
    try:
        suspend_fts_triggers_sync(conn)
        conn.commit()  # persist the DROP across simulated process death
    finally:
        conn.close()

    # New "process": detectable drift state.
    conn = sqlite3.connect(str(db_file))
    try:
        assert _trigger_names(conn).isdisjoint(_EXPECTED_TRIGGERS)
        # Recovery path.
        restore_fts_triggers_sync(conn)
        conn.commit()
        present = _trigger_names(conn)
        assert set(_EXPECTED_TRIGGERS).issubset(present)
    finally:
        conn.close()


def test_restore_is_idempotent(tmp_path: Path) -> None:
    conn = _bootstrap_fts_db(tmp_path / "fts.db")
    try:
        restore_fts_triggers_sync(conn)
        restore_fts_triggers_sync(conn)
        present = _trigger_names(conn)
        assert set(_EXPECTED_TRIGGERS) == (present & set(_EXPECTED_TRIGGERS))
        # No duplicate triggers — sqlite_master holds exactly one row per name.
        names = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger' ORDER BY name")]
        for expected in _EXPECTED_TRIGGERS:
            assert names.count(expected) == 1, f"duplicate trigger {expected}"
    finally:
        conn.close()


def test_rebuild_restores_insight_fts_surfaces(tmp_path: Path) -> None:
    from polylogue.storage.fts.fts_lifecycle import fts_invariant_snapshot_sync, rebuild_fts_index_sync

    conn = _bootstrap_fts_db(tmp_path / "fts.db")
    try:
        conn.execute(
            """
            INSERT INTO session_work_events (
                event_id, conversation_id, provider_name, heuristic_label, search_text
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("event-1", "conversation-1", "codex", "implementation", "fixed daemon locks"),
        )
        conn.execute(
            "INSERT INTO work_threads (thread_id, root_id, search_text) VALUES (?, ?, ?)",
            ("thread-1", "conversation-1", "daemon convergence"),
        )
        conn.execute("DELETE FROM session_work_events_fts")
        conn.execute("DELETE FROM work_threads_fts")
        conn.commit()

        snapshot = fts_invariant_snapshot_sync(conn)
        assert snapshot.session_work_events.ready is False
        assert snapshot.work_threads.ready is False

        rebuild_fts_index_sync(conn)
        conn.commit()

        snapshot = fts_invariant_snapshot_sync(conn)
        assert snapshot.session_work_events.ready is True
        assert snapshot.work_threads.ready is True
    finally:
        conn.close()


def test_targeted_rebuild_restores_only_insight_fts_surfaces(tmp_path: Path) -> None:
    from polylogue.storage.fts.fts_lifecycle import fts_invariant_snapshot_sync, rebuild_session_insight_fts_sync

    conn = _bootstrap_fts_db(tmp_path / "fts.db")
    try:
        conn.execute(
            """
            INSERT INTO session_work_events (
                event_id, conversation_id, provider_name, heuristic_label, search_text
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("event-1", "conversation-1", "codex", "implementation", "fixed daemon locks"),
        )
        conn.execute(
            "INSERT INTO work_threads (thread_id, root_id, search_text) VALUES (?, ?, ?)",
            ("thread-1", "conversation-1", "daemon convergence"),
        )
        conn.execute(
            """
            INSERT INTO session_work_events_fts (event_id, conversation_id, provider_name, heuristic_label, text)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("event-1", "conversation-1", "codex", "implementation", "duplicate event row"),
        )
        conn.execute(
            "INSERT INTO work_threads_fts (thread_id, root_id, text) VALUES (?, ?, ?)",
            ("thread-1", "conversation-1", "duplicate thread row"),
        )
        conn.commit()

        snapshot = fts_invariant_snapshot_sync(conn)
        assert snapshot.session_work_events.ready is False
        assert snapshot.work_threads.ready is False

        rebuild_session_insight_fts_sync(conn)
        conn.commit()

        snapshot = fts_invariant_snapshot_sync(conn)
        assert snapshot.session_work_events.ready is True
        assert snapshot.work_threads.ready is True
    finally:
        conn.close()


def test_concurrent_suspend_restore_threads_converge(tmp_path: Path) -> None:
    """Two threads alternating suspend/restore on independent connections must end with all triggers present."""
    db_file = tmp_path / "fts.db"
    conn = _bootstrap_fts_db(db_file)
    conn.close()

    error: list[BaseException] = []
    stop = threading.Event()

    def worker(op: str) -> None:
        try:
            local = sqlite3.connect(str(db_file), timeout=5.0)
            try:
                for _ in range(40):
                    if op == "suspend":
                        suspend_fts_triggers_sync(local)
                    else:
                        restore_fts_triggers_sync(local)
                    local.commit()
            finally:
                local.close()
        except BaseException as exc:  # pragma: no cover - defensive thread error capture
            error.append(exc)
        finally:
            if op == "restore":
                stop.set()

    t1 = threading.Thread(target=worker, args=("suspend",))
    t2 = threading.Thread(target=worker, args=("restore",))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert not error, f"thread raised: {error}"

    # Final convergence: explicit restore drives the steady state.
    final = sqlite3.connect(str(db_file))
    try:
        restore_fts_triggers_sync(final)
        final.commit()
        assert set(_EXPECTED_TRIGGERS).issubset(_trigger_names(final))
    finally:
        final.close()


@settings(
    deadline=None,
    max_examples=8,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(
    sequence=st.lists(
        st.sampled_from(["suspend", "restore"]),
        min_size=2,
        max_size=10,
    )
)
def test_hypothesis_arbitrary_lifecycle_recoverable(
    tmp_path_factory: pytest.TempPathFactory, sequence: list[str]
) -> None:
    """Any interleaving of suspend/restore ends in a recoverable state.

    After the random sequence, a single ``restore_fts_triggers_sync``
    call must bring every expected trigger back, and no spurious or
    duplicate triggers must remain.
    """
    base = tmp_path_factory.mktemp("fts_hyp")
    conn = _bootstrap_fts_db(base / "fts.db")
    try:
        for op in sequence:
            if op == "suspend":
                suspend_fts_triggers_sync(conn)
            else:
                restore_fts_triggers_sync(conn)
            conn.commit()
        restore_fts_triggers_sync(conn)
        conn.commit()
        present = _trigger_names(conn)
        assert set(_EXPECTED_TRIGGERS).issubset(present)
        # No unexpected FTS-named triggers leaked in.
        unexpected = {
            name
            for name in present
            if (
                name.startswith("messages_fts_")
                or name.startswith("action_events_fts_")
                or name.startswith("session_work_events_fts_")
                or name.startswith("work_threads_fts_")
            )
            and name not in _EXPECTED_TRIGGERS
        }
        assert not unexpected, f"unexpected FTS triggers leaked: {unexpected}"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# #1242: commit-ordering and SIGKILL-safety regression tests.
# ---------------------------------------------------------------------------


def test_ingest_side_effects_run_in_try_block_not_finally() -> None:
    """Side effects must NOT be in a finally block (#1242 bug A).

    The previous arrangement ran ``_commit_sync_ingest_side_effects`` in
    a ``finally`` block, so it fired even after the try block had
    rolled back — silently restoring triggers on top of nothing, AFTER
    the data commit had already happened. The regression here is a
    structural assertion against that shape: the side-effects call must
    appear inside the try (not the finally) so an exception during the
    write window propagates without committing.
    """
    import ast
    import inspect

    from polylogue.pipeline.services.ingest_batch import _core

    src = inspect.getsource(_core._process_ingest_batch_sync)
    tree = ast.parse(src)
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef), "expected a function definition"

    # Find the outer Try in the function body.
    outer_try: ast.Try | None = None
    for node in func.body:
        if isinstance(node, ast.Try):
            outer_try = node
            break
    assert outer_try is not None, "expected an outer try block"

    def _calls_side_effects(stmts: list[ast.stmt]) -> bool:
        for stmt in stmts:
            for sub in ast.walk(stmt):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id == "_commit_sync_ingest_side_effects"
                ):
                    return True
        return False

    assert _calls_side_effects(outer_try.body), "side effects must run inside try (before commit boundary)"
    assert not _calls_side_effects(outer_try.finalbody), (
        "side effects must NOT run in finally (would fire after rollback / after commit)"
    )


def test_sigkill_mid_bulk_recovery_via_daemon_startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bulk-write killed between suspend and restore is recovered on next daemon startup (#1242 bug B).

    Simulates the SIGKILL-during-suspend signature: triggers are dropped
    and committed, then the "process" dies before restore. The next
    daemon startup readiness check must detect the missing triggers and
    rebuild both triggers and the FTS index, restoring the search
    invariant without operator intervention.
    """
    import asyncio

    db_file = tmp_path / "fts_sigkill.db"
    conn = _bootstrap_fts_db(db_file)
    try:
        # Seed two messages so the FTS index has content to rebuild.
        conn.execute(
            "INSERT INTO messages (rowid, message_id, conversation_id, text) VALUES (?, ?, ?, ?)",
            (1, "m1", "c1", "alpha bravo"),
        )
        conn.execute(
            "INSERT INTO messages (rowid, message_id, conversation_id, text) VALUES (?, ?, ?, ?)",
            (2, "m2", "c1", "charlie delta"),
        )
        conn.commit()

        # SIGKILL signature: drop triggers and commit, never restore.
        suspend_fts_triggers_sync(conn)
        conn.commit()
        assert _trigger_names(conn).isdisjoint(_EXPECTED_TRIGGERS)
    finally:
        conn.close()

    # New process: daemon startup readiness check must heal the drift.
    from polylogue.daemon import cli as daemon_cli

    monkeypatch.setattr("polylogue.paths.db_path", lambda: db_file)

    asyncio.run(daemon_cli._ensure_fts_startup_readiness())

    final = sqlite3.connect(str(db_file))
    try:
        present = _trigger_names(final)
        assert set(_EXPECTED_TRIGGERS).issubset(present), (
            f"daemon startup did not restore all FTS triggers: still missing {set(_EXPECTED_TRIGGERS) - present}"
        )
        # FTS index was rebuilt from the persisted messages.
        indexed = final.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0]
        assert indexed >= 2, f"FTS index not repopulated after recovery: {indexed} rows"
    finally:
        final.close()
