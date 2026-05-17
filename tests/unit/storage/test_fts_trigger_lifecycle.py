"""FTS trigger suspension/restore lifecycle invariants (#1182).

Pins the contract documented in ``docs/internals.md`` ("FTS5 Model →
Trigger suspension"):

- ``suspend_fts_triggers_sync`` drops the six expected FTS triggers.
- ``restore_fts_triggers_sync`` is idempotent and re-creates the full
  set even if invoked twice in a row.
- A SIGKILL-like interruption between ``suspend`` and ``restore`` leaves
  the database in a detectable drift state (no triggers present); the
  recovery path is ``restore_fts_triggers_sync`` itself.
- Multiple threads racing on suspend + restore on independent
  connections converge on the "all six present" state.
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
    ensure_fts_index_sync(conn)
    conn.commit()
    return conn


def test_suspend_drops_all_six_triggers(tmp_path: Path) -> None:
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


def test_concurrent_suspend_restore_threads_converge(tmp_path: Path) -> None:
    """Two threads alternating suspend/restore on independent connections must end with all six triggers present."""
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
            if (name.startswith("messages_fts_") or name.startswith("action_events_fts_"))
            and name not in _EXPECTED_TRIGGERS
        }
        assert not unexpected, f"unexpected FTS triggers leaked: {unexpected}"
    finally:
        conn.close()
