"""WAL checkpoint ownership, escalation and hold accounting.

Anti-vacuity: every test here names a mutation of the policy in
``connection_profile.py`` or ``wal_checkpoint.py`` that makes it red -- an
owner that does not disable implicit autocheckpoint, a recurring escalation
that reaches TRUNCATE, a busy result that retries, or a blocker scan on a route
that did not ask for one.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.daemon.write_coordinator import write_hold_budget_s
from polylogue.storage.sqlite import connection_profile, wal_checkpoint
from polylogue.storage.sqlite.connection_profile import (
    CHECKPOINT_ESCALATION_MODES,
    CHECKPOINT_HOLD_BUDGET_S,
    DAEMON_WRITE_CONNECTION_PROFILE,
    OWNED_WAL_AUTOCHECKPOINT_PAGES,
    WAL_AUTOCHECKPOINT_PAGES,
    WRITE_CONNECTION_PROFILE,
    arm_recurring_checkpoint_owner,
    recurring_checkpoint_owner_armed,
    write_connection_pragma_statements,
)


def _autocheckpoint(statements: tuple[str, ...]) -> int:
    declared = [s for s in statements if s.startswith("PRAGMA wal_autocheckpoint")]
    assert len(declared) == 1, statements
    return int(declared[0].rsplit("=", 1)[1])


def _seed_wal(db: Path, *, rows: int) -> None:
    conn = sqlite3.connect(db)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint = 0")
        conn.execute("CREATE TABLE IF NOT EXISTS payload (id INTEGER PRIMARY KEY, body BLOB)")
        conn.executemany("INSERT INTO payload (body) VALUES (?)", [(b"x" * 4096,) for _ in range(rows)])
        conn.commit()
    finally:
        conn.close()


# -- ownership ---------------------------------------------------------------


def test_unowned_process_keeps_bounded_autocheckpoint() -> None:
    assert not recurring_checkpoint_owner_armed()
    assert _autocheckpoint(write_connection_pragma_statements(WRITE_CONNECTION_PROFILE)) == WAL_AUTOCHECKPOINT_PAGES
    assert (
        _autocheckpoint(write_connection_pragma_statements(DAEMON_WRITE_CONNECTION_PROFILE)) == WAL_AUTOCHECKPOINT_PAGES
    )


def test_owned_process_disables_implicit_autocheckpoint_for_every_writer() -> None:
    with arm_recurring_checkpoint_owner():
        assert recurring_checkpoint_owner_armed()
        for profile in (WRITE_CONNECTION_PROFILE, DAEMON_WRITE_CONNECTION_PROFILE):
            assert _autocheckpoint(write_connection_pragma_statements(profile)) == OWNED_WAL_AUTOCHECKPOINT_PAGES
    assert not recurring_checkpoint_owner_armed()


def test_owned_autocheckpoint_reaches_the_opened_connection(tmp_path: Path) -> None:
    db = tmp_path / "owned.db"
    _seed_wal(db, rows=1)
    with arm_recurring_checkpoint_owner():
        conn = connection_profile.open_daemon_connection(db, validate_schema=False)
        try:
            assert conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == OWNED_WAL_AUTOCHECKPOINT_PAGES
        finally:
            conn.close()
    conn = connection_profile.open_daemon_connection(db, validate_schema=False)
    try:
        assert conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == WAL_AUTOCHECKPOINT_PAGES
    finally:
        conn.close()


def test_read_profiles_are_never_reclassified_as_owned_writers() -> None:
    with arm_recurring_checkpoint_owner():
        statements = write_connection_pragma_statements(connection_profile.READ_CONNECTION_PROFILE)
    assert statements == connection_profile.READ_CONNECTION_PROFILE.pragma_statements


# -- escalation --------------------------------------------------------------


def test_recurring_escalation_never_reaches_restart_or_truncate() -> None:
    assert CHECKPOINT_ESCALATION_MODES["recurring"] == ("PASSIVE",)
    assert "TRUNCATE" not in CHECKPOINT_ESCALATION_MODES["quiescent"]
    assert CHECKPOINT_ESCALATION_MODES["exclusive"][-1] == "TRUNCATE"
    for modes in CHECKPOINT_ESCALATION_MODES.values():
        assert modes[0] == "PASSIVE"


def test_checkpoint_below_the_warn_threshold_does_not_open_a_connection(tmp_path: Path) -> None:
    db = tmp_path / "quiet.db"
    _seed_wal(db, rows=1)
    observation = wal_checkpoint.checkpoint_wal(db, reason="unit")
    assert observation.mode == "none"
    assert not observation.ran
    assert observation.wal_bytes_after == observation.wal_bytes_before


def test_recurring_checkpoint_stops_at_passive(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    _seed_wal(db, rows=64)
    observation = wal_checkpoint.checkpoint_wal(db, reason="unit", warn_bytes=1, escalation_bytes=1)
    assert observation.escalation == "recurring"
    assert observation.mode == "passive"


def test_exclusive_escalation_reaches_truncate(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    _seed_wal(db, rows=64)
    observation = wal_checkpoint.checkpoint_wal(
        db, reason="unit", escalation="exclusive", warn_bytes=1, escalation_bytes=2**40
    )
    assert observation.mode == "truncate"
    assert observation.wal_bytes_after == 0


def test_busy_reader_retains_the_wal_and_reports_evidence(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    _seed_wal(db, rows=64)
    reader = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM payload").fetchone()
        observation = wal_checkpoint.checkpoint_wal(
            db, reason="unit", escalation="exclusive", warn_bytes=1, escalation_bytes=2**40
        )
    finally:
        reader.close()
    # A held read snapshot leaves frames the checkpoint cannot reclaim: the
    # WAL survives and the observation says so, rather than the helper looping
    # or escalating until the reader loses.
    assert observation.blocked
    assert observation.wal_bytes_after > 0


def test_blockers_are_not_scanned_unless_the_route_asks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "index.db"
    _seed_wal(db, rows=64)

    def refuse(_db: Path) -> tuple[str, ...]:
        raise AssertionError("an interactive route must not walk every process on the host")

    monkeypatch.setattr(wal_checkpoint, "_sqlite_file_holders", refuse)
    observation = wal_checkpoint.checkpoint_wal(db, reason="unit", warn_bytes=1, escalation_bytes=2**40)
    assert observation.blocking_processes == ()


def test_missing_tiers_are_skipped_rather_than_failing_the_sweep(tmp_path: Path) -> None:
    (tmp_path / "index.db").touch()
    _seed_wal(tmp_path / "index.db", rows=1)
    observations = wal_checkpoint.checkpoint_archive_wals(tmp_path, reason="unit", warn_bytes=1)
    assert len(observations) == 1


def test_unsupported_checkpoint_mode_is_refused(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    _seed_wal(db, rows=1)
    conn = sqlite3.connect(db)
    try:
        with pytest.raises(ValueError, match="unsupported checkpoint mode"):
            wal_checkpoint.checkpoint_connection(conn, "FULL")
    finally:
        conn.close()


# -- hold accounting ---------------------------------------------------------


def test_checkpoint_hold_budget_is_separate_from_publication() -> None:
    checkpoint_budget = write_hold_budget_s("maintenance.wal_checkpoint")
    assert checkpoint_budget == CHECKPOINT_HOLD_BUDGET_S
    # A budget it shared with general maintenance or with a publication hold
    # could not show that checkpointing was the hold that grew.
    assert checkpoint_budget < write_hold_budget_s("maintenance.drive_catchup")
    assert checkpoint_budget < write_hold_budget_s("derivation.session")


def test_observation_reports_its_own_hold_against_the_budget(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    _seed_wal(db, rows=64)
    observation = wal_checkpoint.checkpoint_wal(db, reason="unit", warn_bytes=1, escalation_bytes=1)
    assert observation.elapsed_s >= 0.0
    assert not observation.over_hold_budget


# -- restart -----------------------------------------------------------------


def test_restart_from_a_large_wal_recovers_the_committed_rows(tmp_path: Path) -> None:
    """A process that dies without checkpointing leaves the WAL as the record.

    With ``wal_autocheckpoint = 0`` the WAL is the only place the last commits
    live, so recovery on reopen is what makes owned autocheckpoint safe rather
    than a data-loss trade.
    """
    db = tmp_path / "index.db"
    with arm_recurring_checkpoint_owner():
        conn = connection_profile.open_connection(db, validate_schema=False)
        try:
            conn.execute("CREATE TABLE payload (id INTEGER PRIMARY KEY, body BLOB)")
            conn.executemany("INSERT INTO payload (body) VALUES (?)", [(b"y" * 4096,) for _ in range(256)])
            conn.commit()
            assert conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 0
        finally:
            # Closing without checkpointing: the abandoned WAL is the state a
            # restart must recover from.
            conn.close()

    assert (db.with_suffix(".db-wal")).exists()
    recovered = connection_profile.open_readonly_connection(db, validate_schema=False)
    try:
        assert recovered.execute("SELECT count(*) FROM payload").fetchone()[0] == 256
    finally:
        recovered.close()

    observation = wal_checkpoint.checkpoint_wal(
        db, reason="restart", escalation="exclusive", warn_bytes=1, escalation_bytes=2**40
    )
    assert observation.mode == "truncate"
    assert observation.wal_bytes_after == 0
