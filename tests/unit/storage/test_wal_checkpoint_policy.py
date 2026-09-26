"""WAL checkpoint ownership, escalation and hold accounting.

Anti-vacuity: every test here names a mutation of the policy in
``connection_profile.py`` or ``wal_checkpoint.py`` that makes it red -- an
owner that does not disable implicit autocheckpoint, a recurring escalation
that reaches TRUNCATE, a busy result that retries, or a blocker scan on a route
that did not ask for one.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from polylogue.daemon.write_coordinator import write_hold_budget_s
from polylogue.storage.sqlite import connection_profile, wal_checkpoint
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import (
    CHECKPOINT_ESCALATION_MODES,
    CHECKPOINT_HOLD_BUDGET_S,
    DAEMON_WRITE_CONNECTION_PROFILE,
    ISOLATED_TIER_WRITE_PROFILE,
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


@pytest.fixture
def seed_wal() -> Iterator[Callable[..., sqlite3.Connection]]:
    """Seed a WAL and leave a connection holding it open.

    SQLite deletes the ``-wal`` file when the *last* connection to a database
    closes, so a helper that closes its own seeding connection leaves no WAL
    behind at all. Every escalation assertion in this file would then be
    reading ``checkpoint_wal``'s absent-WAL size gate
    (``wal_checkpoint.py``: ``before < warn_bytes`` returns ``mode="none"``)
    rather than the escalation policy it names -- which is exactly how
    ``mode == "passive"`` and ``mode == "truncate"`` both came back ``"none"``.

    The connection is returned so a test can decide *who* keeps the WAL
    alive; the fixture closes whatever is still open at teardown.
    """
    held: list[sqlite3.Connection] = []

    def _seed(db: Path, *, rows: int) -> sqlite3.Connection:
        conn = sqlite3.connect(db)
        held.append(conn)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint = 0")
        conn.execute("CREATE TABLE IF NOT EXISTS payload (id INTEGER PRIMARY KEY, body BLOB)")
        conn.executemany("INSERT INTO payload (body) VALUES (?)", [(b"x" * 4096,) for _ in range(rows)])
        conn.commit()
        # Assert the premise rather than trusting it: a seeding route that
        # stops producing a WAL must fail here, not quietly turn every
        # escalation assertion below into a test of the size gate.
        wal = db.with_name(f"{db.name}-wal")
        assert wal.exists() and wal.stat().st_size > 0, f"no WAL seeded for {db}"
        return conn

    yield _seed
    for conn in held:
        conn.close()


# -- ownership ---------------------------------------------------------------


def test_unowned_process_keeps_bounded_autocheckpoint() -> None:
    assert not recurring_checkpoint_owner_armed()
    for profile in (WRITE_CONNECTION_PROFILE, DAEMON_WRITE_CONNECTION_PROFILE, ISOLATED_TIER_WRITE_PROFILE):
        assert _autocheckpoint(write_connection_pragma_statements(profile)) == WAL_AUTOCHECKPOINT_PAGES


def test_owned_process_disables_implicit_autocheckpoint_for_every_writer() -> None:
    with arm_recurring_checkpoint_owner():
        assert recurring_checkpoint_owner_armed()
        for profile in (WRITE_CONNECTION_PROFILE, DAEMON_WRITE_CONNECTION_PROFILE, ISOLATED_TIER_WRITE_PROFILE):
            assert _autocheckpoint(write_connection_pragma_statements(profile)) == OWNED_WAL_AUTOCHECKPOINT_PAGES
    assert not recurring_checkpoint_owner_armed()


def test_owned_autocheckpoint_reaches_the_opened_connection(
    tmp_path: Path, seed_wal: Callable[..., sqlite3.Connection]
) -> None:
    db = tmp_path / "owned.db"
    seed_wal(db, rows=1)
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


def test_checkpoint_below_the_warn_threshold_does_not_open_a_connection(
    tmp_path: Path, seed_wal: Callable[..., sqlite3.Connection]
) -> None:
    db = tmp_path / "quiet.db"
    seed_wal(db, rows=1)
    observation = wal_checkpoint.checkpoint_wal(db, reason="unit")
    assert observation.mode == "none"
    assert not observation.ran
    assert observation.wal_bytes_after == observation.wal_bytes_before


def test_recurring_checkpoint_stops_at_passive(tmp_path: Path, seed_wal: Callable[..., sqlite3.Connection]) -> None:
    db = tmp_path / "index.db"
    seed_wal(db, rows=64)
    observation = wal_checkpoint.checkpoint_wal(db, reason="unit", warn_bytes=1, escalation_bytes=1)
    assert observation.escalation == "recurring"
    assert observation.mode == "passive"


def test_exclusive_escalation_reaches_truncate(tmp_path: Path, seed_wal: Callable[..., sqlite3.Connection]) -> None:
    db = tmp_path / "index.db"
    seed_wal(db, rows=64)
    observation = wal_checkpoint.checkpoint_wal(
        db, reason="unit", escalation="exclusive", warn_bytes=1, escalation_bytes=1
    )
    assert observation.mode == "truncate"
    assert observation.wal_bytes_after == 0


def test_busy_reader_retains_the_wal_and_reports_evidence(
    tmp_path: Path, seed_wal: Callable[..., sqlite3.Connection]
) -> None:
    db = tmp_path / "index.db"
    writer = seed_wal(db, rows=64)
    reader = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM payload").fetchone()
        # The reader now holds the snapshot *and* is the only thing keeping the
        # WAL on disk. Dropping the writer first is what makes this test red if
        # the busy reader is ever removed: without it the WAL disappears with
        # the last connection and the observation reports mode="none", not a
        # blocked checkpoint. A second idle connection would have masked that.
        writer.close()
        observation = wal_checkpoint.checkpoint_wal(
            db, reason="unit", escalation="exclusive", warn_bytes=1, escalation_bytes=1
        )
    finally:
        reader.close()
    # A held read snapshot leaves frames the checkpoint cannot reclaim: the
    # WAL survives and the observation says so, rather than the helper looping
    # or escalating until the reader loses.
    assert observation.blocked
    assert observation.wal_bytes_after > 0


def test_blockers_are_not_scanned_unless_the_route_asks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed_wal: Callable[..., sqlite3.Connection]
) -> None:
    db = tmp_path / "index.db"
    seed_wal(db, rows=64)

    def refuse(_db: Path) -> tuple[str, ...]:
        raise AssertionError("an interactive route must not walk every process on the host")

    monkeypatch.setattr(wal_checkpoint, "_sqlite_file_holders", refuse)
    observation = wal_checkpoint.checkpoint_wal(db, reason="unit", warn_bytes=1, escalation_bytes=2**40)
    assert observation.blocking_processes == ()


def test_missing_tiers_are_skipped_rather_than_failing_the_sweep(
    tmp_path: Path, seed_wal: Callable[..., sqlite3.Connection]
) -> None:
    (tmp_path / "index.db").touch()
    seed_wal(tmp_path / "index.db", rows=1)
    observations = wal_checkpoint.checkpoint_archive_wals(tmp_path, reason="unit", warn_bytes=1)
    assert len(observations) == 1


def test_recurring_owner_includes_the_durable_audit_wal(
    tmp_path: Path, seed_wal: Callable[..., sqlite3.Connection]
) -> None:
    audit_db = tmp_path / "audit.db"
    writer = seed_wal(audit_db, rows=64)
    writer.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.AUDIT]}")
    observations = wal_checkpoint.checkpoint_archive_wals(tmp_path, reason="unit", warn_bytes=1)
    assert len(observations) == 1
    assert observations[0].mode == "passive"
    assert observations[0].log_pages > 0


def test_a_failed_open_reports_no_mode_and_the_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed_wal: Callable[..., sqlite3.Connection]
) -> None:
    """Nothing ran, so no mode is claimed -- but the failure is still evidence."""
    db = tmp_path / "index.db"
    seed_wal(db, rows=64)

    def refuse(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(wal_checkpoint, "open_daemon_connection", refuse)
    observation = wal_checkpoint.checkpoint_wal(db, reason="unit", warn_bytes=1)
    assert observation.mode == "none"
    assert not observation.ran
    assert observation.error == "database is locked"


def test_known_but_unauthorized_checkpoint_mode_is_refused(
    tmp_path: Path, seed_wal: Callable[..., sqlite3.Connection]
) -> None:
    db = tmp_path / "index.db"
    seed_wal(db, rows=1)
    conn = sqlite3.connect(db)
    try:
        with pytest.raises(ValueError, match="not permitted at exclusive boundary"):
            wal_checkpoint.checkpoint_connection(conn, "FULL", boundary="exclusive")
    finally:
        conn.close()


def test_recurring_boundary_refuses_exclusive_checkpoint_modes(
    tmp_path: Path, seed_wal: Callable[..., sqlite3.Connection]
) -> None:
    db = tmp_path / "index.db"
    seed_wal(db, rows=1)
    conn = sqlite3.connect(db)
    try:
        with pytest.raises(ValueError, match="not permitted at recurring boundary"):
            wal_checkpoint.checkpoint_connection(conn, "TRUNCATE", boundary="recurring")
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


def test_observation_reports_its_own_hold_against_the_budget(
    tmp_path: Path, seed_wal: Callable[..., sqlite3.Connection]
) -> None:
    db = tmp_path / "index.db"
    seed_wal(db, rows=64)
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
        conn.execute("CREATE TABLE payload (id INTEGER PRIMARY KEY, body BLOB)")
        conn.commit()
        # A second connection outlives the writer, so closing the writer cannot
        # run SQLite's last-connection checkpoint: the WAL survives exactly as
        # it would after a process died mid-run.
        survivor = connection_profile.open_readonly_connection(db, validate_schema=False)
        try:
            conn.executemany("INSERT INTO payload (body) VALUES (?)", [(b"y" * 4096,) for _ in range(256)])
            conn.commit()
            assert conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 0
            conn.close()
            assert db.with_suffix(".db-wal").stat().st_size > 0
            recovered = connection_profile.open_readonly_connection(db, validate_schema=False)
            try:
                assert recovered.execute("SELECT count(*) FROM payload").fetchone()[0] == 256
            finally:
                recovered.close()
        finally:
            survivor.close()

    observation = wal_checkpoint.checkpoint_wal(
        db, reason="restart", escalation="exclusive", warn_bytes=1, escalation_bytes=1
    )
    assert observation.mode == "truncate"
    assert observation.wal_bytes_after == 0


# -- the cold-build pass boundary --------------------------------------------


def test_the_cold_build_pass_boundary_checkpoints_through_the_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``finish_active_cold_build`` is a checkpoint caller like any other.

    It ran a raw ``PRAGMA wal_checkpoint(TRUNCATE)`` on the daemon's *live*
    writer connection, outside the escalation policy entirely. The active
    generation is read concurrently by the CLI, MCP and the daemon's own
    readers (see ``COLD_BUILD_ACTIVE_WRITE_CONNECTION_PROFILE``), and TRUNCATE
    takes the writer lock and waits for those readers to drain -- against a
    30 s busy timeout, once per ingest pass.

    Anti-vacuity, three mutations:
      * restoring the raw ``PRAGMA wal_checkpoint(TRUNCATE)`` leaves ``calls``
        empty;
      * naming ``boundary="exclusive"`` records the wrong boundary;
      * asking for TRUNCATE at the recurring boundary makes the owner raise
        ``ValueError`` out of ``finish_active_cold_build``.
    """
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    calls: list[tuple[str, str, tuple[int, int, int]]] = []
    real = wal_checkpoint.checkpoint_connection

    def record(conn: sqlite3.Connection, mode: str, *, boundary: str) -> tuple[int, int, int]:
        result = real(conn, mode, boundary=boundary)  # type: ignore[arg-type]
        calls.append((mode, boundary, result))
        return result

    monkeypatch.setattr(wal_checkpoint, "checkpoint_connection", record)

    with ArchiveStore.open_active_cold_build(tmp_path) as archive:
        assert archive.active_cold_build_engaged is True
        archive._conn.executemany(
            "INSERT INTO sessions (native_id, origin, content_hash) VALUES (?, ?, ?)",
            [(f"cold-{index}", "codex-session", bytes([index % 256]) * 32) for index in range(64)],
        )
        archive._conn.commit()
        wal = tmp_path / "index.db-wal"
        assert wal.exists() and wal.stat().st_size > 0, "no WAL to drain; the boundary would be vacuous"
        checkpoint_result = archive.finish_active_cold_build()
        assert archive.active_cold_build_engaged is False

    assert len(calls) == 1, calls
    assert checkpoint_result == calls[0][2]
    mode, boundary, (busy_pages, log_pages, checkpointed_pages) = calls[0]
    assert (mode, boundary) == ("PASSIVE", "recurring")
    # PASSIVE is a real drain here, not a downgrade to a no-op: nothing else
    # pins frames, so every logged frame is copied back and none is busy. The
    # WAL *file* keeps its size because a reset WAL is reused in place --
    # shrinking it is what ``journal_size_limit`` owns, not this boundary.
    assert log_pages > 0
    assert (busy_pages, checkpointed_pages) == (0, log_pages)
    assert "TRUNCATE" not in CHECKPOINT_ESCALATION_MODES["recurring"]
