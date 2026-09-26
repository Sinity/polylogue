"""Read-snapshot lifetime measured against what it actually costs: the WAL.

The premise these tests encode was measured, not assumed, on a synthetic WAL
archive with ``wal_autocheckpoint=0``. An *idle* read-only connection costs
nothing: a PASSIVE checkpoint recycled 6079 of 6079 frames straight past it, and
across eight 5k-row bursts the WAL plateaued at 2,978,792 bytes. A reader
holding one lazily stepped cursor made the same PASSIVE reclaim 0 of 6079, and
the WAL grew monotonically to 23,776,552 bytes -- the entire concurrent write
volume, for the cursor's lifetime.

Anti-vacuity, per test:

* ``test_a_streamed_frame_pins_the_wal_until_it_is_released`` is the measurement
  itself. It is red if ``ReadFrame.stream`` ever stopped holding a real cursor
  (then nothing would pin and the two arms would agree), and its second half is
  red if releasing the cursor did not restore reclamation.
* ``test_stream_refuses_past_the_declared_age_and_ends_the_pin`` is red if the
  per-row ``check()`` inside ``stream`` is removed: the export completes, no
  typed error is raised, and the checkpoint still reclaims nothing. Deleting the
  ``finally``/``cursor.close()`` instead keeps the raise but leaves the pin, so
  the reclamation half stays red. The opposite direction is pinned by
  ``test_stream_under_its_bound_returns_every_row``: a blanket refusal that
  never yields cannot pass.
* ``test_live_generation_frame_must_declare_a_maximum_age`` is red if the
  constructor accepts a live profile with the bound removed -- the exact shape
  that reintroduces an unbounded snapshot through the front door.
* the ``read_frame`` extension tests are red if the bound becomes silently
  droppable or extensible without a declared reason.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.storage.sqlite.connection_profile import (
    READ_PROFILES,
    SEALED_READ_CONNECTION_PROFILE,
    ReadFrame,
    ReadFrameCancelledError,
    ReadFrameExpiredError,
    live_read_frames,
    pinning_read_frames,
    read_frame,
)
from polylogue.storage.sqlite.wal_checkpoint import WalCheckpointObservation, checkpoint_wal

_PAYLOAD = "x" * 512
_ROWS = 4000


@pytest.fixture
def wal_db(tmp_path: Path) -> Path:
    """A WAL archive with implicit checkpointing off, as the daemon owner leaves it."""
    db = tmp_path / "index.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute("CREATE TABLE rows_ (position INTEGER PRIMARY KEY, body TEXT NOT NULL)")
        conn.executemany("INSERT INTO rows_ (body) VALUES (?)", [(_PAYLOAD,) for _ in range(_ROWS)])
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    return db


def _write_burst(db: Path, rows: int = _ROWS) -> None:
    conn = sqlite3.connect(db)
    try:
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.executemany("INSERT INTO rows_ (body) VALUES (?)", [(_PAYLOAD,) for _ in range(rows)])
        conn.commit()
    finally:
        conn.close()


def _recurring_checkpoint(db: Path) -> WalCheckpointObservation:
    """Run the production recurring owner's only permitted escalation.

    ``blocked`` is the production definition of a checkpoint that could not
    reclaim the whole log (``checkpointed_pages < log_pages``); SQLite reports
    both counters cumulatively since the WAL was last reset, so the comparison
    -- not the raw count -- is what says a reader held frames.
    """
    return checkpoint_wal(db, reason="unit", warn_bytes=1, escalation_bytes=1)


def _reset_wal(db: Path) -> None:
    conn = sqlite3.connect(db)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
    finally:
        conn.close()


# -- what a held snapshot costs ----------------------------------------------


def test_a_streamed_frame_pins_the_wal_until_it_is_released(wal_db: Path) -> None:
    """An idle frame is free; a streaming one stops PASSIVE reclaiming anything."""
    with read_frame(wal_db, timeout_class="background-read") as idle:
        idle.connection.execute("SELECT count(*) FROM rows_").fetchall()
        _write_burst(wal_db)
        idle_observation = _recurring_checkpoint(wal_db)
    assert idle_observation.log_pages > 0
    assert not idle_observation.blocked, "an idle read-only connection must not hold the log"

    _reset_wal(wal_db)
    with read_frame(wal_db, timeout_class="background-read") as streamed:
        rows = streamed.stream("SELECT position, body FROM rows_ ORDER BY position")
        next(rows)
        _write_burst(wal_db)
        pinned = _recurring_checkpoint(wal_db)
        assert pinned.log_pages > 0
        assert pinned.checkpointed_pages == 0, "a stream in flight must hold every frame the checkpoint wanted"
        assert pinned.blocked

        rows.close()
        released = _recurring_checkpoint(wal_db)
        assert released.checkpointed_pages == released.log_pages > 0, "releasing the cursor must restore reclamation"
        assert not released.blocked


def test_a_blocked_checkpoint_names_the_frame_that_held_it(wal_db: Path) -> None:
    with read_frame(wal_db, timeout_class="background-read") as frame:
        rows = frame.stream("SELECT position, body FROM rows_ ORDER BY position")
        next(rows)
        _write_burst(wal_db)
        observation = _recurring_checkpoint(wal_db)
        assert observation.blocked
        assert observation.blocking_read_frames, "the in-process snapshot must be named, not only a PID"
        assert "index.db:background-read" in observation.blocking_read_frames[0]
        rows.close()

    quiet = _recurring_checkpoint(wal_db)
    assert quiet.blocking_read_frames == ()


# -- the bound fails loudly and ends the pin ---------------------------------


def test_stream_under_its_bound_returns_every_row(wal_db: Path) -> None:
    """The opposite direction: a bound that refuses everything is not a bound."""
    with read_frame(wal_db, timeout_class="background-read") as frame:
        assert len(list(frame.stream("SELECT position FROM rows_ ORDER BY position"))) == _ROWS


def test_stream_refuses_past_the_declared_age_and_ends_the_pin(wal_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with read_frame(wal_db, timeout_class="interactive-read") as frame:
        rows = frame.stream("SELECT position, body FROM rows_ ORDER BY position")
        next(rows)
        _write_burst(wal_db)
        assert _recurring_checkpoint(wal_db).blocked

        monkeypatch.setattr(type(frame), "age_s", property(lambda _self: 1_000.0))
        with pytest.raises(ReadFrameExpiredError) as raised:
            next(rows)

        message = str(raised.value)
        assert "index.db" in message
        assert "1000.0s" in message
        assert "30.0s maximum" in message
        assert "interactive-read" in message

        assert not frame.streaming
        released = _recurring_checkpoint(wal_db)
        assert not released.blocked, "the typed expiry must end the pin, not just report it"
        assert released.checkpointed_pages == released.log_pages > 0


def test_one_expensive_sqlite_step_cannot_outlive_the_frame(wal_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with read_frame(wal_db, timeout_class="interactive-read") as frame:
        age_checks = 0

        def age_during_step(_frame: ReadFrame) -> float:
            nonlocal age_checks
            age_checks += 1
            return 0.0 if age_checks < 3 else 1_000.0

        monkeypatch.setattr(ReadFrame, "age_s", property(age_during_step))
        with pytest.raises(ReadFrameExpiredError):
            list(
                frame.stream(
                    "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n WHERE x < 100000) "
                    "SELECT sum(x) FROM n"
                )
            )
        assert age_checks >= 3
        assert not frame.streaming


def test_cancellation_during_one_expensive_sqlite_step_is_typed(wal_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with read_frame(wal_db, timeout_class="interactive-read") as frame:
        age_checks = 0

        def cancel_during_step(_frame: ReadFrame) -> float:
            nonlocal age_checks
            age_checks += 1
            if age_checks == 3:
                frame.cancel()
            return 0.0

        monkeypatch.setattr(ReadFrame, "age_s", property(cancel_during_step))
        with pytest.raises(ReadFrameCancelledError):
            list(
                frame.stream(
                    "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n WHERE x < 100000) "
                    "SELECT sum(x) FROM n"
                )
            )
        assert age_checks >= 3
        assert not frame.streaming


def test_a_frame_cannot_rebind_out_from_under_an_in_flight_stream(wal_db: Path) -> None:
    with read_frame(wal_db, timeout_class="background-read") as frame:
        rows = frame.stream("SELECT position FROM rows_ ORDER BY position")
        next(rows)
        with pytest.raises(ReadFrameExpiredError, match="stream is in flight"):
            frame.rebind()
        rows.close()
        frame.rebind()


# -- the bound cannot be dropped ---------------------------------------------


def test_live_generation_frame_must_declare_a_maximum_age(wal_db: Path) -> None:
    unbounded = replace(READ_PROFILES["background-read"], max_snapshot_age_s=None)
    with pytest.raises(ValueError, match="must declare max_snapshot_age_s"):
        ReadFrame(wal_db, profile=unbounded)


def test_a_sealed_generation_is_the_only_unbounded_frame(wal_db: Path) -> None:
    frame = ReadFrame(wal_db, profile=SEALED_READ_CONNECTION_PROFILE)
    try:
        assert frame.status().max_snapshot_age_s is None
        assert not frame.status().overdue
    finally:
        frame.close()


def test_extending_the_bound_requires_a_declared_reason(wal_db: Path) -> None:
    with pytest.raises(ValueError, match="requires a reason"):
        read_frame(wal_db, timeout_class="background-read", max_snapshot_age_s=1800.0)
    with pytest.raises(ValueError, match="positive number of seconds"):
        read_frame(wal_db, timeout_class="background-read", max_snapshot_age_s=0.0, reason="whole-archive export")
    with pytest.raises(ValueError, match="pass max_snapshot_age_s"):
        read_frame(wal_db, timeout_class="background-read", reason="whole-archive export")


def test_an_extended_bound_is_carried_into_the_refusal_and_the_registry(
    wal_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with read_frame(
        wal_db,
        timeout_class="background-read",
        max_snapshot_age_s=1800.0,
        reason="whole-archive transcript export",
    ) as frame:
        status = frame.status()
        assert status.max_snapshot_age_s == 1800.0
        assert status.reason == "whole-archive transcript export"

        monkeypatch.setattr(type(frame), "age_s", property(lambda _self: 3_600.0))
        with pytest.raises(ReadFrameExpiredError, match="whole-archive transcript export"):
            frame.check()


# -- the registry is the snapshot owner --------------------------------------


def test_the_registry_holds_every_open_frame_and_releases_closed_ones(wal_db: Path) -> None:
    before = len(live_read_frames())
    frame = read_frame(wal_db, timeout_class="background-read")
    try:
        described = [status.describe() for status in live_read_frames()]
        assert len(described) == before + 1
        assert any("index.db:background-read" in entry for entry in described)
        # Open but idle is not pinning; only a stream is.
        assert pinning_read_frames(wal_db) == ()
    finally:
        frame.close()
    assert len(live_read_frames()) == before


def test_overdue_is_reported_against_the_declared_maximum(wal_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with read_frame(wal_db, timeout_class="interactive-read") as frame:
        assert not frame.status().overdue
        monkeypatch.setattr(type(frame), "age_s", property(lambda _self: 1_000.0))
        status = frame.status()
        assert status.overdue
        assert "max=30s" in status.describe()
