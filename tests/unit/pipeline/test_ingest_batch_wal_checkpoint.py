"""Ingest-batch upkeep: planner statistics, and no checkpointing of its own."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.sqlite.wal_checkpoint import (
    WalCheckpointObservation,
    checkpoint_archive_wals,
    checkpoint_wal,
)


def test_maybe_optimize_sqlite_runs_bounded_pragma(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.maintenance import maybe_optimize_sqlite

    db_path = tmp_path / "optimize.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE sample (id INTEGER PRIMARY KEY, value TEXT)")
        conn.executemany("INSERT INTO sample(value) VALUES (?)", [("a",), ("b",)])
        observation = maybe_optimize_sqlite(conn, reason="test", analysis_limit=17)

    assert observation.ran is True
    assert observation.reason == "test"
    assert observation.analysis_limit == 17
    assert observation.error is None


def test_maybe_optimize_archive_tiers_covers_existing_split_tiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite.maintenance import maybe_optimize_archive_tiers

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    for filename in ("source.db", "index.db", "ops.db"):
        (archive_root / filename).write_bytes(b"sqlite placeholder")

    class FakeConnection:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.closed = False

        def execute(self, _sql: str) -> object:
            return object()

        def close(self) -> None:
            self.closed = True

    opened: list[FakeConnection] = []

    def fake_open(path: Path, *, timeout: float, archive_root: Path) -> FakeConnection:
        # An armed daemon write lease refuses a tier open that omits its archive.
        assert timeout == 11.0
        assert archive_root == path.parent
        conn = FakeConnection(path)
        opened.append(conn)
        return conn

    monkeypatch.setattr("polylogue.storage.sqlite.connection_profile.open_daemon_connection", fake_open)

    observations = maybe_optimize_archive_tiers(archive_root, reason="test", analysis_limit=19, timeout_s=11.0)

    assert [conn.path.name for conn in opened] == ["source.db", "index.db", "ops.db"]
    assert [observation.ran for observation in observations] == [True, True, True]
    assert {observation.analysis_limit for observation in observations} == {19}
    assert all(conn.closed for conn in opened)


def test_checkpoint_archive_wals_covers_existing_split_tiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    for filename in ("source.db", "index.db", "user.db"):
        (archive_root / filename).write_bytes(b"sqlite placeholder")

    calls: list[tuple[Path, str, str]] = []

    def fake_checkpoint(
        db: Path,
        *,
        reason: str,
        escalation: str = "recurring",
        **_: object,
    ) -> WalCheckpointObservation:
        calls.append((db, reason, escalation))
        return WalCheckpointObservation(
            reason=reason,
            mode="passive",
            escalation=escalation,  # type: ignore[arg-type]
            wal_bytes_before=100,
            wal_bytes_after=0,
        )

    monkeypatch.setattr("polylogue.storage.sqlite.wal_checkpoint.checkpoint_wal", fake_checkpoint)

    observations = checkpoint_archive_wals(archive_root, reason="periodic")

    assert [path.name for path, _reason, _escalation in calls] == ["source.db", "index.db", "user.db"]
    assert {reason for _path, reason, _escalation in calls} == {"periodic"}
    assert {escalation for _path, _reason, escalation in calls} == {"recurring"}
    assert [observation.mode for observation in observations] == ["passive", "passive", "passive"]


def test_checkpoint_wal_reports_blocking_processes_when_the_route_asks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "index.db"

    class FakeConnection:
        def execute(self, sql: str) -> object:
            # ``main`` only: an attached read-only tier is never backfilled.
            assert sql == "PRAGMA main.wal_checkpoint(PASSIVE)"
            return self

        def fetchone(self) -> tuple[int, int, int]:
            return (1, 25, 12)

        def close(self) -> None:
            return None

    monkeypatch.setattr("polylogue.storage.sqlite.wal_checkpoint._wal_size", lambda db: 1024)
    monkeypatch.setattr(
        "polylogue.storage.sqlite.wal_checkpoint.open_daemon_connection", lambda *_args, **_kwargs: FakeConnection()
    )
    monkeypatch.setattr(
        "polylogue.storage.sqlite.wal_checkpoint._sqlite_file_holders",
        lambda db: ("1234:polylogue-mcp",) if db == db_path else (),
    )

    observation = checkpoint_wal(db_path, reason="test", warn_bytes=0, escalation_bytes=0, collect_blockers=True)

    assert observation.busy_pages == 1
    assert observation.blocking_processes == ("1234:polylogue-mcp",)
