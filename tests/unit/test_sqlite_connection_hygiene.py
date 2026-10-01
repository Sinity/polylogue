"""Regression coverage for the ``with sqlite3.connect(...)`` connection-leak sweep.

``with sqlite3.connect(path) as conn: ...`` is a well-known Python trap: the
context manager commits/rolls back the *transaction* on ``__exit__`` but never
closes the underlying connection
for the canonical documented fix). Left bare, the connection object is only
closed later by CPython's refcounting/GC — which is unreliable under held
references, reference cycles, or long-lived daemon loops — leaking file
descriptors under sustained load (Ref polylogue-a7xr.1).

Each helper under test here was swept from a bare ``with sqlite3.connect(...)``
to ``with closing(sqlite3.connect(...))`` (or ``contextlib.closing`` — see
per-module import style). This test proves the production dependency: it
patches the target module's ``sqlite3.connect`` to capture the live
``Connection`` object the helper creates, calls the real helper against a tiny
on-disk fixture database, and then asserts the captured connection is closed
(a closed ``sqlite3.Connection`` raises ``ProgrammingError`` on any further
operation). Reverting any one of these call sites back to a bare
``with sqlite3.connect(...) as conn:`` makes the connection outlive the
function returning it (since the wrapping test keeps the sole external
reference alive) and this test fails.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

MODULE_TARGETS: dict[str, str] = {
    "polylogue.storage.backup_package": "sqlite3",
    "polylogue.storage.sqlite.migration_runner": "sqlite3",
    "polylogue.cli.commands.paths": "sqlite3",
    "polylogue.storage.index_generation": "sqlite3",
    "polylogue.storage.archive_readiness": "sqlite3",
}


def _capture_connections(monkeypatch: pytest.MonkeyPatch, module_path: str) -> list[sqlite3.Connection]:
    """Patch ``sqlite3.connect`` inside ``module_path`` to record live connections."""

    import importlib

    module = importlib.import_module(module_path)
    captured: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def _tracking_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn: sqlite3.Connection = real_connect(*args, **kwargs)
        captured.append(conn)
        return conn

    monkeypatch.setattr(module.sqlite3, "connect", _tracking_connect)
    return captured


def _assert_all_closed(conns: list[sqlite3.Connection]) -> None:
    assert conns, "target helper never called sqlite3.connect — test fixture drifted from source"
    for conn in conns:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


@pytest.fixture
def versioned_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "versioned.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA user_version = 7")
        conn.commit()
    finally:
        conn.close()
    return db_path


def test_daemon_backup_sqlite_user_version_closes_connection(
    monkeypatch: pytest.MonkeyPatch, versioned_db: Path
) -> None:
    from polylogue.storage.backup_package import _sqlite_user_version

    captured = _capture_connections(monkeypatch, "polylogue.storage.backup_package")
    assert _sqlite_user_version(versioned_db) == 7
    _assert_all_closed(captured)


def test_migration_runner_sqlite_user_version_closes_connection(
    monkeypatch: pytest.MonkeyPatch, versioned_db: Path
) -> None:
    from polylogue.storage.sqlite.migration_runner import _sqlite_user_version

    captured = _capture_connections(monkeypatch, "polylogue.storage.sqlite.migration_runner")
    assert _sqlite_user_version(versioned_db) == 7
    _assert_all_closed(captured)


def test_migration_runner_reads_a_live_tier_user_version_through_its_wal(versioned_db: Path) -> None:
    """A retried restore can retain committed WAL state on the live tier.

    Anti-vacuity: the sealed read skips the WAL and would report 7; the owner
    refuses it, and the ``live`` read returns the committed 8.
    """
    from polylogue.storage.sqlite.connection_profile import LiveGenerationImmutableError
    from polylogue.storage.sqlite.migration_runner import _sqlite_user_version

    writer = sqlite3.connect(versioned_db)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint = 0")
        writer.execute("PRAGMA user_version = 8")
        writer.commit()
        with pytest.raises(LiveGenerationImmutableError):
            _sqlite_user_version(versioned_db)
        assert _sqlite_user_version(versioned_db, live=True) == 8
    finally:
        writer.close()


def test_immutable_read_accepts_an_inactive_persistent_journal(versioned_db: Path) -> None:
    """A PERSIST-mode journal left after commit has a zeroed header and is not live state.

    Anti-vacuity: judging the journal by size alone refuses this committed
    snapshot; a journal with a non-zero header is still refused.
    """
    from polylogue.storage.sqlite.connection_profile import LiveGenerationImmutableError
    from polylogue.storage.sqlite.migration_runner import _sqlite_user_version

    writer = sqlite3.connect(versioned_db)
    try:
        writer.execute("PRAGMA journal_mode=PERSIST")
        writer.execute("PRAGMA user_version = 9")
        writer.commit()
    finally:
        writer.close()
    journal = versioned_db.with_name(versioned_db.name + "-journal")
    assert journal.stat().st_size > 0
    assert _sqlite_user_version(versioned_db) == 9
    journal.write_bytes(bytes.fromhex("d9d505f920a163d7") + bytes(504))
    with pytest.raises(LiveGenerationImmutableError):
        _sqlite_user_version(versioned_db)


def test_immutable_read_checks_the_wal_beside_a_symlink_target(versioned_db: Path, tmp_path: Path) -> None:
    """Sidecars are judged beside the resolved target, not beside the symlink.

    Anti-vacuity: checking ``<link>-wal`` finds nothing and the immutable open
    silently skips the target's committed WAL row.
    """
    from polylogue.storage.sqlite.connection_profile import LiveGenerationImmutableError
    from polylogue.storage.sqlite.migration_runner import _sqlite_user_version

    link = tmp_path / "linked.db"
    link.symlink_to(versioned_db)
    writer = sqlite3.connect(versioned_db)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint = 0")
        writer.execute("PRAGMA user_version = 8")
        writer.commit()
        with pytest.raises(LiveGenerationImmutableError):
            _sqlite_user_version(link)
    finally:
        writer.close()


def test_cli_paths_read_user_version_closes_connection(monkeypatch: pytest.MonkeyPatch, versioned_db: Path) -> None:
    from polylogue.cli.commands.paths import _read_user_version

    captured = _capture_connections(monkeypatch, "polylogue.cli.commands.paths")
    assert _read_user_version(versioned_db) == 7
    _assert_all_closed(captured)


def test_index_generation_checkpoint_truncate_closes_connection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from polylogue.storage.index_generation import _checkpoint_truncate

    db_path = tmp_path / "checkpoint.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.commit()
    finally:
        conn.close()

    captured = _capture_connections(monkeypatch, "polylogue.storage.index_generation")
    _checkpoint_truncate(db_path, label="test-checkpoint", archive_root=tmp_path)
    _assert_all_closed(captured)
