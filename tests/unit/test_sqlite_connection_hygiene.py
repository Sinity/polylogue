"""Regression coverage for the ``with sqlite3.connect(...)`` connection-leak sweep.

``with sqlite3.connect(path) as conn: ...`` is a well-known Python trap: the
context manager commits/rolls back the *transaction* on ``__exit__`` but never
closes the underlying connection (see ``polylogue/insights/otlp_correlation.py``
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
    "polylogue.daemon.backup": "sqlite3",
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
    from polylogue.daemon.backup import _sqlite_user_version

    captured = _capture_connections(monkeypatch, "polylogue.daemon.backup")
    assert _sqlite_user_version(versioned_db) == 7
    _assert_all_closed(captured)


def test_migration_runner_sqlite_user_version_closes_connection(
    monkeypatch: pytest.MonkeyPatch, versioned_db: Path
) -> None:
    from polylogue.storage.sqlite.migration_runner import _sqlite_user_version

    captured = _capture_connections(monkeypatch, "polylogue.storage.sqlite.migration_runner")
    assert _sqlite_user_version(versioned_db) == 7
    _assert_all_closed(captured)


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
    _checkpoint_truncate(db_path, label="test-checkpoint")
    _assert_all_closed(captured)


def test_archive_readiness_active_rebuild_attempts_closes_connection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from polylogue.storage.archive_readiness import active_rebuild_index_attempts

    ops_db = tmp_path / "ops.db"
    conn = sqlite3.connect(ops_db)
    try:
        conn.execute(
            """
            CREATE TABLE ingest_attempts (
                attempt_id TEXT, phase TEXT, status TEXT,
                started_at_ms INTEGER, heartbeat_at_ms INTEGER,
                parsed_raw_count INTEGER, materialized_count INTEGER
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    captured = _capture_connections(monkeypatch, "polylogue.storage.archive_readiness")
    result = active_rebuild_index_attempts(ops_db)
    assert result == []
    _assert_all_closed(captured)


@pytest.mark.parametrize("module_path", sorted(MODULE_TARGETS))
def test_swept_modules_do_not_reference_bare_with_connect(module_path: str) -> None:
    """Static companion check: source no longer contains the leak pattern.

    Complements the runtime captures above by asserting the exact textual
    pattern (``with sqlite3.connect(``) is gone from each swept module's
    source, so a careless partial revert (e.g. restoring one call site by
    hand while leaving the import) is caught even before the more expensive
    connection-capture tests run.
    """

    import importlib
    import inspect

    module = importlib.import_module(module_path)
    source = inspect.getsource(module)
    assert "with sqlite3.connect(" not in source, (
        f"{module_path} regressed to a bare 'with sqlite3.connect(...)' — wrap in "
        "contextlib.closing()/contextlib's closing so the connection is closed on exit, "
        "not just committed (polylogue-a7xr.1)."
    )
