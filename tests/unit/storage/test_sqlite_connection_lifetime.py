"""Production sqlite connections are closed, not merely committed.

Anti-vacuity: restore any ``with sqlite_connection(...)`` in
``durable_change_train`` (or elsewhere in the archive bootstrap) to the builtin
``with sqlite3.connect(...)`` form and ``test_archive_bootstrap_closes_every_tier_connection``
goes red -- that form commits the transaction and leaks the descriptor. The
same applies to a helper that hands a bare connection to ``with``:
``test_reopening_an_existing_archive_closes_startup_reconcile_connections``
covers the reopen route, which creation-only coverage cannot reach.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.sqlite.managed_connection import sqlite_connection


def _open_fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


@pytest.mark.skipif(not Path("/proc/self/fd").exists(), reason="descriptor count needs /proc")
def test_builtin_connect_context_manager_leaks_the_descriptor(tmp_path: Path) -> None:
    """The premise: the builtin form commits but does not close."""
    before = _open_fd_count()
    for index in range(20):
        with sqlite3.connect(tmp_path / f"builtin-{index}.db") as connection:
            connection.execute("CREATE TABLE t (x)")
    assert _open_fd_count() - before == 20


@pytest.mark.skipif(not Path("/proc/self/fd").exists(), reason="descriptor count needs /proc")
def test_sqlite_connection_closes_and_still_commits(tmp_path: Path) -> None:
    path = tmp_path / "managed.db"
    before = _open_fd_count()
    for index in range(20):
        with sqlite_connection(path) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS t (x)")
            connection.execute("INSERT INTO t VALUES (?)", (index,))
    assert _open_fd_count() == before
    # The writes are durable: the commit half of the builtin form is preserved.
    with sqlite_connection(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 20


def test_sqlite_connection_rolls_back_and_closes_on_error(tmp_path: Path) -> None:
    path = tmp_path / "rollback.db"
    with sqlite_connection(path) as connection:
        connection.execute("CREATE TABLE t (x)")
    with pytest.raises(RuntimeError):
        with sqlite_connection(path) as connection:
            connection.execute("INSERT INTO t VALUES (1)")
            raise RuntimeError("boom")
    with sqlite_connection(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0


@pytest.mark.skipif(not Path("/proc/self/fd").exists(), reason="descriptor count needs /proc")
def test_archive_bootstrap_closes_every_tier_connection(tmp_path: Path) -> None:
    """The production route that exhausted an xdist worker's descriptor table."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    # One warm call so lazily opened caches are not counted as a leak.
    initialize_active_archive_root(tmp_path / "warm")
    before = _open_fd_count()
    for index in range(5):
        initialize_active_archive_root(tmp_path / f"root-{index}")
    assert _open_fd_count() == before


@pytest.mark.skipif(not Path("/proc/self/fd").exists(), reason="descriptor count needs /proc")
def test_reopening_an_existing_archive_closes_startup_reconcile_connections(tmp_path: Path) -> None:
    """Reopening an archive reconciles durable trains against existing tiers.

    Creation takes the fresh-bootstrap route and opens no existing tier, so
    only a reopen exercises ``_open_existing_tier``. Anti-vacuity: return the
    bare connection from that helper again and each reopen retains one
    descriptor per durable tier.
    """
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    root = tmp_path / "existing"
    initialize_active_archive_root(root)
    # One warm reopen so lazily opened caches are not counted as a leak.
    initialize_active_archive_root(root)
    before = _open_fd_count()
    for _ in range(5):
        initialize_active_archive_root(root)
    assert _open_fd_count() == before
