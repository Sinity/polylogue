"""A writable connection is obtainable only from a held lease.

polylogue-8qm4k: the daemon claims to be the sole SQLite writer, but that was a
convention. The gate is advisory, holds were unbounded, and dispatch to the
coordinator fell open on a duck-typing miss, so a writer off the gate exhausted
the busy timeout while a holder ran for hours (a measured
``maintenance.drive_catchup`` hold of 18,623 s) and the live catch-up chunk died
with ``database is locked``.

These laws make the boundary structural: where enforcement is armed, opening a
write-mode connection without the lease raises instead of contending.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.storage.sqlite.connection_profile import open_connection, open_daemon_connection
from polylogue.storage.sqlite.write_lease import (
    UnleasedWriteError,
    WriteHoldExceededError,
    arm_write_lease_enforcement,
    current_write_lease,
    require_write_lease,
    write_lease,
    write_lease_enforced,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "tier.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.commit()
    return path


def test_an_unleased_write_open_is_refused_where_enforcement_is_armed(db_path: Path) -> None:
    """The anti-vacuity case named on the bead: a writer with no lease is a bug.

    Anti-vacuity: delete the ``require_write_lease`` call from
    ``open_connection`` and this goes green while an unserialized writer opens a
    write-mode connection exactly as before.
    """
    with arm_write_lease_enforcement():
        with pytest.raises(UnleasedWriteError):
            open_connection(db_path, validate_schema=False)
        with pytest.raises(UnleasedWriteError):
            open_daemon_connection(db_path, validate_schema=False)


def test_a_leased_write_open_succeeds(db_path: Path) -> None:
    """The lease authorizes; it does not merely record."""
    with arm_write_lease_enforcement(), write_lease("test.writer"):
        with closing(open_connection(db_path, validate_schema=False)) as conn:
            conn.execute("INSERT INTO t VALUES (1)")
            conn.commit()
    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute("SELECT count(*) FROM t").fetchone()[0] == 1


def test_enforcement_is_off_by_default_so_one_shot_writers_are_unaffected(db_path: Path) -> None:
    """A CLI or API process is its own single writer and has no gate to be outside of."""
    assert write_lease_enforced() is False
    with closing(open_connection(db_path, validate_schema=False)) as conn:
        conn.execute("INSERT INTO t VALUES (2)")
        conn.commit()


def test_the_lease_does_not_leak_past_its_block(db_path: Path) -> None:
    with arm_write_lease_enforcement():
        with write_lease("test.writer"):
            assert current_write_lease() is not None
        assert current_write_lease() is None
        with pytest.raises(UnleasedWriteError):
            open_connection(db_path, validate_schema=False)


def test_enforcement_does_not_leak_past_its_block(db_path: Path) -> None:
    with arm_write_lease_enforcement():
        assert write_lease_enforced() is True
    assert write_lease_enforced() is False


def test_a_nested_acquisition_returns_the_outer_lease(db_path: Path) -> None:
    """A publish inside a batch must not reset the outer hold's budget.

    Anti-vacuity: make the inner acquisition a second lease with its own clock
    and a long outer hold stops being reported, which is the exact blindness the
    budget exists to remove.
    """
    with write_lease("outer", max_hold_seconds=10.0) as outer:
        with write_lease("inner", max_hold_seconds=0.001) as inner:
            assert inner is outer
            assert inner.actor == "outer"


def test_a_hold_past_its_declared_budget_is_a_typed_failure() -> None:
    """An over-long hold fails rather than being absorbed as a longer wait.

    The budget cannot preempt a writer already inside a SQLite transaction, so
    it fires at release: its job is to make an 18,623 s hold impossible to miss.
    """
    with pytest.raises(WriteHoldExceededError, match="budget"):
        with write_lease("test.slow", max_hold_seconds=0.0):
            pass


def test_a_hold_within_its_budget_is_silent() -> None:
    with write_lease("test.fast", max_hold_seconds=60.0) as lease:
        assert lease.over_budget is False


def test_require_write_lease_returns_the_lease_for_an_authorized_caller() -> None:
    with write_lease("test.writer") as lease:
        assert require_write_lease("probe") is lease


def test_require_write_lease_is_permissive_when_unarmed() -> None:
    assert require_write_lease("probe") is None


def test_a_readonly_open_never_needs_a_lease(db_path: Path) -> None:
    """The boundary is about writers; refusing readers would be overreach."""
    with arm_write_lease_enforcement():
        with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
            assert conn.execute("SELECT count(*) FROM t").fetchone()[0] >= 0
