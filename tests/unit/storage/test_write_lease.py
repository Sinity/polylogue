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

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import (
    open_connection,
    open_daemon_connection,
    open_isolated_write_connection,
)
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
        with pytest.raises(UnleasedWriteError):
            open_isolated_write_connection(db_path, purpose="probe")


def test_a_leased_write_open_succeeds(db_path: Path) -> None:
    """The lease authorizes; it does not merely record."""
    with arm_write_lease_enforcement(), write_lease("test.writer"):
        with closing(open_connection(db_path, validate_schema=False)) as conn:
            conn.execute("INSERT INTO t VALUES (1)")
            conn.commit()
    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute("SELECT count(*) FROM t").fetchone()[0] == 1


def test_archive_insight_writer_refuses_a_different_archive_root(tmp_path: Path) -> None:
    """A generation path cannot turn one archive's lease into another's writer.

    Anti-vacuity: omitting the explicit ``archive_root`` from the convergence
    writer lets this open succeed because the factory has no root to compare.
    """
    from polylogue.daemon.convergence_stages import _open_archive_insight_write_connection

    owner_root = tmp_path / "owner"
    target_root = tmp_path / "target"
    owner_root.mkdir()
    target_root.mkdir()
    target_db = target_root / "index.db"
    initialize_archive_database(target_db, ArchiveTier.INDEX)

    with (
        write_lease("test.owner", archive_root=owner_root),
        pytest.raises(UnleasedWriteError, match="outside the archive"),
    ):
        _open_archive_insight_write_connection(target_db, archive_root=target_root)


def test_checkpoint_writer_refuses_a_different_archive_root(tmp_path: Path) -> None:
    """The periodic checkpoint route carries the root it was admitted for.

    Anti-vacuity: dropping ``archive_root=archive_root`` from
    ``checkpoint_archive_wals`` reopens the target database under this lease.
    """
    from polylogue.storage.sqlite.wal_checkpoint import checkpoint_archive_wals

    owner_root = tmp_path / "owner"
    target_root = tmp_path / "target"
    owner_root.mkdir()
    target_root.mkdir()
    initialize_archive_database(target_root / "index.db", ArchiveTier.INDEX)

    with (
        write_lease("test.owner", archive_root=owner_root),
        pytest.raises(UnleasedWriteError, match="outside the archive"),
    ):
        checkpoint_archive_wals(target_root, reason="test", warn_bytes=0)


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


def test_a_child_task_cannot_inherit_its_parents_write_lease(db_path: Path) -> None:
    """A copied context marker is not authority to start another writer.

    Anti-vacuity: omitting the task-owner check in ``require_write_lease`` lets
    ``create_task`` inherit the context variable and open an overlapping
    writer while the owning task still holds the coordinator lease.
    """

    async def scenario() -> None:
        with arm_write_lease_enforcement(), write_lease("test.owner"):

            async def child_writer() -> None:
                with closing(open_connection(db_path, validate_schema=False)):
                    pass

            child = asyncio.create_task(child_writer())
            with pytest.raises(UnleasedWriteError, match="inherited by a child task"):
                await child

            # The owning task retains its own authority after refusing the
            # inherited child context.
            with closing(open_connection(db_path, validate_schema=False)):
                pass

    asyncio.run(scenario())


def test_require_write_lease_is_permissive_when_unarmed() -> None:
    assert require_write_lease("probe") is None


def test_a_readonly_open_never_needs_a_lease(db_path: Path) -> None:
    """The boundary is about writers; refusing readers would be overreach."""
    with arm_write_lease_enforcement():
        with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
            assert conn.execute("SELECT count(*) FROM t").fetchone()[0] >= 0


def test_every_write_mode_factory_in_storage_routes_through_the_lease() -> None:
    """The census the bead asks for: no second door into a writable tier.

    A new write-mode factory that forgets ``require_write_lease`` reintroduces
    exactly the defect this closes -- an in-process writer outside the gate,
    exhausting the busy timeout of whoever holds it. Enumerating the factories
    is what makes that a test failure rather than a review miss.

    Anti-vacuity: drop the ``require_write_lease`` call from any listed factory
    and this fails naming it.
    """
    import ast
    from pathlib import Path as _Path

    #: Every function in ``polylogue/storage`` that opens a write-mode SQLite
    #: connection to an archive tier. Read-only opens are deliberately absent.
    write_mode_factories = {
        "polylogue/storage/sqlite/connection_profile.py": {
            "open_connection",
            "open_daemon_connection",
            "open_isolated_write_connection",
        },
        "polylogue/storage/sqlite/connection.py": {"_get_cached_connection"},
    }

    repo_root = _Path(__file__).resolve().parents[3]
    unguarded: list[str] = []
    for relative, functions in write_mode_factories.items():
        module = ast.parse((repo_root / relative).read_text(encoding="utf-8"))
        found = {
            node.name: node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name in functions
        }
        assert set(found) == functions, f"{relative}: {functions - set(found)} no longer exist; update the census"
        for name, node in found.items():
            calls = {
                call.func.id
                for call in ast.walk(node)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            }
            if "require_write_lease" not in calls:
                unguarded.append(f"{relative}:{name}")
    assert unguarded == [], f"write-mode factories that do not take the lease: {unguarded}"


def test_the_cached_write_connection_is_refused_without_a_lease(tmp_path: Path) -> None:
    """The async runtime's thread-local writer is on the lease like any other."""
    from polylogue.storage.sqlite.connection import open_connection as cached_write_connection

    with arm_write_lease_enforcement():
        with pytest.raises(UnleasedWriteError):
            with cached_write_connection(tmp_path / "cached.db"):
                pass
