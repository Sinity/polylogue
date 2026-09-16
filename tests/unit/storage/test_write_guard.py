"""No archive-tier writer exists outside the lease, factory or not.

polylogue-8qm4k asks for the anti-vacuity case in the form the rehearsal
produced it: a writer that never touches a declared write-mode factory, opens
``source.db`` directly, and locks out the daemon's live catch-up chunk. The
factory census in ``test_write_lease.py`` cannot see such a writer, because it
enumerates functions rather than connections.

These laws pin the connection-level boundary: while the guard is installed, a
writable archive-tier open without the lease raises, a reader is untouched, and
nothing but the one declared seam gets through.
"""

from __future__ import annotations

import ast
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.storage.sqlite.write_guard import (
    ARCHIVE_TIER_FILENAMES,
    archive_write_guard_installed,
    declared_unguarded_write,
    guarded_archive_tier_path,
    install_archive_write_guard,
)
from polylogue.storage.sqlite.write_lease import (
    UnleasedWriteError,
    arm_write_lease_enforcement,
    write_lease,
)


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    """A scratch archive with a real ``source.db``, built before the guard arms."""
    source = tmp_path / "source.db"
    with closing(sqlite3.connect(source)) as conn:
        conn.execute("CREATE TABLE raw_sessions (raw_id TEXT PRIMARY KEY)")
        conn.commit()
    return tmp_path


def test_an_unleased_writer_that_never_used_a_factory_is_refused(archive: Path) -> None:
    """The bead's named double: a writer straight onto ``source.db``.

    Anti-vacuity: drop ``install_archive_write_guard`` from the block and this
    test opens the connection exactly as the 2026-09-05 rehearsal writer did,
    with nothing raising.
    """
    with install_archive_write_guard(), arm_write_lease_enforcement():
        with pytest.raises(UnleasedWriteError) as raised:
            sqlite3.connect(archive / "source.db")
    assert "source.db" in str(raised.value)


def test_the_same_writer_succeeds_under_the_lease(archive: Path) -> None:
    with install_archive_write_guard(), arm_write_lease_enforcement():
        with write_lease("test.writer", archive_root=archive):
            with closing(sqlite3.connect(archive / "source.db")) as conn:
                conn.execute("INSERT INTO raw_sessions (raw_id) VALUES ('r1')")
                conn.commit()
    with closing(sqlite3.connect(archive / "source.db")) as conn:
        assert conn.execute("SELECT count(*) FROM raw_sessions").fetchone()[0] == 1


def test_readers_are_untouched(archive: Path) -> None:
    """Refusing readers would be overreach; the boundary is about writers."""
    with install_archive_write_guard(), arm_write_lease_enforcement():
        uri = f"file:{archive / 'source.db'}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            assert conn.execute("SELECT count(*) FROM raw_sessions").fetchone()[0] >= 0
        immutable = f"file:{archive / 'source.db'}?mode=ro&immutable=1"
        with closing(sqlite3.connect(immutable, uri=True)) as conn:
            assert conn.execute("SELECT count(*) FROM raw_sessions").fetchone()[0] >= 0


def test_non_archive_databases_are_untouched(tmp_path: Path) -> None:
    """Spill files, provider caches and scratch databases are not tiers.

    Anti-vacuity: widen the guard to every ``.db`` file and this fails, because
    a revision-backfill spill and the Sinex database are legitimately written
    without the archive lease.
    """
    with install_archive_write_guard(), arm_write_lease_enforcement():
        with closing(sqlite3.connect(tmp_path / "spill.db")) as conn:
            conn.execute("CREATE TABLE t (x INTEGER)")
        with closing(sqlite3.connect(":memory:")) as conn:
            conn.execute("CREATE TABLE t (x INTEGER)")


def test_every_declared_tier_name_is_guarded(archive: Path) -> None:
    """The guard covers all six tiers, not just the one the rehearsal hit."""
    expected = {"source.db", "index.db", "embeddings.db", "user.db", "audit.db", "ops.db"}
    assert set(ARCHIVE_TIER_FILENAMES) == expected
    with install_archive_write_guard(), arm_write_lease_enforcement():
        for name in sorted(ARCHIVE_TIER_FILENAMES):
            with pytest.raises(UnleasedWriteError):
                sqlite3.connect(archive / name)


def test_the_declared_seam_is_the_only_bypass_and_does_not_leak(archive: Path) -> None:
    """Bootstrap, offline rebuild and fixtures own archives without a daemon."""
    with install_archive_write_guard(), arm_write_lease_enforcement():
        with declared_unguarded_write("test fixture construction"):
            with closing(sqlite3.connect(archive / "source.db")) as conn:
                conn.execute("SELECT 1")
        with pytest.raises(UnleasedWriteError):
            sqlite3.connect(archive / "source.db")


def test_the_seam_does_not_authorize_other_threads(archive: Path) -> None:
    """A thread-local bypass must not become a process-wide hole.

    Anti-vacuity: make ``_BYPASS`` a module global and this goes green while
    one bootstrap unlocks every concurrent writer in the process.
    """
    refusals: list[BaseException | None] = []

    def other_thread() -> None:
        try:
            sqlite3.connect(archive / "source.db").close()
            refusals.append(None)
        except BaseException as exc:
            refusals.append(exc)

    with install_archive_write_guard(), arm_write_lease_enforcement(process_wide=True):
        with declared_unguarded_write("test fixture construction"):
            worker = threading.Thread(target=other_thread)
            worker.start()
            worker.join()
    assert isinstance(refusals[0], UnleasedWriteError)


def test_the_seam_requires_a_reason() -> None:
    with pytest.raises(ValueError):
        with declared_unguarded_write(""):
            pass


def test_installation_is_reentrant_and_restores_sqlite3_connect() -> None:
    original = sqlite3.connect
    with install_archive_write_guard():
        assert archive_write_guard_installed()
        with install_archive_write_guard():
            assert archive_write_guard_installed()
        assert sqlite3.connect is not original
    assert sqlite3.connect is original
    assert not archive_write_guard_installed()


def test_the_guard_is_inert_where_enforcement_is_not_armed(archive: Path) -> None:
    """A one-shot CLI or embedded caller is its own single writer."""
    with install_archive_write_guard():
        with closing(sqlite3.connect(archive / "source.db")) as conn:
            conn.execute("SELECT 1")


def test_a_second_writer_is_refused_before_it_can_contend_for_the_lock(archive: Path) -> None:
    """The rehearsal race, reproduced on a scratch archive by two writers.

    Writer A holds an exclusive transaction on ``source.db``. Without the
    boundary writer B waits out its busy timeout and then dies with
    ``database is locked`` -- the failure that killed the daemon on
    2026-09-05. This proves both halves: the raw contention is real here, and
    the guard converts it into an immediate typed refusal instead.

    Determinism: the contending open uses a zero busy timeout, so the lock
    race resolves without sleeping.
    """
    source = archive / "source.db"
    with closing(sqlite3.connect(source)) as writer_a:
        writer_a.execute("BEGIN EXCLUSIVE")
        writer_a.execute("INSERT INTO raw_sessions (raw_id) VALUES ('a')")

        with closing(sqlite3.connect(source, timeout=0)) as writer_b:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                writer_b.execute("INSERT INTO raw_sessions (raw_id) VALUES ('b')")

        with install_archive_write_guard(), arm_write_lease_enforcement():
            with pytest.raises(UnleasedWriteError):
                sqlite3.connect(source, timeout=0)
        writer_a.rollback()


def test_the_daemon_arms_the_guard_with_its_writer_boundary() -> None:
    """The daemon's declared writers enter through the lease because the guard
    is armed for the same block as enforcement itself.

    This is the census in its total form: rather than enumerating writers, it
    proves the process boundary under which *any* archive-tier writer must
    hold the lease.

    Anti-vacuity: remove ``install_archive_write_guard()`` from the daemon's
    writer-boundary ``with`` block and this fails naming it; the factory-level
    arming alone would still pass ``test_write_lease.py``.
    """
    repo_root = Path(__file__).resolve().parents[3]
    module = ast.parse((repo_root / "polylogue" / "daemon" / "cli.py").read_text(encoding="utf-8"))
    armed_blocks = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.AsyncWith | ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Name)
            and item.context_expr.func.id == "arm_write_lease_enforcement"
            for item in node.items
        )
    ]
    assert armed_blocks, "the daemon no longer arms write-lease enforcement; update this census"
    for block in armed_blocks:
        guards = {
            item.context_expr.func.id
            for item in block.items
            if isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Name)
        }
        assert "install_archive_write_guard" in guards, (
            f"daemon/cli.py:{block.lineno} arms enforcement without installing the connection guard, "
            "so a writer that skips the declared factories is unserialized again"
        )


def test_classification_reads_the_file_name_not_the_directory(tmp_path: Path) -> None:
    """A generation or staging copy named ``index.db`` is still a tier open.

    The guard deliberately asserts only that *a* lease is held; archive-root
    binding stays with the factories, which know which archive they were asked
    for. Classifying by directory would miss a generation build entirely.
    """
    assert guarded_archive_tier_path(tmp_path / "generations" / "g1" / "index.db") is not None
    assert guarded_archive_tier_path(tmp_path / "index.db.tmp") is None
    assert guarded_archive_tier_path(f"file:{tmp_path / 'index.db'}?mode=ro", uri=True) is None
    assert guarded_archive_tier_path(f"file:{tmp_path / 'index.db'}?mode=rw", uri=True) is not None
