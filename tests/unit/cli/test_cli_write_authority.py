"""The CLI process's archive writer-ownership boundary (polylogue-8qm4k AC1).

Enforcement used to be armed in two processes only -- ``polylogued run`` and
the MCP stdio bridge with a write/maintenance capability. The
``polylogue``/``plg``/``plog`` console scripts armed neither, so
``require_write_lease`` returned ``None`` for every writable archive-tier open
an ordinary CLI invocation made, and ``ops maintenance archive-init --yes``
created and wrote all six durable tiers beside a live daemon.

These tests drive the real console-script route (Click's root callback, which
``main()``, ``python -m polylogue`` and an embedded ``cli`` caller all reach)
against a real resident process the pidfile claims, and check both
directions: refused when a daemon owns the archive, still executed when
nothing does.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from polylogue.cli.click_app import cli
from polylogue.cli.write_authority import (
    ArchiveWriterOwnershipError,
    ArchiveWriterOwnershipUndecidableError,
    cli_archive_writer_ownership,
)
from polylogue.core.write_lease import UnleasedWriteError

#: Every durable/derived tier ``archive-init`` creates. The refusal must leave
#: none of them behind: a partially initialized archive is the silent-damage
#: outcome the boundary exists to prevent.
TIER_FILES = ("source.db", "index.db", "embeddings.db", "user.db", "audit.db", "ops.db")


@pytest.fixture
def resident_daemon(tmp_path: Path) -> Iterator[Callable[[Path], int]]:
    """Start a process holding the daemon's own exclusive lock on a pidfile.

    ``polylogued run`` proves ownership by taking ``fcntl.flock(fd, LOCK_EX)``
    on ``<root>/daemon.pid`` and holding it for its whole run
    (``polylogue.daemon.cli._acquire_pidfile``). This fixture reproduces that
    token rather than patching the probe, so the probe itself stays under test.

    The helper process is deliberately **not** named ``polylogued``: the probe
    must decide residency from the held lock, never from a command line it can
    only read on Linux.
    """
    processes: list[subprocess.Popen[str]] = []
    script = tmp_path / "hold_pidfile.py"
    script.write_text(
        "import fcntl, os, sys, time\n"
        "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)\n"
        "fcntl.flock(fd, fcntl.LOCK_EX)\n"
        "os.write(fd, str(os.getpid()).encode())\n"
        "os.fsync(fd)\n"
        "sys.stdout.write('ready\\n')\n"
        "sys.stdout.flush()\n"
        "time.sleep(300)\n"
    )

    def start(pidfile: Path) -> int:
        process = subprocess.Popen([sys.executable, str(script), str(pidfile)], stdout=subprocess.PIPE, text=True)
        processes.append(process)
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        return process.pid

    try:
        yield start
    finally:
        for process in processes:
            process.kill()
            process.wait(timeout=30)
            if process.stdout is not None:
                process.stdout.close()


def _archive_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    root.mkdir()
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    return root


def _init_archive(runner: CliRunner) -> Result:
    return runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "archive-init", "--yes", "--output-format", "json"],
        catch_exceptions=False,
    )


def test_resident_daemon_refuses_a_writable_tier_open(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resident_daemon: Callable[[Path], int],
) -> None:
    """An ordinary CLI process may not create tier files a live daemon owns.

    The refusal is asserted as the *boundary's* relabelled error, not as the
    command's own "Blocked:" line. ``archive-init`` used to catch
    ``RuntimeError`` around ``initialize_archive_tier_files_from_plan``, which
    swallowed ``UnleasedWriteError`` and printed its internal text ("open it
    inside ``write_lease(...)``") as a blocked plan -- so this assertion was
    satisfied by a message that never named the resident daemon and never
    reached the relabelling the boundary exists to do (polylogue-re6s3 AC4).
    The catch is now narrowed to ``ArchiveInitBlockedError``.

    Anti-vacuity: drop ``ctx.with_resource(cli_archive_writer_ownership())``
    from the root callback in ``polylogue/cli/click_app.py`` and this test goes
    red with ``executed: true`` and all six tier files on disk -- the defect.
    """
    root = _archive_root(monkeypatch, tmp_path)
    resident_pid = resident_daemon(root / "daemon.pid")

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "archive-init", "--yes", "--output-format", "json"],
        catch_exceptions=True,
    )

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, ArchiveWriterOwnershipError), result.output
    assert f"PID {resident_pid}" in str(result.exception)
    assert "write lease" in str(result.exception)
    assert [name for name in TIER_FILES if (root / name).exists()] == []


def test_offline_cli_still_initializes_the_archive(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """With no resident daemon the CLI is the archive's single writer.

    The opposite direction, so "refuse every CLI write" cannot pass as a fix:
    a boundary that refused unconditionally would break the declared offline
    authorities (archive initialization, durable tier migration, embedding
    backfill) that legitimately own an archive no daemon is serving.
    """
    root = _archive_root(monkeypatch, tmp_path)

    result = _init_archive(cli_runner)

    assert result.exit_code == 0, result.output
    assert sorted(name for name in TIER_FILES if (root / name).exists()) == sorted(TIER_FILES)


def test_resident_daemon_does_not_refuse_reads(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resident_daemon: Callable[[Path], int],
) -> None:
    """Arming bounds writers only; a read under a resident daemon still answers.

    Anti-vacuity: widen the boundary to refuse read-only opens too and this
    goes red, because ``status`` reaches the archive without ever holding the
    write lease.
    """
    root = _archive_root(monkeypatch, tmp_path)
    resident_daemon(root / "daemon.pid")

    result = cli_runner.invoke(cli, ["--plain", "status"], catch_exceptions=False)

    assert "write lease" not in result.output
    assert "cannot write" not in result.output


def test_refusal_names_the_resident_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resident_daemon: Callable[[Path], int],
) -> None:
    """A refusal that escapes the command names the PID that holds the archive.

    Anti-vacuity: re-raise the original ``UnleasedWriteError`` unchanged and
    this goes red, because that message names only "the daemon write lease"
    and never which process is actually holding this archive.
    """
    root = _archive_root(monkeypatch, tmp_path)
    resident_pid = resident_daemon(root / "daemon.pid")

    with pytest.raises(ArchiveWriterOwnershipError) as caught:
        with cli_archive_writer_ownership():
            raise UnleasedWriteError("open_connection(source.db) requires the daemon write lease")

    message = str(caught.value)
    assert str(root) in message
    assert f"PID {resident_pid}" in message
    assert isinstance(caught.value.__cause__, UnleasedWriteError)


def test_unowned_archive_leaves_the_boundary_unarmed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No resident daemon means no second writer, so nothing is armed.

    Anti-vacuity: arm unconditionally and this goes red -- which is also what
    breaks the offline authorities, since an invocation-wide lease is bound to
    the thread and task that minted it while the CLI writes from ``asyncio``
    tasks and worker threads.
    """
    from polylogue.storage.sqlite.write_guard import archive_write_guard_installed
    from polylogue.storage.sqlite.write_lease import write_lease_enforced

    _archive_root(monkeypatch, tmp_path)

    with cli_archive_writer_ownership():
        assert write_lease_enforced() is False
        assert archive_write_guard_installed() is False


def test_resident_archive_writer_arms_the_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resident_daemon: Callable[[Path], int],
) -> None:
    """A resident daemon arms both halves: the factories and ``sqlite3.connect``."""
    from polylogue.storage.sqlite.write_guard import archive_write_guard_installed
    from polylogue.storage.sqlite.write_lease import write_lease_enforced

    root = _archive_root(monkeypatch, tmp_path)
    resident_daemon(root / "daemon.pid")

    with cli_archive_writer_ownership():
        assert write_lease_enforced() is True
        assert archive_write_guard_installed() is True
    assert write_lease_enforced() is False
    assert archive_write_guard_installed() is False


def test_escaping_refusal_is_translated_through_click(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resident_daemon: Callable[[Path], int],
) -> None:
    """The translation reaches a real command, not only a direct ``with`` block.

    Click's ``Context`` forwards the in-flight exception to resources entered
    with ``ctx.with_resource``, which is what lets the boundary re-label a
    refusal it did not raise. Anti-vacuity: register the boundary with
    ``ctx.call_on_close`` instead (teardown only, no exception forwarding) and
    this goes red with the unlabelled ``UnleasedWriteError``.
    """
    import polylogue.storage.embeddings.reconcile as reconcile

    root = _archive_root(monkeypatch, tmp_path)
    resident_pid = resident_daemon(root / "daemon.pid")

    def _refuse(*args: object, **kwargs: object) -> None:
        raise UnleasedWriteError("open_connection(embeddings.db) requires the daemon write lease")

    monkeypatch.setattr(reconcile, "reconcile_embedding_orphans", _refuse)

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "embedding-orphan-reconcile", "--output-format", "json"],
        catch_exceptions=True,
    )

    assert isinstance(result.exception, ArchiveWriterOwnershipError), result.output
    assert f"PID {resident_pid}" in str(result.exception)


def test_residency_is_proved_by_the_held_lock_not_a_command_line(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resident_daemon: Callable[[Path], int],
) -> None:
    """The probe reads the daemon's own lock, never ``/proc/<pid>/cmdline``.

    ``/proc`` does not exist on macOS, a supported install target
    (``docs/installation.md``), so a probe that confirmed residency by reading
    a command line answered "no daemon" for every pid on every macOS install
    and the boundary armed nothing.

    Anti-vacuity: restore the ``b"polylogued" in Path(f"/proc/{pid}/cmdline")
    .read_bytes()`` confirmation in ``resident_daemon_pid`` and this goes red
    even on Linux -- the fixture's holder is named ``hold_pidfile.py``, so the
    only evidence of residency available to *any* platform is the held lock.
    """
    from polylogue.maintenance.offline_guard import resident_daemon_pid

    root = _archive_root(monkeypatch, tmp_path)
    resident_pid = resident_daemon(root / "daemon.pid")

    assert resident_daemon_pid(root) == resident_pid


def test_an_unheld_pidfile_is_not_a_resident_daemon(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The opposite direction: a pidfile nobody holds must not arm the boundary.

    Anti-vacuity: report residency whenever the pidfile merely exists and this
    goes red -- which is also the failure mode that would refuse every offline
    authority after a crashed daemon left its pidfile behind.
    """
    from polylogue.maintenance.offline_guard import resident_daemon_pid

    root = _archive_root(monkeypatch, tmp_path)
    (root / "daemon.pid").write_text(f"{os.getpid()}\n")

    assert resident_daemon_pid(root) is None


def test_a_platform_that_cannot_prove_ownership_refuses_loudly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An unprovable owner is refused, never treated as an absent one.

    This is the shape of the original defect rather than its cause: the
    boundary must not quietly disappear on a platform whose residency probe
    cannot answer. ``fcntl`` is the seam -- a build without it cannot observe
    the daemon's lock at all.

    Anti-vacuity: swallow ``DaemonResidencyUndecidableError`` in
    ``resident_archive_writer`` (or return ``None`` from ``_pidfile_holder_is_live``
    when ``fcntl`` is missing, which is exactly what the old ``except OSError:
    return None`` did for ``/proc``) and this goes red, because the body then
    runs with nothing armed.
    """
    import sys

    _archive_root(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, "fcntl", None)

    with pytest.raises(ArchiveWriterOwnershipUndecidableError) as caught:
        with cli_archive_writer_ownership():
            pass

    assert "cannot prove" in str(caught.value)


def test_a_daemon_that_arrives_mid_invocation_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resident_daemon: Callable[[Path], int],
) -> None:
    """Residency is re-asked at the open, not sampled once at entry.

    ``polylogue ops embed backfill`` can pause at its confirmation prompt for
    minutes. A daemon started in that window used to go unnoticed for the
    whole command, and the backfill then opened writable embedding/index/ops
    tiers beside it.

    Anti-vacuity: decide residency once in ``cli_archive_writer_ownership``
    and yield unguarded when it is absent -- the shipped shape -- and this
    goes red with a live connection to ``index.db``.
    """
    import sqlite3

    root = _archive_root(monkeypatch, tmp_path)

    with cli_archive_writer_ownership():
        resident_pid = resident_daemon(root / "daemon.pid")
        with pytest.raises(ArchiveWriterOwnershipError) as caught:
            sqlite3.connect(str(root / "index.db"))

    assert f"PID {resident_pid}" in str(caught.value)
    assert not (root / "index.db").exists()


def test_revalidation_leaves_an_unowned_archive_writable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The opposite direction: re-asking must not refuse an archive nobody owns.

    Anti-vacuity: refuse unconditionally inside the interception and this goes
    red -- which is also what would break every declared offline authority.
    """
    import sqlite3

    root = _archive_root(monkeypatch, tmp_path)

    with cli_archive_writer_ownership():
        connection = sqlite3.connect(str(root / "index.db"))
        connection.close()

    assert (root / "index.db").exists()
