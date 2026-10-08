"""The CLI process's archive writer-ownership boundary (polylogue-8qm4k AC1).

Enforcement used to be armed in two processes only -- ``polylogued run`` and
the MCP stdio bridge with a write/maintenance capability. The
``polylogue``/``plg``/``plog`` console scripts armed neither, so
``require_write_lease`` returned ``None`` for every writable archive-tier open
an ordinary CLI invocation made, so an offline writer could create tier files
beside a live daemon.

These tests prove the boundary mechanism against a real resident process the
pidfile claims, in both directions: refused when a daemon owns the archive,
refused offline too unless a leased scratch owner writes. ``tests/unit/cli/test_offline_writers.py``
drives each remaining offline writer through the console-script route.
Offline, the configured archive itself is never writable from the CLI.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.cli.write_authority import (
    ArchiveWriterOwnershipError,
    ArchiveWriterOwnershipUndecidableError,
    cli_archive_writer_ownership,
)
from polylogue.core.write_lease import UnleasedWriteError


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


def test_offline_archive_leaves_global_lease_guards_unarmed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A scratch writer keeps the process-wide lease guards unarmed.

    The per-open hook still classifies every writable tier; this checks that a
    command-scoped lease was not added, since CLI writes may run in asyncio
    tasks and worker threads.
    """
    from polylogue.storage.sqlite.write_lease import write_lease_enforced

    _archive_root(monkeypatch, tmp_path)

    with cli_archive_writer_ownership():
        assert write_lease_enforced() is False


def test_unresolved_configured_root_refuses_all_cli_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An unknown configured root cannot prove any target is a scratch archive."""
    import sqlite3

    import polylogue.cli.write_authority as authority

    monkeypatch.setattr(authority, "_archive_root", lambda: None)
    scratch = tmp_path / "unresolved"
    scratch.mkdir()

    with pytest.raises(ArchiveWriterOwnershipUndecidableError):
        with cli_archive_writer_ownership():
            sqlite3.connect(str(scratch / "source.db"))

    assert not (scratch / "source.db").exists()


def test_resident_archive_writer_uses_the_same_root_ownership_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resident_daemon: Callable[[Path], int],
) -> None:
    """Residency does not install a second global SQLite wrapper."""
    from polylogue.maintenance.offline_guard import writable_tier_opens_are_checked
    from polylogue.storage.sqlite.write_lease import write_lease_enforced

    root = _archive_root(monkeypatch, tmp_path)
    resident_daemon(root / "daemon.pid")

    with cli_archive_writer_ownership():
        assert writable_tier_opens_are_checked() is True
        assert write_lease_enforced() is False
    assert write_lease_enforced() is False
    assert writable_tier_opens_are_checked() is False


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


def test_configured_archive_has_no_offline_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """With no daemon, a writable open of the configured archive is ``daemon_required``.

    The configured archive is written only by ``polylogued``; the CLI is never
    its offline owner, even while it is empty (polylogue-5vps8 AC5).

    Anti-vacuity: restore the branch that takes archive custody for the
    configured root when no daemon is resident and the connection opens,
    creating ``index.db``.
    """
    import sqlite3

    from polylogue.cli.shared.helpers import DaemonRequiredError

    root = _archive_root(monkeypatch, tmp_path)

    with cli_archive_writer_ownership():
        with pytest.raises(DaemonRequiredError) as caught:
            sqlite3.connect(str(root / "index.db"))

    assert caught.value.code == "daemon_required"
    assert caught.value.archive_root == str(root.resolve())
    assert not (root / "index.db").exists()


def test_an_unleased_scratch_open_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Outside the configured archive only a leased scratch owner may write.

    Anti-vacuity: admit unleased opens outside the configured root and this
    connection opens, creating the stray tier.
    """
    import sqlite3

    _archive_root(monkeypatch, tmp_path)
    stray = tmp_path / "stray"
    stray.mkdir()

    with cli_archive_writer_ownership():
        with pytest.raises(ArchiveWriterOwnershipError):
            sqlite3.connect(str(stray / "index.db"))

    assert not (stray / "index.db").exists()


@pytest.mark.parametrize("configured_has_daemon", [False, True])
def test_scoped_offline_archive_owner_can_write_its_separate_scratch_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resident_daemon: Callable[[Path], int],
    configured_has_daemon: bool,
) -> None:
    """The one-shot demo owner is an explicit scratch authority without a write lease."""
    import sqlite3

    from polylogue.maintenance.offline_guard import scoped_offline_archive_writer

    configured = _archive_root(monkeypatch, tmp_path)
    if configured_has_daemon:
        resident_daemon(configured / "daemon.pid")
    scratch = tmp_path / "demo-scratch"

    with cli_archive_writer_ownership():
        with scoped_offline_archive_writer(scratch, owner_id="test-demo-scratch"):
            connection = sqlite3.connect(str(scratch / "index.db"))
            try:
                connection.execute("CREATE TABLE owned (id INTEGER PRIMARY KEY)")
            finally:
                connection.close()

    assert (scratch / "index.db").exists()
    assert not (configured / "index.db").exists()


def test_scratch_lease_root_may_not_contain_the_configured_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A scratch file outside the configured tree is still unsafe under an overlapping root."""
    import sqlite3

    from polylogue.core.write_lease import write_lease

    _archive_root(monkeypatch, tmp_path)
    stray = tmp_path / "stray"
    stray.mkdir()

    with cli_archive_writer_ownership():
        with write_lease("test.overlapping-scratch", archive_root=tmp_path):
            with pytest.raises(ArchiveWriterOwnershipError):
                sqlite3.connect(str(stray / "source.db"))

    assert not (stray / "source.db").exists()


def test_a_separate_archive_answers_to_its_own_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resident_daemon: Callable[[Path], int],
) -> None:
    """A tier of a separately leased archive is bounded by that archive.

    A demo seed/receipts/tour operation may build an explicit scratch archive
    while ``POLYLOGUE_ARCHIVE_ROOT`` names another. Its own write lease names
    the scratch root, which the boundary used to compare with the configured
    root and refuse as "a different archive".

    Anti-vacuity: key the boundary on the configured root again and the
    leased open is refused while the daemon-owned one is let through.
    """
    import sqlite3

    from polylogue.core.write_lease import archive_write_custody, write_lease

    _archive_root(monkeypatch, tmp_path)
    separate = tmp_path / "separate"
    separate.mkdir()
    owned = tmp_path / "owned"
    owned.mkdir()

    with cli_archive_writer_ownership():
        with archive_write_custody(separate):
            with write_lease("test.separate-archive", archive_root=separate):
                sqlite3.connect(str(separate / "index.db")).close()
        resident_daemon(owned / "daemon.pid")
        with write_lease("test.owned-archive", archive_root=owned):
            with pytest.raises(ArchiveWriterOwnershipError):
                sqlite3.connect(str(owned / "index.db"))

    assert (separate / "index.db").exists()
    assert not (owned / "index.db").exists()


def test_bound_write_lease_rejects_a_missing_archive_identity(tmp_path: Path) -> None:
    """Every write under an archive-bound lease must name its archive.

    Anti-vacuity: leave ``archive_root`` optional in ``require_write_lease``
    and a caller can authorize a write to an unrelated archive by omitting it.
    """
    from polylogue.storage.sqlite.write_lease import arm_write_lease_enforcement, require_write_lease, write_lease

    with arm_write_lease_enforcement(), write_lease("bound", archive_root=tmp_path):
        with pytest.raises(UnleasedWriteError, match="omitted archive identity"):
            require_write_lease("misrouted archive writer")


def test_cli_rejects_an_inherited_lease_without_actual_thread_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import contextvars
    import sqlite3
    import threading

    from polylogue.storage.sqlite.write_lease import write_lease
    from tests.infra.archive_custody_probe import archive_custody_available

    _archive_root(monkeypatch, tmp_path)
    root = tmp_path / "scratch"
    root.mkdir()
    failures: list[BaseException] = []

    def attempt_open() -> None:
        try:
            connection = sqlite3.connect(root / "index.db")
        except BaseException as error:
            failures.append(error)
        else:
            connection.close()

    with write_lease("test.cli_owner", archive_root=root), cli_archive_writer_ownership():
        copied = contextvars.copy_context()
        thread = threading.Thread(target=lambda: copied.run(attempt_open))
        thread.start()
        thread.join()
        assert len(failures) == 1
        assert isinstance(failures[0], UnleasedWriteError)
        assert not (root / "index.db").exists()
        assert not archive_custody_available(root)
    assert archive_custody_available(root)


@pytest.mark.asyncio
async def test_configured_coordinator_does_not_admit_a_foreign_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sqlite3

    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator

    root = _archive_root(monkeypatch, tmp_path)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    coordinator = DaemonWriteCoordinator(archive_root=root)

    def attempt_foreign_open() -> None:
        with pytest.raises(ArchiveWriterOwnershipError):
            sqlite3.connect(foreign / "source.db")

    try:
        with cli_archive_writer_ownership():
            await coordinator.run_sync("test.cli.foreign_archive", attempt_foreign_open)
        assert not (foreign / "source.db").exists()
    finally:
        assert await coordinator.shutdown(timeout=1.0)


def test_offline_residency_classifies_only_writable_tier_paths(tmp_path: Path) -> None:
    from polylogue.maintenance.offline_guard import guarded_archive_tier_path

    for tier in ("source", "index", "embeddings", "user", "audit", "ops"):
        path = tmp_path / "candidate" / f"{tier}.db"
        assert guarded_archive_tier_path(path) == path
        assert guarded_archive_tier_path(f"{path.as_uri()}?mode=rw", uri=True) == path
        assert guarded_archive_tier_path(f"{path.as_uri()}?mode=ro", uri=True) is None
        assert guarded_archive_tier_path(f"{path.as_uri()}?immutable=1", uri=True) is None
    assert guarded_archive_tier_path(tmp_path / "spill.db") is None
    assert guarded_archive_tier_path(":memory:") is None
    assert guarded_archive_tier_path("/proc/self/fd/7") is None
