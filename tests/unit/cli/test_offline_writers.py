"""The CLI commands that still write archive tiers in this process.

``polylogue-re6s3`` / ``polylogue-5vps8`` require that no CLI route opens a
writable tier beside a running daemon. Four command families are not lowered
onto a declared daemon operation and are not going to be: they are declared
*offline* authorities (archive initialization, backup, demo seeding, the
secret sweep's coverage ledger). For those the requirement is not "route it
through the daemon" but "own the archive exclusively, or refuse" -- design D8.

``tests/unit/cli/test_cli_write_authority.py`` proves the boundary mechanism
on one command. This module is the *coverage* question the acceptance asks:
for each family that still writes, does the production console-script route
actually refuse while a resident daemon owns the archive, and does the refusal
arrive as a decision rather than as a crash?

Two directions per row, and the second is what keeps the first honest:

* Refused beside a resident daemon, naming the holder and a next action.
* **Still a writer.** Offline, the same invocation is observed opening a
  writable archive tier through the production interception seam. Without
  this, a row whose command stopped writing -- or never wrote -- would keep
  passing the refusal test forever while proving nothing, which is exactly
  how a matrix outlives the behaviour it was written for.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from polylogue.cli.click_app import cli
from polylogue.cli.machine_main import run_machine_entry

#: A row that reaches a writable tier open only after a real archive exists.
_NEEDS_TIERS = "initialized"
#: A row that reaches one on a bare directory.
_NEEDS_NOTHING = "bare"


class _WritableTierOpened(BaseException):
    """Raised inside the interception seam to stop an offline row at its write.

    Derived from :class:`BaseException` on purpose: several of these commands
    wrap their work in ``except Exception``, and a probe that a command could
    swallow would report "this row never writes" for a row that writes.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(str(path))


def _argv_archive_init(root: Path, scratch: Path) -> tuple[str, ...]:
    del root, scratch
    return ("ops", "maintenance", "archive-init", "--yes")


def _argv_backup(root: Path, scratch: Path) -> tuple[str, ...]:
    del root
    return ("ops", "backup", "--output-dir", str(scratch / "backup-out"))


def _argv_demo_seed(root: Path, scratch: Path) -> tuple[str, ...]:
    del scratch
    return ("demo", "seed", "--root", str(root))


def _argv_scan_secrets(root: Path, scratch: Path) -> tuple[str, ...]:
    del root, scratch
    return ("ops", "scan-secrets", "--all")


#: ``(id, argv builder, archive state the row needs, machine-format argv tail)``.
#:
#: The last element is ``None`` for a command that has no ``--format json``
#: route at all, so there is no machine envelope for the refusal to travel in.
#: That is recorded rather than worked around: inventing a format for the sake
#: of a test row would assert a surface no operator has.
_OFFLINE_WRITERS: tuple[tuple[str, Callable[[Path, Path], tuple[str, ...]], str, tuple[str, ...] | None], ...] = (
    ("archive-init", _argv_archive_init, _NEEDS_NOTHING, ("--output-format", "json")),
    ("backup", _argv_backup, _NEEDS_TIERS, None),
    ("demo-seed", _argv_demo_seed, _NEEDS_NOTHING, None),
    ("scan-secrets", _argv_scan_secrets, _NEEDS_TIERS, ("--format", "json")),
)

#: Rows carrying a machine-format leg, and the flag spelling each one accepts.
#: ``archive-init`` uses ``--output-format``, which the argv probe in
#: ``machine_errors.wants_json`` did not recognise, so its refusal reached a
#: machine caller as a prose ``Error:`` line on stderr with an empty stdout.
_MACHINE_FORMAT_ROWS = tuple(row for row in _OFFLINE_WRITERS if row[3] is not None)

#: Why a row carries no machine-format leg. Read by
#: :func:`test_every_machine_format_exemption_is_still_true`, so an exemption
#: cannot outlive the gap it describes.
_NO_MACHINE_FORMAT: dict[str, str] = {
    "backup": "has no machine output mode at all",
    "demo-seed": "has no machine output mode at all",
}


@pytest.fixture
def resident_daemon(tmp_path: Path) -> Iterator[Callable[[Path], int]]:
    """Start a process holding the daemon's own exclusive lock on a pidfile.

    The same token ``polylogued run`` takes in
    ``polylogue.daemon.cli._acquire_pidfile``, reproduced rather than patched
    so the residency probe itself stays under test. The helper is deliberately
    not named ``polylogued``: residency is decided by the held lock.
    """
    import subprocess

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


def _prepare_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, needs: str) -> Path:
    root = tmp_path / "archive"
    root.mkdir()
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    monkeypatch.setenv("POLYLOGUE_DB_PATH", str(root / "index.db"))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    if needs == _NEEDS_TIERS:
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

        initialize_active_archive_root(root)
    return root


def _tier_digest(root: Path) -> str:
    """Digest every tier file and journal under ``root``."""
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.db*")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _invoke(argv: tuple[str, ...], monkeypatch: pytest.MonkeyPatch) -> int:
    """Run one invocation through the real machine entry point.

    ``CliRunner().invoke(cli, ...)`` is deliberately not used: the mapping
    from a raised refusal to an operator-visible line and a machine error code
    lives in :func:`polylogue.cli.machine_main.run_machine_entry`, and Click's
    test runner skips all of it.
    """
    full_argv = ["polylogue", "--plain", *argv]
    monkeypatch.setattr(sys, "argv", full_argv)
    with pytest.raises(SystemExit) as exit_info:
        run_machine_entry(cli, full_argv[1:])
    code = exit_info.value.code
    return code if isinstance(code, int) else 1


@pytest.mark.parametrize(
    ("row_id", "build_argv", "needs", "machine_tail"),
    _OFFLINE_WRITERS,
    ids=[row[0] for row in _OFFLINE_WRITERS],
)
def test_refused_beside_resident_daemon(
    row_id: str,
    build_argv: Callable[[Path, Path], tuple[str, ...]],
    needs: str,
    machine_tail: tuple[str, ...] | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    resident_daemon: Callable[[Path], int],
) -> None:
    """No offline writer may touch tiers a live daemon owns.

    Anti-vacuity, per row: ``backup`` goes red by deleting the
    ``_require_exclusive_archive_ownership(root)`` call in ``backup_archive``
    (``polylogue/daemon/backup.py``) -- it exits 0 and writes a complete
    backup, because ``backup_archive`` mints its own
    ``write_lease("maintenance.backup")`` and so satisfies the armed guard.
    That refusal lives at the function that mints the lease rather than in
    this command, so an embedded importer of the public ``backup_archive`` is
    refused on the same terms; see
    ``tests/unit/daemon/test_backup.py::test_embedded_backup_refused_beside_resident_daemon``.
    The other three go red by dropping
    ``ctx.with_resource(cli_archive_writer_ownership())`` from the root
    callback in ``polylogue/cli/click_app.py``.
    """
    del machine_tail
    root = _prepare_root(monkeypatch, tmp_path, needs)
    resident_pid = resident_daemon(root / "daemon.pid")
    before = _tier_digest(root)
    capsys.readouterr()

    exit_code = _invoke(build_argv(root, tmp_path), monkeypatch)

    captured = capsys.readouterr()
    text = f"{captured.out}\n{captured.err}"
    assert exit_code != 0, text
    assert f"PID {resident_pid}" in text, text
    assert "Traceback" not in text, text
    # The refusal is a decision, not a crash: the generic branch of
    # ``run_machine_entry`` labels anything it does not recognise "unexpected
    # error", which is what this boundary's refusal used to reach an operator
    # as (polylogue-re6s3 AC4).
    assert "unexpected error" not in text, text
    assert _tier_digest(root) == before, f"{row_id} mutated tiers behind the refusal"


@pytest.mark.parametrize(
    ("row_id", "build_argv", "needs", "machine_tail"),
    _OFFLINE_WRITERS,
    ids=[row[0] for row in _OFFLINE_WRITERS],
)
def test_row_still_opens_a_writable_tier(
    row_id: str,
    build_argv: Callable[[Path, Path], tuple[str, ...]],
    needs: str,
    machine_tail: tuple[str, ...] | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Offline, every row in the matrix really does open a writable tier.

    The pin that stops the refusal test above from going vacuous. A row whose
    command stopped writing -- or that never wrote in the first place -- would
    keep passing a refusal assertion while certifying nothing at all.

    Anti-vacuity for this test itself: point a row at ``find`` and it goes
    red, because no writable tier open reaches the interception seam.

    ``find`` and not ``ops status``, which looks like the obvious read-only
    control and is not one: ``status`` writes a route-observation receipt to
    the disposable ``ops.db`` through
    ``polylogue.operations.route_observation._emit_best_effort``, so it opens
    a writable tier on every run and would make this test pass while proving
    nothing.
    """
    del machine_tail
    from polylogue.maintenance.offline_guard import refuse_writable_tier_opens

    root = _prepare_root(monkeypatch, tmp_path, needs)
    opened: list[Path] = []

    def record(path: Path) -> None:
        opened.append(path)
        raise _WritableTierOpened(path)

    with pytest.raises(_WritableTierOpened), refuse_writable_tier_opens(record):
        _invoke(build_argv(root, tmp_path), monkeypatch)

    assert opened, f"{row_id} opened no writable archive tier offline"


def test_every_machine_format_exemption_is_still_true() -> None:
    """An exemption may not outlive the gap it records.

    Anti-vacuity: give ``backup`` a ``--format json`` option without removing
    its row here and this goes red, which is the point -- an exemption list
    that nothing re-checks becomes a permanent excuse.
    """
    from polylogue.cli.commands.backup import backup_command

    declared = {row_id for row_id, _, _, machine_tail in _OFFLINE_WRITERS if machine_tail is None}
    assert declared == set(_NO_MACHINE_FORMAT), declared ^ set(_NO_MACHINE_FORMAT)

    option_names = {name for param in backup_command.params for name in param.opts}
    assert "--format" not in option_names, (
        "`ops backup` gained a machine format: give it a machine-format row in "
        "_OFFLINE_WRITERS and drop its _NO_MACHINE_FORMAT entry"
    )


@pytest.mark.parametrize(
    ("row_id", "build_argv", "needs", "machine_tail"),
    _MACHINE_FORMAT_ROWS,
    ids=[row[0] for row in _MACHINE_FORMAT_ROWS],
)
def test_machine_format_refusal_is_typed(
    row_id: str,
    build_argv: Callable[[Path, Path], tuple[str, ...]],
    needs: str,
    machine_tail: tuple[str, ...] | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    resident_daemon: Callable[[Path], int],
) -> None:
    """A machine caller gets the ownership code, not ``runtime_error`` or prose.

    Two failures met here. The refusal reached a ``--format json`` client as
    ``runtime_error`` -- the code a corrupt tier and a genuine crash also
    produce -- with the resident writer and the remedy nowhere on the wire.
    And ``archive-init`` never produced an envelope at all: it spells its
    machine mode ``--output-format json``, which ``wants_json`` did not
    recognise, so the terminal branch ran and the caller got an empty stdout
    and prose on stderr.

    Anti-vacuity: delete the ``ArchiveWriterOwnershipError`` branch from the
    JSON path of ``run_machine_entry`` and both rows go red on ``code`` while
    ``test_refused_beside_resident_daemon`` stays green, because the terminal
    branch is a separate mapping -- the asymmetry that let the gap survive a
    green suite. Drop ``--output-format`` from ``_JSON_FORMAT_FLAGS`` and the
    ``archive-init`` row alone goes red, on the JSON decode.
    """
    assert machine_tail is not None
    root = _prepare_root(monkeypatch, tmp_path, needs)
    resident_pid = resident_daemon(root / "daemon.pid")
    capsys.readouterr()

    exit_code = _invoke((*build_argv(root, tmp_path), *machine_tail), monkeypatch)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code != 0, payload
    assert payload["status"] == "error", payload
    assert payload["code"] == "archive_writer_ownership_unavailable", (row_id, payload)
    details = payload["details"]
    assert details["archive_root"] == str(root), payload
    assert f"PID {resident_pid}" in str(details["resident_writer"]), payload
    assert "stop it" in str(details["remedy"]), payload
