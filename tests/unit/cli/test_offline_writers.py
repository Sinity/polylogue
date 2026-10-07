"""The CLI commands that still write archive tiers in this process.

``polylogue-re6s3`` / ``polylogue-5vps8`` require that no CLI route opens a
writable tier beside a running daemon. Every mutating verb now lowers onto a
declared daemon operation (``tests/unit/cli/test_cli_operation_authority.py``
proves each opens no writable tier when the daemon is absent); ``ops backup``
and ``ops scan-secrets`` left this module when they became
``maintenance.backup`` and ``maintenance.secret_scan``. One family remains a
declared *offline* authority: demo seeding (``demo seed``, and ``demo
receipts``/``demo tour`` through the same guarded ``seed_demo_archive``),
which builds a synthetic archive in a scratch root that no daemon serves and
that is never the configured archive. For it the requirement is not "route it
through the daemon" but "own the scratch root exclusively, or refuse" --
design D8. The configured archive itself has no offline writer: a demo row
aimed at it is refused with ``daemon_required`` before anything is written.

``tests/unit/cli/test_cli_write_authority.py`` proves the boundary mechanism
on one command. This module is the *coverage* question the acceptance asks:
for each family that still writes, does the production console-script route
actually refuse while a resident daemon owns the archive, and does the refusal
arrive as a decision rather than as a crash?

Two directions per row, and the second is what keeps the first honest:

* Refused beside a resident daemon on its scratch root, naming the holder
  and a next action.
* **Still a writer.** Offline, the same invocation is observed opening a
  writable archive tier through the production interception seam. Without
  this, a row whose command stopped writing -- or never wrote -- would keep
  passing the refusal test forever while proving nothing, which is exactly
  how a matrix outlives the behaviour it was written for.
"""

from __future__ import annotations

import contextlib
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


def _argv_demo_seed(root: Path, scratch: Path) -> tuple[str, ...]:
    del scratch
    return ("demo", "seed", "--root", str(root))


def _argv_demo_receipts(root: Path, scratch: Path) -> tuple[str, ...]:
    del scratch
    return ("demo", "receipts", "--root", str(root), "--seed")


def _argv_demo_tour(root: Path, scratch: Path) -> tuple[str, ...]:
    return ("demo", "tour", "--out-dir", str(scratch / "tour"), "--root", str(root), "--no-force")


#: ``(id, argv builder, archive state the row needs, machine-format argv tail)``.
_OFFLINE_WRITERS: tuple[tuple[str, Callable[[Path, Path], tuple[str, ...]], str, tuple[str, ...]], ...] = (
    ("demo seed", _argv_demo_seed, _NEEDS_NOTHING, ("--format", "json")),
    ("demo receipts", _argv_demo_receipts, _NEEDS_NOTHING, ("--format", "json")),
    ("demo tour", _argv_demo_tour, _NEEDS_NOTHING, ("--format", "json")),
)


def _cli_offline_writer_commands() -> set[str]:
    """The CLI commands whose own body reaches the guarded demo seeder.

    Derived from the source rather than from a registry beside it: every CLI
    module that imports ``seed_demo_archive`` is an in-process writer, and in
    the one module that does, each Click command whose body calls the local
    ``_seed_demo_archive`` wrapper is one offline-writer route.
    """
    import ast

    import polylogue.cli

    cli_root = Path(polylogue.cli.__file__).parent
    importers: set[Path] = set()
    for path in cli_root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and any(alias.name == "seed_demo_archive" for alias in node.names):
                importers.add(path.relative_to(cli_root))
    assert importers == {Path("commands/demo.py")}, importers

    commands: set[str] = set()
    tree = ast.parse((cli_root / "commands" / "demo.py").read_text(encoding="utf-8"))
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        names = [
            str(decorator.args[0].value)
            for decorator in function.decorator_list
            if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "command"
            and decorator.args
            and isinstance(decorator.args[0], ast.Constant)
            and isinstance(decorator.args[0].value, str)
        ]
        calls_seeder = any(
            isinstance(node, ast.Name) and node.id == "_seed_demo_archive" for node in ast.walk(function)
        )
        if names and calls_seeder:
            commands.update(f"demo {name}" for name in names)
    return commands


def test_rows_are_exactly_the_cli_offline_writers() -> None:
    """The CLI's actual offline-writer routes and this matrix name the same commands.

    Anti-vacuity: make another command call ``_seed_demo_archive`` (or import
    ``seed_demo_archive`` into another CLI module) without a row here and this
    is red, so no in-process writer exists without its refusal proof.
    """
    assert {row[0] for row in _OFFLINE_WRITERS} == _cli_offline_writer_commands()


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


def _configure_archive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Name the configured archive; demo rows never write it."""
    configured = tmp_path / "archive"
    configured.mkdir()
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(configured))
    monkeypatch.setenv("POLYLOGUE_DB_PATH", str(configured / "index.db"))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    return configured


def _prepare_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, needs: str) -> Path:
    """Configure the archive and return the separate scratch root a demo row seeds."""
    _configure_archive(monkeypatch, tmp_path)
    root = tmp_path / "demo-root"
    root.mkdir()
    if needs == _NEEDS_TIERS:
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

        initialize_active_archive_root(root)
    return root


def _tier_digest(root: Path) -> str:
    """Digest tier bytes and nonempty journals, excluding reader bookkeeping.

    SQLite WAL readers may create shared-memory files and empty WAL files;
    neither is an archive write. A nonempty WAL remains mutation evidence.
    """
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.db*")):
        if path.name.endswith("-shm"):
            continue
        payload = path.read_bytes()
        if path.name.endswith("-wal") and not payload:
            continue
        digest.update(path.name.encode())
        digest.update(payload)
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
    machine_tail: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    resident_daemon: Callable[[Path], int],
) -> None:
    """No offline writer may touch tiers a live daemon owns.

    Anti-vacuity: drop the ``scoped_offline_archive_writer`` claim from
    ``scoped_one_shot_archive_owner`` (``operations/canonical_archive_ingest.py``)
    together with ``ctx.with_resource(cli_archive_writer_ownership())`` in
    ``polylogue/cli/click_app.py`` and demo seeding writes tiers beside the
    resident daemon.
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
    machine_tail: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Offline, every row in the matrix really does open a writable tier.

    The pin that stops the refusal test above from going vacuous. A row whose
    command stopped writing -- or that never wrote in the first place -- would
    keep passing a refusal assertion while certifying nothing at all.

    Anti-vacuity for this test itself: point a row at ``find`` and it goes
    red, because no writable tier open reaches the interception seam.

    ``test_status_and_agent_views_open_no_writable_tier`` below is the read
    side of the same seam.
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


@pytest.mark.parametrize(
    ("row_id", "build_argv", "needs", "machine_tail"),
    _OFFLINE_WRITERS,
    ids=[row[0] for row in _OFFLINE_WRITERS],
)
def test_machine_format_refusal_is_typed(
    row_id: str,
    build_argv: Callable[[Path, Path], tuple[str, ...]],
    needs: str,
    machine_tail: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    resident_daemon: Callable[[Path], int],
) -> None:
    """A machine caller gets the ownership code, not ``runtime_error`` or prose.

    Two failures met here. The refusal reached a ``--format json`` client as
    ``runtime_error`` -- the code a corrupt tier and a genuine crash also
    produce -- with the resident writer and the remedy nowhere on the wire.

    Anti-vacuity: delete the ``ArchiveWriterOwnershipError`` branch from the
    JSON path of ``run_machine_entry`` and the row goes red on ``code`` while
    ``test_refused_beside_resident_daemon`` stays green, because the terminal
    branch is a separate mapping -- the asymmetry that let the gap survive a
    green suite.
    """
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
    assert "resident polylogued" in str(details["remedy"]), payload


@pytest.mark.parametrize(
    ("row_id", "build_argv", "needs", "machine_tail"),
    _OFFLINE_WRITERS,
    ids=[row[0] for row in _OFFLINE_WRITERS],
)
def test_demo_row_aimed_at_the_configured_archive_is_daemon_required(
    row_id: str,
    build_argv: Callable[[Path, Path], tuple[str, ...]],
    needs: str,
    machine_tail: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No demo route seeds the configured archive, even while it is empty.

    The reset hazard: before the reset the live root is empty, so the
    content-based guard in ``polylogue.demo.seed`` sees a fresh root and
    seeding it would write synthetic raws into the durable ``source.db``.

    Anti-vacuity: drop the ``_require_scratch_target`` calls from
    ``polylogue/cli/commands/demo.py`` and the row reaches the write boundary
    instead; drop that boundary's configured-root refusal too and it seeds.
    """
    del needs
    configured = _configure_archive(monkeypatch, tmp_path)
    before = _tier_digest(configured)
    capsys.readouterr()

    exit_code = _invoke((*build_argv(configured, tmp_path), *machine_tail), monkeypatch)

    payload = json.loads(capsys.readouterr().out)
    assert exit_code != 0, payload
    assert payload["code"] == "daemon_required", (row_id, payload)
    assert payload["details"]["archive_root"] == str(configured.resolve()), payload
    assert _tier_digest(configured) == before
    assert not list(configured.glob("*.db")), f"{row_id} created tiers in the configured archive"


def test_cli_writer_against_configured_archive_without_daemon_is_daemon_required(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The write boundary itself refuses the configured archive with no daemon.

    ``demo seed`` is driven past its own scratch-root check, so the refusal
    observed here comes from the per-open boundary in
    ``polylogue/cli/write_authority.py`` against a real in-process writer.

    Anti-vacuity: restore an offline-owner branch that takes archive custody
    for the configured root when no daemon is resident and this row seeds the
    configured archive (exit 0, tiers created).
    """
    import polylogue.cli.commands.demo as demo_cli

    configured = _configure_archive(monkeypatch, tmp_path)
    monkeypatch.setattr(demo_cli, "_require_scratch_target", lambda target, *, purpose: target.resolve())
    capsys.readouterr()

    exit_code = _invoke(("demo", "seed", "--root", str(configured), "--format", "json"), monkeypatch)

    payload = json.loads(capsys.readouterr().out)
    assert exit_code != 0, payload
    assert payload["code"] == "daemon_required", payload
    assert payload["details"]["archive_root"] == str(configured.resolve()), payload
    assert "polylogued run" in payload["message"], payload
    assert not list(configured.glob("*.db")), "the configured archive gained tiers offline"


@pytest.mark.parametrize(
    "argv",
    [("status",), ("ops", "status"), ("agents", "status"), ("agents", "work-item")],
    ids=["status", "ops status", "agents status", "agents work-item"],
)
def test_status_and_agent_views_open_no_writable_tier(
    argv: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The read-only views write no tier from the CLI process (polylogue-k5iaf).

    They used to time themselves through ``observe_route`` and write the
    receipt into ``ops.db`` with a plain ``sqlite3.connect``: a second writer
    beside the daemon when none was resident, and a refused open (a dropped
    receipt) when one was. The CLI process is not the ops tier's owner, so it
    records no route observation at all.

    Anti-vacuity: wrap any of these commands in ``observe_route`` again and
    its row records ``ops.db`` here.
    """
    from polylogue.maintenance.offline_guard import refuse_writable_tier_opens

    root = _prepare_root(monkeypatch, tmp_path, _NEEDS_TIERS)
    before = _tier_digest(root)
    opened: list[Path] = []
    monkeypatch.setattr(sys, "argv", ["polylogue", "--plain", *argv])
    with refuse_writable_tier_opens(opened.append), contextlib.suppress(SystemExit):
        run_machine_entry(cli, ["--plain", *argv])

    assert opened == []
    assert _tier_digest(root) == before
