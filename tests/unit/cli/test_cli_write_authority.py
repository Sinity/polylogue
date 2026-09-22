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

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from polylogue.cli.click_app import cli
from polylogue.cli.write_authority import (
    ArchiveWriterOwnershipError,
    cli_archive_writer_ownership,
)
from polylogue.core.write_lease import UnleasedWriteError

#: Every durable/derived tier ``archive-init`` creates. The refusal must leave
#: none of them behind: a partially initialized archive is the silent-damage
#: outcome the boundary exists to prevent.
TIER_FILES = ("source.db", "index.db", "embeddings.db", "user.db", "audit.db", "ops.db")


@pytest.fixture
def resident_daemon(tmp_path: Path) -> Iterator[int]:
    """A live process whose ``/proc`` cmdline names ``polylogued``.

    ``resident_daemon_pid`` requires all three facts -- a readable pidfile, a
    signalable PID, and ``polylogued`` in that PID's command line -- so a
    stale pidfile or an unrelated process cannot masquerade as the owner.
    This fixture supplies a real one instead of patching the probe, which
    would leave the probe itself untested.
    """
    script = tmp_path / "polylogued.py"
    script.write_text("import sys, time\nsys.stdout.write('ready\\n')\nsys.stdout.flush()\ntime.sleep(300)\n")
    process = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        yield process.pid
    finally:
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
    resident_daemon: int,
) -> None:
    """An ordinary CLI process may not create tier files a live daemon owns.

    Anti-vacuity: drop ``ctx.with_resource(cli_archive_writer_ownership())``
    from the root callback in ``polylogue/cli/click_app.py`` and this test goes
    red with ``executed: true`` and all six tier files on disk -- the defect.
    """
    root = _archive_root(monkeypatch, tmp_path)
    (root / "daemon.pid").write_text(f"{resident_daemon}\n")

    result = _init_archive(cli_runner)

    assert result.exit_code == 1, result.output
    assert "write lease" in result.output
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
    resident_daemon: int,
) -> None:
    """Arming bounds writers only; a read under a resident daemon still answers.

    Anti-vacuity: widen the boundary to refuse read-only opens too and this
    goes red, because ``status`` reaches the archive without ever holding the
    write lease.
    """
    root = _archive_root(monkeypatch, tmp_path)
    (root / "daemon.pid").write_text(f"{resident_daemon}\n")

    result = cli_runner.invoke(cli, ["--plain", "status"], catch_exceptions=False)

    assert "write lease" not in result.output
    assert "cannot write" not in result.output


def test_refusal_names_the_resident_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resident_daemon: int,
) -> None:
    """A refusal that escapes the command names the PID that holds the archive.

    Anti-vacuity: re-raise the original ``UnleasedWriteError`` unchanged and
    this goes red, because that message names only "the daemon write lease"
    and never which process is actually holding this archive.
    """
    root = _archive_root(monkeypatch, tmp_path)
    (root / "daemon.pid").write_text(f"{resident_daemon}\n")

    with pytest.raises(ArchiveWriterOwnershipError) as caught:
        with cli_archive_writer_ownership():
            raise UnleasedWriteError("open_connection(source.db) requires the daemon write lease")

    message = str(caught.value)
    assert str(root) in message
    assert f"PID {resident_daemon}" in message
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
    resident_daemon: int,
) -> None:
    """A resident daemon arms both halves: the factories and ``sqlite3.connect``."""
    from polylogue.storage.sqlite.write_guard import archive_write_guard_installed
    from polylogue.storage.sqlite.write_lease import write_lease_enforced

    root = _archive_root(monkeypatch, tmp_path)
    (root / "daemon.pid").write_text(f"{resident_daemon}\n")

    with cli_archive_writer_ownership():
        assert write_lease_enforced() is True
        assert archive_write_guard_installed() is True
    assert write_lease_enforced() is False
    assert archive_write_guard_installed() is False


def test_escaping_refusal_is_translated_through_click(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resident_daemon: int,
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
    (root / "daemon.pid").write_text(f"{resident_daemon}\n")

    def _refuse(*args: object, **kwargs: object) -> None:
        raise UnleasedWriteError("open_connection(embeddings.db) requires the daemon write lease")

    monkeypatch.setattr(reconcile, "reconcile_embedding_orphans", _refuse)

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "embedding-orphan-reconcile", "--output-format", "json"],
        catch_exceptions=True,
    )

    assert isinstance(result.exception, ArchiveWriterOwnershipError), result.output
    assert f"PID {resident_daemon}" in str(result.exception)
