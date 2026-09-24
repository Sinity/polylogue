"""Production-route checks for embedded archive writer ownership."""

from __future__ import annotations

import inspect
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import polylogue.api.archive as archive_module
import polylogue.context.preamble as preamble_module
from polylogue.analysis.judgment.types import ComparativeJudgment, JudgeIdentity
from polylogue.config import Config
from polylogue.context.preamble import _record_preamble_ledger
from polylogue.context.scheduler import ContextAssembly, schedule_context
from polylogue.core.enums import ComparativeVerdict
from polylogue.core.refs import ExecutionContextRef
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator
from polylogue.maintenance.offline_guard import ArchiveWriterOwnershipError


@pytest.fixture
def resident_daemon(tmp_path: Path) -> Iterator[Callable[[Path], int]]:
    """Hold the archive pidfile lock with a process whose name is irrelevant."""

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
    processes: list[subprocess.Popen[str]] = []

    def start(pidfile: Path) -> int:
        process = subprocess.Popen(
            [sys.executable, str(script), str(pidfile)],
            stdout=subprocess.PIPE,
            text=True,
        )
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


def _config(root: Path) -> Config:
    root.mkdir()
    return Config(archive_root=root, render_root=root.parent / "render", sources=[])


def _judgment() -> ComparativeJudgment:
    return ComparativeJudgment(
        judgment_id="embedded-boundary-judgment",
        items=("session:codex:left", "session:codex:right"),
        dimension="quality",
        verdict=ComparativeVerdict.PREFER_LEFT,
        judge=JudgeIdentity(actor_ref="user:local", execution_context_id="embedded-boundary"),
        blinded=True,
        rubric_id="boundary-rubric",
        rubric_version=1,
        decided_at_ms=1_700_000_000_000,
    )


def _assembly() -> ContextAssembly:
    return schedule_context(
        (),
        moment="context-preamble",
        target_session=None,
        execution_context=ExecutionContextRef.from_legacy_id("embedded-boundary"),
        token_budget=1,
    )


def test_comparative_judgment_writes_when_no_daemon_is_resident(tmp_path: Path) -> None:
    """The boundary is not a blanket refusal of entitled offline writes."""

    root = tmp_path / "archive"
    result = archive_module._archive_record_comparative_judgment(_config(root), _judgment(), author_kind="user")

    assert result.assertion_id
    assert (root / "user.db").exists()


def test_comparative_judgment_refuses_before_user_tier_bootstrap_under_daemon(
    tmp_path: Path,
    resident_daemon: Callable[[Path], int],
) -> None:
    """A resident daemon is named and the route does not initialize user.db."""

    root = tmp_path / "archive"
    config = _config(root)
    resident_pid = resident_daemon(root / "daemon.pid")

    with pytest.raises(ArchiveWriterOwnershipError) as caught:
        archive_module._archive_record_comparative_judgment(config, _judgment(), author_kind="user")

    assert f"PID {resident_pid}" in str(caught.value)
    assert caught.value.resident_writer is not None
    assert not (root / "user.db").exists()


@pytest.mark.asyncio
async def test_comparative_judgment_allows_the_active_daemon_writer_lease(
    tmp_path: Path,
    resident_daemon: Callable[[Path], int],
) -> None:
    """The daemon-owned route remains valid once its coordinator lease is held."""

    root = tmp_path / "archive"
    config = _config(root)
    resident_daemon(root / "daemon.pid")
    coordinator = DaemonWriteCoordinator(archive_root=root)

    async def write() -> object:
        return archive_module._archive_record_comparative_judgment(config, _judgment(), author_kind="user")

    result = await coordinator.run("api.comparative-boundary-test", write)

    assert getattr(result, "assertion_id", None)


def test_context_preamble_ledger_refuses_before_ops_tier_bootstrap_under_daemon(
    tmp_path: Path,
    resident_daemon: Callable[[Path], int],
) -> None:
    """The context surface uses the same per-route boundary as the facade."""

    root = tmp_path / "archive"
    config = _config(root)
    resident_pid = resident_daemon(root / "daemon.pid")

    with pytest.raises(ArchiveWriterOwnershipError) as caught:
        _record_preamble_ledger(SimpleNamespace(config=config), _assembly())

    assert f"PID {resident_pid}" in str(caught.value)
    assert not (root / "ops.db").exists()


def test_disabling_the_route_boundary_reproduces_the_unowned_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resident_daemon: Callable[[Path], int],
) -> None:
    """Anti-vacuity: removing the production guard restores the old write."""

    root = tmp_path / "archive"
    config = _config(root)
    resident_daemon(root / "daemon.pid")
    monkeypatch.setattr(archive_module, "_require_archive_write_authority", lambda *_args: None)

    result = archive_module._archive_record_comparative_judgment(config, _judgment(), author_kind="user")

    assert result.assertion_id
    assert (root / "user.db").exists()


def test_context_preamble_ledger_still_succeeds_offline(tmp_path: Path) -> None:
    """The disposable context receipt remains functional without a daemon."""

    root = tmp_path / "archive"
    config = _config(root)

    _record_preamble_ledger(SimpleNamespace(config=config), _assembly())

    assert (root / "ops.db").exists()


def test_embedded_writable_route_inventory_uses_the_shared_guard() -> None:
    """Every known embedded writable chain crosses one guard before opening."""

    routes = {
        "context delivery user.db": archive_module._archive_record_context_delivery,
        "single assertion judgment user.db": archive_module._archive_judge_assertion_candidate,
        "captured assertion user.db": archive_module._archive_capture_assertion_candidate,
        "bulk assertion judgment user.db": archive_module._archive_judge_assertion_candidates,
        "comparative judgment user.db": archive_module._archive_record_comparative_judgment,
        "facade mutation index/user.db": archive_module.PolylogueArchiveMixin._execute_facade_mutation,
        "manual continuation index/user.db": archive_module.PolylogueArchiveMixin.record_manual_continuation,
        "context injection ledger ops.db": archive_module.PolylogueArchiveMixin.compile_context,
        "work event index.db": archive_module.PolylogueArchiveMixin.record_work_event,
        "session deletion index.db": archive_module.PolylogueArchiveMixin.delete_session_safe,
        "context preamble ledger ops.db": preamble_module._record_preamble_ledger,
    }
    missing = [
        name
        for name, route in routes.items()
        if "_require_archive_write_authority" not in inspect.getsource(cast(Any, route))
    ]
    assert missing == []
