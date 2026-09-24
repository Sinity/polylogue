"""MCP judge coverage for the shared embedded writer boundary."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import pytest

import polylogue.api.archive as archive_module
from polylogue.mcp.declarations.models import MCPCapabilities
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, upsert_assertion
from tests.infra.mcp import MCPServerUnderTest, installed_runtime_services, invoke_surface_async


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


def _user_archive(root: Path) -> None:
    root.mkdir()
    initialize_archive_database(root / "user.db", ArchiveTier.USER)


def _seed_candidate(root: Path, assertion_id: str) -> None:
    with sqlite3.connect(root / "user.db") as conn:
        upsert_assertion(
            conn,
            assertion_id=assertion_id,
            target_ref="session:embedded-boundary",
            kind=AssertionKind.LESSON,
            body_text="A synthetic candidate for the embedded MCP writer boundary.",
            author_ref="agent:test",
            author_kind="agent",
            evidence_refs=["session:embedded-boundary"],
            status="candidate",
            now_ms=1_700_000_000_000,
        )


@pytest.mark.asyncio
async def test_judge_only_mcp_refuses_a_resident_daemon_before_user_write(
    tmp_path: Path,
    resident_daemon: Callable[[Path], int],
) -> None:
    """The judge-only handler reaches the real facade and gets a typed refusal."""

    from polylogue.mcp.server import build_server

    archive_root = tmp_path / "archive"
    _user_archive(archive_root)
    _seed_candidate(archive_root, "resident-boundary-candidate")
    resident_pid = resident_daemon(archive_root / "daemon.pid")
    server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(judge=True)))
    judge_fn = server._tool_manager._tools["judge"].fn

    with installed_runtime_services(archive_root):
        result = json.loads(
            await invoke_surface_async(
                judge_fn,
                candidate_ref="assertion:resident-boundary-candidate",
                decision="accept",
            )
        )

    assert result.get("is_error") is True, result
    assert result.get("detail") == "ArchiveWriterOwnershipError", result
    assert resident_pid > 0


@pytest.mark.asyncio
async def test_judge_only_mcp_remains_functional_without_a_daemon(tmp_path: Path) -> None:
    """The shared boundary does not blanket-refuse entitled offline MCP writes."""

    from polylogue.mcp.server import build_server

    archive_root = tmp_path / "archive"
    _user_archive(archive_root)
    _seed_candidate(archive_root, "offline-boundary-candidate")
    server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(judge=True)))
    judge_fn = server._tool_manager._tools["judge"].fn

    with installed_runtime_services(archive_root):
        result = json.loads(
            await invoke_surface_async(
                judge_fn,
                candidate_ref="assertion:offline-boundary-candidate",
                decision="accept",
            )
        )

    assert result.get("is_error") is not True, result


@pytest.mark.asyncio
async def test_judge_only_mcp_anti_vacuity_restores_the_write_when_boundary_is_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resident_daemon: Callable[[Path], int],
) -> None:
    """Removing the shared API guard restores the formerly unowned judge path."""

    from polylogue.mcp.server import build_server

    archive_root = tmp_path / "archive"
    _user_archive(archive_root)
    _seed_candidate(archive_root, "anti-vacuity-candidate")
    resident_daemon(archive_root / "daemon.pid")
    monkeypatch.setattr(archive_module, "_require_archive_write_authority", lambda *_args: None)
    server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(judge=True)))
    judge_fn = server._tool_manager._tools["judge"].fn

    with installed_runtime_services(archive_root):
        result = json.loads(
            await invoke_surface_async(
                judge_fn,
                candidate_ref="assertion:anti-vacuity-candidate",
                decision="accept",
            )
        )

    assert result.get("is_error") is not True, result
    with sqlite3.connect(archive_root / "user.db") as conn:
        status = conn.execute(
            "SELECT status FROM assertions WHERE assertion_id = ?",
            ("anti-vacuity-candidate",),
        ).fetchone()
    assert status == ("accepted",)


def test_mcp_capability_inventory_is_declaration_derived() -> None:
    """Every capability whose handlers can write is enumerated with its tools.

    ``write`` routes through facade mutations, ``judge`` reaches the direct
    user-tier judgment route, and ``maintenance`` dispatches daemon-owned
    maintenance operations. Keeping the declared tool mapping here makes a
    newly added capability or privileged handler fail until its route is
    audited against the shared ownership boundary.
    """

    from polylogue.mcp.declarations.registry import MCP_TOOL_DECLARATIONS

    tools_by_capability = {
        capability: {
            declaration.name for declaration in MCP_TOOL_DECLARATIONS if declaration.required_capability == capability
        }
        for capability in ("write", "judge", "maintenance")
    }
    assert tools_by_capability == {
        "write": {"write", "record_work_event", "emit_decision", "run"},
        "judge": {"judge"},
        "maintenance": {"maintenance"},
    }
