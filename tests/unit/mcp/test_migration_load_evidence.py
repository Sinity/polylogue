"""Evidence for the consolidated MCP read migration.

These tests deliberately use the registered six-tool server and real
``RuntimeServices`` backed by a temporary SQLite archive. The migration
anti-vacuity test disables the shared transaction boundary and expects the
real query route to fail. The load test then drives all six read tools from
multiple OS threads and checks request identity, admission, cleanup, and
archive integrity after the calls finish.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.archive.query.execution_control import (
    DEFAULT_CAPACITY,
    QueryAdmissionController,
    QueryExecutionContext,
    default_admission_controller,
    reset_default_admission_controller_for_tests,
)
from polylogue.archive.query.transaction import QueryTransaction
from polylogue.mcp.declarations.models import MCPCapabilities
from polylogue.mcp.payloads import MCPArchiveStatsPayload
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async


def _seed_archive(archive_root: Path, *, count: int) -> tuple[str, ...]:
    """Create distinct searchable sessions through the real archive writer."""
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    with ArchiveStore(archive_root) as archive:
        return tuple(
            archive.write_parsed(
                ParsedSession(
                    source_name=Provider.CHATGPT,
                    provider_session_id=f"mcp-load-{index}",
                    title=f"MCP read isolation {index}",
                    messages=[
                        ParsedMessage(
                            provider_message_id="m1",
                            role=Role.USER,
                            text=f"needle-mcp-load-{index:03d}",
                            blocks=[
                                ParsedContentBlock(
                                    type=BlockType.TEXT,
                                    text=f"needle-mcp-load-{index:03d}",
                                )
                            ],
                        )
                    ],
                )
            )
            for index in range(count)
        )


@contextmanager
def _installed_runtime_services(archive_root: Path) -> Iterator[None]:
    """Point the real MCP facade at one isolated archive root."""
    from polylogue.config import Config
    from polylogue.mcp import server_support
    from polylogue.services import RuntimeServices

    services = RuntimeServices(
        config=Config(archive_root=archive_root, render_root=archive_root.parent / "render", sources=[]),
    )
    try:
        original: RuntimeServices | None = server_support._get_runtime_services()
    except RuntimeError:
        original = None
    server_support._set_runtime_services(services)
    try:
        yield
    finally:
        server_support._set_runtime_services(original)


@pytest.mark.asyncio
async def test_read_migration_fails_closed_without_shared_query_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The canonical MCP query cannot succeed through a bypassing adapter."""
    from polylogue.mcp.server import build_server

    archive_root = tmp_path / "archive"
    _seed_archive(archive_root, count=1)
    server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities()))
    query_fn = server._tool_manager._tools["query"].fn

    def disabled_transaction(self: QueryTransaction, work: object) -> object:
        del self, work
        raise RuntimeError("mcp-migration-proof: shared query transaction disabled")

    monkeypatch.setattr(QueryTransaction, "run", disabled_transaction)

    with _installed_runtime_services(archive_root):
        result = json.loads(
            await invoke_surface_async(
                query_fn,
                expression="messages where text:needle-mcp-load-000",
                limit=1,
            )
        )

    assert result["is_error"] is True
    assert result["code"] == "internal_error"
    assert result["detail"] == "RuntimeError"


def _archive_fd_targets(archive_root: Path) -> frozenset[str]:
    proc_fd = Path("/proc/self/fd")
    if not proc_fd.exists():
        return frozenset()
    root = str(archive_root.resolve())
    targets: set[str] = set()
    for entry in proc_fd.iterdir():
        try:
            target = os.readlink(entry)
        except OSError:
            continue
        if target == root or target.startswith(root + os.sep):
            targets.add(target)
    return frozenset(targets)


def _archive_files(archive_root: Path) -> frozenset[str]:
    return frozenset(path.relative_to(archive_root).as_posix() for path in archive_root.rglob("*") if path.is_file())


async def _read_bundle(
    server: MCPServerUnderTest,
    session_id: str,
    marker: str,
    snapshot_ref: str,
) -> dict[str, dict[str, Any]]:
    """Call every registered base read tool once for one request identity."""
    tools = server._tool_manager._tools
    session_ref = f"session:{session_id}"
    return {
        "query": json.loads(
            await invoke_surface_async(
                tools["query"].fn,
                expression=f"messages where text:{marker}",
                limit=1,
            )
        ),
        "read": json.loads(await invoke_surface_async(tools["read"].fn, ref=session_ref)),
        "get": json.loads(await invoke_surface_async(tools["get"].fn, ref=session_ref)),
        "explain": json.loads(
            await invoke_surface_async(
                tools["explain"].fn,
                subject="query",
                expression=f"messages where text:{marker}",
            )
        ),
        "context": json.loads(
            await invoke_surface_async(
                tools["context"].fn,
                intent="lookup",
                result_ref=snapshot_ref,
                recipient_ref=f"agent:{marker}",
            )
        ),
        "status": json.loads(await invoke_surface_async(tools["status"].fn, scope="archive")),
    }


async def _seed_context_deliveries(
    server: MCPServerUnderTest,
    markers: tuple[str, ...],
) -> dict[str, str]:
    """Create one durable, marker-bearing context receipt per request."""
    write_fn = server._tool_manager._tools["write"].fn
    snapshot_refs: dict[str, str] = {}
    for marker in markers:
        delivered = json.loads(
            await invoke_surface_async(
                write_fn,
                operation="deliver_context",
                fields={
                    "recipient_ref": f"agent:{marker}",
                    "delivered_by_ref": "user:local",
                    "boundary": "mcp-evidence",
                    "query": marker,
                    "max_sessions": 1,
                    "include_messages": True,
                    "include_assertions": True,
                },
            )
        )
        assert delivered.get("is_error") is not True, delivered
        assert delivered["recipient_ref"] == f"agent:{marker}"
        assert marker in json.dumps(delivered["context_image"], sort_keys=True)
        snapshot_refs[marker] = delivered["snapshot_ref"]
    return snapshot_refs


def test_concurrent_consolidated_read_surface_is_isolated_and_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Incident-shaped concurrency keeps request identity and read resources isolated."""
    from polylogue.mcp.server import build_server

    request_count = 32
    worker_count = 16
    archive_root = tmp_path / "archive"
    session_ids = _seed_archive(archive_root, count=request_count)
    markers = tuple(f"needle-mcp-load-{index:03d}" for index in range(request_count))
    server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))

    with _installed_runtime_services(archive_root):
        snapshot_refs = asyncio.run(_seed_context_deliveries(server, markers))

        reset_default_admission_controller_for_tests()
        observed_in_flight: list[int] = []
        observed_admission_owners: list[str | None] = []
        observation_lock = threading.Lock()
        capacity_barrier = threading.Barrier(DEFAULT_CAPACITY)
        barrier_participants = 0
        participant_lock = threading.Lock()
        original_admit = QueryAdmissionController.admit_blocking

        @contextmanager
        def observe_admission(self: QueryAdmissionController, ctx: QueryExecutionContext) -> Iterator[None]:
            nonlocal barrier_participants
            with original_admit(self, ctx):
                with observation_lock:
                    observed_in_flight.append(self.in_flight_weight)
                    observed_admission_owners.append(ctx.owner_ref)
                with participant_lock:
                    is_participant = barrier_participants < DEFAULT_CAPACITY
                    if is_participant:
                        barrier_participants += 1
                if is_participant:
                    capacity_barrier.wait(timeout=10)
                yield

        monkeypatch.setattr(QueryAdmissionController, "admit_blocking", observe_admission)

        before_files = _archive_files(archive_root)
        before_fds = _archive_fd_targets(archive_root)

        def run_request(index: int) -> tuple[int, dict[str, dict[str, Any]]]:
            return index, asyncio.run(
                _read_bundle(server, session_ids[index], markers[index], snapshot_refs[markers[index]])
            )

        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="mcp-load") as executor:
            results = dict(executor.map(run_request, range(request_count)))

    assert set(results) == set(range(request_count))
    for index, bundle in results.items():
        marker = markers[index]
        foreign_markers = set(markers) - {marker}
        for tool_name, body in bundle.items():
            assert body.get("is_error") is not True, f"{tool_name} failed for {marker}: {body}"
            serialized = json.dumps(body, sort_keys=True)
            assert not foreign_markers.intersection(serialized), (
                f"{tool_name} leaked another request identity for {marker}: {body}"
            )
        for tool_name in ("query", "explain"):
            assert marker in json.dumps(bundle[tool_name], sort_keys=True), (
                f"{tool_name} dropped the request identity for {marker}: {bundle[tool_name]}"
            )
        for tool_name in ("read", "get"):
            assert session_ids[index] in json.dumps(bundle[tool_name], sort_keys=True), (
                f"{tool_name} dropped the request identity for {session_ids[index]}: {bundle[tool_name]}"
            )
        context_serialized = json.dumps(bundle["context"], sort_keys=True)
        assert marker in context_serialized, f"context dropped its own marker {marker}: {bundle['context']}"
        status = bundle["status"]
        assert status["scope"] == "archive", status
        archive_status = MCPArchiveStatsPayload.model_validate(status["archive"])
        assert archive_status.total_sessions == request_count
        assert archive_status.total_messages == request_count
        status_serialized = json.dumps(status, sort_keys=True)
        assert not set(markers).intersection(status_serialized), (
            f"status exposed request-specific data for {marker}: {status}"
        )

    controller = default_admission_controller()
    assert observed_in_flight
    assert barrier_participants == DEFAULT_CAPACITY
    assert max(observed_in_flight) == DEFAULT_CAPACITY
    assert all(value <= DEFAULT_CAPACITY for value in observed_in_flight)
    assert "context" not in observed_admission_owners
    assert controller.in_flight_weight == 0

    after_files = _archive_files(archive_root)
    allowed_new_sidecars = {
        f"{database.name}-{sidecar}" for database in archive_root.glob("*.db") for sidecar in ("wal", "shm")
    }
    assert after_files - before_files <= allowed_new_sidecars
    assert _archive_fd_targets(archive_root) == before_fds

    for database in archive_root.glob("*.db"):
        with sqlite3.connect(database) as connection:
            assert connection.execute("PRAGMA quick_check").fetchone() == ("ok",)
