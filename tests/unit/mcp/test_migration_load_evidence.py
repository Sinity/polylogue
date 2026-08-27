"""Evidence for the consolidated MCP read migration.

These tests deliberately use the registered six-tool server and real
``RuntimeServices`` backed by a temporary SQLite archive. The migration
anti-vacuity test disables the shared transaction boundary and expects the
real query route to fail. The load test then drives all six read tools from
multiple OS threads and checks request identity, admission, cleanup, and
archive integrity after the calls finish. Additional registered-route tests
cover cancellation during result assembly and the response-budget path for
large results.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import tracemalloc
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import pytest

# These tests exercise the incident-scale query response and cancellation
# paths. Keep them in the isolated lane so their large transient payloads do
# not contend with ordinary selected tests.
pytestmark = pytest.mark.storage_scale

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
from polylogue.mcp.server_support import MCP_RESPONSE_BUDGET_BYTES
from tests.infra.mcp import MCPServerUnderTest, installed_runtime_services, invoke_surface_async


def _seed_archive(
    archive_root: Path,
    *,
    count: int,
    marker_prefix: str = "needle-mcp-load",
    message_bytes: int = 0,
) -> tuple[str, ...]:
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
                    provider_session_id=f"mcp-load-{index:03d}",
                    title=f"MCP read isolation {index}",
                    messages=[
                        ParsedMessage(
                            provider_message_id="m1",
                            role=Role.USER,
                            text=(marker := f"{marker_prefix}-{index:03d}")
                            + (f" {'x' * message_bytes}" if message_bytes else ""),
                            blocks=[
                                ParsedContentBlock(
                                    type=BlockType.TEXT,
                                    text=marker + (f" {'x' * message_bytes}" if message_bytes else ""),
                                )
                            ],
                        )
                    ],
                )
            )
            for index in range(count)
        )


@pytest.mark.asyncio
async def test_read_migration_fails_closed_without_shared_query_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The canonical MCP query cannot succeed through a bypassing adapter."""
    from polylogue.mcp.server import build_server

    monkeypatch.setenv("POLYLOGUE_NO_DAEMON", "1")
    archive_root = tmp_path / "archive"
    _seed_archive(archive_root, count=1)
    server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities()))
    query_fn = server._tool_manager._tools["query"].fn

    def disabled_transaction(self: QueryTransaction, work: object) -> object:
        del self, work
        raise RuntimeError("mcp-migration-proof: shared query transaction disabled")

    monkeypatch.setattr(QueryTransaction, "run", disabled_transaction)

    with installed_runtime_services(archive_root):
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


def _archive_fd_targets(archive_root: Path) -> dict[str, int] | None:
    proc_fd = Path("/proc/self/fd")
    if not proc_fd.is_dir():
        return None
    root = str(archive_root.resolve())
    targets: dict[str, int] = {}
    for entry in proc_fd.iterdir():
        try:
            target = os.readlink(entry)
        except OSError:
            continue
        if target == root or target.startswith(root + os.sep):
            targets[target] = targets.get(target, 0) + 1
    return targets


def _archive_files(archive_root: Path) -> frozenset[str]:
    return frozenset(path.relative_to(archive_root).as_posix() for path in archive_root.rglob("*") if path.is_file())


def _archive_file_sizes(archive_root: Path) -> dict[str, int]:
    return {
        path.relative_to(archive_root).as_posix(): path.stat().st_size
        for path in archive_root.rglob("*")
        if path.is_file()
    }


def _require_fd_probe(archive_root: Path) -> dict[str, int]:
    snapshot = _archive_fd_targets(archive_root)
    if snapshot is None:
        pytest.skip("archive descriptor cleanup evidence requires /proc/self/fd")
    return snapshot


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


@pytest.mark.asyncio
async def test_registered_query_disconnect_drains_real_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancelled registered query stamps disconnect and drains its reader."""
    from polylogue.archive.query import unit_results
    from polylogue.mcp.server import build_server

    monkeypatch.setenv("POLYLOGUE_NO_DAEMON", "1")
    archive_root = tmp_path / "archive"
    _seed_archive(archive_root, count=1)
    server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities()))
    query_fn = server._tool_manager._tools["query"].fn
    entered = threading.Event()
    observed_contexts: list[QueryExecutionContext] = []
    original_envelope = unit_results.query_unit_envelope

    def hold_after_real_query(*args: Any, **kwargs: Any) -> Any:
        execution_context = kwargs.get("execution_context")
        assert isinstance(execution_context, QueryExecutionContext)
        result = original_envelope(*args, **kwargs)
        observed_contexts.append(execution_context)
        entered.set()
        assert execution_context.cancel_event.wait(timeout=5), (
            "registered query did not observe disconnect cancellation"
        )
        return result

    monkeypatch.setattr(unit_results, "query_unit_envelope", hold_after_real_query)
    reset_default_admission_controller_for_tests()
    before_fds = _require_fd_probe(archive_root)

    with installed_runtime_services(archive_root):
        task = asyncio.create_task(
            invoke_surface_async(
                query_fn,
                expression="messages where text:needle-mcp-load-000",
                limit=1,
            )
        )
        assert await asyncio.to_thread(entered.wait, 5), "registered query never reached real result assembly"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert len(observed_contexts) == 1
    context = observed_contexts[0]
    assert context.owner_ref == "query_units"
    assert context.receipt.rows_emitted == 1
    assert context.receipt.state == "disconnected"
    assert context.receipt.cleanup_complete is True
    assert default_admission_controller().in_flight_weight == 0
    assert _archive_fd_targets(archive_root) == before_fds


@pytest.mark.asyncio
async def test_registered_large_query_bounds_transient_bytes_and_cleans_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The registered query's formatted wire envelope leaves bounded temp state."""
    from polylogue.mcp.server import build_server

    monkeypatch.setenv("POLYLOGUE_NO_DAEMON", "1")
    archive_root = tmp_path / "archive"
    _seed_archive(
        archive_root,
        count=20,
        marker_prefix="needle-mcp-large-result",
        message_bytes=3_500,
    )
    server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities()))
    query_fn = server._tool_manager._tools["query"].fn

    async def invoke_large_query() -> str:
        return await invoke_surface_async(
            query_fn,
            expression="messages where text:needle-mcp-large-result",
            limit=20,
        )

    outer_before_fds = _require_fd_probe(archive_root)
    with installed_runtime_services(archive_root):
        warm_response = json.loads(await invoke_large_query())
        assert warm_response["status"] == "response_budget_exceeded"

        before_files = _archive_file_sizes(archive_root)
        before_fds = _require_fd_probe(archive_root)
        tracemalloc.start()
        baseline_bytes, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        response_text = await invoke_large_query()
        response = json.loads(response_text)
        continuation = cast(dict[str, Any], response["continuation"])
        resumed = json.loads(await invoke_surface_async(query_fn, **cast(dict[str, Any], continuation["arguments"])))
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        after_files = _archive_file_sizes(archive_root)
        after_fds = _require_fd_probe(archive_root)
        assert all(after_fds.get(path, 0) <= count for path, count in before_fds.items())
        assert set(after_fds) <= set(before_fds)

    response_bytes = len(response_text.encode("utf-8"))
    compact_response_bytes = len(json.dumps(response, sort_keys=True).encode("utf-8"))
    assert response["status"] == "response_budget_exceeded"
    assert response["original_bytes"] > MCP_RESPONSE_BUDGET_BYTES
    assert response["returned_items"] > 0
    assert response["continuation"] is not None
    assert response_bytes <= MCP_RESPONSE_BUDGET_BYTES
    assert response_bytes > compact_response_bytes
    assert response_text == json.dumps(response, indent=2, ensure_ascii=False, default=str)

    resumed_page = cast(
        dict[str, Any], resumed["page"] if resumed.get("status") == "response_budget_exceeded" else resumed
    )
    first_item_ids = {item["message_id"] for item in cast(list[dict[str, str]], response["page"]["items"])}
    resumed_item_ids = {item["message_id"] for item in cast(list[dict[str, str]], resumed_page["items"])}
    assert resumed_item_ids
    assert first_item_ids.isdisjoint(resumed_item_ids)

    transient_bytes = max(0, peak_bytes - baseline_bytes)
    assert transient_bytes <= 8 * 1024 * 1024

    new_files = set(after_files) - set(before_files)
    allowed_sidecars = {
        f"{database.name}-{sidecar}" for database in archive_root.glob("*.db") for sidecar in ("wal", "shm")
    }
    assert new_files <= allowed_sidecars
    temporary_sidecar_bytes = sum(after_files[path] for path in new_files if path in allowed_sidecars)
    assert temporary_sidecar_bytes <= 1 * 1024 * 1024
    assert not any(path.endswith((".tmp", ".spill", ".partial")) for path in after_files)
    assert _archive_fd_targets(archive_root) == outer_before_fds


def test_concurrent_consolidated_read_surface_is_isolated_and_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Incident-shaped concurrency keeps request identity and read resources isolated."""
    from polylogue.mcp.server import build_server

    request_count = 32
    worker_count = 16
    archive_root = tmp_path / "archive"
    state_root = tmp_path / "state"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_root))
    monkeypatch.setenv("POLYLOGUE_NO_DAEMON", "1")
    session_ids = _seed_archive(archive_root, count=request_count)
    markers = tuple(f"needle-mcp-load-{index:03d}" for index in range(request_count))
    server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))

    with installed_runtime_services(archive_root):
        snapshot_refs = asyncio.run(_seed_context_deliveries(server, markers))
        before_fds = _require_fd_probe(archive_root)

        reset_default_admission_controller_for_tests()
        observed_in_flight: list[int] = []
        observed_admission_owners: list[str | None] = []
        observation_lock = threading.Lock()
        capacity_barrier = threading.Barrier(DEFAULT_CAPACITY)
        barrier_participants = 0
        participant_lock = threading.Lock()
        original_admit_async = QueryAdmissionController._admit_async

        async def observe_admission(self: QueryAdmissionController, ctx: QueryExecutionContext) -> int:
            """Observe after async admission, then synchronize admitted readers.

            The consolidated MCP tools acquire through ``_admit_async`` before
            transferring lease release to their executor completion callback;
            observing the public context manager would miss that ownership
            route. The barrier runs in a worker thread so all request loops can
            reach the rendezvous without blocking one another.
            """
            nonlocal barrier_participants
            weight = await original_admit_async(self, ctx)
            with observation_lock:
                observed_in_flight.append(self.in_flight_weight)
                observed_admission_owners.append(ctx.owner_ref)
            with participant_lock:
                is_participant = barrier_participants < DEFAULT_CAPACITY
                if is_participant:
                    barrier_participants += 1
            if is_participant:
                await asyncio.to_thread(capacity_barrier.wait, timeout=10)
            return weight

        monkeypatch.setattr(QueryAdmissionController, "_admit_async", observe_admission)

        # The production route is async; the barrier above deliberately holds
        # the first capacity-sized set after admission so the max-in-flight
        # assertion measures overlapping leases rather than scheduling order.

        before_files = _archive_files(archive_root)

        def run_request(index: int) -> tuple[int, dict[str, dict[str, Any]]]:
            return index, asyncio.run(
                _read_bundle(server, session_ids[index], markers[index], snapshot_refs[markers[index]])
            )

        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="mcp-load") as executor:
            results = dict(executor.map(run_request, range(request_count)))
        after_fds = _require_fd_probe(archive_root)
        assert after_fds == before_fds

    assert set(results) == set(range(request_count))
    for index, bundle in results.items():
        marker = markers[index]
        session_id = session_ids[index]
        foreign_identities = (set(markers) | set(session_ids)) - {marker, session_id}
        for tool_name, body in bundle.items():
            assert body.get("is_error") is not True, f"{tool_name} failed for {marker}: {body}"
            serialized = json.dumps(body, sort_keys=True)
            assert not any(identity in serialized for identity in foreign_identities), (
                f"{tool_name} leaked another request identity for {marker}: {body}"
            )
        for tool_name in ("query", "explain"):
            assert marker in json.dumps(bundle[tool_name], sort_keys=True), (
                f"{tool_name} dropped the request identity for {marker}: {bundle[tool_name]}"
            )
        for tool_name in ("read", "get"):
            assert session_id in json.dumps(bundle[tool_name], sort_keys=True), (
                f"{tool_name} dropped the request identity for {session_id}: {bundle[tool_name]}"
            )
        context_serialized = json.dumps(bundle["context"], sort_keys=True)
        assert marker in context_serialized, f"context dropped its own marker {marker}: {bundle['context']}"
        status = bundle["status"]
        assert status["scope"] == "archive", status
        archive_status = MCPArchiveStatsPayload.model_validate(status["archive"])
        assert archive_status.total_sessions == request_count
        assert archive_status.total_messages == request_count
        status_serialized = json.dumps(status, sort_keys=True)
        assert not any(marker in status_serialized for marker in markers), (
            f"status exposed request-specific data for {marker}: {status}"
        )

    controller = default_admission_controller()
    assert observed_in_flight
    assert barrier_participants == DEFAULT_CAPACITY
    assert max(observed_in_flight) == DEFAULT_CAPACITY
    assert all(value <= DEFAULT_CAPACITY for value in observed_in_flight)
    assert "context" not in observed_admission_owners
    owner_counts = Counter(observed_admission_owners)
    assert set(owner_counts) == {"query_units", "archive.resolve_ref", "status"}
    assert owner_counts["query_units"] == request_count
    assert owner_counts["archive.resolve_ref"] == request_count * 2
    assert owner_counts["status"] == request_count
    assert owner_counts.get("explain", 0) == 0
    assert controller.in_flight_weight == 0

    after_files = _archive_files(archive_root)
    allowed_new_sidecars = {
        f"{database.name}-{sidecar}" for database in archive_root.glob("*.db") for sidecar in ("wal", "shm")
    }
    assert after_files - before_files <= allowed_new_sidecars
    assert not (state_root / "polylogue" / "mcp-call-log").exists()

    for database in archive_root.glob("*.db"):
        with sqlite3.connect(database) as connection:
            assert connection.execute("PRAGMA quick_check").fetchone() == ("ok",)
