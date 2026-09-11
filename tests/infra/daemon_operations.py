"""Production-stack fixtures for declared daemon operation tests.

The fixture deliberately starts the independent machine UDS listener rather
than borrowing the browser HTTP handler.  It also owns a real coordinator loop
so mutation tests exercise the same writer retention rules as the daemon.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest

from polylogue.daemon.execution import BoundedComputeAdapter
from polylogue.daemon.operation_runtime import DaemonOperationRuntime
from polylogue.daemon.uds import DaemonAPIUnixHTTPServer
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
from polylogue.daemon_client import DaemonClient
from polylogue.operations.mutation_transaction import recover_interrupted_operations
from polylogue.operations.operation_context import prepare_operation_journals
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


class _CoordinatorLoop:
    """One dedicated event loop thread for a real write coordinator."""

    def __init__(self, archive_root: Path) -> None:
        self._archive_root = archive_root
        self.loop: asyncio.AbstractEventLoop | None = None
        self.coordinator: DaemonWriteCoordinator | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="test-daemon-coordinator", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=2):
            raise TimeoutError("test daemon coordinator loop did not start")

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.loop = loop
        self.coordinator = DaemonWriteCoordinator(archive_root=self._archive_root)
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            loop.close()

    def close(self) -> None:
        assert self.loop is not None
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=2)
        if self._thread.is_alive():
            raise TimeoutError("test daemon coordinator loop did not stop")


@dataclass(slots=True)
class DaemonOperationStack:
    """Live machine listener plus its independently-owned production runtime."""

    archive_root: Path
    socket_path: Path
    client: DaemonClient
    server: DaemonAPIUnixHTTPServer
    runtime: DaemonOperationRuntime
    write_coordinator: DaemonWriteCoordinator
    write_bridge: DaemonWriteThreadBridge
    execution_kernel: BoundedComputeAdapter
    _loop: _CoordinatorLoop
    _server_thread: threading.Thread

    def session_exists(self, session_id: str) -> bool:
        """Read the seeded archive through its canonical archive reader."""

        with ArchiveStore.open_existing(self.archive_root) as archive:
            try:
                archive.resolve_session_id(session_id)
            except KeyError:
                return False
        return True

    def close(self) -> None:
        """Stop ingress, drain the lifecycle owner, then release resources."""

        self.server.shutdown()
        self.server.server_close()
        self._server_thread.join(timeout=2)
        if self._server_thread.is_alive():
            raise TimeoutError("test machine operation listener did not stop")

        loop = self._loop.loop
        assert loop is not None
        drained = asyncio.run_coroutine_threadsafe(self.write_coordinator.shutdown(timeout=2), loop).result(timeout=3)
        if not drained:
            raise RuntimeError("test daemon writer did not drain before loop close")
        self.execution_kernel.shutdown(wait=True)
        self._loop.close()


@contextlib.contextmanager
def running_daemon_operations(
    archive_root: Path,
    *,
    seed_archive: Callable[[Path], None] | None = None,
) -> Iterator[DaemonOperationStack]:
    """Start one real machine operation stack rooted at ``archive_root``.

    ``seed_archive`` runs after bootstrap and before daemon startup. The shared
    compute adapter is passed to both the runtime and UDS server explicitly;
    no global adapter, browser route, configuration root, or daemon singleton
    is used.
    """

    archive_root = archive_root.resolve()
    initialize_active_archive_root(archive_root)
    if seed_archive is not None:
        seed_archive(archive_root)
    socket_path = Path("/tmp") / f"plg-op-{os.getpid()}-{uuid4().hex}.sock"
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        try:
            probe.bind(str(socket_path))
        except PermissionError:
            pytest.skip("sandbox denies AF_UNIX listeners required for the production operation stack")
    finally:
        probe.close()
        socket_path.unlink(missing_ok=True)

    coordinator_loop = _CoordinatorLoop(archive_root)
    assert coordinator_loop.loop is not None
    assert coordinator_loop.coordinator is not None
    coordinator = coordinator_loop.coordinator
    bridge = DaemonWriteThreadBridge(coordinator, coordinator_loop.loop, timeout=5)
    bridge.run_sync("daemon.operation_journals.startup", prepare_operation_journals, archive_root)
    bridge.run_sync("daemon.operation_recovery.startup", recover_interrupted_operations, archive_root)
    kernel = BoundedComputeAdapter(max_workers=2, queue_units=4, thread_name_prefix="test-daemon-operation")
    runtime = DaemonOperationRuntime(archive_root, write_bridge=bridge, execution_kernel=kernel)
    server = DaemonAPIUnixHTTPServer(
        socket_path,
        archive_root=archive_root,
        auth_token=None,
        write_bridge=bridge,
        execution_kernel=kernel,
        operation_runtime=runtime,
    )
    thread = threading.Thread(target=server.serve_forever, name="test-machine-operation-listener", daemon=True)
    thread.start()
    stack = DaemonOperationStack(
        archive_root=archive_root,
        socket_path=socket_path,
        client=DaemonClient(socket_path, timeout_s=5),
        server=server,
        runtime=runtime,
        write_coordinator=coordinator,
        write_bridge=bridge,
        execution_kernel=kernel,
        _loop=coordinator_loop,
        _server_thread=thread,
    )
    try:
        yield stack
    finally:
        stack.close()


@pytest.fixture
def daemon_operation_stack(tmp_path: Path) -> Iterator[DaemonOperationStack]:
    """Pytest fixture providing an empty, synthetic, production operation stack."""

    with running_daemon_operations(tmp_path / "archive") as stack:
        yield stack


__all__ = ["DaemonOperationStack", "daemon_operation_stack", "running_daemon_operations"]
