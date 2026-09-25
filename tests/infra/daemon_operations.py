"""Production-stack fixtures for declared daemon operation tests.

The fixture deliberately starts the independent machine UDS listener rather
than borrowing the browser HTTP handler.  It also owns a real coordinator loop
so mutation tests exercise the same writer retention rules as the daemon.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import queue
import socket
import sys
import threading
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest

from polylogue.daemon.execution import BoundedComputeAdapter
from polylogue.daemon.operation_runtime import DaemonOperationRuntime
from polylogue.daemon.socket_path import ensure_private_socket_dir
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
            return session_id in archive.resolve_exact_session_ids((session_id,))

    def close(self) -> None:
        """Stop ingress, drain the lifecycle owner, then release resources."""

        self.server.shutdown()
        self.server.server_close()
        self._server_thread.join(timeout=2)
        if self._server_thread.is_alive():
            raise TimeoutError("test machine operation listener did not stop")

        loop = self._loop.loop
        assert loop is not None
        asyncio.run_coroutine_threadsafe(self.runtime.shutdown(), loop).result(timeout=5)
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
    server_error_sink: queue.SimpleQueue[str] | None = None,
    compute_workers: int = 2,
    compute_queue_units: int = 4,
    socket_path: Path | None = None,
) -> Iterator[DaemonOperationStack]:
    """Start one real machine operation stack rooted at ``archive_root``.

    ``seed_archive`` runs after bootstrap and before daemon startup. The shared
    compute adapter is passed to both the runtime and UDS server explicitly;
    no global adapter, browser route, configuration root, or daemon singleton
    is used.  ``compute_workers``/``compute_queue_units`` size that one bounded
    kernel: a test that deliberately saturates admission and one that measures
    service under load need different sizes, and both must state which.
    """

    archive_root = archive_root.resolve()
    initialize_active_archive_root(archive_root)
    if seed_archive is not None:
        seed_archive(archive_root)
    socket_path = socket_path or (Path("/tmp") / f"plg-op-{os.getpid()}-{uuid4().hex}.sock")
    if socket_path.parent != Path("/tmp"):
        ensure_private_socket_dir(socket_path.parent)
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
    kernel = BoundedComputeAdapter(
        max_workers=compute_workers, queue_units=compute_queue_units, thread_name_prefix="test-daemon-operation"
    )
    runtime = DaemonOperationRuntime(
        archive_root, write_bridge=bridge, execution_kernel=kernel, owner_loop=bridge.owner_loop
    )
    server = DaemonAPIUnixHTTPServer(
        socket_path,
        archive_root=archive_root,
        auth_token=None,
        write_bridge=bridge,
        execution_kernel=kernel,
        operation_runtime=runtime,
    )
    if server_error_sink is not None:

        def capture_server_error(_request: object, _client_address: object) -> None:
            """Retain one unhandled handler traceback for a test assertion."""

            exc_type, exc, trace = sys.exc_info()
            assert exc_type is not None and exc is not None
            server_error_sink.put("".join(traceback.format_exception(exc_type, exc, trace)))

        object.__setattr__(server, "handle_error", capture_server_error)
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


@contextlib.contextmanager
def cli_daemon_archive(
    archive_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    seed_archive: Callable[[Path], None] | None = None,
    home: Path | None = None,
) -> Iterator[DaemonOperationStack]:
    """Run a real daemon and point the CLI's mutation route at it.

    CLI mutation commands own syntax, preview and refusal only; the resident
    daemon is the sole writer (``configured_mutation_operation``). A CLI test
    that wants to observe what a mutation actually *does* therefore has to
    supply the daemon the command requires, which is what this does: a real
    ``DaemonOperationRuntime`` behind a real UDS listener, reached over the
    ordinary socket the CLI probes.

    ``POLYLOGUE_ARCHIVE_ROOT`` and the XDG roots are set rather than patching
    ``polylogue.cli.commands.*`` path helpers, because the daemon-side handlers
    resolve their own paths through :mod:`polylogue.paths`; patching a CLI
    module attribute would move only the preview and leave the writer pointed
    at the developer's real home.
    """

    archive_root = archive_root.resolve()
    with running_daemon_operations(archive_root, seed_archive=seed_archive) as stack:
        monkeypatch.setattr("polylogue.daemon.socket_path.daemon_socket_path", lambda _root: stack.socket_path)
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
        if home is not None:
            home.mkdir(parents=True, exist_ok=True)
            monkeypatch.setenv("HOME", str(home))
            monkeypatch.setenv("XDG_DATA_HOME", str(home / ".local" / "share"))
            monkeypatch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
            monkeypatch.setenv("XDG_STATE_HOME", str(home / ".local" / "state"))
            monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
        yield stack


@pytest.fixture
def daemon_operation_stack(tmp_path: Path) -> Iterator[DaemonOperationStack]:
    """Pytest fixture providing an empty, synthetic, production operation stack."""

    with running_daemon_operations(tmp_path / "archive") as stack:
        yield stack


__all__ = [
    "DaemonOperationStack",
    "cli_daemon_archive",
    "daemon_operation_stack",
    "running_daemon_operations",
]
