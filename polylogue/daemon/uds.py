"""Unix-domain transport for the daemon's existing HTTP handler."""

from __future__ import annotations

import asyncio
import socketserver
import threading
from collections import deque
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from polylogue.daemon.execution import BoundedComputeAdapter
from polylogue.daemon.socket_path import daemon_socket_path
from polylogue.daemon.web_auth import WebCredentialRegistry
from polylogue.daemon.write_coordinator import DaemonWriteThreadBridge

_ARCHIVE_QUERY_MAX_WORKERS = 8
_ARCHIVE_QUERY_MAX_QUEUED = 16


def machine_operation_handler() -> type[BaseHTTPRequestHandler]:
    """Return the single-route machine handler without exposing browser GETs.

    The import is lazy so importing the machine transport does not import the
    browser route registry.  The returned adapter deliberately admits only
    ``POST /api/operation``; browser routes remain on the TCP listener.
    """
    from polylogue.daemon.http import DaemonAPIHandler

    class MachineOperationHandler(DaemonAPIHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_error(405, "machine endpoint accepts POST only")

        def do_HEAD(self) -> None:  # noqa: N802
            self.send_error(405, "machine endpoint accepts POST only")

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path != "/api/operation":
                self.send_error(404, "machine route not found")
                return
            super().do_POST()

    return MachineOperationHandler


class _StandaloneWriteRuntime:
    """Minimal coordinator for standalone machine-socket tests."""

    def __init__(self) -> None:
        ready = threading.Event()
        self.loop = asyncio.new_event_loop()
        self.coordinator = None

        def run() -> None:
            asyncio.set_event_loop(self.loop)
            from polylogue.daemon.write_coordinator import DaemonWriteCoordinator

            self.coordinator = DaemonWriteCoordinator()
            ready.set()
            self.loop.run_forever()
            self.loop.close()

        self.thread = threading.Thread(target=run, name="daemon-uds-writer", daemon=True)
        self.thread.start()
        if not ready.wait(5.0):
            raise RuntimeError("standalone daemon UDS writer loop failed to start")
        self.bridge = DaemonWriteThreadBridge(self.coordinator, self.loop)

    def close(self) -> None:
        if self.coordinator is None:
            return
        future = asyncio.run_coroutine_threadsafe(self.coordinator.shutdown(timeout=5.0), self.loop)
        try:
            idle = future.result(timeout=5.5)
        except TimeoutError:
            idle = False
        if idle:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=1.0)
        else:
            asyncio.run_coroutine_threadsafe(self._stop_when_idle(), self.loop)

    async def _stop_when_idle(self) -> None:
        while self.coordinator is not None and not await self.coordinator.shutdown(timeout=5.0):
            await asyncio.sleep(0)
        self.loop.call_soon(self.loop.stop)


class DaemonAPIUnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """AF_UNIX peer for :class:`DaemonAPIHTTPServer`; routing stays identical."""

    daemon_threads = True

    def __init__(
        self,
        socket_path: Path,
        handler_class: type[BaseHTTPRequestHandler] | None = None,
        *,
        auth_token: str | None = None,
        write_bridge: DaemonWriteThreadBridge | None = None,
    ) -> None:
        # UnixStreamServer may call server_close() while super().__init__ is
        # unwinding a failed bind.  Establish every attribute that cleanup
        # reads before crossing that boundary so the original OSError remains
        # authoritative.
        self.socket_path = socket_path
        self.execution_kernel: BoundedComputeAdapter | None = None
        self.archive_query_executor = None
        self._owned_write_runtime: _StandaloneWriteRuntime | None = None
        socket_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with __import__("contextlib").suppress(FileNotFoundError):
            socket_path.unlink()
        if handler_class is None:
            handler_class = machine_operation_handler()
        super().__init__(str(socket_path), handler_class)
        self.auth_token = auth_token
        self.api_host = "127.0.0.1"
        self.started_at = datetime.now(UTC).isoformat()
        self.web_credentials = WebCredentialRegistry()
        if write_bridge is None:
            self._owned_write_runtime = _StandaloneWriteRuntime()
            write_bridge = self._owned_write_runtime.bridge
        self.write_bridge = write_bridge
        self.execution_kernel = BoundedComputeAdapter(
            max_workers=_ARCHIVE_QUERY_MAX_WORKERS,
            queue_units=_ARCHIVE_QUERY_MAX_QUEUED,
            thread_name_prefix="polylogue-compute",
        )
        self.archive_query_executor = self.execution_kernel.executor
        self.coordination_cache: dict[tuple[str, int], Any] = {}
        self.coordination_cache_lock = threading.Lock()
        self.coordination_cache_condition = threading.Condition(self.coordination_cache_lock)
        self.coordination_cache_building: set[tuple[str, int]] = set()
        self.operation_ids_seen: set[str] = set()
        self.operation_ids_order: deque[str] = deque(maxlen=4096)
        # request_id -> (request fingerprint, HTTP status, immutable envelope)
        # lets a client recover a lost response without re-running a write.
        self.operation_results: dict[str, tuple[str, int, dict[str, object]]] = {}
        self.operation_ids_lock = threading.Lock()

    def server_close(self) -> None:
        kernel = getattr(self, "execution_kernel", None)
        if kernel is not None:
            kernel.shutdown(wait=False, cancel_futures=True)
        owned_write_runtime = getattr(self, "_owned_write_runtime", None)
        if owned_write_runtime is not None:
            owned_write_runtime.close()
            self._owned_write_runtime = None
        super().server_close()
        with __import__("contextlib").suppress(FileNotFoundError):
            self.socket_path.unlink()


__all__ = ["DaemonAPIUnixHTTPServer", "daemon_socket_path", "machine_operation_handler"]
