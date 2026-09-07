"""The daemon's bounded, archive-scoped machine HTTP endpoint.

The UDS listener is deliberately a different ``BaseHTTPRequestHandler`` from
the browser listener.  It has one route and never constructs a browser route
table.  ``operation_adapter`` is a temporary semantic adapter: it supplies
the canonical operation implementations, not HTTP dispatch.
"""

from __future__ import annotations

import asyncio
import socketserver
import threading
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, cast

from polylogue.daemon.execution import BoundedComputeAdapter
from polylogue.daemon.socket_path import daemon_socket_path
from polylogue.daemon.web_auth import WebCredentialRegistry
from polylogue.daemon.write_coordinator import DaemonWriteThreadBridge

_ARCHIVE_QUERY_MAX_WORKERS = 8
_ARCHIVE_QUERY_MAX_QUEUED = 16


def machine_operation_handler(operation_adapter: type[object] | None = None) -> type[BaseHTTPRequestHandler]:
    """Return the single-route machine handler without exposing browser GETs.

    The handler deliberately admits only ``POST /api/operation``; browser
    routes remain on the TCP listener.  An adapter is copied as a set of
    operation methods rather than inherited.  This is important: Python's
    MRO must not make browser ``do_*`` methods or its route registry reachable
    from the machine socket.
    """

    class MachineOperationHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _format: str, *_args: object) -> None:
            """Machine clients receive typed errors; do not emit access logs."""

        def do_GET(self) -> None:
            self.send_error(405, "machine endpoint accepts POST only")

        def do_HEAD(self) -> None:
            self.send_error(405, "machine endpoint accepts POST only")

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if path != "/api/operation":
                self.send_error(404, "machine route not found")
                return
            # The machine listener has no browser dispatch.  Calling the
            # browser handler's ``do_POST`` here would consult its complete
            # route table before arriving at this operation adapter.
            if not self._check_host_admission():
                return
            if self._reject_credential_query():
                return
            if not self._check_auth("read", allow_web=False):
                return
            self._handle_daemon_operation()

        def _operation_helper(self, name: str) -> Callable[..., object]:
            adapter = getattr(self.server, "operation_adapter", None)
            if adapter is None:
                raise RuntimeError("machine operation executor is unavailable")
            member = getattr(adapter, name)
            descriptor = getattr(member, "__get__", None)
            return cast(Callable[..., object], descriptor(self, type(self)) if descriptor is not None else member)

        def _check_host_admission(self) -> bool:
            if operation_adapter is None:
                return True
            return bool(self._operation_helper("_check_host_admission")())

        def _reject_credential_query(self) -> bool:
            if operation_adapter is None:
                return False
            return bool(self._operation_helper("_reject_credential_query")())

        def _check_auth(self, *args: object, **kwargs: object) -> bool:
            if operation_adapter is None:
                return True
            return bool(self._operation_helper("_check_auth")(*args, **kwargs))

        def _handle_daemon_operation(self) -> None:
            if operation_adapter is None:
                self.send_error(503, "machine operation executor is unavailable")
                return
            self._operation_helper("_handle_daemon_operation")()

    if operation_adapter is not None:
        # The operation code has a large set of small private helpers.  Bind
        # one only when the canonical implementation asks for it; copying the
        # browser class's route methods would make the machine handler's
        # surface depend on that registry again.
        def _machine_getattr(self: BaseHTTPRequestHandler, name: str) -> object:
            adapter = getattr(self.server, "operation_adapter", None)
            if adapter is None:
                raise AttributeError(name)
            member = getattr(adapter, name)
            descriptor = getattr(member, "__get__", None)
            return descriptor(self, type(self)) if descriptor is not None else member

        MachineOperationHandler.__getattr__ = _machine_getattr  # type: ignore[attr-defined]
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
        assert self.coordinator is not None
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
        # ``handler_class`` remains a source-compatible constructor argument
        # for embedding tests.  It is an operation adapter, never the UDS
        # request handler itself.
        adapter = None if handler_class is BaseHTTPRequestHandler else handler_class
        super().__init__(str(socket_path), machine_operation_handler(adapter))
        self.operation_adapter = adapter
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
