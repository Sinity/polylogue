"""Unix-domain transport for the daemon's existing HTTP handler."""

from __future__ import annotations

import socketserver
import threading
from collections import deque
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from polylogue.daemon.execution import BoundedComputeAdapter
from polylogue.daemon.http import (
    _ARCHIVE_QUERY_MAX_QUEUED,
    _ARCHIVE_QUERY_MAX_WORKERS,
    _StandaloneWriteRuntime,
)
from polylogue.daemon.socket_path import daemon_socket_path
from polylogue.daemon.web_auth import WebCredentialRegistry
from polylogue.daemon.write_coordinator import DaemonWriteThreadBridge


class DaemonOperationHandler(BaseHTTPRequestHandler):
    """Archive-scoped machine endpoint with no browser route surface.

    The operation implementation is attached by the daemon composition root;
    this handler accepts only the one typed POST endpoint. Browser HTTP keeps
    its own handler and route registry.
    """

    daemon_handler_class: type[BaseHTTPRequestHandler] | None = None

    def do_GET(self) -> None:
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/api/operation":
            self.send_error(404)
            return
        handler_class = getattr(self.server, "daemon_handler_class", None)
        if handler_class is None:
            self.send_error(500)
            return
        # The browser implementation is used as a semantic operation
        # executor only. Its route dispatcher is never entered on this socket.
        handler_class(self.request, self.client_address, self.server)

    def log_message(self, format: str, *args: object) -> None:
        return


class DaemonAPIUnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """AF_UNIX peer for :class:`DaemonAPIHTTPServer`; routing stays identical."""

    daemon_threads = True

    def __init__(
        self,
        socket_path: Path,
        handler_class: type[BaseHTTPRequestHandler],
        *,
        auth_token: str | None = None,
        write_bridge: DaemonWriteThreadBridge | None = None,
    ) -> None:
        # UnixStreamServer may call server_close() while super().__init__ is
        # unwinding a failed bind.  Establish every attribute that cleanup
        # reads before crossing that boundary so the original OSError remains
        # authoritative.
        self.socket_path = socket_path
        self.daemon_handler_class = handler_class
        self.execution_kernel: BoundedComputeAdapter | None = None
        self.archive_query_executor = None
        self._owned_write_runtime: _StandaloneWriteRuntime | None = None
        socket_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with __import__("contextlib").suppress(FileNotFoundError):
            socket_path.unlink()
        # The socket itself is served by the machine-only handler. The
        # supplied class is retained solely as the semantic executor for the
        # typed operation, so browser routes cannot be reached over AF_UNIX.
        # Keep test-specific handler subclasses usable while the production
        # browser handler gets the machine-only wrapper.
        request_handler = DaemonOperationHandler if handler_class.__name__ == "DaemonAPIHandler" else handler_class
        super().__init__(str(socket_path), request_handler)
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
        self.archive_query_admission = threading.BoundedSemaphore(
            _ARCHIVE_QUERY_MAX_WORKERS + _ARCHIVE_QUERY_MAX_QUEUED
        )
        self.coordination_cache: dict[tuple[str, int], Any] = {}
        self.coordination_cache_lock = threading.Lock()
        self.coordination_cache_condition = threading.Condition(self.coordination_cache_lock)
        self.coordination_cache_building: set[tuple[str, int]] = set()
        self.operation_ids_seen: set[str] = set()
        self.operation_ids_order: deque[str] = deque(maxlen=4096)
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


__all__ = ["DaemonAPIUnixHTTPServer", "DaemonOperationHandler", "daemon_socket_path"]
