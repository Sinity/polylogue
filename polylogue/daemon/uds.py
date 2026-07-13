"""Unix-domain transport for the daemon's existing HTTP handler."""

from __future__ import annotations

import os
import socketserver
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from polylogue.daemon.http import (
    _ARCHIVE_QUERY_MAX_QUEUED,
    _ARCHIVE_QUERY_MAX_WORKERS,
    _StandaloneWriteRuntime,
)
from polylogue.daemon.web_auth import WebCredentialRegistry
from polylogue.daemon.write_coordinator import DaemonWriteThreadBridge


def daemon_socket_path(runtime_dir: str | None = None) -> Path:
    """Return the per-user UDS path without creating it."""

    return Path(runtime_dir or os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "polylogue" / "daemon.sock"


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
        socket_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with __import__("contextlib").suppress(FileNotFoundError):
            socket_path.unlink()
        super().__init__(str(socket_path), handler_class)
        self.socket_path = socket_path
        self.auth_token = auth_token
        self.api_host = "127.0.0.1"
        self.started_at = datetime.now(UTC).isoformat()
        self.web_credentials = WebCredentialRegistry()
        self._owned_write_runtime: _StandaloneWriteRuntime | None = None
        if write_bridge is None:
            self._owned_write_runtime = _StandaloneWriteRuntime()
            write_bridge = self._owned_write_runtime.bridge
        self.write_bridge = write_bridge
        self.archive_query_executor = ThreadPoolExecutor(
            max_workers=_ARCHIVE_QUERY_MAX_WORKERS, thread_name_prefix="archive-query"
        )
        self.archive_query_admission = threading.BoundedSemaphore(
            _ARCHIVE_QUERY_MAX_WORKERS + _ARCHIVE_QUERY_MAX_QUEUED
        )
        self.coordination_cache: dict[tuple[str, int], Any] = {}
        self.coordination_cache_lock = threading.Lock()
        self.coordination_cache_condition = threading.Condition(self.coordination_cache_lock)
        self.coordination_cache_building: set[tuple[str, int]] = set()

    def server_close(self) -> None:
        executor = getattr(self, "archive_query_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        if self._owned_write_runtime is not None:
            self._owned_write_runtime.close()
            self._owned_write_runtime = None
        super().server_close()
        with __import__("contextlib").suppress(FileNotFoundError):
            self.socket_path.unlink()


__all__ = ["DaemonAPIUnixHTTPServer", "daemon_socket_path"]
