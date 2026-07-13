"""Minimal stdlib UDS client for the hot daemon read path.

This module deliberately has no archive/storage imports: importing it must be
cheap enough to decide whether the daemon can answer before direct CLI setup.
"""

from __future__ import annotations

import http.client
import json
import socket
from pathlib import Path
from time import perf_counter
from typing import Any


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: Path, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(str(self.socket_path))


class DaemonClient:
    def __init__(self, socket_path: Path, *, timeout_s: float = 0.1, auth_token: str | None = None) -> None:
        self.socket_path = socket_path
        self.timeout_s = timeout_s
        self.auth_token = auth_token
        self.last_elapsed_ms: int | None = None

    def request_json(self, method: str, path: str, body: dict[str, object] | None = None) -> dict[str, Any] | None:
        if not self.socket_path.exists():
            return None
        connection = _UnixHTTPConnection(self.socket_path, self.timeout_s)
        raw = json.dumps(body, separators=(",", ":")).encode() if body is not None else None
        started_at = perf_counter()
        try:
            headers = {"Host": "127.0.0.1", "Content-Type": "application/json"}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            connection.request(method, path, body=raw, headers=headers)
            response = connection.getresponse()
            if response.status != 200:
                return None
            payload = json.loads(response.read().decode())
            self.last_elapsed_ms = round((perf_counter() - started_at) * 1000)
            return payload if isinstance(payload, dict) else None
        except (OSError, TimeoutError, ValueError, http.client.HTTPException):
            return None
        finally:
            connection.close()

    def cli_query(self, params: dict[str, object]) -> dict[str, Any] | None:
        """Run one root-request parameter dictionary through the daemon."""

        return self.request_json("POST", "/api/cli/query", {"params": params})

    def probe(self, *, archive_root: str, index_schema_version: int, daemon_version: str) -> dict[str, Any] | None:
        health = self.request_json("GET", "/api/health")
        if health is None:
            return None
        if health.get("archive_root") != archive_root:
            return None
        if health.get("index_schema_version") != index_schema_version:
            return None
        if health.get("daemon_version") != daemon_version:
            return None
        return health


__all__ = ["DaemonClient"]
