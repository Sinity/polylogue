"""Minimal stdlib UDS client for daemon-owned maintenance and read routes."""

from __future__ import annotations

import http.client
import json
import socket
from pathlib import Path
from time import perf_counter
from typing import Any


class DaemonResponseError(RuntimeError):
    """A daemon response with a typed non-success HTTP envelope."""

    def __init__(self, *, status: int, code: str | None, detail: str | None) -> None:
        self.status = status
        self.code = code
        self.detail = detail or code or f"daemon returned HTTP {status}"
        super().__init__(self.detail)


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: Path, timeout: float | None) -> None:
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(str(self.socket_path))


class DaemonClient:
    """Transport adapter for the daemon's existing AF_UNIX HTTP routes."""

    def __init__(self, socket_path: Path, *, timeout_s: float | None = 0.1, auth_token: str | None = None) -> None:
        self.socket_path = socket_path
        self.timeout_s = timeout_s
        self.auth_token = auth_token
        self.last_elapsed_ms: int | None = None

    def request_json(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None = None,
        *,
        raise_for_status: bool = False,
        accepted_statuses: frozenset[int] = frozenset({200}),
    ) -> dict[str, Any] | None:
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
            payload = json.loads(response.read().decode())
            self.last_elapsed_ms = round((perf_counter() - started_at) * 1000)
            if response.status not in accepted_statuses:
                if raise_for_status:
                    envelope = payload if isinstance(payload, dict) else {}
                    code = envelope.get("error")
                    detail = envelope.get("detail")
                    raise DaemonResponseError(
                        status=response.status,
                        code=code if isinstance(code, str) else None,
                        detail=detail if isinstance(detail, str) else None,
                    )
                return None
            return payload if isinstance(payload, dict) else None
        except (OSError, TimeoutError, ValueError, http.client.HTTPException):
            return None
        finally:
            connection.close()

    def cli_query(self, params: dict[str, object]) -> dict[str, Any] | None:
        return self.request_json("POST", "/api/cli/query", {"params": params})

    def probe(
        self,
        *,
        archive_root: str,
        index_schema_version: int,
        daemon_version: str,
        accept_degraded: bool = False,
    ) -> dict[str, Any] | None:
        """Return identity only for the daemon serving the requested archive.

        Maintenance callers may accept the health endpoint's 503 envelope in
        order to reach the daemon-owned repair route.  This does not authorize
        the repair: the write endpoint still runs its typed preflight.  Query
        callers retain the strict 200-only default.
        """
        health = (
            self.request_json("GET", "/api/health", accepted_statuses=frozenset({200, 503}))
            if accept_degraded
            else self.request_json("GET", "/api/health")
        )
        if health is None:
            return None
        if health.get("archive_root") != archive_root:
            return None
        if health.get("index_schema_version") != index_schema_version:
            return None
        if health.get("daemon_version") != daemon_version:
            return None
        return health


__all__ = ["DaemonClient", "DaemonResponseError"]
