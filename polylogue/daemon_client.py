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


class DaemonMutationIndeterminateError(RuntimeError):
    """A confirmed mutation may have reached the daemon without a receipt."""

    def __init__(self, *, method: str, path: str) -> None:
        self.method = method
        self.path = path
        super().__init__(f"daemon outcome is indeterminate after {method} {path}")


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: Path, timeout: float | None) -> None:
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path
        self.connected = False

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(str(self.socket_path))
        self.connected = True


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
        response = self._request_json_response(method, path, body, mutation=False)
        if response is None:
            return None
        status, payload = response
        if status not in accepted_statuses:
            if raise_for_status:
                self._raise_response_error(status, payload)
            return None
        return payload

    def request_mutation_json(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None = None,
    ) -> dict[str, Any] | None:
        """Submit a confirmed mutation without conflating no-daemon and no-receipt."""

        response = self._request_json_response(method, path, body, mutation=True)
        if response is None:
            return None
        status, payload = response
        if status != 200:
            self._raise_response_error(status, payload)
        return payload

    @staticmethod
    def _raise_response_error(status: int, payload: dict[str, Any] | None) -> None:
        envelope = payload if isinstance(payload, dict) else {}
        code = envelope.get("error")
        detail = envelope.get("detail")
        raise DaemonResponseError(
            status=status,
            code=code if isinstance(code, str) else None,
            detail=detail if isinstance(detail, str) else None,
        )

    def _request_json_response(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None = None,
        *,
        mutation: bool = False,
    ) -> tuple[int, dict[str, Any] | None] | None:
        """Return the response status with its decoded JSON object, if any."""

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
            response_body = response.read()
            try:
                decoded = json.loads(response_body.decode())
            except (UnicodeDecodeError, ValueError):
                decoded = None
            self.last_elapsed_ms = round((perf_counter() - started_at) * 1000)
            return response.status, decoded if isinstance(decoded, dict) else None
        except KeyboardInterrupt as exc:
            if mutation and connection.connected:
                raise DaemonMutationIndeterminateError(method=method, path=path) from exc
            raise
        except (OSError, TimeoutError, ValueError, http.client.HTTPException) as exc:
            if mutation and connection.connected:
                raise DaemonMutationIndeterminateError(method=method, path=path) from exc
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
        """Return identity only for the daemon serving the requested archive."""

        response = self._request_json_response("GET", "/api/health")
        if response is None:
            return None
        status, health = response
        if status == 503:
            if not accept_degraded or health is None or health.get("raw_failure_lifecycle_state") != "degraded":
                return None
        elif status != 200:
            return None
        if health is None:
            return None
        if health.get("archive_root") != archive_root:
            return None
        if health.get("index_schema_version") != index_schema_version:
            return None
        if health.get("daemon_version") != daemon_version:
            return None
        return health


__all__ = ["DaemonClient", "DaemonMutationIndeterminateError", "DaemonResponseError"]
