"""The archive's single bounded HTTP/1.1 machine operation endpoint."""

from __future__ import annotations

import hashlib
import hmac
import json
import socket
import socketserver
import struct
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING

from polylogue.daemon.socket_path import daemon_socket_path
from polylogue.operations.daemon_protocol import (
    DAEMON_OPERATION_PROTOCOL,
    DAEMON_OPERATION_SPECS,
    MAX_DECLARED_OPERATION_BODY_BYTES,
    MAX_OPERATION_RESULT_BYTES,
    DaemonOperationRequest,
)
from polylogue.operations.mutation_transaction import MutationPrincipal

if TYPE_CHECKING:
    from polylogue.daemon.execution import BoundedComputeAdapter
    from polylogue.daemon.operation_runtime import DaemonOperationRuntime
    from polylogue.daemon.write_coordinator import DaemonWriteThreadBridge


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"invalid JSON constant: {value}")


def _peer_principal(connection: socket.socket, token: str | None) -> MutationPrincipal:
    """Bind authority to a bearer or the kernel-authenticated local user."""
    if token:
        actor = f"daemon:bearer:{hashlib.sha256(token.encode()).hexdigest()}"
        role = "daemon-authenticated"
    else:
        try:
            credentials = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
            _pid, uid, _gid = struct.unpack("3i", credentials)
        except (AttributeError, OSError, struct.error) as exc:
            raise PermissionError("Unix peer credentials are unavailable") from exc
        if uid < 0:
            raise PermissionError("Unix peer credentials are invalid")
        actor = f"daemon:unix:uid:{uid}"
        role = "daemon-unix-peer"
    return MutationPrincipal(
        actor_ref=actor,
        capabilities=frozenset(spec.capability for spec in DAEMON_OPERATION_SPECS),
        surface="cli",
        role_label=role,
    )


class MachineOperationHandler(BaseHTTPRequestHandler):
    """One exchange per bounded connection, independent of browser routing."""

    protocol_version = "HTTP/1.1"
    server: DaemonAPIUnixHTTPServer

    def setup(self) -> None:
        self.request.settimeout(5.0)
        super().setup()

    def log_message(self, _format: str, *_args: object) -> None:
        pass

    def _send(self, status: int, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        if len(encoded) > MAX_OPERATION_RESULT_BYTES:
            payload = {
                **payload,
                "result": None,
                "outcome": "indeterminate" if payload.get("accepted_reference") else "failed",
                "error": {"code": "result_too_large", "detail": "result exceeds the operation response bound"},
            }
            encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
            status = 413
        self.close_connection = True
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return

    def _reject(self, status: int, code: str, detail: str) -> None:
        self._send(
            status,
            {
                "protocol": DAEMON_OPERATION_PROTOCOL,
                "outcome": "rejected",
                "error": {"code": code, "detail": detail, "retryable": False},
            },
        )

    def do_GET(self) -> None:
        self._reject(405, "method_not_allowed", "machine operations require POST")

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_PUT(self) -> None:
        self.do_GET()

    def do_DELETE(self) -> None:
        self.do_GET()

    def do_PATCH(self) -> None:
        self.do_GET()

    def do_OPTIONS(self) -> None:
        self.do_GET()

    def do_POST(self) -> None:
        started = monotonic()
        if self.path != "/api/operation":
            self._reject(404, "operation_endpoint_required", "machine endpoint is /api/operation")
            return
        token = self.server.auth_token
        authorization = self.headers.get("Authorization", "")
        if token and not hmac.compare_digest(authorization, f"Bearer {token}"):
            self._reject(401, "unauthorized", "machine authentication required")
            return
        if self.headers.get("Transfer-Encoding") is not None:
            self._reject(400, "invalid_framing", "chunked operation requests are unsupported")
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            self._reject(415, "unsupported_media_type", "operation body must be application/json")
            return
        lengths = self.headers.get_all("Content-Length", [])
        try:
            if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdecimal():
                raise ValueError("one positive Content-Length is required")
            length = int(lengths[0])
            if length <= 0:
                raise ValueError("operation body is missing")
        except ValueError as exc:
            self._reject(400, "invalid_framing", str(exc))
            return
        if length > MAX_DECLARED_OPERATION_BODY_BYTES:
            self._reject(413, "request_too_large", "operation body exceeds the declared bound")
            return
        try:
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("partial operation body")
            raw = json.loads(body, parse_constant=_reject_json_constant)
            request = DaemonOperationRequest.from_dict(raw)
        except (ValueError, TypeError, UnicodeDecodeError, TimeoutError) as exc:
            self._reject(400, "invalid_request", str(exc))
            return
        try:
            principal = _peer_principal(self.connection, token)
        except PermissionError as exc:
            self._reject(401, "peer_authentication_unavailable", str(exc))
            return
        from polylogue.daemon.operation_disconnect import observe_peer_disconnect

        with observe_peer_disconnect(self.connection) as disconnected:
            envelope = self.server.operation_runtime.call(
                request, principal, started_at=started, client_disconnect=disconnected
            )
        outcome = envelope.get("outcome")
        status = 202 if outcome in {"accepted", "running", "indeterminate"} else 200
        if outcome in {"failed", "rejected", "timed-out", "cancelled"}:
            status = 409
        self._send(status, envelope)


class DaemonAPIUnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    request_queue_size = 24

    def __init__(
        self,
        socket_path: Path,
        *,
        archive_root: Path,
        auth_token: str | None,
        write_bridge: DaemonWriteThreadBridge,
        execution_kernel: BoundedComputeAdapter,
        operation_runtime: DaemonOperationRuntime | None = None,
    ) -> None:
        from polylogue.daemon.operation_runtime import DaemonOperationRuntime

        self.socket_path = socket_path
        self.auth_token = auth_token
        self._connections = threading.BoundedSemaphore(self.request_queue_size)
        self._socket_identity: tuple[int, int] | None = None
        self.operation_runtime = operation_runtime or DaemonOperationRuntime(
            archive_root,
            write_bridge=write_bridge,
            execution_kernel=execution_kernel,
        )
        socket_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        super().__init__(str(socket_path), MachineOperationHandler)
        socket_path.chmod(0o600)
        metadata = socket_path.stat()
        self._socket_identity = (metadata.st_dev, metadata.st_ino)

    def process_request(self, request: socket.socket, client_address: object) -> None:
        if not self._connections.acquire(blocking=False):
            try:
                request.settimeout(0.1)
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._connections.release()
            raise

    def process_request_thread(self, request: socket.socket, client_address: object) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connections.release()

    def server_close(self) -> None:
        super().server_close()
        if self._socket_identity is not None:
            try:
                metadata = self.socket_path.stat()
                if (metadata.st_dev, metadata.st_ino) == self._socket_identity:
                    self.socket_path.unlink()
            except FileNotFoundError:
                pass


__all__ = ["DaemonAPIUnixHTTPServer", "MachineOperationHandler", "daemon_socket_path"]
