"""One browser operation over a locally authenticated, owned receiver socket."""

from __future__ import annotations

import base64
import binascii
import errno
import http.client
import json
import os
import queue
import select
import socket
import sqlite3
import struct
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlparse

import ijson
from ijson.common import JSONError

from polylogue.browser_capture.models import BROWSER_CAPTURE_API_SCHEMA
from polylogue.browser_capture.native_host import _authenticate_receiver
from polylogue.browser_capture.receiver import (
    ReceiverCredentialError,
    load_or_mint_receiver_identity,
    load_or_mint_receiver_token,
)
from polylogue.core.json import JSONValue
from polylogue.core.staged_body import (
    BodyIncompleteError,
    BodyStorageExhaustedError,
    StagedBody,
    stage_body_chunks,
)
from polylogue.schemas.observation_spill import StreamedJSONDocument, StreamedJSONReadError
from polylogue.storage.sqlite.connection_profile import NativeConnectionSettlementError

_CHUNK_BYTES = 64 * 1024
_FORBIDDEN_HEADERS = frozenset(
    {"authorization", "host", "connection", "content-length", "transfer-encoding", "origin", "proxy-authorization"}
)


class NativeTransportError(RuntimeError):
    """A named refusal before or during a native operation."""


class NativeStorageError(RuntimeError):
    """Native request or frame custody failed locally."""


def _send(output: BinaryIO, value: dict[str, object]) -> None:
    data = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
    output.write(struct.pack("<I", len(data)))
    output.write(data)
    output.flush()


class NativeInput:
    """A single pending staged frame and disconnect-aware socket custody.

    The reader watches a stop pipe as well as browser stdin. EOF cancels the
    owned socket even while its HTTP response is blocked. Frames have no
    total-body ceiling; their JSON documents are decoded off the wire on disk.
    """

    def __init__(self, input_fd: int, root: Path) -> None:
        self.input_fd = input_fd
        self.root = root
        self.pending: queue.Queue[StagedBody | BaseException] = queue.Queue(maxsize=1)
        self.cancelled = threading.Event()
        self.stopped = threading.Event()
        self.failure: BaseException | None = None
        self._stop_read, self._stop_write = os.pipe()
        self._socket_lock = threading.Lock()
        self._socket: socket.socket | None = None
        self._thread = threading.Thread(target=self._read_frames, daemon=True)

    def __enter__(self) -> NativeInput:
        self._thread.start()
        return self

    def bind_socket(self, peer: socket.socket) -> None:
        with self._socket_lock:
            self._socket = peer
            if self.cancelled.is_set():
                self._interrupt_socket()

    def _interrupt_socket(self) -> None:
        if self._socket is not None:
            with suppress(OSError):
                self._socket.shutdown(socket.SHUT_RDWR)

    def cancel(self) -> None:
        self.cancelled.set()
        with self._socket_lock:
            self._interrupt_socket()

    def _read(self, length: int) -> bytes:
        if self.stopped.is_set():
            return b""
        ready, _, _ = select.select([self.input_fd, self._stop_read], [], [])
        if self._stop_read in ready:
            return b""
        return os.read(self.input_fd, length)

    def _read_exact(self, length: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < length:
            part = self._read(length - len(chunks))
            if not part:
                raise NativeTransportError("native_input_incomplete")
            chunks.extend(part)
        return bytes(chunks)

    def _frame_chunks(self, length: int) -> Iterator[bytes]:
        remaining = length
        while remaining:
            chunk = self._read(min(_CHUNK_BYTES, remaining))
            if not chunk:
                raise NativeTransportError("native_input_incomplete")
            remaining -= len(chunk)
            yield chunk

    def _put(self, item: StagedBody | BaseException) -> bool:
        while not self.stopped.is_set():
            try:
                self.pending.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def _read_frames(self) -> None:
        try:
            while not self.stopped.is_set():
                size = struct.unpack("<I", self._read_exact(4))[0]
                try:
                    frame = stage_body_chunks(
                        self._frame_chunks(size), spool_root=self.root, durable=False, reserved_length=size
                    )
                except OSError as exc:
                    raise NativeStorageError("receiver_observation_storage_failed") from exc
                # Cancellation is a control signal, not a request queued behind
                # an HTTP read. Inspect only its declared type under reader custody.
                try:
                    with frame.path.open("rb") as source:
                        for prefix, event, value in ijson.parse(source):
                            if prefix == "type" and event == "string":
                                if value == "cancel":
                                    frame.discard()
                                    raise NativeTransportError("native_operation_cancelled")
                                break
                except OSError as exc:
                    frame.discard()
                    raise NativeStorageError("receiver_observation_storage_failed") from exc
                except BaseException:
                    frame.discard()
                    raise
                if not self._put(frame):
                    frame.discard()
                    return
        except BaseException as exc:
            self.failure = exc
            self.cancel()
            self._put(exc)

    def _next_frame(self) -> StagedBody | BaseException:
        while True:
            try:
                return self.pending.get(timeout=0.05)
            except queue.Empty:
                if self.cancelled.is_set():
                    if self.failure is not None:
                        raise self.failure from None
                    raise NativeTransportError("native_operation_cancelled") from None

    @contextmanager
    def frame(self) -> Iterator[dict[str, JSONValue]]:
        staged = self._next_frame()
        if isinstance(staged, BaseException):
            raise staged
        try:
            with StreamedJSONDocument(staged.path) as value:
                if not isinstance(value, dict):
                    raise NativeTransportError("native_request_invalid")
                if value.get("type") == "cancel":
                    self.cancel()
                    raise NativeTransportError("native_operation_cancelled")
                yield value
        except (sqlite3.Error, NativeConnectionSettlementError) as exc:
            raise NativeStorageError("receiver_observation_storage_failed") from exc
        except (ValueError, JSONError) as exc:
            raise NativeTransportError("native_request_invalid") from exc
        finally:
            staged.discard()

    def __exit__(self, *args: object) -> None:
        self.stopped.set()
        os.write(self._stop_write, b"x")
        self.cancel()
        self._thread.join()
        os.close(self._stop_read)
        os.close(self._stop_write)
        while not self.pending.empty():
            item = self.pending.get_nowait()
            if isinstance(item, StagedBody):
                item.discard()


def _request(frame: dict[str, JSONValue]) -> tuple[str, int, str, str, str | None, dict[str, str]]:
    if frame.get("type") != "request" or frame.get("version") != 1:
        raise NativeTransportError("native_request_invalid")
    endpoint = frame.get("endpoint")
    if not isinstance(endpoint, str):
        raise NativeTransportError("loopback_endpoint_required")
    parsed = urlparse(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise NativeTransportError("loopback_endpoint_required")
    method, path, expected = frame.get("method"), frame.get("path"), frame.get("receiver_id")
    if (
        method not in {"GET", "POST", "PUT"}
        or not isinstance(path, str)
        or not path.startswith("/v1/")
        or urlparse(path).netloc
        or urlparse(path).fragment
        or urlparse(path).path == "/v1/pairing/redeem"
        or (expected is not None and not isinstance(expected, str))
    ):
        raise NativeTransportError("native_request_invalid")
    headers = frame.get("headers")
    if not isinstance(headers, dict):
        raise NativeTransportError("native_request_invalid")
    copied: dict[str, str] = {}
    for key in headers:
        value = headers[key]
        if (
            not isinstance(value, str)
            or key.lower() in _FORBIDDEN_HEADERS
            or "\r" in key
            or "\n" in key
            or "\r" in value
            or "\n" in value
        ):
            raise NativeTransportError("native_request_invalid")
        copied[key] = value
    assert isinstance(method, str)
    return parsed.hostname, parsed.port or 80, method, path, expected, copied


def _body(frames: NativeInput, output: BinaryIO) -> Iterator[bytes]:
    sequence = 0
    while True:
        with frames.frame() as frame:
            if frame.get("type") == "body_end":
                return
            data = frame.get("data")
            if frame.get("type") != "body" or frame.get("sequence") != sequence or not isinstance(data, str):
                raise NativeTransportError("native_request_invalid")
            try:
                chunk = base64.b64decode(data, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise NativeTransportError("native_request_invalid") from exc
        yield chunk
        _send(output, {"type": "body_ack", "sequence": sequence})
        sequence += 1


def _connect(connection: http.client.HTTPConnection, frames: NativeInput, host: str, port: int) -> None:
    """Connect only to loopback, with cancellation rather than a work deadline."""
    import ipaddress

    failure: OSError | None = None
    for family, kind, protocol, _name, address in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM):
        if not ipaddress.ip_address(str(address[0])).is_loopback:
            raise NativeTransportError("loopback_endpoint_required")
        peer = socket.socket(family, kind, protocol)
        try:
            frames.bind_socket(peer)
            peer.setblocking(False)
            code = peer.connect_ex(address)
            while code in {errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EALREADY}:
                if frames.cancelled.is_set():
                    raise NativeTransportError("native_operation_cancelled")
                _readable, writable, _errors = select.select([], [peer], [peer], 0.05)
                if writable or _errors:
                    code = peer.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            if code:
                raise OSError(code, os.strerror(code))
            if frames.cancelled.is_set():
                raise NativeTransportError("native_operation_cancelled")
            peer.setblocking(True)
            connection.sock = peer
            connection.auto_open = 0
            return
        except OSError as exc:
            failure = exc
            peer.close()
        except BaseException:
            peer.close()
            raise
    if failure is not None:
        raise failure
    raise NativeTransportError("receiver_unreachable")


def serve_native_operation(extension_id: str, *, input_fd: int, output: BinaryIO) -> int:
    """Run one operation; the persisted bearer never enters a native frame."""
    connection: http.client.HTTPConnection | None = None
    body: StagedBody | None = None
    try:
        if not extension_id:
            raise NativeTransportError("native_sender_identity_required")
        with (
            tempfile.TemporaryDirectory(prefix="polylogue-native-operation-") as temporary,
            NativeInput(input_fd, Path(temporary)) as frames,
        ):
            with frames.frame() as request:
                host, port, method, path, expected, headers = _request(request)
            identity = load_or_mint_receiver_identity()
            if expected is not None and expected != identity:
                raise NativeTransportError("receiver_identity_mismatch")
            secret = load_or_mint_receiver_token()
            try:
                body = stage_body_chunks(_body(frames, output), spool_root=Path(temporary), durable=False)
            except OSError as exc:
                raise NativeStorageError("receiver_observation_storage_failed") from exc
            connection = http.client.HTTPConnection(host, port, timeout=None)
            # Bind cancellation before proof or operation response reads.
            _connect(connection, frames, host, port)
            assert connection.sock is not None
            frames.bind_socket(connection.sock)
            refusal = _authenticate_receiver(connection, identity, secret)
            if refusal is not None:
                raise NativeTransportError(refusal)
            assert connection.sock is not None
            connection.sock.settimeout(None)
            headers["Authorization"] = f"Bearer {secret}"
            headers["Origin"] = f"chrome-extension://{extension_id}"
            headers["Content-Length"] = str(body.size_bytes)
            with body.path.open("rb") as source:
                connection.request(method, path, body=source if body.size_bytes else None, headers=headers)
            response = connection.getresponse()
            _send(
                output,
                {
                    "type": "response",
                    "status": response.status,
                    "headers": dict(response.getheaders()),
                    "receiver_id": identity,
                    "api_schema": BROWSER_CAPTURE_API_SCHEMA,
                },
            )
            sequence = 0
            try:
                while True:
                    with frames.frame() as frame:
                        if frame.get("type") != "response_next":
                            raise NativeTransportError("native_request_invalid")
                    chunk = response.read(_CHUNK_BYTES)
                    if not chunk:
                        _send(output, {"type": "response_end"})
                        return 0
                    _send(
                        output,
                        {
                            "type": "response_body",
                            "sequence": sequence,
                            "data": base64.b64encode(chunk).decode("ascii"),
                        },
                    )
                    sequence += 1
            finally:
                response.close()
    except NativeTransportError as exc:
        with suppress(BrokenPipeError):
            _send(output, {"type": "error", "error": str(exc)})
    except (BodyIncompleteError, EOFError):
        _send(output, {"type": "error", "error": "native_input_incomplete"})
    except (BodyStorageExhaustedError, StreamedJSONReadError, NativeStorageError):
        _send(output, {"type": "error", "error": "receiver_observation_storage_failed"})
    except ReceiverCredentialError:
        _send(output, {"type": "error", "error": "receiver_credential_unavailable"})
    except ValueError:
        _send(output, {"type": "error", "error": "native_request_invalid"})
    except (OSError, http.client.HTTPException):
        _send(output, {"type": "error", "error": "receiver_unreachable"})
    finally:
        if connection is not None:
            connection.close()
        if body is not None:
            body.discard()
    return 1
