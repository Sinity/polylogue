"""Native bootstrap proof: exact browser allowlist, identity-bound replies, receiver authentication."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import socket
import sqlite3
import struct
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread
from typing import BinaryIO, cast
from urllib.parse import urlparse

import pytest

import polylogue.browser_capture.receiver as receiver_module
from devtools.isolated_environment import isolated_home_environment
from polylogue.browser_capture import native_host, native_transport
from polylogue.browser_capture.server import BrowserCaptureHandler, make_server


def test_install_native_host_is_scoped_to_exact_extension_ids(tmp_path: Path) -> None:
    target = tmp_path / "host.json"
    native_host.install_native_host(
        ("z-extension", "a-extension", "a-extension"), executable="/bin/host", destination=target
    )
    manifest = json.loads(target.read_text())
    assert manifest["allowed_origins"] == ["chrome-extension://a-extension/", "chrome-extension://z-extension/"]
    assert manifest["path"] == "/bin/host"
    assert "auth_token" not in target.read_text()


def test_firefox_manifest_uses_allowed_extensions(tmp_path: Path) -> None:
    """Anti-vacuity: a Chrome-only manifest cannot launch under Firefox."""
    target = tmp_path / "firefox-host.json"
    native_host.install_native_host(
        ("addon@example.invalid",), executable="/bin/host", browser="firefox", destination=target
    )
    manifest = json.loads(target.read_text())
    assert manifest["allowed_extensions"] == ["addon@example.invalid"]
    assert "allowed_origins" not in manifest


def test_native_host_rejects_missing_browser_sender(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps({"endpoint": "http://127.0.0.1:8765"}).encode()
    output = io.BytesIO()
    monkeypatch.setattr(sys, "argv", ["host"])
    monkeypatch.setattr(
        sys,
        "stdin",
        type("S", (), {"buffer": io.BytesIO(struct.pack("<I", len(payload)) + payload)})(),
    )
    monkeypatch.setattr(sys, "stdout", type("S", (), {"buffer": output})())
    assert native_host.main() == 1
    size = struct.unpack("<I", output.getvalue()[:4])[0]
    assert json.loads(output.getvalue()[4 : 4 + size])["error"] == "native_sender_identity_required"


def test_native_host_binds_expected_receiver_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    code, reply, raw = _run_native_host(monkeypatch, "http://127.0.0.1:8765", expected="rx-other")
    assert code == 1
    assert reply["error"] == "receiver_identity_mismatch"
    assert _SECRET.encode() not in raw


_SECRET = "receiver-secret-bearer"


def _run_native_host(
    monkeypatch: pytest.MonkeyPatch, endpoint: str, *, expected: str | None = None
) -> tuple[int, dict[str, object], bytes]:
    """Run the actual native frame owner for one operation with real pipes."""
    import polylogue.runtime

    input_read, input_write = os.pipe()
    output_read, output_write = os.pipe()
    incoming = os.fdopen(input_read, "rb")
    outgoing = os.fdopen(output_write, "wb")
    monkeypatch.setattr(polylogue.runtime, "require_free_threaded_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(sys, "argv", ["host", "chrome-extension://good-id/"])
    monkeypatch.setattr(sys, "stdin", type("S", (), {"buffer": incoming})())
    monkeypatch.setattr(sys, "stdout", type("S", (), {"buffer": outgoing})())
    monkeypatch.setattr(native_transport, "load_or_mint_receiver_identity", lambda: "rx-actual")
    monkeypatch.setattr(receiver_module, "load_or_mint_receiver_identity", lambda path=None: "rx-actual")
    monkeypatch.setattr(native_transport, "load_or_mint_receiver_token", lambda: _SECRET)
    observed: list[tuple[dict[str, object], bytes]] = []
    failures: list[BaseException] = []

    def browser() -> None:
        with os.fdopen(input_write, "wb") as sender, os.fdopen(output_read, "rb") as receiver:
            try:
                _send_frame(
                    sender,
                    {
                        "type": "request",
                        "version": 1,
                        "endpoint": endpoint,
                        "receiver_id": expected,
                        "method": "GET",
                        "path": "/v1/status",
                        "headers": {},
                    },
                )
                _send_frame(sender, {"type": "body_end"})
                reply, raw = _read_frame(receiver)
                if reply["type"] == "response":
                    parts: list[bytes] = []
                    while True:
                        _send_frame(sender, {"type": "response_next"})
                        frame, encoded = _read_frame(receiver)
                        raw += encoded
                        if frame["type"] == "response_end":
                            break
                        if frame["type"] == "error":
                            reply = frame
                            break
                        assert frame["type"] == "response_body"
                        parts.append(base64.b64decode(str(frame["data"])))
                    if reply["type"] == "response":
                        reply = json.loads(b"".join(parts))
                observed.append((reply, raw))
            except BaseException as exc:
                failures.append(exc)

    thread = Thread(target=browser)
    thread.start()
    try:
        code = native_host.main()
    finally:
        outgoing.close()
        incoming.close()
        thread.join()
    assert not failures, failures
    reply, raw = observed[0]
    return code, reply, raw


def _send_frame(stream: BinaryIO, value: dict[str, object]) -> None:
    payload = json.dumps(value).encode()
    stream.write(struct.pack("<I", len(payload)) + payload)
    stream.flush()


def _read_frame(stream: BinaryIO) -> tuple[dict[str, object], bytes]:
    header = stream.read(4)
    assert len(header) == 4
    length = struct.unpack("<I", header)[0]
    payload = stream.read(length)
    assert len(payload) == length
    return json.loads(payload), header + payload


@contextmanager
def _serving(server: ThreadingHTTPServer) -> Iterator[str]:
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.server_address[:2])
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_native_host_operates_on_authenticated_socket_without_releasing_bearer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legitimate receiver answers the challenge, so a fresh profile is paired."""
    with _serving(make_server("127.0.0.1", 0, spool_path=tmp_path, auth_token=_SECRET)) as endpoint:
        code, response, raw = _run_native_host(monkeypatch, endpoint)

    assert code == 0
    assert "auth_token" not in response
    assert _SECRET.encode() not in raw
    assert response["receiver_id"] == "rx-actual"


def test_native_bootstrap_and_next_health_request_do_not_disclose_bearer_to_relay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relay can forward a possession proof without owning its secret."""
    seen_authorization: list[str | None] = []
    with _serving(make_server("127.0.0.1", 0, spool_path=tmp_path, auth_token=_SECRET)) as genuine:
        target = urlparse(genuine)

        class Relay(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                upstream = HTTPConnection(target.hostname or "", target.port or 80, timeout=5)
                try:
                    upstream.request("POST", self.path, body=body)
                    response = upstream.getresponse()
                    payload = response.read()
                    self.send_response(response.status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                finally:
                    upstream.close()

            def do_GET(self) -> None:
                seen_authorization.append(self.headers.get("Authorization"))
                self.send_response(401)
                self.send_header("Content-Length", "0")
                self.end_headers()

        with _serving(ThreadingHTTPServer(("127.0.0.1", 0), Relay)) as endpoint:
            _code, bootstrap, _raw = _run_native_host(monkeypatch, endpoint)
            token = bootstrap.get("auth_token")
            if isinstance(token, str):
                # runtime.js bootstrapReceiverCredential saves this token;
                # checkReceiverHealth then calls probeReceiverStatus with it.
                parsed = urlparse(endpoint)
                peer = HTTPConnection(parsed.hostname or "", parsed.port or 80, timeout=5)
                try:
                    peer.request("GET", "/v1/status", headers={"Authorization": f"Bearer {token}"})
                    peer.getresponse().read()
                finally:
                    peer.close()

    assert f"Bearer {_SECRET}" not in seen_authorization
    assert _code == 1
    assert bootstrap["error"] == "receiver_authentication_failed"


def test_native_host_withholds_the_bearer_when_no_receiver_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stopped daemon leaves the port unanswered: nothing is released."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    code, response, raw = _run_native_host(monkeypatch, f"http://127.0.0.1:{port}")

    assert code == 1
    assert response["error"] == "receiver_unreachable"
    assert "auth_token" not in response
    assert _SECRET.encode() not in raw


def test_native_host_withholds_the_bearer_from_an_impostor_on_the_receiver_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another local process on the port can neither answer nor harvest the bearer.

    Anti-vacuity: at the reported head the host checked only that the endpoint
    was loopback, so this impostor's port received a paired extension's
    bearer and the reply below carried it.
    """
    seen: list[tuple[dict[str, str], bytes]] = []

    class _Impostor(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            seen.append((dict(self.headers), body))
            reply = json.dumps({"ok": True, "receiver_id": "rx-actual", "proof": "forged"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

    with _serving(ThreadingHTTPServer(("127.0.0.1", 0), _Impostor)) as endpoint:
        code, response, raw = _run_native_host(monkeypatch, endpoint)

    assert code == 1
    assert response["error"] == "receiver_authentication_failed"
    assert "auth_token" not in response
    assert _SECRET.encode() not in raw
    assert len(seen) == 1
    headers, body = seen[0]
    assert "Authorization" not in headers
    assert _SECRET.encode() not in body


def test_native_host_withholds_a_bearer_the_receiver_does_not_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receiver run with a different explicit token cannot be paired with the persisted one.

    This is the state behind the extension's former endless 401 refresh: the
    host now reports it instead of releasing a bearer the receiver rejects.
    """
    with _serving(make_server("127.0.0.1", 0, spool_path=tmp_path, auth_token="another-explicit-token")) as endpoint:
        code, response, raw = _run_native_host(monkeypatch, endpoint)

    assert code == 1
    assert response["error"] == "receiver_authentication_failed"
    assert _SECRET.encode() not in raw


def test_receiver_attestation_proves_possession_without_revealing_the_bearer(tmp_path: Path) -> None:
    """The route answers a fresh challenge with the bearer-keyed MAC and nothing else."""
    challenge = "c" * 43
    with _serving(make_server("127.0.0.1", 0, spool_path=tmp_path, auth_token=_SECRET)) as endpoint:
        parsed = urlparse(endpoint)
        connection = HTTPConnection(parsed.hostname or "", parsed.port or 80, timeout=5)
        connection.request("POST", "/v1/receiver/attest", body=json.dumps({"challenge": challenge}))
        response = connection.getresponse()
        raw = response.read()
        connection.request("POST", "/v1/receiver/attest", body=json.dumps({"challenge": "short"}))
        malformed = connection.getresponse()
        malformed.read()
        connection.close()

    body = json.loads(raw)
    assert response.status == 200
    assert body["proof"] == receiver_module.receiver_attestation_proof(
        _SECRET, body["receiver_id"], challenge, body["endpoint"]
    )
    assert _SECRET.encode() not in raw
    assert malformed.status == 400


def test_receiver_attestation_is_refused_without_a_bearer(tmp_path: Path) -> None:
    """An auth-disabled receiver has nothing to prove and says so."""
    with _serving(make_server("127.0.0.1", 0, spool_path=tmp_path, auth_token=None)) as endpoint:
        parsed = urlparse(endpoint)
        connection = HTTPConnection(parsed.hostname or "", parsed.port or 80, timeout=5)
        connection.request("POST", "/v1/receiver/attest", body=json.dumps({"challenge": "c" * 43}))
        response = connection.getresponse()
        body = json.loads(response.read())
        connection.close()

    assert response.status == 409
    assert body["error"] == "receiver_auth_disabled"


def test_install_resolves_a_bare_command_to_an_absolute_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A native-messaging manifest's `path` must be absolute, not a command name.

    The installer's own `--executable` default is the bare console-script name
    `polylogue-browser-capture-native-host`, and it was written verbatim, so
    `browser-capture native-host install` reported success while installing a
    host neither Chrome nor Firefox can launch on Linux or macOS.

    Anti-vacuity: writing `executable` straight into the record makes the
    manifest `path` the bare name below.
    """
    launcher = tmp_path / "bin" / "polylogue-browser-capture-native-host"
    launcher.parent.mkdir()
    launcher.write_text("#!/bin/sh\n")
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH", str(launcher.parent))

    target = tmp_path / "host.json"
    native_host.install_native_host(
        ("a-extension",), executable="polylogue-browser-capture-native-host", destination=target
    )

    manifest = json.loads(target.read_text())
    assert manifest["path"] == str(launcher)
    assert Path(manifest["path"]).is_absolute()


def test_install_refuses_an_unresolvable_executable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unresolvable launcher is refused, not written as a broken manifest."""
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    target = tmp_path / "host.json"

    with pytest.raises(ValueError, match="not on PATH"):
        native_host.install_native_host(("a-extension",), executable="no-such-launcher", destination=target)

    assert not target.exists()


def test_native_host_scratch_failure_returns_error_envelope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(self: object) -> None:
        raise sqlite3.OperationalError("neutral scratch full")

    monkeypatch.setattr("polylogue.browser_capture.native_host.StreamedJSONDocument.__enter__", fail)
    server = make_server("127.0.0.1", 0, spool_path=tmp_path, auth_token=_SECRET)
    with _serving(server) as endpoint:
        code, reply, raw = _run_native_host(monkeypatch, endpoint)
    assert code == 1
    assert reply == {"type": "error", "error": "receiver_observation_storage_failed"}
    assert _SECRET.encode() not in raw


@pytest.mark.parametrize("phase", ["execute", "step"])
def test_native_host_lazy_spill_read_failure_is_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    from typing import Any, cast

    from polylogue.schemas.observation_spill import StreamedJSONDocument

    actual_enter = StreamedJSONDocument.__enter__

    class FailedRows:
        def __iter__(self) -> FailedRows:
            return self

        def __next__(self) -> object:
            raise sqlite3.OperationalError("neutral lazy read stepping failed")

        def close(self) -> None:
            pass

    class FailedConnection:
        def execute(self, *args: Any, **kwargs: Any) -> Any:
            if phase == "execute":
                raise sqlite3.OperationalError("neutral lazy read execute failed")
            return FailedRows()

    def fail_read(self: StreamedJSONDocument) -> object:
        from polylogue.schemas.observation_spill import SpilledObject

        document = actual_enter(self)
        assert isinstance(document, SpilledObject)
        document._connection = cast(sqlite3.Connection, FailedConnection())
        return document

    monkeypatch.setattr(StreamedJSONDocument, "__enter__", fail_read)
    server = make_server("127.0.0.1", 0, spool_path=tmp_path, auth_token=_SECRET)
    with _serving(server) as endpoint:
        code, reply, raw = _run_native_host(monkeypatch, endpoint)
    assert code == 1
    assert reply == {"type": "error", "error": "receiver_observation_storage_failed"}
    assert _SECRET.encode() not in raw


def test_stale_accepted_signer_cannot_attest_after_listener_tuple_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replacement: list[ThreadingHTTPServer] = []
    server = make_server("127.0.0.1", 0, spool_path=tmp_path, auth_token=_SECRET)

    class ReboundHandler(BrowserCaptureHandler):
        def _receiver_attest(self) -> None:
            server.socket.close()
            replacement.append(ThreadingHTTPServer(server.server_address, BaseHTTPRequestHandler))
            super()._receiver_attest()

    server.RequestHandlerClass = ReboundHandler
    try:
        with _serving(server) as endpoint:
            code, result, raw = _run_native_host(monkeypatch, endpoint)
        assert replacement
        assert replacement[0].server_address == server.server_address
        assert code == 1
        assert result["error"] == "receiver_authentication_failed"
        assert _SECRET.encode() not in raw
    finally:
        for listener in replacement:
            listener.server_close()


@pytest.mark.parametrize("signal", ["cancel", "eof"])
def test_native_input_interrupts_blocked_socket_and_settles_stages(tmp_path: Path, signal: str) -> None:
    input_read, input_write = os.pipe()
    local, peer = socket.socketpair()
    settled = Event()
    failures: list[BaseException] = []
    with native_transport.NativeInput(input_read, tmp_path) as frames:
        frames.bind_socket(local)

        def blocked_read() -> None:
            try:
                assert local.recv(1) == b""
            except BaseException as exc:
                failures.append(exc)
            finally:
                settled.set()

        worker = Thread(target=blocked_read)
        worker.start()
        if signal == "cancel":
            with os.fdopen(input_write, "wb") as source:
                _send_frame(source, {"type": "cancel"})
                assert settled.wait(5)
        else:
            os.close(input_write)
            assert settled.wait(5)
        worker.join()
    os.close(input_read)
    local.close()
    peer.close()
    assert not failures
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.uses_real_clock("native subprocess and HTTP own their cancellation and byte-stream clocks")
@pytest.mark.parametrize("blocked", [False, True])
def test_actual_native_process_streams_large_bytes_or_settles_input_eof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blocked: bool
) -> None:
    archive = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive))
    receiver_module.persist_receiver_token(_SECRET)
    identity = receiver_module.load_or_mint_receiver_identity()
    started, release = Event(), Event()
    acquired: list[bytes] = []
    data = bytes((index * 31 + 7) % 256 for index in range(1024 * 1024 + 17))
    server = make_server("127.0.0.1", 0, spool_path=tmp_path / "spool", auth_token=_SECRET)

    class OwnedHandler(BrowserCaptureHandler):
        def _do_put(self) -> None:
            if self._reject_origin() or self._reject_token():
                return
            acquired.append(self.rfile.read(int(self.headers["Content-Length"])))
            started.set()
            if blocked:
                release.wait(10)
                return
            self._send_attachment(
                io.BytesIO(acquired[0]),
                len(acquired[0]),
                content_type="application/octet-stream",
                filename="neutral.bin",
            )

    server.RequestHandlerClass = OwnedHandler
    environment = isolated_home_environment(os.environ, home=tmp_path / "home")
    environment["POLYLOGUE_ARCHIVE_ROOT"] = str(archive)
    environment["TMPDIR"] = str(tmp_path)
    command = [
        sys.executable,
        "-c",
        "from polylogue.browser_capture.native_host import main; raise SystemExit(main())",
        "chrome-extension://native-neutral/",
    ]
    with _serving(server) as endpoint:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment
        )
        assert process.stdin is not None and process.stdout is not None and process.stderr is not None
        source_pipe = cast(BinaryIO, process.stdin)
        response_pipe = cast(BinaryIO, process.stdout)
        raw_frames = bytearray()
        try:
            _send_frame(
                source_pipe,
                {
                    "type": "request",
                    "version": 1,
                    "endpoint": endpoint,
                    "receiver_id": identity,
                    "method": "PUT",
                    "path": "/v1/neutral-bytes",
                    "headers": {},
                },
            )
            for sequence, offset in enumerate(range(0, len(data), 64 * 1024)):
                _send_frame(
                    source_pipe,
                    {
                        "type": "body",
                        "sequence": sequence,
                        "data": base64.b64encode(data[offset : offset + 64 * 1024]).decode("ascii"),
                    },
                )
                frame, raw = _read_frame(response_pipe)
                raw_frames.extend(raw)
                assert frame == {"type": "body_ack", "sequence": sequence}
            _send_frame(source_pipe, {"type": "body_end"})
            if blocked:
                assert started.wait(5)
                process.stdin.close()
                frame, raw = _read_frame(response_pipe)
                raw_frames.extend(raw)
                assert frame == {"type": "error", "error": "native_input_incomplete"}
                assert process.wait(timeout=5) == 1
            else:
                frame, raw = _read_frame(response_pipe)
                raw_frames.extend(raw)
                assert frame["type"] == "response" and frame["status"] == 200
                received = bytearray()
                chunks = 0
                while True:
                    _send_frame(source_pipe, {"type": "response_next"})
                    frame, raw = _read_frame(response_pipe)
                    raw_frames.extend(raw)
                    if frame["type"] == "response_end":
                        break
                    assert frame["type"] == "response_body" and frame["sequence"] == chunks
                    received.extend(base64.b64decode(str(frame["data"])))
                    chunks += 1
                assert chunks > 16
                assert hashlib.sha256(received).digest() == hashlib.sha256(data).digest()
                assert received == data
                assert process.wait(timeout=5) == 0
            assert acquired == [data]
            assert _SECRET.encode() not in raw_frames
            assert not list(tmp_path.glob("polylogue-native-operation-*"))
        finally:
            release.set()
            if process.poll() is None:
                process.kill()
                process.wait()
            if not process.stdin.closed:
                process.stdin.close()
            process.stdout.close()
            process.stderr.close()


@pytest.mark.parametrize("field,value", [("method", []), ("method", {}), ("version", True)])
def test_malformed_native_request_fields_have_a_named_refusal(field: str, value: object) -> None:
    from polylogue.core.json import JSONValue

    request: dict[str, JSONValue] = {
        "type": "request",
        "version": 1,
        "endpoint": "http://127.0.0.1:8765",
        "method": "GET",
        "path": "/v1/status",
        "receiver_id": None,
        "headers": {},
    }
    request[field] = cast(JSONValue, value)
    with pytest.raises(native_transport.NativeTransportError, match="^native_request_invalid$"):
        native_transport._request(request)
