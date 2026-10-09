"""Native bootstrap proof: exact browser allowlist, identity-bound replies, receiver authentication."""

from __future__ import annotations

import io
import json
import socket
import sqlite3
import struct
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import cast
from urllib.parse import urlparse

import pytest

import polylogue.browser_capture.receiver as receiver_module
from polylogue.browser_capture import native_host
from polylogue.browser_capture.server import make_server


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
    payload = json.dumps({"endpoint": "http://127.0.0.1:8765", "receiver_id": "rx-other"}).encode()
    output = io.BytesIO()
    monkeypatch.setattr(sys, "argv", ["host", "chrome-extension://good-id/"])
    monkeypatch.setattr(
        sys,
        "stdin",
        type("S", (), {"buffer": io.BytesIO(struct.pack("<I", len(payload)) + payload)})(),
    )
    monkeypatch.setattr(sys, "stdout", type("S", (), {"buffer": output})())
    monkeypatch.setattr(native_host, "load_or_mint_receiver_identity", lambda: "rx-actual")
    assert native_host.main() == 1
    size = struct.unpack("<I", output.getvalue()[:4])[0]
    assert json.loads(output.getvalue()[4 : 4 + size])["error"] == "receiver_identity_mismatch"


_SECRET = "receiver-secret-bearer"


def _run_native_host(monkeypatch: pytest.MonkeyPatch, endpoint: str) -> tuple[int, dict[str, object], bytes]:
    """Run the host for one bootstrap request; return exit, reply, and raw stdout."""
    import polylogue.runtime

    payload = json.dumps({"endpoint": endpoint}).encode()
    output = io.BytesIO()
    monkeypatch.setattr(polylogue.runtime, "require_free_threaded_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(sys, "argv", ["host", "chrome-extension://good-id/"])
    monkeypatch.setattr(
        sys, "stdin", type("S", (), {"buffer": io.BytesIO(struct.pack("<I", len(payload)) + payload)})()
    )
    monkeypatch.setattr(sys, "stdout", type("S", (), {"buffer": output})())
    monkeypatch.setattr(native_host, "load_or_mint_receiver_identity", lambda: "rx-actual")
    monkeypatch.setattr(receiver_module, "load_or_mint_receiver_identity", lambda path=None: "rx-actual")
    monkeypatch.setattr(native_host, "load_or_mint_receiver_token", lambda: _SECRET)
    code = native_host.main()
    raw = output.getvalue()
    size = struct.unpack("<I", raw[:4])[0]
    return code, json.loads(raw[4 : 4 + size]), raw


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


def test_native_host_releases_the_bearer_to_the_receiver_that_proves_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legitimate receiver answers the challenge, so a fresh profile is paired."""
    with _serving(make_server("127.0.0.1", 0, spool_path=tmp_path, auth_token=_SECRET)) as endpoint:
        code, response, _raw = _run_native_host(monkeypatch, endpoint)

    assert code == 0
    assert response["auth_token"] == _SECRET
    assert response["receiver_id"] == "rx-actual"


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
    assert body["proof"] == receiver_module.receiver_attestation_proof(_SECRET, body["receiver_id"], challenge)
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
    assert reply == {"ok": False, "error": "receiver_observation_storage_failed", "receiver_id": "rx-actual"}
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
    assert reply == {"ok": False, "error": "receiver_observation_storage_failed", "receiver_id": "rx-actual"}
    assert _SECRET.encode() not in raw
