from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread
from typing import cast
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from polylogue.browser_capture.receiver import (
    BrowserCaptureReceiverConfig,
    receiver_identity,
    receiver_status_payload,
    receiver_status_proof,
    resolve_receiver_auth_token,
)
from polylogue.browser_capture.server import make_server
from polylogue.core.json import JSONValue
from polylogue.daemon.browser_capture import status_command
from polylogue.daemon.status import browser_capture_status_payload, format_daemon_status_lines
from polylogue.daemon.status_snapshot import (
    configure_browser_capture_status,
    configure_runtime_components,
    refresh_status_snapshot,
)
from polylogue.paths import browser_capture_receiver_identity_path, browser_capture_receiver_token_path
from tests.infra.frozen_clock import FrozenClock


@pytest.mark.parametrize("allow_no_auth", [False, True])
def test_summary_matches_bound_receiver_policy_and_auth_gate(tmp_path: Path, allow_no_auth: bool) -> None:
    token = resolve_receiver_auth_token(None, allow_no_auth=allow_no_auth, token_path=tmp_path / "token")
    server = make_server("127.0.0.1", 0, spool_path=tmp_path / "capture", auth_token=token)
    configure_runtime_components(browser_capture_enabled=True)
    configure_browser_capture_status(server.config)
    receiver_identity(server.config)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.server_address[:2])
    try:
        connection = HTTPConnection(host, port)
        connection.request("GET", "/v1/status")
        response = connection.getresponse()
        unauthenticated_status = response.status
        response.read()
        connection.close()
        assert unauthenticated_status == (200 if allow_no_auth else 401)
        headers = {"Authorization": "Bearer " + token} if token is not None else {}
        connection = HTTPConnection(host, port)
        connection.request("GET", "/v1/status", headers=headers)
        response = connection.getresponse()
        assert response.status == 200
        direct = json.loads(response.read())
        connection.close()
        summary = browser_capture_status_payload()
        snapshot = refresh_status_snapshot(rich=False).payload
        states = snapshot["component_state"]
        readiness = snapshot["component_readiness"]
        assert isinstance(states, dict) and isinstance(readiness, dict)
        assert states["browser_capture"] == "running"
        capture_readiness = readiness["browser_capture"]
        assert isinstance(capture_readiness, dict)
        assert capture_readiness["state"] == "ready"
        minimal = snapshot["browser_capture"]
        assert isinstance(minimal, dict)
        for key in ("receiver_id", "auth_required", "allow_remote", "allowed_origins", "spool_ready", "active"):
            assert summary[key] == direct[key]
            assert minimal[key] == direct[key]
        assert summary["auth_required"] is (not allow_no_auth)
        assert "spool_path" not in summary
        assert "auth_token" not in summary
        assert "api_auth_token" not in summary
        if token is not None:
            assert token not in json.dumps(summary)
            assert token not in json.dumps(minimal)
        summary["allowed_origins"] = []
        assert browser_capture_status_payload()["allowed_origins"] == direct["allowed_origins"]
    finally:
        configure_browser_capture_status(None)
        server.shutdown()
        server.server_close()
        thread.join()
    stopped = browser_capture_status_payload()
    assert stopped["active"] is False
    assert stopped["auth_required"] is None
    assert stopped["allow_remote"] is None
    assert stopped["spool_ready"] is None
    assert stopped["reason"] == "receiver_not_observed"


@pytest.mark.parametrize("allow_no_auth", [False, True])
def test_browser_capture_status_cli_observes_standalone_receiver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, allow_no_auth: bool
) -> None:
    from polylogue.paths import browser_capture_receiver_token_path

    token = resolve_receiver_auth_token(None, allow_no_auth=allow_no_auth)
    before = browser_capture_receiver_token_path().read_bytes() if token is not None else None
    # The bound standalone override, not stale separately configured credentials, owns authentication.
    monkeypatch.setenv("POLYLOGUE_BROWSER_CAPTURE_AUTH_TOKEN", "neutral-stale-config-token")
    server = make_server("127.0.0.1", 0, spool_path=tmp_path / "capture", auth_token=token)
    receiver_identity(server.config)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.server_address[:2])
    endpoint_args = ["--host", host, "--port", str(port)] + (["--allow-no-auth"] if allow_no_auth else [])
    # Standalone serving has no resident operation socket and no caller-local observation.
    configure_browser_capture_status(None)
    try:
        with patch(
            "polylogue.daemon.commands._live_daemon_status_payload",
            side_effect=AssertionError("standalone status must use the receiver"),
        ):
            result = CliRunner().invoke(status_command, ["--format", "json", *endpoint_args])
            assert result.exit_code == 0, result.output
            observed = json.loads(result.stdout)
            assert observed["active"] is True
            assert observed["auth_required"] is (not allow_no_auth)
            text = CliRunner().invoke(status_command, endpoint_args)
            assert text.exit_code == 0, text.output
            assert "Browser capture authentication: " + ("disabled" if allow_no_auth else "required") in text.output
            assert "Browser capture remote access: disabled" in text.output
        if token is not None:
            assert browser_capture_receiver_token_path().read_bytes() == before
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.frozen_clock_modules("polylogue.daemon.status_snapshot")
def test_receiver_observation_timestamp_refreshes_without_policy_change(
    tmp_path: Path, frozen_clock: FrozenClock
) -> None:
    configure_browser_capture_status(BrowserCaptureReceiverConfig(spool_path=tmp_path, auth_token="neutral"))
    first = browser_capture_status_payload()
    frozen_clock.advance(60)
    later = browser_capture_status_payload()
    assert first["checked_at"] != later["checked_at"]
    assert {k: v for k, v in first.items() if k != "checked_at"} == {
        k: v for k, v in later.items() if k != "checked_at"
    }


@pytest.mark.parametrize(
    ("auth", "remote", "auth_text", "remote_text"),
    [(True, True, "required", "allowed"), (False, False, "disabled", "disabled"), (None, None, "unknown", "unknown")],
)
def test_plain_daemon_status_renders_receiver_policy(
    auth: bool | None, remote: bool | None, auth_text: str, remote_text: str
) -> None:
    lines = list(format_daemon_status_lines({"browser_capture": {"auth_required": auth, "allow_remote": remote}}))
    assert "Browser capture authentication: " + auth_text in lines
    assert "Browser capture remote access: " + remote_text in lines


def test_unobserved_receiver_policy_stays_unknown_without_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse_default(*args: object, **kwargs: object) -> None:
        raise AssertionError("status must not reconstruct or mint receiver configuration")

    monkeypatch.setattr("polylogue.browser_capture.receiver.BrowserCaptureReceiverConfig.default", refuse_default)
    configure_browser_capture_status(None)
    configure_runtime_components(browser_capture_enabled=True)
    summary = browser_capture_status_payload()
    assert summary["active"] is False
    assert summary["auth_required"] is None
    assert summary["allow_remote"] is None
    assert summary["spool_ready"] is None


@pytest.mark.parametrize("spoof", ["unauthenticated_success", "wrong_identity", "tampered_response", "malformed_proof"])
def test_status_rejects_schema_valid_impostor_before_trusting_policy(tmp_path: Path, spoof: str) -> None:
    from http.server import BaseHTTPRequestHandler, HTTPServer

    token = resolve_receiver_auth_token("neutral-persisted-token")
    assert token is not None
    secret = token
    config = BrowserCaptureReceiverConfig(spool_path=tmp_path, auth_token=token)
    identity = receiver_identity(config)
    payload = receiver_status_payload(config)
    payload["receiver_id"] = "rx-wrong-identity"
    received_tokens: list[str | None] = []

    class Impostor(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            challenge = request["challenge"]
            received_tokens.append(self.headers.get("Authorization"))
            body = json.dumps(payload).encode()
            proof = receiver_status_proof(secret, identity, challenge, payload_sha256=hashlib.sha256(body).hexdigest())
            if spoof == "unauthenticated_success":
                proof = "invalid"
            if spoof == "malformed_proof":
                proof = "invalid-unicode-\u00e9"
            if spoof == "tampered_response":
                body += b" "
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Polylogue-Status-Proof", proof)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return None

    server = HTTPServer(("127.0.0.1", 0), Impostor)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = CliRunner().invoke(status_command, ["--port", str(server.server_port), "--format", "json"])
        assert result.exit_code == 1
        reason = {
            "wrong_identity": "receiver_identity_mismatch",
            "unauthenticated_success": "receiver_authentication_failed",
            "tampered_response": "receiver_authentication_failed",
            "malformed_proof": "receiver_authentication_failed",
        }[spoof]
        assert reason in result.output
        assert received_tokens == [None]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("exclusion", ["halt", "profile"])
def test_daemon_does_not_publish_unscheduled_receiver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exclusion: str
) -> None:
    import asyncio

    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.service_halt import HaltReason, HaltRegistry, UnitKind, unit_id
    from polylogue.daemon.services import ServiceProfile
    from polylogue.paths import archive_root

    if exclusion == "halt":
        HaltRegistry(archive_root()).halt(
            unit_id(UnitKind.SERVICE, "browser_capture_server"),
            reason=HaltReason.TERMINAL_REFUSAL,
            message="neutral test halt",
            frame="neutral",
        )
    observations: list[dict[str, object]] = []

    def observe(config: BrowserCaptureReceiverConfig | None) -> None:
        configure_browser_capture_status(config)
        snapshot = refresh_status_snapshot(rich=False).payload
        states = snapshot["component_state"]
        readiness = snapshot["component_readiness"]
        assert isinstance(states, dict) and isinstance(readiness, dict)
        assert states["browser_capture"] == "stopped"
        capture_readiness = readiness["browser_capture"]
        assert isinstance(capture_readiness, dict)
        assert capture_readiness["state"] == "missing"
        assert snapshot["browser_capture_active"] is False
        observations.append(dict(browser_capture_status_payload()))
        if len(observations) == 1:
            raise RuntimeError("observation reached")

    monkeypatch.setattr("polylogue.daemon.status_snapshot.configure_browser_capture_status", observe)
    with pytest.raises(RuntimeError, match="observation reached"):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(),
                enable_watch=False,
                enable_browser_capture=True,
                browser_capture_host="127.0.0.1",
                browser_capture_port=0,
                service_profile=ServiceProfile.SURFACES if exclusion == "halt" else ServiceProfile.INTAKE,
            )
        )
    assert observations
    assert all(observation["active"] is False for observation in observations)
    assert all(observation["auth_required"] is None for observation in observations)


@pytest.mark.parametrize("invalid", ["malformed", "schema", "origin"])
def test_status_invalid_payload_is_a_named_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    from http.server import BaseHTTPRequestHandler, HTTPServer

    config = BrowserCaptureReceiverConfig(spool_path=tmp_path)
    receiver_identity(config)
    payload = receiver_status_payload(config)
    if invalid == "schema":
        payload["schema_version"] = 2
    elif invalid == "origin":
        payload["allowed_origins"] = ["https://neutral.example", None]
    body = b"{invalid" if invalid == "malformed" else json.dumps(payload).encode()

    class InvalidReceiver(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return None

    server = HTTPServer(("127.0.0.1", 0), InvalidReceiver)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = CliRunner().invoke(status_command, ["--port", str(server.server_port), "--allow-no-auth"])
        assert result.exit_code == 1
        assert "receiver_status_invalid_payload" in result.output
        assert "Browser capture receiver" not in result.stdout
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_status_requires_auth_override_and_streams_large_roster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from http.client import HTTPResponse

    monkeypatch.setenv("POLYLOGUE_BROWSER_CAPTURE_ALLOW_NO_AUTH", "1")
    token = resolve_receiver_auth_token("neutral-explicit-token", allow_no_auth=True)
    assert token is not None
    origins = tuple(f"https://neutral-{index:06d}.{'x' * 50}.{'y' * 50}.example" for index in range(35000))
    server = make_server("127.0.0.1", 0, spool_path=tmp_path, auth_token=token, extra_origins=origins)
    receiver_identity(server.config)
    original_read = HTTPResponse.read
    status_reads: list[int] = []

    def bounded_read(self: HTTPResponse, amt: int | None = None) -> bytes:
        assert amt is not None and 0 < amt <= 65536
        status_reads.append(amt)
        return original_read(self, amt)

    monkeypatch.setattr(HTTPResponse, "read", bounded_read)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = CliRunner().invoke(
            status_command, ["--port", str(server.server_address[1]), "--require-auth", "--format", "json"]
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert set(origins).issubset(payload["allowed_origins"])
        assert payload["auth_required"] is True
        assert len(status_reads) > 64
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.uses_real_clock("the real loopback attestation responds after the former socket timeout")
def test_status_waits_for_valid_slow_attestation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    from polylogue.browser_capture.server import BrowserCaptureHandler

    original = BrowserCaptureHandler._receiver_status_attest

    def delayed(self: BrowserCaptureHandler) -> None:
        time.sleep(5.1)
        original(self)

    monkeypatch.setattr(BrowserCaptureHandler, "_receiver_status_attest", delayed)
    token = resolve_receiver_auth_token("neutral-slow-token")
    server = make_server("127.0.0.1", 0, spool_path=tmp_path, auth_token=token)
    receiver_identity(server.config)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = CliRunner().invoke(
            status_command, ["--port", str(server.server_address[1]), "--require-auth", "--format", "json"]
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["auth_required"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_signed_status_relay_never_receives_bearer(tmp_path: Path) -> None:
    from http.server import BaseHTTPRequestHandler, HTTPServer

    token = resolve_receiver_auth_token("neutral-relay-secret")
    assert token is not None
    genuine = make_server("127.0.0.1", 0, spool_path=tmp_path, auth_token=token)
    receiver_identity(genuine.config)
    observed: list[tuple[str, str | None, bytes]] = []

    class Relay(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            observed.append((self.path, self.headers.get("Authorization"), body))
            with closing(HTTPConnection("127.0.0.1", genuine.server_port)) as connection:
                connection.request("POST", self.path, body=body, headers={"Content-Type": "application/json"})
                response = connection.getresponse()
                data = response.read()
                self.send_response(response.status)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("X-Polylogue-Status-Proof", response.getheader("X-Polylogue-Status-Proof", ""))
                self.end_headers()
                self.wfile.write(data)

        def log_message(self, format: str, *args: object) -> None:
            return None

    relay = HTTPServer(("127.0.0.1", 0), Relay)
    threads = [Thread(target=server.serve_forever, daemon=True) for server in (genuine, relay)]
    for thread in threads:
        thread.start()
    try:
        result = CliRunner().invoke(status_command, ["--port", str(relay.server_port), "--format", "json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["receiver_id"] == receiver_identity(genuine.config)
        assert len(observed) == 1
        path, authorization, body = observed[0]
        assert path == "/v1/receiver/status-attest"
        assert authorization is None
        assert token.encode() not in body
        # A missing request proof never discloses status, even on the genuine owner.
        with closing(HTTPConnection("127.0.0.1", genuine.server_port)) as connection:
            connection.request(
                "POST", path, body=b'{"challenge":"neutral-challenge"}', headers={"Content-Type": "application/json"}
            )
            response = connection.getresponse()
            assert response.status == 400
            assert b"spool_path" not in response.read()
    finally:
        for server in (relay, genuine):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join()


@pytest.mark.parametrize("failure", ["sqlite", "scratch_directory", "spill_directory", "network"])
def test_status_scratch_sqlite_failure_is_typed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    token = resolve_receiver_auth_token("neutral-scratch-token")
    server = make_server("127.0.0.1", 0, spool_path=tmp_path, auth_token=token)
    receiver_identity(server.config)

    def fail(self: object) -> None:
        raise sqlite3.OperationalError("neutral full scratch volume")

    if failure == "sqlite":
        monkeypatch.setattr("polylogue.browser_capture.native_host.StreamedJSONDocument.__enter__", fail)
    elif failure == "spill_directory":

        def no_spill(self: object) -> None:
            raise OSError("neutral spill temporary filesystem unavailable")

        monkeypatch.setattr("polylogue.browser_capture.native_host.StreamedJSONDocument.__enter__", no_spill)
    elif failure == "scratch_directory":

        def no_scratch(*args: object, **kwargs: object) -> None:
            raise OSError("neutral temporary filesystem unavailable")

        monkeypatch.setattr("polylogue.browser_capture.native_host.tempfile.TemporaryDirectory", no_scratch)
    else:

        def no_network(self: object, amt: int | None = None) -> bytes:
            raise OSError("neutral connection reset")

        monkeypatch.setattr("http.client.HTTPResponse.read", no_network)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = CliRunner().invoke(status_command, ["--port", str(server.server_port)])
        assert result.exit_code == 1
        assert (
            "receiver_unreachable" if failure == "network" else "receiver_observation_storage_failed"
        ) in result.stderr
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_origin_formatter_yields_before_consuming_roster() -> None:
    class Origins(list[str]):
        def __iter__(self) -> Iterator[str]:
            yield "https://first.example"
            raise AssertionError("remaining origins must not be collected before emission")

    iterator = format_daemon_status_lines(
        {"browser_capture": {"allowed_origins": cast(list[JSONValue], Origins()), "spool_ready": True}}
    )
    assert next(iterator) == "Polylogue daemon"
    assert next(iterator) == "Browser capture spool: ready"
    assert next(iterator) == "Browser capture origins:"
    assert next(iterator) == "  https://first.example"


@pytest.mark.parametrize("permissions", [None, 0o644])
def test_status_observes_default_or_public_identity_with_large_roster(tmp_path: Path, permissions: int | None) -> None:
    token = resolve_receiver_auth_token("neutral-default-identity-token")
    origins = tuple(f"https://origin-{number}.example" for number in range(10000))
    server = make_server("127.0.0.1", 0, spool_path=tmp_path, auth_token=token, extra_origins=origins)
    expected = receiver_identity(server.config)
    identity_path = browser_capture_receiver_identity_path()
    if permissions is not None:
        identity_path.chmod(permissions)
    configure_browser_capture_status(server.config)
    first = browser_capture_status_payload()
    second = browser_capture_status_payload()
    assert first["allowed_origins"] is second["allowed_origins"]
    minimal = refresh_status_snapshot(rich=False).payload["browser_capture"]
    assert isinstance(minimal, dict)
    assert minimal["allowed_origins"] is first["allowed_origins"]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = CliRunner().invoke(status_command, ["--port", str(server.server_port), "--format", "json"])
        assert result.exit_code == 0, result.output
        actual = json.loads(result.stdout)
        assert actual["receiver_id"] == expected
        assert set(origins).issubset(actual["allowed_origins"])
        assert actual["auth_required"] is True
        assert json.loads(json.dumps(first))["allowed_origins"] == actual["allowed_origins"]
    finally:
        configure_browser_capture_status(None)
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    "mutation",
    ["setitem", "delitem", "iadd", "imul", "append", "extend", "insert", "remove", "pop", "clear", "reverse", "sort"],
)
def test_published_origin_roster_is_immutable(tmp_path: Path, mutation: str) -> None:
    config = BrowserCaptureReceiverConfig(spool_path=tmp_path, allowed_origins=frozenset({"https://neutral.example"}))
    configure_browser_capture_status(config)
    roster = browser_capture_status_payload()["allowed_origins"]
    assert isinstance(roster, list)
    with pytest.raises(TypeError, match="immutable"):
        if mutation == "setitem":
            roster[0] = "https://wrong.example"
        elif mutation == "delitem":
            del roster[0]
        elif mutation == "iadd":
            roster += ["https://wrong.example"]
        elif mutation == "imul":
            roster *= 2
        elif mutation == "append":
            roster.append("https://wrong.example")
        elif mutation == "extend":
            roster.extend(["https://wrong.example"])
        elif mutation == "insert":
            roster.insert(0, "https://wrong.example")
        elif mutation == "remove":
            roster.remove("https://neutral.example")
        elif mutation == "pop":
            roster.pop()
        elif mutation == "clear":
            roster.clear()
        elif mutation == "reverse":
            roster.reverse()
        else:
            roster.sort(key=str)
    assert browser_capture_status_payload()["allowed_origins"] == ["https://neutral.example"]
    configure_browser_capture_status(None)


@pytest.mark.parametrize("credential", ["identity", "token"])
def test_local_credential_read_failure_precedes_any_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, credential: str
) -> None:
    resolve_receiver_auth_token("neutral-local-token")
    receiver_identity(BrowserCaptureReceiverConfig(spool_path=tmp_path))
    target = (
        browser_capture_receiver_identity_path() if credential == "identity" else browser_capture_receiver_token_path()
    )
    original_open = os.open

    def unavailable(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == target:
            raise FileNotFoundError("neutral local read race")
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def no_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("local credential refusal must happen before network setup")

    monkeypatch.setattr("polylogue.daemon.browser_capture.os.open", unavailable)
    monkeypatch.setattr("polylogue.daemon.browser_capture.http.client.HTTPConnection", no_connection)
    result = CliRunner().invoke(status_command, [])
    assert result.exit_code == 1
    reason = "receiver_identity_unavailable" if credential == "identity" else "receiver_credential_unavailable"
    assert reason in result.stderr
    assert "receiver_unreachable" not in result.stderr


def test_published_origin_roster_cannot_be_reinitialized(tmp_path: Path) -> None:
    configure_browser_capture_status(BrowserCaptureReceiverConfig(spool_path=tmp_path))
    snapshot = browser_capture_status_payload()
    roster = snapshot["allowed_origins"]
    assert isinstance(roster, list)
    original = list(roster)
    with pytest.raises(TypeError, match="immutable"):
        type(roster).__init__(roster, ["https://wrong.example"])
    assert list(roster) == original
    assert browser_capture_status_payload()["allowed_origins"] is roster
    configure_browser_capture_status(None)
