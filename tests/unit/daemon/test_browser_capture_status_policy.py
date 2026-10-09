from __future__ import annotations

import json
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread
from typing import cast
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from polylogue.browser_capture.receiver import (
    BrowserCaptureReceiverConfig,
    receiver_attestation_proof,
    receiver_identity,
    receiver_status_payload,
    resolve_receiver_auth_token,
)
from polylogue.browser_capture.server import make_server
from polylogue.daemon.browser_capture import status_command
from polylogue.daemon.status import browser_capture_status_payload, format_daemon_status_lines
from polylogue.daemon.status_snapshot import (
    configure_browser_capture_status,
    configure_runtime_components,
    refresh_status_snapshot,
)
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
        minimal = refresh_status_snapshot(rich=False).payload["browser_capture"]
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
    lines = format_daemon_status_lines({"browser_capture": {"auth_required": auth, "allow_remote": remote}})
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


@pytest.mark.parametrize("spoof", ["unauthenticated_success", "wrong_identity"])
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
            challenge = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["challenge"]
            proof = receiver_attestation_proof(secret, identity, challenge) if spoof == "wrong_identity" else "invalid"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"proof": proof}).encode())

        def do_GET(self) -> None:
            received_tokens.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, format: str, *args: object) -> None:
            return None

    server = HTTPServer(("127.0.0.1", 0), Impostor)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = CliRunner().invoke(status_command, ["--port", str(server.server_port), "--format", "json"])
        assert result.exit_code == 1
        assert (
            "receiver_identity_mismatch" if spoof == "wrong_identity" else "receiver_authentication_failed"
        ) in result.output
        assert received_tokens == (["Bearer " + token] if spoof == "wrong_identity" else [])
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
