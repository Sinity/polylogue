from __future__ import annotations

import json
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread
from typing import cast
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from polylogue.browser_capture.receiver import BrowserCaptureReceiverConfig, resolve_receiver_auth_token
from polylogue.browser_capture.server import make_server
from polylogue.config import resolve_runtime_config
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
    server = make_server("127.0.0.1", 0, spool_path=tmp_path / "capture", auth_token=token)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.server_address[:2])
    runtime = resolve_runtime_config(cli_overrides={"browser_capture_host": host, "browser_capture_port": port})
    monkeypatch.setattr("polylogue.config.resolve_runtime_config", lambda: runtime)
    # Standalone serving has no resident operation socket and no caller-local observation.
    configure_browser_capture_status(None)
    try:
        with patch(
            "polylogue.daemon.commands._live_daemon_status_payload",
            side_effect=AssertionError("standalone status must use the receiver"),
        ):
            result = CliRunner().invoke(status_command, ["--format", "json"])
            assert result.exit_code == 0, result.output
            observed = json.loads(result.stdout)
            assert observed["active"] is True
            assert observed["auth_required"] is (not allow_no_auth)
            text = CliRunner().invoke(status_command, [])
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
