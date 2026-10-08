"""Resolved credentials reach the ordinary machine CLI transport unchanged."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from polylogue.cli.click_app import cli
from polylogue.cli.machine_main import run_machine_entry
from polylogue.config import resolve_runtime_config
from polylogue.daemon.uds import MachineOperationHandler
from tests.infra.daemon_operations import cli_daemon_archive
from tests.infra.storage_records import SessionBuilder

_TOKEN = "neutral-machine-token"
_SESSION_ID = "claude-code-session:ext-auth-neutral"


@pytest.mark.parametrize("token_source", ["environment", "toml"])
@pytest.mark.parametrize("valid_token", [True, False])
@pytest.mark.uses_real_clock
def test_real_cli_selection_preserves_configured_machine_token(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    token_source: str,
    valid_token: bool,
) -> None:
    """Removing either auth projection field breaks valid authenticated selection."""
    root = workspace_env["archive_root"]
    token = _TOKEN if valid_token else "neutral-wrong-token"
    config_file = workspace_env["data_root"] / "machine.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(f'[daemon.api]\nauth_token = "{token}"\n' if token_source == "toml" else "")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_file))
    monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", "")
    monkeypatch.delenv("POLYLOGUE_API_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("POLYLOGUE_API_ALLOW_NO_AUTH", raising=False)
    monkeypatch.delenv("POLYLOGUE_NO_DAEMON", raising=False)
    if token_source == "environment":
        monkeypatch.setenv("POLYLOGUE_API_AUTH_TOKEN", token)

    def seed(archive_root: Path) -> None:
        SessionBuilder(archive_root / "index.db", "auth-neutral").provider("claude-code").title(
            "Neutral authenticated selection"
        ).add_message("m0", role="user", text="neutral message").save()

    refusals: list[tuple[int, dict[str, Any]]] = []
    original_send = MachineOperationHandler._send

    def observe_send(handler: MachineOperationHandler, status: int, payload: dict[str, Any]) -> None:
        if status == 401:
            refusals.append((status, payload))
        original_send(handler, status, payload)

    monkeypatch.setattr(MachineOperationHandler, "_send", observe_send)
    with cli_daemon_archive(root, monkeypatch, seed_archive=seed) as stack:
        stack.server.auth_token = _TOKEN
        argv = ["--id", _SESSION_ID, "select", "--format", "json"]
        monkeypatch.setattr(sys, "argv", ["polylogue", *argv])
        capsys.readouterr()
        exit_code = 0
        try:
            run_machine_entry(cli, argv)
        except SystemExit as exc:
            assert isinstance(exc.code, int)
            exit_code = exc.code
        payload = json.loads(capsys.readouterr().out)
        if valid_token:
            assert exit_code == 0, payload
            assert payload["id"] == _SESSION_ID
        else:
            assert exit_code != 0
            assert payload["code"] == "runtime_error"
            assert refusals
            assert all(status == 401 and response["error"]["code"] == "unauthorized" for status, response in refusals)


@pytest.mark.parametrize("allow_no_auth", [False, True])
def test_pure_runtime_projection_retains_explicit_auth_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, allow_no_auth: bool
) -> None:
    runtime = resolve_runtime_config(
        environment={"HOME": str(tmp_path), "POLYLOGUE_SITE_CONFIG": ""},
        cli_overrides={"api_auth_token": _TOKEN, "api_allow_no_auth": allow_no_auth},
    )
    monkeypatch.setenv("POLYLOGUE_API_AUTH_TOKEN", "different-ambient-token")
    config = runtime.as_config()
    assert config.api_auth_token == _TOKEN
    assert config.api_allow_no_auth is allow_no_auth
    assert config.with_sources([]).api_auth_token == _TOKEN
    assert config.with_sources([]).api_allow_no_auth is allow_no_auth
    assert config.with_sources(config.sources) == config
    assert _TOKEN not in repr(config)


def test_receiver_factory_preserves_distinct_machine_and_pairing_credentials(tmp_path: Path) -> None:
    from polylogue.browser_capture.server import make_server

    pairing_token = "neutral-receiver-pairing-token"
    server = make_server(
        "127.0.0.1",
        0,
        spool_path=tmp_path / "spool",
        archive_root=tmp_path,
        auth_token=pairing_token,
        api_auth_token=_TOKEN,
        api_allow_no_auth=True,
    )
    try:
        assert server.config.auth_token == pairing_token
        assert server.config.api_auth_token == _TOKEN
        assert server.config.api_allow_no_auth is True
        assert _TOKEN not in repr(server.config)
        assert pairing_token not in repr(server.config)
    finally:
        server.server_close()
