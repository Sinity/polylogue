# mypy: disable-error-code="assignment,arg-type,comparison-overlap"

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from polylogue.cli.commands.mcp import mcp_command
from polylogue.sources.drive.source_factory import build_drive_source_client


def test_mcp_command_runs_stdio_server_and_handles_missing_dependency() -> None:
    runner = CliRunner()
    env = SimpleNamespace(ui=SimpleNamespace(console=SimpleNamespace(print=lambda *_args: None)), services="services")

    with patch("polylogue.mcp.server.serve_stdio") as mock_serve:
        result = runner.invoke(mcp_command, ["--transport", "stdio"], obj=env)

    assert result.exit_code == 0
    mock_serve.assert_called_once_with("services", role="read")

    console = SimpleNamespace(print=MagicMock())
    env = SimpleNamespace(ui=SimpleNamespace(console=console), services="services")
    original_module = sys.modules.get("polylogue.mcp.server")

    try:
        sys.modules["polylogue.mcp.server"] = None
        result = runner.invoke(mcp_command, [], obj=env)
    finally:
        if original_module is None:
            sys.modules.pop("polylogue.mcp.server", None)
        else:
            sys.modules["polylogue.mcp.server"] = original_module

    assert result.exit_code == 1
    assert "MCP dependencies not installed" in console.print.call_args_list[0].args[0]
    assert "Install the base polylogue package" in console.print.call_args_list[1].args[0]


def test_mcp_command_rejects_unsupported_transport_via_callback() -> None:
    console = SimpleNamespace(print=MagicMock())
    env = SimpleNamespace(ui=SimpleNamespace(console=console), services="services")

    wrapped = getattr(mcp_command.callback, "__wrapped__", None)
    assert callable(wrapped)

    try:
        wrapped(env, "http", "read")
    except SystemExit as exc:
        assert exc.code == 1
    else:  # pragma: no cover - defensive
        raise AssertionError("expected SystemExit")

    console.print.assert_called_once_with("Unsupported transport: http")


def test_build_drive_source_client_wires_auth_gateway_and_client(tmp_path: Path) -> None:
    ui = object()
    config = object()

    with (
        patch(
            "polylogue.sources.drive.source_factory.resolve_drive_retry_policy", return_value="retry-policy"
        ) as mock_policy,
        patch("polylogue.sources.drive.source_factory.DriveAuthManager", return_value="auth-manager") as mock_auth,
        patch("polylogue.sources.drive.source_factory.DriveServiceGateway", return_value="gateway") as mock_gateway,
        patch("polylogue.sources.drive.source_factory.DriveSourceClient", return_value="client") as mock_client,
    ):
        client = build_drive_source_client(
            ui=ui,
            credentials_path=tmp_path / "credentials.json",
            token_path=tmp_path / "token.json",
            retries=5,
            retry_base=2.5,
            config=config,
        )

    assert client == "client"
    mock_policy.assert_called_once_with(retries=5, retry_base=2.5, config=config)
    mock_auth.assert_called_once_with(
        ui=ui,
        credentials_path=tmp_path / "credentials.json",
        token_path=tmp_path / "token.json",
        config=config,
    )
    mock_gateway.assert_called_once_with(auth_manager="auth-manager", retry_policy="retry-policy")
    mock_client.assert_called_once_with(gateway="gateway")
