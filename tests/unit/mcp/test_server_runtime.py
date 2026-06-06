# mypy: disable-error-code="comparison-overlap,arg-type"

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from polylogue.mcp import server as server_module


def test_build_server_registers_tools_resources_and_prompts() -> None:
    fake_mcp = MagicMock()
    fake_fast_mcp = MagicMock(return_value=fake_mcp)

    with (
        patch("mcp.server.fastmcp.FastMCP", fake_fast_mcp),
        patch("polylogue.mcp.server.register_tools") as mock_tools,
        patch("polylogue.mcp.server.register_resources") as mock_resources,
        patch("polylogue.mcp.server.register_prompts") as mock_prompts,
    ):
        built = server_module.build_server()

    assert built is fake_mcp
    fake_fast_mcp.assert_called_once()
    hooks = mock_tools.call_args.args[1]
    assert callable(hooks.json_payload)
    assert callable(hooks.clamp_limit)
    mock_resources.assert_called_once_with(fake_mcp, hooks)
    mock_prompts.assert_called_once_with(fake_mcp, hooks)


def test_get_server_caches_instance_and_updates_runtime_services() -> None:
    original = server_module._server_instance
    server_module._server_instance = None
    services = SimpleNamespace()

    try:
        with (
            patch("polylogue.mcp.server._set_runtime_services") as mock_set_services,
            patch("polylogue.mcp.server.build_server", return_value="server") as mock_build,
        ):
            assert server_module._get_server(services) == "server"
            assert server_module._get_server() == "server"

        mock_set_services.assert_called_once_with(services)
        mock_build.assert_called_once_with(role="read")
    finally:
        server_module._server_instance = original


def test_serve_stdio_runs_cached_server() -> None:
    server = MagicMock()

    with patch("polylogue.mcp.server._get_server", return_value=server) as mock_get_server:
        server_module.serve_stdio(services="services")

    mock_get_server.assert_called_once_with("services", role="read")
    server.run.assert_called_once_with(transport="stdio")
