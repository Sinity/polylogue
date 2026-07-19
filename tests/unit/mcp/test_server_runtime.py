# mypy: disable-error-code="comparison-overlap,arg-type"

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from polylogue.mcp import server as server_module


def test_build_server_registers_tools_resources_and_prompts() -> None:
    fake_mcp = MagicMock()
    fake_fast_mcp = MagicMock(return_value=fake_mcp)
    declared_mcp = MagicMock()

    with (
        patch("mcp.server.fastmcp.FastMCP", fake_fast_mcp),
        patch("polylogue.mcp.server.DeclaredToolRegistrar", return_value=declared_mcp) as registrar_type,
        patch("polylogue.mcp.server.register_tools") as mock_tools,
        patch("polylogue.mcp.server.register_resources") as mock_resources,
        patch("polylogue.mcp.server.register_prompts") as mock_prompts,
    ):
        built = server_module.build_server()

    assert built is fake_mcp
    fake_fast_mcp.assert_called_once()
    registrar_type.assert_called_once_with(fake_mcp, role="read")
    hooks = mock_tools.call_args.args[1]
    assert mock_tools.call_args.args[0] is declared_mcp
    assert callable(hooks.json_payload)
    assert callable(hooks.clamp_limit)
    declared_mcp.finalize.assert_called_once_with()
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
        mock_build.assert_called_once_with(role="read", services=services)
    finally:
        server_module._server_instance = original


def test_get_server_singleton_is_race_safe_under_concurrent_first_access() -> None:
    """Concurrent first access must build exactly one FastMCP server (polylogue-xikl.2).

    ``_get_server()``'s check-then-set on ``_server_instance``/
    ``_server_instance_role`` used to be unguarded. The daemon's real
    ``archive_query_executor`` already dispatches concurrent request
    handlers onto real OS threads today, so this is a live hazard, not a
    hypothetical one. A short delay in ``build_server`` forces the
    interleaving window open: before the fix this reliably produced
    multiple distinct server builds; with the lock guarding the whole
    check-then-build section, exactly one thread ever builds a server.
    """
    original_instance = server_module._server_instance
    original_role = server_module._server_instance_role
    server_module._server_instance = None
    server_module._server_instance_role = None

    build_calls: list[object] = []
    build_calls_lock = threading.Lock()

    def delayed_build(*, role: str, services: object | None = None) -> object:
        del services
        time.sleep(0.02)
        built = SimpleNamespace(role=role)
        with build_calls_lock:
            build_calls.append(built)
        return built

    servers: list[object] = []
    servers_lock = threading.Lock()

    def worker() -> None:
        server = server_module._get_server()
        with servers_lock:
            servers.append(server)

    try:
        with patch("polylogue.mcp.server.build_server", side_effect=delayed_build):
            threads = [threading.Thread(target=worker) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

        assert len(servers) == 8
        assert len(build_calls) == 1, f"concurrent first access built {len(build_calls)} servers, expected exactly 1"
        assert len({id(server) for server in servers}) == 1
    finally:
        server_module._server_instance = original_instance
        server_module._server_instance_role = original_role


def test_serve_stdio_runs_cached_server() -> None:
    server = MagicMock()

    with patch("polylogue.mcp.server._get_server", return_value=server) as mock_get_server:
        server_module.serve_stdio(services="services")

    mock_get_server.assert_called_once_with("services", role="read")
    server.run.assert_called_once_with(transport="stdio")
