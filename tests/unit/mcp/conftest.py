from __future__ import annotations

import asyncio

import pytest

from tests.infra.mcp import MCPServerUnderTest


@pytest.fixture
def mcp_server() -> MCPServerUnderTest:
    """Build and return an MCP server instance for testing.

    Ensures a fresh event loop policy so that stale/closed loops left behind
    by ``asyncio.run()`` calls in earlier tests (common in the xdist worker
    process) do not cause ``RuntimeError: Event loop is closed`` in
    pytest-asyncio managed async tests that depend on this fixture.
    """
    # Reset the event loop policy so pytest-asyncio creates a fresh loop
    # rather than inheriting a closed one from a prior asyncio.run() call.
    asyncio.set_event_loop_policy(None)

    from polylogue.mcp.server import build_server

    server = build_server()
    assert isinstance(server, MCPServerUnderTest)
    return server
