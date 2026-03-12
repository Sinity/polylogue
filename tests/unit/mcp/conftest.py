from __future__ import annotations

import pytest


@pytest.fixture
def mcp_server():
    """Build and return an MCP server instance for testing."""
    from polylogue.mcp.server import build_server

    return build_server()
