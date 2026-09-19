"""CLI and MCP share one session-projection vocabulary (polylogue-mjupn).

MCP's ``get(ref, projection=X)`` carried four hand-written branches, each
re-spelling the projection name, the archive method and the payload key, with
nothing tying them to the registry-checked CLI ``read --view`` vocabulary.

Anti-vacuity: adding a projection name MCP serves that the shared read-view
vocabulary does not declare raises at import; renaming an archive method
without updating the table makes the method-existence assertion red.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from polylogue.archive.viewport import READ_VIEW_PROFILE_BY_ID
from polylogue.cli.read_view_handlers import READ_VIEW_HANDLERS
from polylogue.cli.read_view_registry import READ_VIEW_HANDLER_METADATA
from polylogue.mcp.session_projections import SESSION_LIST_PROJECTION_NAMES, SESSION_LIST_PROJECTIONS
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async, make_polylogue_mock


def test_every_mcp_projection_is_a_declared_read_view() -> None:
    """The shared table must point at the real CLI dispatch table.

    Anti-vacuity: add a table entry without an executable CLI handler and this
    fails before an MCP route can advertise a name the CLI cannot dispatch.
    """
    assert set(SESSION_LIST_PROJECTIONS) <= set(READ_VIEW_PROFILE_BY_ID)
    assert set(SESSION_LIST_PROJECTIONS) <= set(READ_VIEW_HANDLER_METADATA)
    assert set(SESSION_LIST_PROJECTIONS) <= set(READ_VIEW_HANDLERS)


def test_every_declared_method_exists_on_the_archive_facade() -> None:
    from polylogue import Polylogue

    for projection in SESSION_LIST_PROJECTIONS.values():
        assert hasattr(Polylogue, projection.method), projection


def test_payload_keys_are_distinct_and_named_after_their_rows() -> None:
    keys = [projection.payload_key for projection in SESSION_LIST_PROJECTIONS.values()]
    assert len(set(keys)) == len(keys)
    assert SESSION_LIST_PROJECTIONS["file-edits"].payload_key == "file_edits"


@pytest.mark.asyncio
async def test_session_list_projection_routes_through_read_and_get(mcp_server: MCPServerUnderTest) -> None:
    """Both MCP routes must look up an entry in the table.

    Anti-vacuity: restore a hand-written ``if projection == \"events\"``
    branch, or remove either table lookup, and this does not observe the same
    facade method and response key from both production tool handlers.
    """
    projection = SESSION_LIST_PROJECTIONS["events"]
    poly = make_polylogue_mock()
    method = AsyncMock(return_value=[{"kind": "event"}])
    setattr(poly, projection.method, method)

    with patch("polylogue.mcp.server._get_polylogue", return_value=poly):
        read = json.loads(
            await invoke_surface_async(
                mcp_server._tool_manager._tools["read"].fn,
                ref="session:codex:projection-registry",
                view=projection.name,
            )
        )
        get = json.loads(
            await invoke_surface_async(
                mcp_server._tool_manager._tools["get"].fn,
                ref="session:codex:projection-registry",
                projection=projection.name,
            )
        )

    assert read[projection.payload_key] == get[projection.payload_key] == [{"kind": "event"}]
    assert method.await_count == 2


@pytest.mark.asyncio
async def test_unknown_session_projection_is_rejected(mcp_server: MCPServerUnderTest) -> None:
    """An unknown projection must not silently resolve the default object."""
    with patch("polylogue.mcp.server._get_polylogue", return_value=make_polylogue_mock()):
        payload = json.loads(
            await invoke_surface_async(
                mcp_server._tool_manager._tools["get"].fn,
                ref="session:codex:projection-registry",
                projection="not-a-projection",
            )
        )

    assert payload["code"] == "invalid_argument"


@pytest.mark.asyncio
async def test_capability_explanation_is_derived_from_session_projection_table(
    mcp_server: MCPServerUnderTest,
) -> None:
    """Adding or removing a table entry changes capability discovery too."""
    poly = make_polylogue_mock()
    poly.stats = AsyncMock(return_value=SimpleNamespace(session_count=0, message_count=0))

    with patch("polylogue.mcp.server._get_polylogue", return_value=poly):
        payload = json.loads(
            await invoke_surface_async(
                mcp_server._tool_manager._tools["explain"].fn,
                subject="capability",
                limit=1,
            )
        )

    assert payload["read_views"] == list(SESSION_LIST_PROJECTION_NAMES)
