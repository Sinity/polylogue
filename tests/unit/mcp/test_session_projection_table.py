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
from polylogue.operations.session_projections import (
    SESSION_LIST_PROJECTION_NAMES,
    SESSION_LIST_PROJECTIONS,
    SessionListProjection,
    mcp_get_session_projection_names,
    mcp_read_view_names,
)
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async, make_polylogue_mock


def test_every_mcp_projection_is_a_declared_read_view() -> None:
    """The shared table must point at the real CLI dispatch table.

    Anti-vacuity: add a table entry without an executable CLI handler and this
    fails before an MCP route can advertise a name the CLI cannot dispatch.
    """
    assert set(SESSION_LIST_PROJECTIONS) <= set(READ_VIEW_PROFILE_BY_ID)
    assert set(SESSION_LIST_PROJECTIONS) <= set(READ_VIEW_HANDLER_METADATA)
    assert set(SESSION_LIST_PROJECTIONS) <= set(READ_VIEW_HANDLERS)


def test_cli_and_mcp_projection_vocabulary_has_one_source() -> None:
    """CLI and MCP cannot advertise divergent table-backed projection names.

    Anti-vacuity: omitting the CLI table injection or hand-maintaining either
    MCP vocabulary makes one equality fail when the projection table changes.
    """
    from polylogue.cli.read_view_handlers import session_list_read_view_handlers

    assert tuple(session_list_read_view_handlers()) == SESSION_LIST_PROJECTION_NAMES
    assert set(mcp_read_view_names()) - {"summary", "topology", "messages"} == set(SESSION_LIST_PROJECTIONS)
    assert set(mcp_get_session_projection_names()) - {"orchestration"} == set(SESSION_LIST_PROJECTIONS)


def test_every_declared_method_exists_on_the_archive_facade() -> None:
    from polylogue import Polylogue

    for projection in SESSION_LIST_PROJECTIONS.values():
        assert hasattr(Polylogue, projection.method), projection


def test_payload_keys_are_distinct_and_named_after_their_rows() -> None:
    keys = [projection.payload_key for projection in SESSION_LIST_PROJECTIONS.values()]
    assert len(set(keys)) == len(keys)
    assert SESSION_LIST_PROJECTIONS["file-edits"].payload_key == "file_edits"


@pytest.mark.asyncio
async def test_projection_table_drives_cli_and_mcp_name_vocabulary(
    monkeypatch: pytest.MonkeyPatch, mcp_server: MCPServerUnderTest
) -> None:
    """One added row reaches every dispatch vocabulary without a name branch.

    Anti-vacuity: restoring a hand-written MCP name tuple, or retaining the
    CLI's separate list-projection rows, leaves ``projection-fixture`` out of
    one of these production dispatch inputs.
    """
    from polylogue.cli.read_view_handlers import session_list_read_view_handlers
    from polylogue.cli.read_views.events import run_read_events

    fixture = SessionListProjection(
        "projection-fixture",
        "get_session_events",
        "events",
        cli_handler="events",
    )
    monkeypatch.setitem(SESSION_LIST_PROJECTIONS, fixture.name, fixture)

    handlers = session_list_read_view_handlers()
    assert handlers[fixture.name].handler is run_read_events
    assert fixture.name in mcp_read_view_names()
    assert fixture.name in mcp_get_session_projection_names()

    poly = make_polylogue_mock()
    method = AsyncMock(return_value=[{"kind": "fixture-event"}])
    setattr(poly, fixture.method, method)
    with patch("polylogue.mcp.server._get_polylogue", return_value=poly):
        read = json.loads(
            await invoke_surface_async(
                mcp_server._tool_manager._tools["read"].fn,
                ref="session:codex:projection-registry",
                view=fixture.name,
            )
        )
        get = json.loads(
            await invoke_surface_async(
                mcp_server._tool_manager._tools["get"].fn,
                ref="session:codex:projection-registry",
                projection=fixture.name,
            )
        )

    assert read[fixture.payload_key] == get[fixture.payload_key] == [{"kind": "fixture-event"}]
    assert method.await_count == 2


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
    """An unknown projection must not silently resolve the default object.

    Anti-vacuity: deleting either boundary check falls through to
    ``resolve_ref`` and returns an object payload instead of invalid_argument.
    """
    with patch("polylogue.mcp.server._get_polylogue", return_value=make_polylogue_mock()):
        get_payload = json.loads(
            await invoke_surface_async(
                mcp_server._tool_manager._tools["get"].fn,
                ref="session:codex:projection-registry",
                projection="not-a-projection",
            )
        )
        read_payload = json.loads(
            await invoke_surface_async(
                mcp_server._tool_manager._tools["read"].fn,
                ref="session:codex:projection-registry",
                view="not-a-projection",
            )
        )

    assert get_payload["code"] == "invalid_argument"
    assert read_payload["code"] == "invalid_argument"


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
