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
from typing import Literal, get_args, get_origin, get_type_hints
from unittest.mock import AsyncMock, patch

import pytest

from polylogue.archive.viewport import READ_VIEW_PROFILE_BY_ID
from polylogue.cli.read_view_handlers import READ_VIEW_HANDLERS
from polylogue.cli.read_view_registry import READ_VIEW_HANDLER_METADATA
from polylogue.mcp.server_cutover import mcp_get_projection_names, mcp_query_projection_names
from polylogue.operations.evidence_window import EVIDENCE_WINDOW_FAMILIES
from polylogue.operations.session_projections import (
    SESSION_LIST_PROJECTION_NAMES,
    SESSION_LIST_PROJECTIONS,
    SessionListProjection,
    mcp_get_session_projection_names,
    mcp_read_view_names,
)
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async, make_polylogue_mock


def _literal_values(annotation: object) -> set[str]:
    """Flatten a Literal or an optional Literal into its string choices."""

    if get_origin(annotation) is Literal:
        return {value for value in get_args(annotation) if isinstance(value, str)}
    return {value for argument in get_args(annotation) for value in _literal_values(argument)}


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
    from polylogue.cli.read_views.session_evidence import run_read_events

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

    ``agent-policies`` is the one list projection answered whole; the windowed
    ones are covered by the next test.
    Anti-vacuity: restore a hand-written ``if projection == \"agent-policies\"``
    branch, or remove either table lookup, and this does not observe the same
    facade method and response key from both production tool handlers.
    """
    projection = SESSION_LIST_PROJECTIONS["agent-policies"]
    poly = make_polylogue_mock()
    method = AsyncMock(return_value=[{"kind": "policy"}])
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

    assert read[projection.payload_key] == get[projection.payload_key] == [{"kind": "policy"}]
    assert method.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["events", "file-edits", "web-content", "materials"])
async def test_windowed_session_list_projection_pages_through_the_evidence_window(
    mcp_server: MCPServerUnderTest, name: str
) -> None:
    """A windowed list projection is one bounded page on both MCP routes.

    Anti-vacuity: send a windowed projection back to its whole-list facade
    method (``projection.method``), or drop the ``read`` route's ``limit``,
    and the refused whole-list call or the forwarded limit goes red here.
    """
    projection = SESSION_LIST_PROJECTIONS[name]
    poly = make_polylogue_mock()
    setattr(poly, projection.method, AsyncMock(side_effect=AssertionError("answered the relation whole")))
    window = AsyncMock(
        return_value={
            "rows": [{"kind": "row"}],
            "total": 3,
            "returned": 1,
            "limit": 1,
            "offset": 0,
            "next_offset": 1,
            "continuation": "token",
            "complete": False,
        }
    )
    poly.read_session_evidence_window = window

    with patch("polylogue.mcp.server._get_polylogue", return_value=poly):
        read = json.loads(
            await invoke_surface_async(
                mcp_server._tool_manager._tools["read"].fn,
                ref="session:codex:projection-registry",
                view=projection.name,
                limit=1,
                offset=2,
            )
        )
        get = json.loads(
            await invoke_surface_async(
                mcp_server._tool_manager._tools["get"].fn,
                ref="session:codex:projection-registry",
                projection=projection.name,
            )
        )

    assert read[projection.payload_key] == get[projection.payload_key] == [{"kind": "row"}]
    assert read["continuation"] == get["continuation"] == "token"
    assert read["complete"] is get["complete"] is False
    assert window.await_args_list[0].args[1] == projection.name
    assert window.await_args_list[0].kwargs["limit"] == 1
    assert window.await_args_list[0].kwargs["offset"] == 2


@pytest.mark.asyncio
async def test_budget_trimmed_window_page_narrows_from_the_windows_resolved_offset(
    mcp_server: MCPServerUnderTest,
) -> None:
    """An oversized page reached by continuation narrows from where that page began.

    A continuation overrides the request's coordinates, so the caller's
    ``offset`` (0 here) is not where the page starts. Anti-vacuity: build the
    budget continuation from the request's arguments and it restarts at
    offset 0 (or repeats the same oversized token); leave the page's own
    ``continuation`` in the trimmed page and following it skips the rows the
    trim omitted.
    """
    from polylogue.mcp.server_support import MCP_RESPONSE_BUDGET_BYTES

    projection = SESSION_LIST_PROJECTIONS["file-edits"]
    poly = make_polylogue_mock()
    rows = [{"original_file": "x" * (MCP_RESPONSE_BUDGET_BYTES // 3), "position": i} for i in range(4)]
    poly.read_session_evidence_window = AsyncMock(
        return_value={
            "rows": rows,
            "total": 12,
            "returned": 4,
            "limit": 4,
            "offset": 4,
            "next_offset": 8,
            "continuation": "token-after-8",
            "complete": False,
        }
    )
    with patch("polylogue.mcp.server._get_polylogue", return_value=poly):
        body = json.loads(
            await invoke_surface_async(
                mcp_server._tool_manager._tools["read"].fn,
                ref="session:codex:projection-registry",
                view=projection.name,
                limit=4,
                continuation="token-after-4",
            )
        )

    assert body["budget_exceeded"] is True
    consumed = body["returned_items"]
    assert 0 < consumed < len(rows)
    assert body["page"][projection.payload_key] == rows[:consumed]
    assert body["page"]["continuation"] is None
    assert body["page"]["next_offset"] == 4 + consumed
    assert body["continuation"]["tool"] == "read"
    arguments = body["continuation"]["arguments"]
    assert arguments["view"] == projection.name
    assert arguments["offset"] == 4 + consumed
    assert "continuation" not in arguments


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
async def test_mcp_read_query_and_get_projection_literals_and_guards_are_live(
    mcp_server: MCPServerUnderTest,
) -> None:
    """MCP schemas and runtime rejection use their derived projection vocabularies.

    Anti-vacuity: red if a handler returns to an unrestricted ``str``
    annotation, or an unknown query projection silently falls through to the
    default unit-query route.
    """

    read = mcp_server._tool_manager._tools["read"].fn
    query = mcp_server._tool_manager._tools["query"].fn
    get = mcp_server._tool_manager._tools["get"].fn

    assert _literal_values(get_type_hints(read)["view"]) == set(mcp_read_view_names())
    assert _literal_values(get_type_hints(query)["projection"]) == set(mcp_query_projection_names())
    assert _literal_values(get_type_hints(get)["projection"]) == set(mcp_get_projection_names())
    assert set(mcp_get_session_projection_names()) <= _literal_values(get_type_hints(get)["projection"])

    with patch("polylogue.mcp.server._get_polylogue", return_value=make_polylogue_mock()):
        payload = json.loads(await invoke_surface_async(query, projection="not-a-projection"))

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

    assert payload["read_views"] == list(mcp_read_view_names())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "view", tuple(name for name in SESSION_LIST_PROJECTIONS if name not in EVIDENCE_WINDOW_FAMILIES)
)
async def test_read_list_views_honor_limit_offset_and_next_page(mcp_server: MCPServerUnderTest, view: str) -> None:
    """A list view answered whole is still sliced by ``limit``/``offset``.

    The windowed views page through the evidence window instead (previous
    tests). Anti-vacuity: without the shared read slice, the handler returns
    all five rows on every page.
    """
    projection = SESSION_LIST_PROJECTIONS[view]
    poly = make_polylogue_mock()
    rows = [{"event_index": i} for i in range(5)]
    setattr(poly, projection.method, AsyncMock(return_value=rows))
    read = mcp_server._tool_manager._tools["read"].fn
    with patch("polylogue.mcp.server._get_polylogue", return_value=poly):
        pages = [
            json.loads(
                await invoke_surface_async(read, ref="session:codex-session:w13", view=view, limit=2, offset=offset)
            )
            for offset in (0, 2, 4)
        ]
    assert [page["total"] for page in pages] == [5, 5, 5]
    assert [page["next_offset"] for page in pages] == [2, 4, None]
    assert [item for page in pages for item in page[projection.payload_key]] == rows
