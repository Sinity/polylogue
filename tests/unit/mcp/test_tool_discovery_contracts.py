"""Discovery and budget-continuation contracts of registered MCP tools."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from polylogue.agent_integration.spec import TOOL_CONTRACT_BY_NAME
from polylogue.mcp.declarations import MCP_TOOL_DECLARATIONS
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async, make_polylogue_mock


def test_precompact_is_in_the_published_context_contract() -> None:
    """The published context contract names the precompact intent the route serves.

    Anti-vacuity: drop ``precompact`` from the intent argument and the agent
    contract lists only ``resume`` among the preamble intents.
    """
    intent = next(argument for argument in TOOL_CONTRACT_BY_NAME["context"].arguments if argument.name == "intent")
    assert "precompact" in intent.description


@pytest.mark.asyncio
async def test_clear_corrections_confirmation_is_discoverable_and_enforced(mcp_server: MCPServerUnderTest) -> None:
    """The write declaration names every operation the handler gates on ``confirm``.

    Anti-vacuity: remove ``clear_corrections`` from the registry description
    and callers are not told about the refusal the handler below enforces.
    """
    declaration = next(tool for tool in MCP_TOOL_DECLARATIONS if tool.name == "write")
    assert "clear_corrections" in declaration.description
    with patch("polylogue.mcp.server._get_polylogue", return_value=make_polylogue_mock()):
        body = json.loads(
            await invoke_surface_async(
                mcp_server._tool_manager._tools["write"].fn,
                operation="clear_corrections",
                session_id="codex-session:discovery",
                confirm=False,
            )
        )
    assert body["is_error"] is True
    assert "confirm=true" in body["message"]


@pytest.mark.asyncio
async def test_personal_state_byte_budget_continuation_advances_from_nonzero_offset(
    mcp_server: MCPServerUnderTest,
) -> None:
    """An over-budget personal-state page continues from its retained prefix.

    Anti-vacuity: route personal-state projections through the framed
    transaction rebase and the envelope has no continuation, because these
    payloads carry no transaction request.
    """
    poly = make_polylogue_mock()
    rows = [
        {
            "annotation_id": f"annotation-{i}",
            "target_type": "session",
            "target_id": "codex-session:budget",
            "session_id": "codex-session:budget",
            "message_id": None,
            "note_text": "x" * 8000,
            "created_at": "2026-09-01T00:00:00Z",
            "updated_at": "2026-09-01T00:00:00Z",
        }
        for i in range(12)
    ]
    poly.list_annotations.return_value = rows
    query = mcp_server._tool_manager._tools["query"].fn
    with patch("polylogue.mcp.server._get_polylogue", return_value=poly):
        first = json.loads(await invoke_surface_async(query, projection="annotations", limit=8, offset=3))
        assert first["budget_exceeded"] is True
        assert first["returned_items"] > 0
        follow = first["continuation"]
        assert follow["tool"] == "query"
        assert follow["arguments"]["projection"] == "annotations"
        assert follow["arguments"]["offset"] == 3 + first["returned_items"]
        second = json.loads(await invoke_surface_async(query, **follow["arguments"]))
    page = second.get("page") or second
    assert page["items"][0]["annotation_id"] == f"annotation-{3 + first['returned_items']}"
