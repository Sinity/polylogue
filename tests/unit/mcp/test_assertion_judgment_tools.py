from __future__ import annotations

import json
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

from polylogue.mcp.server import build_server
from polylogue.mcp.server_support import MCPRole
from polylogue.surfaces.payloads import AssertionBulkJudgmentPayload
from tests.infra.mcp import MCPServerUnderTest, invoke_surface


def _tool_names(role: str) -> set[str]:
    server = cast(MCPServerUnderTest, build_server(role=cast(MCPRole, role)))
    return set(server._tool_manager._tools)


def test_assertion_review_tools_require_review_capability() -> None:
    """A caller-supplied actor ref cannot elevate ordinary write capability."""

    read = _tool_names("read")
    write = _tool_names("write")
    review = _tool_names("review")
    assert {"list_assertion_candidates", "list_assertion_candidate_reviews"} <= read
    assert "judge_assertion_candidate" not in write
    assert "judge_assertion_candidates" not in write
    assert {"judge_assertion_candidate", "judge_assertion_candidates"} <= review


def test_mcp_supersede_preserves_operator_replacement_fields() -> None:
    """The review adapter forwards correction fields into the real bulk envelope."""

    poly = MagicMock()
    poly.judge_assertion_candidates = AsyncMock(
        return_value=AssertionBulkJudgmentPayload(items=(), applied_count=0, idempotent_count=0, failed_count=0)
    )
    with patch("polylogue.mcp.server._get_polylogue", return_value=poly):
        server = cast(MCPServerUnderTest, build_server(role="review"))
        tool = server._tool_manager._tools["judge_assertion_candidate"]
        result = invoke_surface(
            tool.fn,
            candidate_ref="assertion:candidate-1",
            decision="supersede",
            replacement_kind="decision",
            replacement_body_text="Operator correction",
            replacement_value={"source": "review"},
        )

    assert json.loads(result)["failed_count"] == 0
    item = poly.judge_assertion_candidates.await_args.kwargs["items"][0]
    assert item.replacement_kind == "decision"
    assert item.replacement_body_text == "Operator correction"
    assert item.replacement_value == {"source": "review"}


def test_mcp_bulk_judgment_rejects_non_boolean_injection() -> None:
    """String JSON values must not obtain reviewer injection authorization."""

    poly = MagicMock()
    poly.judge_assertion_candidates = AsyncMock()
    with patch("polylogue.mcp.server._get_polylogue", return_value=poly):
        server = cast(MCPServerUnderTest, build_server(role="review"))
        tool = server._tool_manager._tools["judge_assertion_candidates"]
        result = invoke_surface(
            tool.fn,
            items=[{"candidate_ref": "assertion:candidate-1", "decision": "accept", "inject": "false"}],
        )

    assert json.loads(result)["code"] == "invalid_assertion_judgment"
    poly.judge_assertion_candidates.assert_not_awaited()


def test_mcp_bulk_judgment_rejects_invalid_supersede_replacement_fields() -> None:
    """Malformed replacement fields must not silently promote original content."""

    poly = MagicMock()
    poly.judge_assertion_candidates = AsyncMock()
    with patch("polylogue.mcp.server._get_polylogue", return_value=poly):
        server = cast(MCPServerUnderTest, build_server(role="review"))
        tool = server._tool_manager._tools["judge_assertion_candidates"]
        result = invoke_surface(
            tool.fn,
            items=[
                {
                    "candidate_ref": "assertion:candidate-1",
                    "decision": "supersede",
                    "replacement_kind": 7,
                    "replacement_body_text": "Corrected claim",
                }
            ],
        )

    assert json.loads(result)["code"] == "invalid_assertion_judgment"
    poly.judge_assertion_candidates.assert_not_awaited()
