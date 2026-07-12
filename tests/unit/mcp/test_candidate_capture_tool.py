"""MCP parity contract for candidate capture."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from polylogue.core.enums import AssertionKind, AssertionStatus
from polylogue.surfaces.payloads import AssertionClaimPayload
from tests.infra.mcp import MCPServerUnderTest, invoke_surface, make_polylogue_mock


def test_capture_candidate_returns_the_shared_assertion_payload(mcp_server: MCPServerUnderTest) -> None:
    payload = AssertionClaimPayload(
        assertion_id="assertion-terminal-note:mcp",
        target_ref="session:codex-session:demo",
        kind=AssertionKind.LESSON,
        body_text="lesson from MCP",
        evidence_refs=("session:codex-session:demo",),
        status=AssertionStatus.CANDIDATE,
        context_policy={"inject": False, "promotion_required": True},
        created_at_ms=1,
        updated_at_ms=1,
    )
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        poly = make_polylogue_mock()
        poly.capture_assertion_candidate = AsyncMock(return_value=payload)
        mock_get_polylogue.return_value = poly
        raw = invoke_surface(
            mcp_server._tool_manager._tools["capture_assertion_candidate"].fn,
            body_text="lesson from MCP",
            kind="lesson",
            refs=["session:codex-session:demo"],
            scope_refs=["repo:polylogue"],
        )

    assert json.loads(raw) == payload.model_dump(mode="json")
    poly.capture_assertion_candidate.assert_awaited_once_with(
        body_text="lesson from MCP",
        kind=AssertionKind.LESSON,
        refs=("session:codex-session:demo",),
        scope_refs=("repo:polylogue",),
        cwd=None,
    )
