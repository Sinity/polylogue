from __future__ import annotations

from typing import cast

from polylogue.mcp.server import build_server
from polylogue.mcp.server_support import MCPRole
from tests.infra.mcp import MCPServerUnderTest


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
