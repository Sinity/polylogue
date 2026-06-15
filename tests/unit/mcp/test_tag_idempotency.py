"""Verify tag mutation operations surface idempotency outcomes.

Refs #862.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from polylogue.surfaces.payloads import TagMutationResult
from tests.infra.mcp import (
    MCPServerUnderTest,
    invoke_surface,
    make_polylogue_mock,
)

_ADD_TAG_TOOL = "add_tag"
_REMOVE_TAG_TOOL = "remove_tag"
_CONV_ID = "test:conv-tag-1"
_TAG = "review"


def _tag_result(outcome: str, detail: str | None = None) -> TagMutationResult:
    """Build a ``TagMutationResult`` matching the centralized facade contract."""
    return TagMutationResult(outcome=outcome, detail=detail)  # type: ignore[arg-type]


class TestTagIdempotencyOutcomes:
    """Each mutation path must surface the correct outcome value."""

    # ── add_tag ──────────────────────────────────────────────────────

    def test_add_tag_absent_yields_added(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock(resolved_id=_CONV_ID)
            mock_poly.add_tag = AsyncMock(return_value=_tag_result("added"))
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools[_ADD_TAG_TOOL].fn,
                session_id=_CONV_ID,
                tag=_TAG,
            )

        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["outcome"] == "added"
        assert parsed["tag"] == _TAG
        assert "detail" not in parsed

    def test_add_tag_already_present_yields_no_op(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock(resolved_id=_CONV_ID)
            mock_poly.add_tag = AsyncMock(return_value=_tag_result("no_op", "already_present"))
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools[_ADD_TAG_TOOL].fn,
                session_id=_CONV_ID,
                tag=_TAG,
            )

        parsed = json.loads(result)
        assert parsed["status"] == "unchanged"
        assert parsed["outcome"] == "no_op"
        assert parsed["detail"] == "already_present"
        assert parsed["tag"] == _TAG

    def test_add_tag_session_not_found(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock(resolved_id=None)
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools[_ADD_TAG_TOOL].fn,
                session_id="nonexistent:id",
                tag=_TAG,
            )

        parsed = json.loads(result)
        assert "message" in parsed
        assert "not found" in parsed["message"].lower()

    # ── remove_tag ───────────────────────────────────────────────────

    def test_remove_tag_present_yields_removed(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock(resolved_id=_CONV_ID)
            mock_poly.remove_tag = AsyncMock(return_value=_tag_result("removed"))
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools[_REMOVE_TAG_TOOL].fn,
                session_id=_CONV_ID,
                tag=_TAG,
            )

        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["outcome"] == "removed"
        assert parsed["tag"] == _TAG
        assert "detail" not in parsed

    def test_remove_tag_absent_yields_not_present(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock(resolved_id=_CONV_ID)
            mock_poly.remove_tag = AsyncMock(return_value=_tag_result("not_present", "tag_not_present"))
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools[_REMOVE_TAG_TOOL].fn,
                session_id=_CONV_ID,
                tag=_TAG,
            )

        parsed = json.loads(result)
        assert parsed["status"] == "not_found"
        assert parsed["outcome"] == "not_present"
        assert parsed["detail"] == "tag_not_present"
        assert parsed["tag"] == _TAG

    def test_remove_tag_session_not_found(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock(resolved_id=None)
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools[_REMOVE_TAG_TOOL].fn,
                session_id="nonexistent:id",
                tag=_TAG,
            )

        parsed = json.loads(result)
        assert "message" in parsed
        assert "not found" in parsed["message"].lower()


class TestBulkTagExcludesOutcome:
    """bulk_tag_sessions does not carry a per-tag outcome — it uses counts."""

    def test_bulk_tag_has_no_outcome_field(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            from polylogue.surfaces.payloads import BulkTagMutationResult

            mock_poly = make_polylogue_mock()
            mock_poly.bulk_tag_sessions = AsyncMock(
                return_value=BulkTagMutationResult(session_count=3, tag_count=1, affected_count=3, skipped_count=0)
            )
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools["bulk_tag_sessions"].fn,
                session_ids=["conv-1", "conv-2", "conv-3"],
                tags=["important"],
            )

        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert "outcome" not in parsed


class TestMutationResultPayloadRoundtrip:
    """ensure outcome serializes correctly and exclude_none omits null."""

    def test_outcome_present_in_serialized(self) -> None:
        from polylogue.mcp.payloads import MutationResultPayload

        payload = MutationResultPayload(
            status="ok",
            session_id=_CONV_ID,
            tag=_TAG,
            outcome="added",
        )
        serialized = payload.model_dump_json(exclude_none=True)
        parsed = json.loads(serialized)
        assert parsed["outcome"] == "added"

    def test_outcome_omitted_when_none(self) -> None:
        from polylogue.mcp.payloads import MutationResultPayload

        payload = MutationResultPayload(
            status="ok",
            session_id=_CONV_ID,
            tag=_TAG,
        )
        serialized = payload.model_dump_json(exclude_none=True)
        parsed = json.loads(serialized)
        assert "outcome" not in parsed
