"""MCP structural annotation join contract."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from polylogue.annotations.join import AnnotationStructuralJoinResult
from polylogue.core.enums import AssertionStatus
from tests.infra.mcp import MCPServerUnderTest, invoke_surface, make_polylogue_mock


def _result() -> AnnotationStructuralJoinResult:
    return AnnotationStructuralJoinResult(
        qualified_schema_id="delegation.discourse@v1",
        requested_statuses=(AssertionStatus.ACTIVE,),
        selected_annotation_count=0,
        matched_annotation_count=0,
        offset=0,
        selection_truncated=False,
        joined_count=0,
        missing_target_count=0,
        ambiguous_target_count=0,
        schema_drift_count=0,
        invalid_value_count=0,
        multi_label_target_count=0,
        duplicate_label_count=0,
        diagnostics_truncated=False,
        diagnostics=(),
        rows=(),
        groups=(),
    )


def test_mcp_annotation_join_maps_the_complete_request(mcp_server: MCPServerUnderTest) -> None:
    poly = make_polylogue_mock()
    poly.join_typed_annotations = AsyncMock(return_value=_result())
    with patch("polylogue.mcp.server._get_polylogue", return_value=poly):
        raw = invoke_surface(
            mcp_server._tool_manager._tools["join_typed_annotations"].fn,
            schema_id="delegation.discourse",
            schema_version=1,
            statuses=["active"],
            target_kind="delegation",
            group_by=["repo", "model"],
            limit=25,
            offset=5,
        )

    payload = json.loads(raw)
    assert payload["joined_count"] == 0
    poly.join_typed_annotations.assert_awaited_once_with(
        schema_id="delegation.discourse",
        schema_version=1,
        statuses=["active"],
        target_kind="delegation",
        group_by=["repo", "model"],
        limit=25,
        offset=5,
    )


def test_mcp_annotation_join_returns_validation_error(mcp_server: MCPServerUnderTest) -> None:
    poly = make_polylogue_mock()
    poly.join_typed_annotations = AsyncMock(side_effect=ValueError("status required"))
    with patch("polylogue.mcp.server._get_polylogue", return_value=poly):
        raw = invoke_surface(
            mcp_server._tool_manager._tools["join_typed_annotations"].fn,
            schema_id="delegation.discourse",
            schema_version=1,
            statuses=[],
        )

    payload = json.loads(raw)
    assert payload["code"] == "invalid_annotation_join"
    assert payload["message"] == "status required"
