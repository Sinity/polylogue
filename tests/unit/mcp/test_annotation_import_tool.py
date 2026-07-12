"""MCP annotation batch import contract."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from polylogue.annotations.importer import (
    AnnotationBatchImportError,
    AnnotationBatchImportRequest,
    AnnotationBatchImportResult,
)
from tests.infra.mcp import MCPServerUnderTest, invoke_surface, make_polylogue_mock


def test_mcp_annotation_import_delegates_to_product_operation(mcp_server: MCPServerUnderTest) -> None:
    result = AnnotationBatchImportResult(
        status="ok",
        batch_ref="annotation-batch:b1",
        qualified_schema_id="delegation.discourse@v1",
        target_ref="delegation:d1",
        total_count=1,
        valid_count=1,
        invalid_count=0,
        abstained_count=0,
        rows=(),
    )
    with (
        patch("polylogue.mcp.server._get_polylogue", return_value=make_polylogue_mock()),
        patch(
            "polylogue.mcp.server_mutation_tools.run_annotation_batch_import",
            new=AsyncMock(return_value=result),
        ) as operation,
    ):
        raw = invoke_surface(
            mcp_server._tool_manager._tools["import_annotation_batch"].fn,
            jsonl='{"row_key":"r","value":{},"evidence_refs":[]}',
            batch_id="b1",
            schema_id="delegation.discourse",
            schema_version=1,
            target_ref="delegation:d1",
            source_result_ref="result-set:r1",
            actor_ref="agent:a1",
            model_ref="agent:m1",
            prompt_ref="block:p1:0",
            metadata={"campaign": "mcp"},
        )

    payload = json.loads(raw)
    assert payload["batch_ref"] == "annotation-batch:b1"
    operation.assert_awaited_once()
    assert operation.await_args is not None
    _, request = operation.await_args.args
    assert request == AnnotationBatchImportRequest(
        jsonl='{"row_key":"r","value":{},"evidence_refs":[]}',
        batch_id="b1",
        schema_id="delegation.discourse",
        schema_version=1,
        target_ref="delegation:d1",
        source_result_ref="result-set:r1",
        actor_ref="agent:a1",
        model_ref="agent:m1",
        prompt_ref="block:p1:0",
        metadata={"campaign": "mcp"},
    )


def test_mcp_annotation_import_returns_validation_error_envelope(mcp_server: MCPServerUnderTest) -> None:
    with (
        patch("polylogue.mcp.server._get_polylogue", return_value=make_polylogue_mock()),
        patch(
            "polylogue.mcp.server_mutation_tools.run_annotation_batch_import",
            new=AsyncMock(side_effect=AnnotationBatchImportError("invalid batch")),
        ),
    ):
        raw = invoke_surface(
            mcp_server._tool_manager._tools["import_annotation_batch"].fn,
            jsonl='{"row_key":"r","value":{},"evidence_refs":[]}',
            batch_id="b1",
            schema_id="delegation.discourse",
            schema_version=1,
            target_ref="delegation:d1",
            source_result_ref="result-set:r1",
            actor_ref="agent:a1",
            model_ref="agent:m1",
            prompt_ref="block:p1:0",
        )

    payload = json.loads(raw)
    assert payload["code"] == "invalid_annotation_batch"
    assert payload["error"] == "invalid_annotation_batch"
    assert payload["message"] == "invalid batch"
