"""CLI annotation import delegates to the shared product operation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from polylogue.annotations.importer import AnnotationBatchImportRequest, AnnotationBatchImportResult
from polylogue.cli.click_app import cli


class _PolylogueContext:
    async def __aenter__(self) -> _PolylogueContext:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


def test_cli_annotation_import_delegates_to_product_operation(tmp_path: Path) -> None:
    source = tmp_path / "labels.jsonl"
    source.write_text('{"row_key":"r1","value":{},"evidence_refs":[]}\n', encoding="utf-8")
    product_result = AnnotationBatchImportResult(
        status="ok",
        batch_ref="annotation-batch:cli-batch",
        qualified_schema_id="delegation.discourse@v1",
        target_ref="delegation:d1",
        total_count=1,
        valid_count=1,
        invalid_count=0,
        abstained_count=0,
        rows=(),
    )
    with (
        patch("polylogue.cli.commands.annotations.Polylogue", return_value=_PolylogueContext()),
        patch(
            "polylogue.cli.commands.annotations.import_annotation_batch",
            new=AsyncMock(return_value=product_result),
        ) as operation,
    ):
        result = CliRunner().invoke(
            cli,
            [
                "annotations",
                "import",
                str(source),
                "--batch-id",
                "cli-batch",
                "--schema-id",
                "delegation.discourse",
                "--schema-version",
                "1",
                "--target-ref",
                "delegation:d1",
                "--source-result-ref",
                "result-set:r1",
                "--actor-ref",
                "agent:a1",
                "--model-ref",
                "agent:m1",
                "--prompt-ref",
                "block:p1:0",
                "--metadata-json",
                '{"campaign":"cli"}',
            ],
        )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["batch_ref"] == "annotation-batch:cli-batch"
    operation.assert_awaited_once()
    assert operation.await_args is not None
    _, request = operation.await_args.args
    assert request == AnnotationBatchImportRequest(
        jsonl='{"row_key":"r1","value":{},"evidence_refs":[]}\n',
        batch_id="cli-batch",
        schema_id="delegation.discourse",
        schema_version=1,
        target_ref="delegation:d1",
        source_result_ref="result-set:r1",
        actor_ref="agent:a1",
        model_ref="agent:m1",
        prompt_ref="block:p1:0",
        metadata={"campaign": "cli"},
    )


def test_cli_annotation_import_rejects_non_object_metadata(tmp_path: Path) -> None:
    source = tmp_path / "labels.jsonl"
    source.write_text('{"row_key":"r1","value":{},"evidence_refs":[]}\n', encoding="utf-8")
    result = CliRunner().invoke(
        cli,
        [
            "annotations",
            "import",
            str(source),
            "--batch-id",
            "cli-batch",
            "--schema-id",
            "delegation.discourse",
            "--schema-version",
            "1",
            "--target-ref",
            "delegation:d1",
            "--source-result-ref",
            "result-set:r1",
            "--actor-ref",
            "agent:a1",
            "--model-ref",
            "agent:m1",
            "--prompt-ref",
            "block:p1:0",
            "--metadata-json",
            "[]",
        ],
    )

    assert result.exit_code != 0
    assert "must decode to a JSON object" in result.output
