"""CLI structural annotation join delegates to the shared product operation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from polylogue.annotations.join_contracts import AnnotationStructuralJoinRequest, AnnotationStructuralJoinResult
from polylogue.cli.click_app import cli
from polylogue.core.enums import AssertionStatus


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


def test_cli_annotation_join_maps_the_complete_request() -> None:
    with patch(
        "polylogue.cli.commands.annotations.dispatch_read",
        return_value=({"result": _result().model_dump(mode="json"), "outcome": {"state": "empty"}}, None),
    ) as operation:
        result = CliRunner().invoke(
            cli,
            [
                "annotations",
                "join",
                "--schema-id",
                "delegation.discourse",
                "--schema-version",
                "1",
                "--status",
                "active",
                "--target-kind",
                "delegation",
                "--group-by",
                "repo",
                "--group-by",
                "model",
                "--limit",
                "25",
                "--offset",
                "5",
            ],
        )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["joined_count"] == 0
    operation.assert_called_once()
    assert operation.call_args is not None
    _, lowered = operation.call_args.args
    assert lowered.operation == "annotation.join"
    request = AnnotationStructuralJoinRequest.model_validate(lowered.payload)
    assert request == AnnotationStructuralJoinRequest(
        schema_id="delegation.discourse",
        schema_version=1,
        statuses=(AssertionStatus.ACTIVE,),
        target_kind="delegation",
        group_by=("repo", "model"),
        limit=25,
        offset=5,
    )


def test_cli_annotation_join_requires_explicit_status() -> None:
    result = CliRunner().invoke(
        cli,
        [
            "annotations",
            "join",
            "--schema-id",
            "delegation.discourse",
            "--schema-version",
            "1",
        ],
    )

    assert result.exit_code == 2
    assert "Missing option '--status'" in result.output


def test_cli_annotation_join_requires_the_resident_without_local_fallback(tmp_path: Path) -> None:
    from polylogue.cli.operation_kernel import OperationUnavailableError

    with patch(
        "polylogue.cli.commands.annotations.dispatch_read",
        side_effect=OperationUnavailableError(operation="annotation.join", archive_root=str(tmp_path)),
    ):
        result = CliRunner().invoke(
            cli,
            [
                "annotations",
                "join",
                "--schema-id",
                "delegation.discourse",
                "--schema-version",
                "1",
                "--status",
                "active",
            ],
        )
    assert result.exit_code != 0
    assert isinstance(result.exception, (OperationUnavailableError, SystemExit))
