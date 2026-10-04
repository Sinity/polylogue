"""Strict resident insight contracts and original reader ownership."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from polylogue.analysis.registry import INSIGHT_REGISTRY, InsightQueryError
from polylogue.operations.insight_contracts import InsightListRequest, InsightListResult
from polylogue.operations.insight_reads import execute_insight_read
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.daemon_operations import running_daemon_operations


@pytest.mark.parametrize("name", tuple(INSIGHT_REGISTRY))
def test_each_registered_insight_query_has_a_closed_wire_branch(name: str) -> None:
    descriptor = INSIGHT_REGISTRY[name]
    request = InsightListRequest.model_validate({"page": {"insight": name, "query": {}}})
    assert descriptor.query_model is not None
    assert descriptor.query_model.model_json_schema()["additionalProperties"] is False
    assert isinstance(request.page.query, descriptor.query_model)
    with pytest.raises((ValidationError, InsightQueryError)):
        InsightListRequest.model_validate({"page": {"insight": name, "query": {"undeclared_filter": True}}})
    with pytest.raises(ValidationError):
        InsightListResult.model_validate(
            {"page": {"insight": name, "items": [], "total": 0, "extra": True}, "outcome": {}}
        )


def test_insight_reader_checks_cancellation_before_original_sql() -> None:
    class CancelledError(RuntimeError):
        pass

    class Untouched:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"cancelled read touched {name}")

    def abort() -> None:
        raise CancelledError()

    with pytest.raises(CancelledError):
        execute_insight_read(
            {"page": {"insight": "session_profiles", "query": {}}},
            archive=cast(ArchiveStore, Untouched()),
            checkpoint=abort,
        )


def test_resident_coverage_returns_the_original_typed_empty_verdict(tmp_path: Path) -> None:
    with running_daemon_operations(tmp_path / "archive") as stack:
        envelope = stack.client.operation(
            "insights.list", {"page": {"insight": "archive_coverage", "query": {"group_by": "origin", "limit": 1}}}
        )
        assert envelope is not None and envelope["outcome"] == "completed"
        result = InsightListResult.model_validate(envelope["result"])
        assert result.page.insight == "archive_coverage"
        assert result.page.items == []
        assert result.page.total == 0
        assert result.outcome.state == "empty"
        assert envelope["archive"]["archive_identity"]
        assert envelope["schema_versions"]["index"]


def test_insight_wire_schema_and_nested_scalar_validation_remain_canonical() -> None:
    from polylogue.operations.daemon_protocol import (
        InsightReadRequest,
        InsightReadResult,
        OperationResultContractError,
        validate_operation_result,
    )
    from polylogue.surfaces.outcome import decide_outcome

    assert InsightReadRequest.model_json_schema() == InsightListRequest.model_json_schema()
    assert InsightReadResult.model_json_schema() == InsightListResult.model_json_schema()
    valid = {
        "page": {
            "insight": "archive_coverage",
            "items": [{"group_by": "origin", "bucket": "synthetic", "session_count": 1}],
            "total": 1,
        },
        "outcome": decide_outcome(matched=1).to_dict(),
    }
    validate_operation_result("insights.list", valid)
    invalid = {
        **valid,
        "page": {**valid["page"], "items": [{"group_by": "origin", "bucket": "synthetic", "session_count": True}]},
    }
    with pytest.raises(OperationResultContractError):
        validate_operation_result("insights.list", invalid)
