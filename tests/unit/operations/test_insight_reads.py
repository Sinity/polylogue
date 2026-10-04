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


def test_resident_readiness_preserves_selected_original_coverage(tmp_path: Path) -> None:
    from polylogue.operations.insight_contracts import InsightReadinessResult

    with running_daemon_operations(tmp_path / "archive") as stack:
        envelope = stack.client.operation(
            "insights.readiness", {"query": {"insights": ["profiles"], "origin": "codex-session"}}
        )
        assert envelope is not None and envelope["outcome"] == "completed"
        result = InsightReadinessResult.model_validate(envelope["result"])
        assert [entry.insight_name for entry in result.report.insights] == ["session_profiles"]
        assert result.report.total_sessions == 0
        assert result.report.origin == "codex-session"
        assert result.report.converged is True
        assert result.outcome.state == "ok"
        assert envelope["archive"]["archive_identity"]
        assert envelope["schema_versions"]["index"]


def test_pending_readiness_is_named_even_without_report_rows() -> None:
    from polylogue.analysis.readiness import InsightReadinessReport
    from polylogue.operations.insight_contracts import InsightReadinessResult
    from polylogue.operations.insight_reads import execute_insight_readiness

    class PendingArchive:
        def insight_readiness_report(self, query: object) -> InsightReadinessReport:
            return InsightReadinessReport(checked_at="2026-01-01T00:00:00Z", converged=False)

    result = execute_insight_readiness(
        {"query": {}}, archive=cast(ArchiveStore, PendingArchive()), checkpoint=lambda: None
    )
    parsed = InsightReadinessResult.model_validate(result)
    assert parsed.outcome.state == "degraded"
    assert parsed.outcome.reason == "insight_convergence_pending"
    assert parsed.report.converged is False


def test_readiness_schema_and_strict_wire_validation_match_canonical_contract() -> None:
    from polylogue.operations.daemon_protocol import (
        InsightReadinessWireRequest,
        InsightReadinessWireResult,
        OperationResultContractError,
        validate_operation_result,
    )
    from polylogue.operations.insight_contracts import InsightReadinessRequest, InsightReadinessResult
    from polylogue.surfaces.outcome import decide_outcome

    assert InsightReadinessWireRequest.model_json_schema() == InsightReadinessRequest.model_json_schema()
    assert InsightReadinessWireResult.model_json_schema() == InsightReadinessResult.model_json_schema()
    with pytest.raises(ValidationError):
        InsightReadinessRequest.model_validate({"query": {"undeclared_filter": True}})
    with pytest.raises(ValidationError):
        InsightReadinessRequest.model_validate({"query": {"insights": ["unknown-insight"]}})
    result = {
        "report": {"checked_at": "2026-01-01T00:00:00Z", "total_sessions": 0, "converged": False},
        "outcome": decide_outcome(matched=0, degraded=("insight_convergence_pending",)).to_dict(),
    }
    validate_operation_result("insights.readiness", result)
    with pytest.raises(OperationResultContractError):
        validate_operation_result(
            "insights.readiness", {**result, "report": {**result["report"], "converged": "false"}}
        )


@pytest.mark.parametrize(
    "fields,reason",
    [
        ({"table_present": False}, "insight_output_diverged"),
        ({"expected_row_count": 1}, "insight_output_incomplete"),
        ({"degraded_count": 1}, "insight_evidence_degraded"),
        ({"schema_contract_issues": ("missing declared field",)}, "insight_evidence_degraded"),
    ],
)
def test_readiness_report_preserves_named_gaps_after_materialization(fields: dict[str, object], reason: str) -> None:
    from polylogue.analysis.readiness import InsightReadinessEntry, InsightReadinessReport
    from polylogue.operations.insight_contracts import InsightReadinessResult
    from polylogue.operations.insight_reads import execute_insight_readiness

    entry = InsightReadinessEntry.model_validate(
        {"insight_name": "session_profiles", "display_name": "Profiles", **fields}
    )

    class ObservedArchive:
        def insight_readiness_report(self, query: object) -> InsightReadinessReport:
            return InsightReadinessReport(checked_at="2026-01-01T00:00:00Z", converged=True, insights=(entry,))

    result = InsightReadinessResult.model_validate(
        execute_insight_readiness({"query": {}}, archive=cast(ArchiveStore, ObservedArchive()), checkpoint=lambda: None)
    )
    assert result.outcome.state == "degraded"
    assert result.outcome.reason == reason
    assert result.report.converged is True
    assert result.report.insights == (entry,)
