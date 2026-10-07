"""Resident Hermes diagnostics preserve availability rather than require Index."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.operations.daemon_protocol import (
    HermesHealthWireRequest,
    HermesHealthWireResult,
    OperationResultContractError,
    validate_operation_result,
)
from polylogue.operations.daemon_reads import DaemonReadDependencies
from polylogue.operations.hermes_health import execute_hermes_health
from polylogue.operations.hermes_health_contracts import (
    HermesHealthRequest,
    HermesHealthResult,
    decode_hermes_health_result,
)
from tests.infra.daemon_operations import running_daemon_operations


def test_resident_disabled_hermes_health_does_not_require_index(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    with running_daemon_operations(
        root, read_dependencies=DaemonReadDependencies(hermes_root=tmp_path / "absent")
    ) as stack:
        (root / "index.db").unlink()
        envelope = stack.client.operation("insights.hermes_health", {})
        assert envelope is not None and envelope["outcome"] == "completed", envelope
        result = decode_hermes_health_result(envelope["result"])
        assert result.report.verdict == "disabled"
        assert result.report.enabled is False
        assert result.outcome.state == "ok"
        assert "index" not in envelope["schema_versions"]
        assert envelope["archive"]["archive_identity"]
        rejected = stack.client.operation("insights.hermes_health", {}, index_schema_version=9999)
        assert rejected is not None and rejected["outcome"] == "rejected", rejected


def test_resident_hermes_health_keeps_missing_tier_coverage(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    hermes = tmp_path / "hermes"
    hermes.mkdir()
    with running_daemon_operations(root, read_dependencies=DaemonReadDependencies(hermes_root=hermes)) as stack:
        (root / "index.db").unlink()
        (root / "ops.db").unlink()
        envelope = stack.client.operation("insights.hermes_health", {})
        assert envelope is not None and envelope["outcome"] == "completed", envelope
        result = decode_hermes_health_result(envelope["result"])
        assert result.report.verdict == "unavailable"
        assert result.report.measurement_coverage.complete is False
        assert result.outcome.state == "degraded"
        assert result.outcome.reason == "hermes_health_unavailable"
        assert result.report.convergence_debt_failed_count == 0
        assert "convergence-debt measurement unavailable" in result.report.caveats


def test_health_cancellation_precedes_every_probe(tmp_path: Path) -> None:
    class CancelledError(RuntimeError):
        pass

    def checkpoint() -> None:
        raise CancelledError()

    with pytest.raises(CancelledError):
        execute_hermes_health(tmp_path / "archive", hermes_root=tmp_path / "hermes", checkpoint=checkpoint)


def test_health_wire_preserves_canonical_closed_schema_and_strict_scalars(tmp_path: Path) -> None:
    assert HermesHealthWireRequest.model_json_schema() == HermesHealthRequest.model_json_schema()
    assert HermesHealthWireResult.model_json_schema() == HermesHealthResult.model_json_schema()
    result = execute_hermes_health(tmp_path, hermes_root=tmp_path / "absent", checkpoint=lambda: None)
    validate_operation_result("insights.hermes_health", result)
    assert decode_hermes_health_result(result).report.sources == ()
    report = result["report"]
    assert isinstance(report, dict)
    assert decode_hermes_health_result(result).report.to_dict() == report
    for changed in ({"enabled": 0}, {"convergence_debt_failed_count": True}, {"undeclared": "value"}):
        with pytest.raises(OperationResultContractError):
            validate_operation_result("insights.hermes_health", {**result, "report": {**report, **changed}})


def test_cancelled_session_sample_physically_closes_original_readers(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    from polylogue.analysis import hermes_integration_health as health

    opened: list[sqlite3.Connection] = []
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection as original_open

    def capture(path: Path) -> sqlite3.Connection:
        connection = original_open(path)
        opened.append(connection)
        return connection

    class CancelledError(RuntimeError):
        pass

    def abort() -> None:
        raise CancelledError()

    monkeypatch.setattr(health, "open_readonly_connection", capture)
    with pytest.raises(CancelledError):
        health._sample_session_debt(workspace_env["archive_root"], ("neutral-session",), checkpoint=abort)
    assert len(opened) == 2
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


def test_python_health_composer_preserves_split_archive_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.analysis.hermes_health_contracts import HermesIntegrationHealth
    from polylogue.config import Config
    from polylogue.operations import hermes_health

    configured = tmp_path / "configured"
    selected = tmp_path / "selected"
    selected.mkdir()
    config = Config(
        archive_root=configured, render_root=configured / "render", sources=[], db_path=selected / "index.db"
    )
    observed: list[Path] = []
    report = HermesIntegrationHealth("2026-01-01T00:00:00Z", False, "absent", "disabled")

    def capture(root: Path, *, hermes_root: Path) -> HermesIntegrationHealth:
        observed.append(root)
        return report

    monkeypatch.setattr(hermes_health, "read_hermes_health", capture)
    assert hermes_health.configured_hermes_health(config) is report
    assert observed == [selected]


def test_resident_health_reports_unreadable_source_without_an_audit_prerequisite(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    hermes = tmp_path / "hermes"
    hermes.mkdir()
    with running_daemon_operations(root, read_dependencies=DaemonReadDependencies(hermes_root=hermes)) as stack:
        (root / "source.db").write_bytes(b"neutral corrupt source tier")
        envelope = stack.client.operation("insights.hermes_health", {})
        assert envelope is not None and envelope["outcome"] == "completed", envelope
        result = decode_hermes_health_result(envelope["result"])
        assert result.report.verdict == "unavailable"
        assert result.report.measurement_coverage.complete is False
        assert result.outcome.state == "degraded"
        assert any("source tier" in reason for reason in result.report.measurement_coverage.unmeasured_reasons)


def test_health_failed_cleanup_does_not_claim_physical_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.archive.query.execution_control import QueryExecutionContext
    from polylogue.operations import hermes_health
    from polylogue.operations.daemon_execution import execute_operation
    from polylogue.operations.daemon_protocol import DaemonOperationRequest
    from polylogue.operations.mutation_transaction import MutationPrincipal
    from polylogue.operations.operation_context_types import OperationContext

    class CleanupError(RuntimeError):
        pass

    def fail(*args: object, **kwargs: object) -> dict[str, object]:
        raise CleanupError("original reader cleanup failed")

    with running_daemon_operations(tmp_path / "archive") as stack:
        request = DaemonOperationRequest(operation="insights.hermes_health", payload={}, request_id="cleanup-fault")
        control = QueryExecutionContext(call_id="cleanup-fault", query_ref=request.fingerprint)
        context = OperationContext(
            stack.archive_root,
            MutationPrincipal("neutral-reader", frozenset({"read"}), "cli"),
            "daemon",
            runtime=stack.runtime,
            read_control=control,
            read_dependencies=DaemonReadDependencies(hermes_root=tmp_path / "absent"),
        )
        monkeypatch.setattr(hermes_health, "execute_hermes_health", fail)
        with pytest.raises(CleanupError):
            execute_operation(request, context)
        assert control.receipt.cleanup_complete is False


@pytest.mark.parametrize(
    ("operation", "payload"),
    [
        ("insights.hermes_health", {}),
        ("user.settings.list", {}),
        ("user.settings.get", {"setting_key": "subscription_tier"}),
    ],
)
def test_independent_read_explicit_index_condition_is_observed_not_invented(
    tmp_path: Path, operation: str, payload: dict[str, object]
) -> None:
    from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION

    root = tmp_path / "archive"
    with running_daemon_operations(root) as stack:
        matching = stack.client.operation(operation, payload, index_schema_version=INDEX_SCHEMA_VERSION)
        assert matching is not None and matching["outcome"] == "completed", matching
        assert matching["schema_versions"]["index"] == INDEX_SCHEMA_VERSION
        wrong = stack.client.operation(operation, payload, index_schema_version=INDEX_SCHEMA_VERSION + 1)
        assert wrong is not None and wrong["outcome"] == "rejected", wrong
        assert wrong["error"]["code"] == "schema_version_mismatch"
        (root / "index.db").unlink()
        missing = stack.client.operation(operation, payload, index_schema_version=INDEX_SCHEMA_VERSION)
        assert missing is not None and missing["outcome"] == "rejected", missing
        assert missing["error"]["code"] == "archive_tier_unavailable"
        assert missing["error"]["data"]["tier"] == "index"
        independent = stack.client.operation(operation, payload)
        assert independent is not None and independent["outcome"] == "completed", independent
        assert "index" not in independent["schema_versions"]
