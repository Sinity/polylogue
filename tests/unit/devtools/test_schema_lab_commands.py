from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from devtools import schema_audit, schema_generate, schema_promote
from polylogue.lib.outcomes import OutcomeCheck, OutcomeStatus
from polylogue.schemas.audit_models import AuditReport
from polylogue.schemas.generation_models import GenerationResult
from polylogue.schemas.operator_models import (
    SchemaAuditRequest,
    SchemaInferRequest,
    SchemaInferResult,
    SchemaPromoteRequest,
    SchemaPromoteResult,
)


@dataclass(frozen=True)
class _ConfigStub:
    db_path: Path


def test_schema_audit_returns_success_for_passing_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_audit(request: SchemaAuditRequest) -> AuditReport:
        assert request.provider == "chatgpt"
        return AuditReport(
            provider="chatgpt",
            checks=[OutcomeCheck(name="package", status=OutcomeStatus.OK, summary="package valid")],
        )

    monkeypatch.setattr(schema_audit, "audit_schemas", fake_audit)

    assert schema_audit.main(["--provider", "chatgpt", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "ok"
    result = payload["result"]
    assert result["provider"] == "chatgpt"
    assert result["summary"]["passed"] == 1


def test_schema_audit_returns_failure_for_failing_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_audit(request: SchemaAuditRequest) -> AuditReport:
        assert request.provider is None
        return AuditReport(checks=[OutcomeCheck(name="package", status=OutcomeStatus.ERROR, summary="missing")])

    monkeypatch.setattr(schema_audit, "audit_schemas", fake_audit)

    assert schema_audit.main(["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "ok"
    assert payload["result"]["summary"]["failed"] == 1


def test_schema_generate_forwards_generation_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[SchemaInferRequest] = []

    def fake_get_config() -> _ConfigStub:
        return _ConfigStub(db_path=tmp_path / "archive.db")

    def fake_infer(request: SchemaInferRequest) -> SchemaInferResult:
        captured.append(request)
        return SchemaInferResult(
            generation=GenerationResult(
                provider=request.provider,
                schema={"type": "object"},
                sample_count=2,
                versions=["v1"],
                default_version="v1",
                package_count=1,
            )
        )

    monkeypatch.setattr(schema_generate, "get_config", fake_get_config)
    monkeypatch.setattr(schema_generate, "infer_schema", fake_infer)

    assert schema_generate.main(["--provider", "chatgpt", "--max-samples", "2"]) == 0

    assert captured == [
        SchemaInferRequest(
            provider="chatgpt",
            db_path=tmp_path / "archive.db",
            max_samples=2,
            privacy_config=None,
            cluster=False,
            full_corpus=False,
        )
    ]
    assert "Generated schema package set for chatgpt" in capsys.readouterr().out


def test_schema_generate_cluster_without_manifest_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_get_config() -> _ConfigStub:
        return _ConfigStub(db_path=tmp_path / "archive.db")

    def fake_infer(request: SchemaInferRequest) -> SchemaInferResult:
        return SchemaInferResult(
            generation=GenerationResult(
                provider=request.provider,
                schema={"type": "object"},
                sample_count=1,
            )
        )

    monkeypatch.setattr(schema_generate, "get_config", fake_get_config)
    monkeypatch.setattr(schema_generate, "infer_schema", fake_infer)

    assert schema_generate.main(["--provider", "chatgpt", "--cluster"]) == 1
    assert "No samples found for clustering" in capsys.readouterr().err


def test_schema_promote_forwards_cluster_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[SchemaPromoteRequest] = []

    def fake_get_config() -> _ConfigStub:
        return _ConfigStub(db_path=tmp_path / "archive.db")

    def fake_promote(request: SchemaPromoteRequest) -> SchemaPromoteResult:
        captured.append(request)
        return SchemaPromoteResult(
            provider=request.provider,
            cluster_id=request.cluster_id,
            package_version="v2",
            package=None,
            schema={"type": "object"},
            versions=["v1", "v2"],
        )

    monkeypatch.setattr(schema_promote, "get_config", fake_get_config)
    monkeypatch.setattr(schema_promote, "promote_schema_cluster", fake_promote)

    assert (
        schema_promote.main(["--provider", "chatgpt", "--cluster", "cluster-1", "--with-samples", "--max-samples", "7"])
        == 0
    )

    assert captured == [
        SchemaPromoteRequest(
            provider="chatgpt",
            cluster_id="cluster-1",
            db_path=tmp_path / "archive.db",
            with_samples=True,
            max_samples=7,
        )
    ]
    assert "Promoted cluster cluster-1" in capsys.readouterr().out


def test_schema_promote_reports_workflow_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_get_config() -> _ConfigStub:
        return _ConfigStub(db_path=tmp_path / "archive.db")

    def fake_promote(request: SchemaPromoteRequest) -> SchemaPromoteResult:
        raise ValueError(f"missing cluster: {request.cluster_id}")

    monkeypatch.setattr(schema_promote, "get_config", fake_get_config)
    monkeypatch.setattr(schema_promote, "promote_schema_cluster", fake_promote)

    assert schema_promote.main(["--provider", "chatgpt", "--cluster", "missing"]) == 1
    assert "schema-promote: missing cluster: missing" in capsys.readouterr().err
