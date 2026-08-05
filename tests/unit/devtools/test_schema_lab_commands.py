from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from devtools import schema_audit, schema_generate, schema_inspect, schema_promote
from polylogue.core.outcomes import OutcomeCheck, OutcomeStatus
from polylogue.maintenance.schema_inference_gate import (
    run_schema_inference_gate,
    schema_inference_gate_receipt_digest,
)
from polylogue.schemas.audit.models import AuditReport
from polylogue.schemas.generation.models import GenerationResult
from polylogue.schemas.operator.models import (
    SchemaAuditRequest,
    SchemaCompareRequest,
    SchemaExplainRequest,
    SchemaInferRequest,
    SchemaInferResult,
    SchemaListRequest,
    SchemaPromoteRequest,
    SchemaPromoteResult,
)
from polylogue.storage.index_generation import RebuildLease
from tests.unit.maintenance.test_schema_inference_gate import _seed_archive


@dataclass(frozen=True)
class _ConfigStub:
    db_path: Path
    archive_root: Path | None = None


@contextmanager
def _allow_schema_generation(*_args: object, **_kwargs: object) -> Iterator[dict[str, object]]:
    yield {}


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


def test_schema_inspect_list_forwards_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[SchemaListRequest] = []
    sentinel = object()

    def fake_list(request: SchemaListRequest) -> object:
        captured.append(request)
        return sentinel

    rendered: dict[str, object] = {}

    def fake_render(*, provider: str | None, result: object, json_output: bool) -> None:
        rendered.update({"provider": provider, "result": result, "json_output": json_output})

    monkeypatch.setattr(schema_inspect, "list_schemas", fake_list)
    monkeypatch.setattr(schema_inspect, "render_schema_list_result", fake_render)

    assert schema_inspect.list_main(["--provider", "chatgpt", "--json"]) == 0

    assert captured == [SchemaListRequest(provider="chatgpt")]
    assert rendered == {"provider": "chatgpt", "result": sentinel, "json_output": True}


def test_schema_inspect_compare_forwards_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[SchemaCompareRequest] = []
    sentinel = object()

    def fake_compare(request: SchemaCompareRequest) -> object:
        captured.append(request)
        return sentinel

    rendered: dict[str, object] = {}

    def fake_render(*, result: object, json_output: bool, md_output: bool) -> None:
        rendered.update({"result": result, "json_output": json_output, "md_output": md_output})

    monkeypatch.setattr(schema_inspect, "compare_schema_versions", fake_compare)
    monkeypatch.setattr(schema_inspect, "render_schema_compare_result", fake_render)

    assert (
        schema_inspect.compare_main(
            ["--provider", "chatgpt", "--from", "v1", "--to", "v2", "--element", "session_document", "--markdown"]
        )
        == 0
    )

    assert captured == [
        SchemaCompareRequest(
            provider="chatgpt",
            from_version="v1",
            to_version="v2",
            element_kind="session_document",
        )
    ]
    assert rendered == {"result": sentinel, "json_output": False, "md_output": True}


def test_schema_inspect_explain_forwards_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[SchemaExplainRequest] = []
    sentinel = object()

    def fake_explain(request: SchemaExplainRequest) -> object:
        captured.append(request)
        return sentinel

    rendered: dict[str, object] = {}

    def fake_render(*, result: object, json_output: bool, verbose: bool) -> None:
        rendered.update({"result": result, "json_output": json_output, "verbose": verbose})

    monkeypatch.setattr(schema_inspect, "explain_schema", fake_explain)
    monkeypatch.setattr(schema_inspect, "render_schema_explain_result", fake_render)

    assert (
        schema_inspect.explain_main(
            ["--provider", "chatgpt", "--version", "latest", "--element", "session_document", "--review-evidence"]
        )
        == 0
    )

    assert captured == [
        SchemaExplainRequest(
            provider="chatgpt",
            version="latest",
            element_kind="session_document",
            review_evidence=True,
        )
    ]
    assert rendered == {"result": sentinel, "json_output": False, "verbose": False}


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
    monkeypatch.setattr(schema_generate, "authorize_schema_generation", _allow_schema_generation)

    assert (
        schema_generate.main(
            [
                "--provider",
                "chatgpt",
                "--max-samples",
                "2",
                "--schema-inference-receipt",
                str(tmp_path / "receipt.json"),
            ]
        )
        == 0
    )

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


def test_schema_generate_writes_aggregate_progress_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_get_config() -> _ConfigStub:
        return _ConfigStub(db_path=tmp_path / "archive.db")

    def fake_infer(request: SchemaInferRequest) -> SchemaInferResult:
        assert request.progress_callback is not None
        request.progress_callback(
            "observe_and_cluster",
            {
                "phase": "observe_and_cluster",
                "state": "progress",
                "unit_count": 128,
                "units_per_s": 32.0,
            },
        )
        return SchemaInferResult(
            generation=GenerationResult(
                provider=request.provider,
                schema={"type": "object"},
                sample_count=2,
                phase_receipt={"version": 1, "status": "succeeded", "phases": []},
            )
        )

    receipt_path = tmp_path / "receipt.json"
    monkeypatch.setattr(schema_generate, "get_config", fake_get_config)
    monkeypatch.setattr(schema_generate, "infer_schema", fake_infer)
    monkeypatch.setattr(schema_generate, "authorize_schema_generation", _allow_schema_generation)

    assert (
        schema_generate.main(
            [
                "--provider",
                "chatgpt",
                "--progress",
                "--receipt",
                str(receipt_path),
                "--schema-inference-receipt",
                str(tmp_path / "schema-inference-gate-receipt.json"),
            ]
        )
        == 0
    )

    receipt = json.loads(receipt_path.read_text())
    assert receipt["generation"]["status"] == "succeeded"
    assert len(receipt["progress_events"]) == 1
    event = receipt["progress_events"][0]
    assert event["phase"] == "observe_and_cluster"
    assert event["state"] == "progress"
    assert event["unit_count"] == 128
    assert event["units_per_s"] == 32.0
    assert "max_rss_bytes" in event["process"]
    assert receipt["input"] == {
        "index_size_bytes": None,
        "source_db_bytes": None,
        "raw_row_count": None,
        "raw_blob_bytes": None,
    }
    assert "max_rss_bytes" in receipt["process_final"]
    assert receipt["resume"]["status"] == "restart_from_acquisition"
    assert "observe_and_cluster" in capsys.readouterr().err


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
    monkeypatch.setattr(schema_generate, "authorize_schema_generation", _allow_schema_generation)

    assert (
        schema_generate.main(
            ["--provider", "chatgpt", "--cluster", "--schema-inference-receipt", str(tmp_path / "receipt.json")]
        )
        == 1
    )
    assert "No samples found for clustering" in capsys.readouterr().err


def _authoritative_schema_gate(tmp_path: Path, name: str) -> tuple[Path, Path]:
    root = tmp_path / name
    ground_truth = _seed_archive(root)
    receipt = tmp_path / f"{name}-receipt" / "schema-inference-gate-receipt.json"
    result = run_schema_inference_gate(
        root,
        receipt_path=receipt,
        ground_truth_roots={"codex-session": (ground_truth,)},
    )
    assert result.passed, result.payload["pass_fail_reasons"]
    return root, receipt


def _successful_inference(request: SchemaInferRequest) -> SchemaInferResult:
    return SchemaInferResult(
        generation=GenerationResult(
            provider=request.provider,
            schema={"type": "object"},
            sample_count=1,
        )
    )


def test_schema_generate_direct_command_bypass_refuses_without_authoritative_receipt() -> None:
    with pytest.raises(SystemExit):
        schema_generate.main(["--provider", "chatgpt"])


def test_schema_generate_valid_route_consumes_fresh_gate_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, receipt = _authoritative_schema_gate(tmp_path, "valid")
    monkeypatch.setattr(schema_generate, "get_config", lambda: _ConfigStub(root / "index.db", root))
    monkeypatch.setattr(schema_generate, "infer_schema", _successful_inference)

    assert schema_generate.main(["--provider", "chatgpt", "--schema-inference-receipt", str(receipt)]) == 0
    assert "Generated schema package set for chatgpt" in capsys.readouterr().out


def test_schema_generate_rejects_receipt_for_a_different_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _archive_a, receipt_a = _authoritative_schema_gate(tmp_path, "archive-a")
    archive_b, _receipt_b = _authoritative_schema_gate(tmp_path, "archive-b")
    monkeypatch.setattr(schema_generate, "get_config", lambda: _ConfigStub(archive_b / "index.db", archive_b))
    monkeypatch.setattr(schema_generate, "infer_schema", pytest.fail)

    assert schema_generate.main(["--provider", "chatgpt", "--schema-inference-receipt", str(receipt_a)]) == 1
    assert "different archive" in capsys.readouterr().err


@pytest.mark.parametrize("mutation", ["replaced", "stale"])
def test_schema_generate_rejects_replaced_or_stale_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, receipt = _authoritative_schema_gate(tmp_path, mutation)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if mutation == "replaced":
        payload["receipt_nonce"] = "attacker-replacement"
    else:
        payload["generated_at"] = datetime(2020, 1, 1, tzinfo=UTC).isoformat()
        payload["receipt_sha256"] = schema_inference_gate_receipt_digest(payload)
    receipt.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(schema_generate, "get_config", lambda: _ConfigStub(root / "index.db", root))
    monkeypatch.setattr(schema_generate, "infer_schema", pytest.fail)

    assert schema_generate.main(["--provider", "chatgpt", "--schema-inference-receipt", str(receipt)]) == 1
    error = capsys.readouterr().err
    assert "digest mismatch" in error or "stale" in error


def test_schema_generate_rejects_stale_non_source_tier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, receipt = _authoritative_schema_gate(tmp_path, "stale-user-tier")
    with sqlite3.connect(root / "user.db") as connection:
        connection.execute("PRAGMA user_version = 0")
    monkeypatch.setattr(schema_generate, "get_config", lambda: _ConfigStub(root / "index.db", root))
    monkeypatch.setattr(schema_generate, "infer_schema", pytest.fail)

    assert schema_generate.main(["--provider", "chatgpt", "--schema-inference-receipt", str(receipt)]) == 1
    assert "tier identity is stale" in capsys.readouterr().err


def test_schema_generate_refuses_concurrent_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, receipt = _authoritative_schema_gate(tmp_path, "concurrent-writer")
    monkeypatch.setattr(schema_generate, "get_config", lambda: _ConfigStub(root / "index.db", root))
    monkeypatch.setattr(schema_generate, "infer_schema", pytest.fail)

    with RebuildLease(root):
        assert schema_generate.main(["--provider", "chatgpt", "--schema-inference-receipt", str(receipt)]) == 1
    assert "exclusive offline archive ownership" in capsys.readouterr().err


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
