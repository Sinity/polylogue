from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from devtools import schema_audit, schema_generate, schema_inspect, schema_promote
from polylogue.core.outcomes import OutcomeCheck, OutcomeStatus
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
from polylogue.storage.archive_identity import ArchiveLocation


@dataclass(frozen=True)
class _ConfigStub:
    archive_root: Path
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
        return _ConfigStub(archive_root=tmp_path, db_path=tmp_path / "index.db")

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

    assert (
        schema_generate.main(
            [
                "--provider",
                "chatgpt",
                "--max-samples",
                "2",
            ]
        )
        == 0
    )

    assert captured == [
        SchemaInferRequest(
            provider="chatgpt",
            db_path=tmp_path / "index.db",
            archive_location=ArchiveLocation.resolve(tmp_path),
            max_samples=2,
            privacy_config=None,
            cluster=False,
            full_corpus=False,
            persist_cluster_manifest=False,
        )
    ]
    assert "Generated schema package set for chatgpt" in capsys.readouterr().out


def test_schema_generate_cluster_preview_keeps_declared_source_manifest_in_memory(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The preview route may cluster a declared fixture but must not publish it.

    Anti-vacuity: clustered source inference normally persists this manifest
    for promotion. Calling the actual devtools entry point proves its explicit
    request policy prevents that otherwise-live write.
    """
    source = Path(__file__).parents[2] / "fixtures" / "origin-capability" / "codex-session.jsonl"
    manifest_path = workspace_env["data_root"] / "polylogue" / "schemas" / "codex" / "manifest.json"

    assert (
        schema_generate.main(
            [
                "--provider",
                "codex",
                "--source",
                f"codex={source}",
                "--source-cache",
                str(tmp_path / "source-cache.sqlite3"),
                "--source-workers",
                "1",
                "--cluster",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)["result"]
    assert payload["manifest"]["provider"] == "codex"
    assert payload["manifest_path"] is None
    assert not manifest_path.exists()


def test_schema_generate_retained_clusters_are_promotable(
    workspace_env: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``generate --cluster --retain-clusters`` is the producer ``promote`` reads.

    Anti-vacuity: wiring ``--retain-clusters`` back to
    ``persist_cluster_manifest=False`` leaves the registry without a manifest,
    and the promotion below raises ``No cluster manifest found``.
    """
    from polylogue.core.enums import Provider
    from polylogue.core.sources import origin_from_provider
    from polylogue.schemas.operator.workflow import promote_schema_cluster
    from polylogue.storage.blob_store import get_blob_store
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
    from tests.infra.storage_records import db_setup

    index_db = db_setup(workspace_env)
    payload = json.dumps(
        {
            "id": "conversation-1",
            "title": "Schema inference",
            "create_time": 1_700_000_000.0,
            "update_time": 1_700_000_060.0,
            "mapping": {
                "node-1": {
                    "id": "node-1",
                    "parent": None,
                    "children": [],
                    "message": {
                        "id": "message-1",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["infer this schema"]},
                        "create_time": 1_700_000_000.0,
                    },
                }
            },
        }
    ).encode()
    get_blob_store().write_from_bytes(payload)
    with sqlite3.connect(workspace_env["archive_root"] / "source.db") as conn:
        write_source_raw_session(
            conn,
            origin=origin_from_provider(Provider.CHATGPT),
            source_path="/fixtures/chatgpt-export.json",
            canonical_source_path="/fixtures/chatgpt-export.json",
            source_index=0,
            payload=payload,
            acquired_at_ms=1_700_000_000_000,
        )

    assert schema_generate.main(["--provider", "chatgpt", "--cluster", "--retain-clusters", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)["result"]
    assert result["manifest_path"] is not None
    cluster_id = result["manifest"]["clusters"][0]["cluster_id"]

    promoted = promote_schema_cluster(SchemaPromoteRequest(provider="chatgpt", cluster_id=cluster_id, db_path=index_db))

    assert promoted.cluster_id == cluster_id
    assert promoted.package_version


def test_schema_generate_refuses_retain_clusters_without_cluster(workspace_env: dict[str, Path]) -> None:
    """Retaining is meaningless without clustering; the flag is refused, not ignored."""
    with pytest.raises(SystemExit) as exit_info:
        schema_generate.main(["--provider", "codex", "--retain-clusters"])
    assert exit_info.value.code == 2


def test_schema_generate_writes_aggregate_progress_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_get_config() -> _ConfigStub:
        return _ConfigStub(archive_root=tmp_path, db_path=tmp_path / "index.db")

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

    assert (
        schema_generate.main(
            [
                "--provider",
                "chatgpt",
                "--progress",
                "--receipt",
                str(receipt_path),
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


def test_schema_generate_preview_binds_external_active_index_to_configured_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A preview must use its configured archive, not generation siblings or ambient blobs."""
    from polylogue.core.sources import origin_from_provider
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session

    def seed(root: Path, label: str) -> tuple[ArchiveLocation, bytes]:
        initialize_active_archive_root(root)
        payload = json.dumps([{"id": label, "mapping": {}}]).encode()
        BlobStore(root / "blob").write_from_bytes(payload)
        with sqlite3.connect(root / "source.db") as connection:
            write_source_raw_session(
                connection,
                origin=origin_from_provider("chatgpt"),
                source_path=f"/{label}.json",
                canonical_source_path=f"/{label}.json",
                source_index=0,
                payload=payload,
                acquired_at_ms=0,
            )
        external_index = root / ".index-generations" / "current" / "index.db"
        external_index.parent.mkdir(parents=True)
        external_index.write_bytes((root / "index.db").read_bytes())
        (root / ".index-active-pointer").write_text(str(external_index), encoding="utf-8")
        return ArchiveLocation.resolve(root), payload

    selected, selected_payload = seed(tmp_path / "selected", "selected")
    seed(tmp_path / "ambient", "ambient")
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "ambient"))
    monkeypatch.setattr(
        schema_generate,
        "get_config",
        lambda: _ConfigStub(archive_root=selected.configured_root, db_path=selected.active_index_path),
    )

    receipt_path = tmp_path / "preview-receipt.json"
    assert schema_generate.main(["--provider", "chatgpt", "--receipt", str(receipt_path)]) == 0

    receipt = json.loads(receipt_path.read_text())
    assert receipt["input"]["raw_row_count"] == 1
    assert receipt["input"]["raw_blob_bytes"] == len(selected_payload)
    assert receipt["input"]["source_db_bytes"] is not None
    assert receipt["generation"]["status"] == "succeeded"


def test_schema_generate_cluster_without_manifest_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_get_config() -> _ConfigStub:
        return _ConfigStub(archive_root=tmp_path, db_path=tmp_path / "index.db")

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
        return _ConfigStub(archive_root=tmp_path, db_path=tmp_path / "index.db")

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
    registry_root = tmp_path / "schemas"
    registry_root.mkdir()
    (registry_root / "schema.json").write_text('{"type": "object"}', encoding="utf-8")
    monkeypatch.setattr(schema_promote, "_schema_registry_root", lambda: registry_root)

    assert (
        schema_promote.main(["--provider", "chatgpt", "--cluster", "cluster-1", "--with-samples", "--max-samples", "7"])
        == 0
    )

    assert captured == [
        SchemaPromoteRequest(
            provider="chatgpt",
            cluster_id="cluster-1",
            db_path=tmp_path / "index.db",
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
        return _ConfigStub(archive_root=tmp_path, db_path=tmp_path / "index.db")

    def fake_promote(request: SchemaPromoteRequest) -> SchemaPromoteResult:
        raise ValueError(f"missing cluster: {request.cluster_id}")

    monkeypatch.setattr(schema_promote, "get_config", fake_get_config)
    monkeypatch.setattr(schema_promote, "promote_schema_cluster", fake_promote)

    assert schema_promote.main(["--provider", "chatgpt", "--cluster", "missing"]) == 1
    assert "schema-promote: missing cluster: missing" in capsys.readouterr().err


def test_promotion_audits_the_tree_promotion_writes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The post-promotion audit root is the registry's storage root.

    ``promote_schema_cluster`` writes through
    ``polylogue.schemas.operator.registry.schema_registry()``, whose
    ``storage_root`` is ``data_home()/schemas`` -- an XDG path. The audit used
    the installed ``polylogue/schemas`` package directory instead, so it
    inspected bundled artifacts promotion never touched and a malformed or
    privacy-unsafe newly promoted artifact returned success uninspected.

    Anti-vacuity: restore ``Path(next(iter(polylogue.schemas.__path__)))`` and
    the first assertion fails, because that path is inside the checkout and
    does not move with ``XDG_DATA_HOME``. The second assertion pins the
    opposite direction: a helper that returned any tmp path would satisfy the
    first one, so the value must be the registry's own storage root.
    """
    from polylogue.schemas.registry import SchemaRegistry

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))

    observed = schema_promote._schema_registry_root()

    assert observed == tmp_path / "share" / "polylogue" / "schemas"
    assert observed == SchemaRegistry().storage_root


def test_schema_generate_handles_missing_archive_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A RuntimeError escapes this production command instead of producing exit code 1."""
    from polylogue.schemas.sampling_db import SchemaArchiveEvidenceError

    monkeypatch.setattr(
        schema_generate, "get_config", lambda: _ConfigStub(archive_root=tmp_path, db_path=tmp_path / "index.db")
    )

    def refuse(request: SchemaInferRequest) -> SchemaInferResult:
        raise SchemaArchiveEvidenceError("synthetic missing source tier")

    monkeypatch.setattr(schema_generate, "infer_schema", refuse)
    assert schema_generate.main(["--provider", "chatgpt"]) == 1
    assert "synthetic missing source tier" in capsys.readouterr().err


def test_schema_module_cli_preserves_configured_archive_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The module CLI binds evidence to the configured root, not the generation directory.

    Anti-vacuity: drop the ``archive_location`` argument and the real binder
    resolves the generation directory as the archive root and refuses.
    """
    from polylogue.schemas.operator import schema_inference
    from polylogue.schemas.sampling_db import _schema_archive_location
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    external_index = root / ".index-generations" / "generation-1" / "index.db"
    external_index.parent.mkdir(parents=True)
    external_index.write_bytes((root / "index.db").read_bytes())
    (root / ".index-active-pointer").write_text(str(external_index), encoding="utf-8")
    selected = ArchiveLocation.resolve(root)
    monkeypatch.setattr(
        schema_inference, "get_config", lambda: _ConfigStub(archive_root=root, db_path=selected.active_index_path)
    )
    observed: list[ArchiveLocation] = []

    def generate(
        *, db_path: Path, archive_location: ArchiveLocation | None = None, **kwargs: object
    ) -> list[GenerationResult]:
        observed.append(_schema_archive_location(db_path=db_path, archive_location=archive_location))
        return []

    monkeypatch.setattr(schema_inference, "generate_all_schemas", generate)
    assert schema_inference.cli_main(["--provider", "chatgpt", "--output-dir", str(tmp_path / "output")]) == 0
    assert observed == [selected]
