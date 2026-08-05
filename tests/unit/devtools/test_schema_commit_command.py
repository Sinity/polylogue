"""``devtools lab schema commit`` -- the real, persisting schema-commit CLI.

Unit-level plumbing tests: the request built from CLI args, and rendering of
the ``SchemaCommitResult``. The commit path's actual file-writing behavior is
covered end-to-end in ``tests/unit/schemas/test_operator_commit.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from devtools import schema_commit
from polylogue.schemas.generation.models import GenerationResult
from polylogue.schemas.operator.models import SchemaCommitRequest, SchemaCommitResult, SchemaVersionCommitReport
from polylogue.schemas.operator.receipt import (
    SchemaInferenceCoverageDecision,
    SchemaInferenceReceipt,
)

_HANDOFF = SchemaInferenceReceipt(
    gate_receipt_digest="a" * 64,
    coverage_decisions=(
        SchemaInferenceCoverageDecision(origin="codex-session", provider="codex", decision="committed", reason=None),
    ),
    packages=(),
)


@dataclass(frozen=True)
class _ConfigStub:
    archive_root: Path
    db_path: Path


def test_schema_commit_forwards_request_and_defaults_output_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: list[SchemaCommitRequest] = []

    def fake_get_config() -> _ConfigStub:
        return _ConfigStub(archive_root=tmp_path / "archive", db_path=tmp_path / "archive.db")

    def fake_commit(request: SchemaCommitRequest) -> SchemaCommitResult:
        captured.append(request)
        return SchemaCommitResult(
            provider=request.provider,
            generation=GenerationResult(provider=request.provider, schema={"type": "object"}, sample_count=3),
            versions=(SchemaVersionCommitReport(version="v1", status="new", sample_count=3),),
            dry_run=request.dry_run,
        )

    monkeypatch.setattr(schema_commit, "get_config", fake_get_config)
    monkeypatch.setattr(schema_commit, "commit_provider_schema", fake_commit)

    assert (
        schema_commit.main(["--provider", "chatgpt", "--schema-inference-gate-receipt", str(tmp_path / "gate.json")])
        == 0
    )

    assert len(captured) == 1
    request = captured[0]
    assert request.provider == "chatgpt"
    assert request.output_dir == schema_commit.DEFAULT_OUTPUT_DIR
    assert request.db_path == tmp_path / "archive.db"
    assert request.full_corpus is True
    assert request.dry_run is False
    assert request.schema_inference_gate_receipt_path == tmp_path / "gate.json"


def test_schema_commit_honors_output_dir_and_dry_run_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: list[SchemaCommitRequest] = []

    monkeypatch.setattr(
        schema_commit,
        "get_config",
        lambda: _ConfigStub(archive_root=tmp_path / "archive", db_path=tmp_path / "archive.db"),
    )

    def fake_commit(request: SchemaCommitRequest) -> SchemaCommitResult:
        captured.append(request)
        return SchemaCommitResult(
            provider=request.provider,
            generation=GenerationResult(provider=request.provider, schema={"type": "object"}, sample_count=1),
            versions=(SchemaVersionCommitReport(version="v1", status="unchanged", sample_count=1),),
            dry_run=request.dry_run,
        )

    monkeypatch.setattr(schema_commit, "commit_provider_schema", fake_commit)

    custom_output = tmp_path / "custom-providers"
    assert (
        schema_commit.main(
            [
                "--provider",
                "chatgpt",
                "--output-dir",
                str(custom_output),
                "--dry-run",
                "--no-full-corpus",
                "--schema-inference-gate-receipt",
                str(tmp_path / "gate.json"),
            ]
        )
        == 0
    )

    assert captured[0].output_dir == custom_output
    assert captured[0].dry_run is True
    assert captured[0].full_corpus is False


def test_schema_commit_json_output_reports_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        schema_commit,
        "get_config",
        lambda: _ConfigStub(archive_root=tmp_path / "archive", db_path=tmp_path / "archive.db"),
    )
    monkeypatch.setattr(
        schema_commit,
        "commit_provider_schema",
        lambda request: SchemaCommitResult(
            provider=request.provider,
            generation=GenerationResult(provider=request.provider, schema={"type": "object"}, sample_count=42),
            versions=(
                SchemaVersionCommitReport(
                    version="v2", status="changed", sample_count=42, added_paths=("session_document.new",)
                ),
            ),
            dry_run=False,
            handoff=_HANDOFF,
            handoff_path=tmp_path / "handoff.json",
        ),
    )

    assert (
        schema_commit.main(
            ["--provider", "chatgpt", "--json", "--schema-inference-gate-receipt", str(tmp_path / "gate.json")]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["provider"] == "chatgpt"
    assert payload["success"] is True
    assert payload["narrowed"] is False
    assert payload["sample_count"] == 42
    assert payload["versions"][0]["status"] == "changed"
    assert payload["versions"][0]["added_paths"] == ["session_document.new"]
    assert payload["handoff"]["gate_receipt_digest"] == "a" * 64
    assert payload["handoff_path"] == str(tmp_path / "handoff.json")


def test_schema_commit_exits_nonzero_on_generation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        schema_commit,
        "get_config",
        lambda: _ConfigStub(archive_root=tmp_path / "archive", db_path=tmp_path / "archive.db"),
    )
    monkeypatch.setattr(
        schema_commit,
        "commit_provider_schema",
        lambda request: SchemaCommitResult(
            provider=request.provider,
            generation=GenerationResult(provider=request.provider, schema=None, sample_count=0, error="No samples"),
            versions=(),
            dry_run=False,
        ),
    )

    assert (
        schema_commit.main(
            [
                "--provider",
                "broken-provider",
                "--json",
                "--schema-inference-gate-receipt",
                str(tmp_path / "gate.json"),
            ]
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is False
    assert payload["error"] == "No samples"


def test_schema_commit_exits_nonzero_when_narrowed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A commit that succeeds but narrows a previously-committed type must
    not report a clean exit code -- the whole point of the report is that a
    bad promotion can't land unnoticed."""
    monkeypatch.setattr(
        schema_commit,
        "get_config",
        lambda: _ConfigStub(archive_root=tmp_path / "archive", db_path=tmp_path / "archive.db"),
    )
    monkeypatch.setattr(
        schema_commit,
        "commit_provider_schema",
        lambda request: SchemaCommitResult(
            provider=request.provider,
            generation=GenerationResult(provider=request.provider, schema={"type": "object"}, sample_count=3),
            versions=(
                SchemaVersionCommitReport(
                    version="v1", status="changed", sample_count=3, narrowed_paths=("session_document.timestamp",)
                ),
            ),
            dry_run=False,
        ),
    )

    assert (
        schema_commit.main(["--provider", "chatgpt", "--schema-inference-gate-receipt", str(tmp_path / "gate.json")])
        == 1
    )
