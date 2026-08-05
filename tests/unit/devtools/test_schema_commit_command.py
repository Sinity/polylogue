"""``devtools lab schema commit`` -- the real, persisting schema-commit CLI.

Unit-level plumbing tests: the request built from CLI args, and rendering of
the ``SchemaCommitResult``. The commit path's actual file-writing behavior is
covered end-to-end in ``tests/unit/schemas/test_operator_commit.py``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

from devtools import schema_commit
from polylogue.schemas.generation.models import GenerationResult
from polylogue.schemas.operator.models import SchemaCommitRequest, SchemaCommitResult, SchemaVersionCommitReport


@dataclass(frozen=True)
class _ConfigStub:
    db_path: Path


@contextmanager
def _allow_schema_generation(*_args: object, **_kwargs: object) -> Iterator[dict[str, object]]:
    yield {}


def test_schema_commit_forwards_request_and_defaults_output_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: list[SchemaCommitRequest] = []

    def fake_get_config() -> _ConfigStub:
        return _ConfigStub(db_path=tmp_path / "archive.db")

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
    authorization_calls: list[tuple[object, ...]] = []

    @contextmanager
    def allow_schema_generation(*args: object, **_kwargs: object) -> Iterator[dict[str, object]]:
        authorization_calls.append(args)
        yield {}

    monkeypatch.setattr(schema_commit, "authorize_schema_generation", allow_schema_generation)

    assert (
        schema_commit.main(["--provider", "chatgpt", "--schema-inference-receipt", str(tmp_path / "receipt.json")]) == 0
    )

    assert len(captured) == 1
    request = captured[0]
    assert request.provider == "chatgpt"
    assert request.output_dir == schema_commit.DEFAULT_OUTPUT_DIR
    assert request.db_path == tmp_path / "archive.db"
    assert request.full_corpus is True
    assert request.dry_run is False
    assert authorization_calls == [(tmp_path, tmp_path / "receipt.json")]


def test_schema_commit_refuses_persistence_without_authoritative_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(schema_commit, "get_config", lambda: _ConfigStub(db_path=tmp_path / "archive.db"))
    monkeypatch.setattr(schema_commit, "commit_provider_schema", pytest.fail)

    assert schema_commit.main(["--provider", "chatgpt"]) == 1
    assert "schema-inference-receipt is required" in capsys.readouterr().err


def test_schema_commit_honors_output_dir_and_dry_run_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: list[SchemaCommitRequest] = []

    monkeypatch.setattr(schema_commit, "get_config", lambda: _ConfigStub(db_path=tmp_path / "archive.db"))

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
            ["--provider", "chatgpt", "--output-dir", str(custom_output), "--dry-run", "--no-full-corpus"]
        )
        == 0
    )

    assert captured[0].output_dir == custom_output
    assert captured[0].dry_run is True
    assert captured[0].full_corpus is False


def test_schema_commit_json_output_reports_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(schema_commit, "get_config", lambda: _ConfigStub(db_path=tmp_path / "archive.db"))
    monkeypatch.setattr(schema_commit, "authorize_schema_generation", _allow_schema_generation)
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
        ),
    )

    assert (
        schema_commit.main(
            ["--provider", "chatgpt", "--json", "--schema-inference-receipt", str(tmp_path / "receipt.json")]
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


def test_schema_commit_exits_nonzero_on_generation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(schema_commit, "get_config", lambda: _ConfigStub(db_path=tmp_path / "archive.db"))
    monkeypatch.setattr(schema_commit, "authorize_schema_generation", _allow_schema_generation)
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
            ["--provider", "broken-provider", "--json", "--schema-inference-receipt", str(tmp_path / "receipt.json")]
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
    monkeypatch.setattr(schema_commit, "get_config", lambda: _ConfigStub(db_path=tmp_path / "archive.db"))
    monkeypatch.setattr(schema_commit, "authorize_schema_generation", _allow_schema_generation)
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
        schema_commit.main(["--provider", "chatgpt", "--schema-inference-receipt", str(tmp_path / "receipt.json")]) == 1
    )
