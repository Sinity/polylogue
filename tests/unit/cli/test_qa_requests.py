from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from polylogue.cli.qa_requests import (
    QACaptureMode,
    QASnapshotPlan,
    build_qa_finalization_plan,
    build_qa_invocation_plan,
    build_qa_snapshot_plan,
)
from polylogue.cli.qa_snapshot import execute_snapshot_plan


def test_build_qa_snapshot_plan_defaults_label_for_snapshot_from(tmp_path: Path) -> None:
    source = tmp_path / "existing"
    source.mkdir()

    plan = build_qa_snapshot_plan(snapshot_label=None, snapshot_from=source)

    assert plan is not None
    assert plan.label == "snapshot"
    assert plan.source_dir == source
    assert plan.skips_qa is True
    assert plan.resolve_source_dir(None) == source


def test_build_qa_snapshot_plan_uses_explicit_label_without_snapshot_from() -> None:
    plan = build_qa_snapshot_plan(snapshot_label="release-v3", snapshot_from=None)

    assert plan is not None
    assert plan.label == "release-v3"
    assert plan.source_dir is None
    assert plan.skips_qa is False
    assert plan.resolve_source_dir(Path("/tmp/report")) == Path("/tmp/report")


def test_capture_mode_values_match_cli_surface() -> None:
    assert tuple(mode.value for mode in QACaptureMode) == ("none", "vhs")


def test_build_qa_finalization_plan_preserves_typed_intent(tmp_path: Path) -> None:
    snapshot_plan = QASnapshotPlan(label="release-v3", source_dir=tmp_path / "report")

    plan = build_qa_finalization_plan(
        capture_mode=QACaptureMode.VHS,
        json_output=True,
        snapshot_plan=snapshot_plan,
    )

    assert plan.capture_mode is QACaptureMode.VHS
    assert plan.json_output is True
    assert plan.snapshot_plan is snapshot_plan


def test_build_qa_invocation_plan_normalizes_snapshot_only(tmp_path: Path) -> None:
    source_dir = tmp_path / "report"
    source_dir.mkdir()

    plan = build_qa_invocation_plan(
        synthetic=True,
        source_names=None,
        fresh=None,
        ingest=None,
        regenerate_schemas=False,
        only_stage=None,
        skip_stages=(),
        workspace=None,
        report_dir=None,
        verbose=False,
        fail_fast=False,
        tier_filter=None,
        capture="vhs",
        json_output=True,
        snapshot_label=None,
        snapshot_from=source_dir,
    )

    assert plan.snapshot_only is True
    assert plan.session_request is None
    assert plan.finalization_plan.capture_mode is QACaptureMode.VHS
    assert plan.finalization_plan.snapshot_plan is plan.snapshot_plan


def test_build_qa_invocation_plan_builds_session_request_for_regular_run() -> None:
    plan = build_qa_invocation_plan(
        synthetic=True,
        source_names=None,
        fresh=None,
        ingest=None,
        regenerate_schemas=False,
        only_stage="audit",
        skip_stages=(),
        workspace=None,
        report_dir=None,
        verbose=False,
        fail_fast=False,
        tier_filter=None,
        capture="none",
        json_output=False,
        snapshot_label="release-v3",
        snapshot_from=None,
    )

    assert plan.snapshot_only is False
    assert plan.session_request is not None
    assert plan.session_request.skip_audit is False
    assert plan.session_request.skip_proof is True
    assert plan.finalization_plan.snapshot_plan is not None


def test_execute_snapshot_plan_uses_fallback_report_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_snapshot_results(
        source_dir: Path,
        *,
        label: str,
        output_root: Path,
        json_output: bool,
        env: Any,
    ) -> None:
        captured.update(
            source_dir=source_dir,
            label=label,
            output_root=output_root,
            json_output=json_output,
            env=env,
        )

    monkeypatch.setattr("polylogue.cli.qa_snapshot.snapshot_results", _fake_snapshot_results)

    result = execute_snapshot_plan(
        QASnapshotPlan(label="release-v3"),
        fallback_source_dir=tmp_path / "report",
        output_root=tmp_path / "snapshots",
        json_output=True,
        env=MagicMock(),
    )

    assert result is True
    assert captured["source_dir"] == tmp_path / "report"
    assert captured["label"] == "release-v3"


def test_execute_snapshot_plan_returns_false_without_source_dir(tmp_path: Path) -> None:
    result = execute_snapshot_plan(
        QASnapshotPlan(label="release-v3"),
        fallback_source_dir=None,
        output_root=tmp_path / "snapshots",
        json_output=False,
        env=MagicMock(),
    )

    assert result is False
