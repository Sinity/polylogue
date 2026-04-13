from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from polylogue.cli.qa_requests import QACaptureMode, QASnapshotPlan, build_qa_snapshot_plan
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


def test_execute_snapshot_plan_uses_fallback_report_dir(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_snapshot_results(source_dir, *, label, output_root, json_output, env) -> None:
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
