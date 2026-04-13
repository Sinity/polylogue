from __future__ import annotations

from pathlib import Path

from polylogue.cli.qa_requests import QACaptureMode, build_qa_snapshot_plan


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
