from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from polylogue.cli.shared.qa_finalization import finalize_qa_run
from polylogue.cli.shared.qa_requests import QACaptureMode, QAFinalizationPlan, QASnapshotPlan
from polylogue.showcase.qa_runner import QAResult


def _string_payload(value: object) -> str:
    assert isinstance(value, str)
    return value


def test_finalize_qa_run_emits_json_and_executes_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    emitted: dict[str, object] = {}

    monkeypatch.setattr(
        "polylogue.cli.shared.qa_finalization.render_qa_session",
        lambda result: {"overall_status": "ok", "report_dir": str(result.report_dir)},
    )
    monkeypatch.setattr(
        "polylogue.cli.shared.qa_finalization.click.echo",
        lambda payload: emitted.setdefault("json_payload", payload),
    )
    monkeypatch.setattr(
        "polylogue.cli.shared.qa_finalization.execute_snapshot_plan",
        lambda plan, *, fallback_source_dir, output_root, json_output, env: emitted.update(
            snapshot_plan=plan,
            fallback_source_dir=fallback_source_dir,
            output_root=output_root,
            snapshot_json_output=json_output,
            snapshot_env=env,
        ),
    )
    monkeypatch.setattr(
        "polylogue.cli.shared.qa_finalization.run_vhs_capture",
        lambda env, showcase_result, json_output: emitted.update(
            capture_env=env,
            capture_showcase_result=showcase_result,
            capture_json_output=json_output,
        ),
    )

    showcase_result = MagicMock(output_dir=tmp_path / "showcase")
    result = QAResult(showcase_result=showcase_result, report_dir=tmp_path / "report")
    env = MagicMock()
    plan = QAFinalizationPlan(
        capture_mode=QACaptureMode.VHS,
        json_output=True,
        snapshot_plan=QASnapshotPlan(label="release-v3"),
    )

    finalize_qa_run(result, plan=plan, archive_root=tmp_path / "archive", env=env)

    assert json.loads(_string_payload(emitted["json_payload"]))["overall_status"] == "ok"
    assert emitted["capture_showcase_result"] is showcase_result
    assert emitted["capture_json_output"] is True
    assert emitted["fallback_source_dir"] == tmp_path / "report"
    assert emitted["output_root"] == tmp_path / "archive" / "qa" / "snapshots"
    assert emitted["snapshot_json_output"] is True


def test_finalize_qa_run_prints_summary_without_capture_or_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("polylogue.cli.shared.qa_finalization.render_qa_summary", lambda result: "qa-summary")
    capture = MagicMock()
    snapshot = MagicMock()
    monkeypatch.setattr("polylogue.cli.shared.qa_finalization.run_vhs_capture", capture)
    monkeypatch.setattr("polylogue.cli.shared.qa_finalization.execute_snapshot_plan", snapshot)

    env = MagicMock()
    result = QAResult(exercises_skipped=True, invariants_skipped=True)

    finalize_qa_run(
        result,
        plan=QAFinalizationPlan(capture_mode=QACaptureMode.NONE, json_output=False, snapshot_plan=None),
        archive_root=tmp_path / "archive",
        env=env,
    )

    env.ui.console.print.assert_called_once_with("qa-summary")
    capture.assert_not_called()
    snapshot.assert_not_called()
