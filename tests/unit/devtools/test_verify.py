"""Semantic verifier contracts independent of execution-host lifecycle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from devtools import verify
from devtools.verify_runs import (
    CURRENT_RUN_PATH,
    VerifyRun,
    aggregate_pytest_statistics,
    env_for_pytest_step,
)


def test_quick_steps_are_static_gates() -> None:
    labels = [label for label, _command in verify.build_verify_steps(quick=True, lab=False)]

    assert "ruff check" in labels
    assert "verify oracle-integrity" in labels
    assert not any(label.startswith("pytest") for label in labels)


def test_native_selection_partitions_semantic_lanes() -> None:
    steps = verify.build_verify_steps(
        quick=False,
        lab=False,
        testmon_mode="affected",
        testmon_environment="polylogue-test",
    )

    labels = [label for label, _command in steps]
    assert labels[-3:] == [
        "pytest native parallel (affected)",
        "pytest native serial (affected)",
        "pytest native storage-scale (affected)",
    ]


def test_declared_operation_requires_the_fixed_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SINNIXD_OPERATION", "verify_quick")

    assert verify._declared_agentctl_operation(["--quick"]) == "verify_quick"
    assert verify._declared_agentctl_operation([]) is None


def test_pytest_receipt_decodes_report_and_selection(tmp_path: Path) -> None:
    run = VerifyRun(tier="test", argv=[], git_head="head", root=tmp_path)
    artifacts = run.start_step(label="pytest focused", cmd=[sys.executable, "-m", "pytest"])
    (artifacts.step_dir / "pytest-report.json").write_text(
        json.dumps({"tests": [{"outcome": "passed"}, {"outcome": "failed"}]}),
        encoding="utf-8",
    )
    artifacts.selection_path.write_text(json.dumps({"selected_count": 2, "deselected_count": 1}), encoding="utf-8")
    artifacts.summary_path.write_text(json.dumps({"exitstatus": 1}), encoding="utf-8")
    artifacts.events_dir.mkdir()
    (artifacts.events_dir / "gw0.jsonl").write_text(
        json.dumps({"event": "test_report", "updated_at": "2026-01-01T00:00:00Z"}) + "\n", encoding="utf-8"
    )

    result = run.finish_step(step_id=artifacts.step_id, result={"exit": 1, "duration_s": 0.1})

    assert result is not None
    statistics = aggregate_pytest_statistics(artifacts.step_dir, command=[], step_result={"exit": 1})
    assert statistics["outcomes"] == {"passed": 1, "failed": 1}
    assert statistics["selected_count"] == 2
    assert statistics["event_count"] == 1


def test_step_environment_is_receipt_scoped(tmp_path: Path) -> None:
    run = VerifyRun(tier="test", argv=[], git_head=None, root=tmp_path)
    artifacts = run.start_step(label="pytest focused", cmd=[])

    env = env_for_pytest_step({}, run=run, artifacts=artifacts)

    assert env["POLYLOGUE_VERIFY_RUN_ID"] == run.run_id
    assert env["POLYLOGUE_PYTEST_RUN_ID"].startswith(run.run_id)
    assert Path(env["POLYLOGUE_PYTEST_EVENTS_DIR"]) == artifacts.events_dir


def test_agentctl_verify_run_omits_mutable_current_receipt(tmp_path: Path) -> None:
    """AgentCTL-owned verification does not create the local UI mirror."""
    VerifyRun(tier="all", argv=["--all"], git_head="head", root=tmp_path, mirror_current=False)

    assert not (tmp_path / CURRENT_RUN_PATH).exists()
