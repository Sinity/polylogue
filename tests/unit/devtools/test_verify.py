"""Semantic verifier contracts independent of execution-host lifecycle."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from devtools import required_gate, verify, verify_runs
from devtools.testmon_bootstrap import NativeTestmonRepairError
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


def test_quick_missing_ruff_is_a_named_failed_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    history: dict[str, Any] = {}
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify, "assert_polylogue_matches_checkout", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(verify, "git_head", lambda _root: "head")
    monkeypatch.setattr(verify, "build_verify_steps", lambda **_kwargs: [("ruff check", ["ruff", "check"])])
    monkeypatch.setattr(required_gate.shutil, "which", lambda name, path=None: None if name == "ruff" else "/bin/true")  # type: ignore[attr-defined]
    monkeypatch.setattr(verify, "append_verify_history", lambda payload: history.update(payload))

    assert verify._main(["--quick"]) == 127
    step = history["steps"][0]
    assert step["diagnosis"] == "gate_missing_executable"
    assert step["required_gate"]["gate_passed"] is False
    assert history["diagnosis"] == "gate_missing_executable"


def test_required_gate_subprocess_launch_failure_is_typed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    history: dict[str, Any] = {}
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify, "assert_polylogue_matches_checkout", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(verify, "git_head", lambda _root: "head")
    monkeypatch.setattr(verify, "build_verify_steps", lambda **_kwargs: [("ruff check", ["ruff", "check"])])
    monkeypatch.setattr(required_gate.shutil, "which", lambda *_args, **_kwargs: "/bin/ruff")  # type: ignore[attr-defined]
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("ruff")),
    )
    monkeypatch.setattr(verify, "append_verify_history", lambda payload: history.update(payload))

    assert verify._main(["--quick"]) == 127
    assert history["diagnosis"] == "gate_subprocess_launch_failed"
    assert history["steps"][0]["error"] == "ruff"


def test_render_all_diagnosis_is_carried_into_existing_receipt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    run = VerifyRun(tier="quick", argv=["--quick"], git_head="head", root=tmp_path)
    command = [
        sys.executable,
        "-c",
        "import sys; print('render all: cli-reference failed; diagnosis: render_input_missing', file=sys.stderr); sys.exit(1)",
    ]

    exit_code, _elapsed, metadata = verify._run("render all", command, run=run)

    assert exit_code == 1
    assert metadata["diagnosis"] == "render_input_missing"
    payload = json.loads((tmp_path / str(run.relative_run_dir) / "run.json").read_text(encoding="utf-8"))
    assert payload["steps"][0]["diagnosis"] == "render_input_missing"


def test_early_gate_failure_exit_is_authoritative() -> None:
    result = verify._early_gate_failure_result(0.0, {"exit": 0, "diagnosis": "gate_missing_executable"})

    assert result["exit"] == 127


def test_finish_step_does_not_retry_unavailable_pytest_statistics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mutation: retrying the same unavailable evidence reader repeats an input failure."""
    run = VerifyRun(tier="test", argv=[], git_head="head", root=tmp_path)
    artifacts = run.start_step(label="pytest focused", cmd=["pytest"])
    calls = 0

    def unavailable(*_args: object, **_kwargs: object) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise OSError("synthetic unavailable report")

    monkeypatch.setattr(verify_runs, "aggregate_pytest_statistics", unavailable)

    result = run.finish_step(step_id=artifacts.step_id, result={"exit": 1, "duration_s": 0.1})

    assert result is not None
    assert calls == 1
    assert "statistics" not in result


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
    (artifacts.step_dir / "scratch-metrics.json").write_text(
        json.dumps({"high_water_usage": {"apparent_bytes": 128}}), encoding="utf-8"
    )

    result = run.finish_step(step_id=artifacts.step_id, result={"exit": 1, "duration_s": 0.1})

    assert result is not None
    statistics = aggregate_pytest_statistics(artifacts.step_dir, command=[], step_result={"exit": 1})
    assert statistics["outcomes"] == {"passed": 1, "failed": 1}
    assert statistics["selected_count"] == 2
    assert statistics["event_count"] == 1
    assert result["scratch_metrics"]["high_water_usage"]["apparent_bytes"] == 128


def test_zero_exit_without_a_report_is_a_failed_pytest_step(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify, "scratch_root_from_environment", lambda _env: tmp_path / "scratch")
    monkeypatch.setattr(verify, "_clear_pytest_report", lambda _command: None)
    monkeypatch.setattr(
        verify,
        "run_managed_pytest",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(["pytest"], 0),
    )
    run = VerifyRun(tier="test", argv=[], git_head="head", root=tmp_path)

    exit_code, _elapsed, metadata = verify._run("pytest native serial (affected)", ["pytest"], run=run)

    assert exit_code != 0
    assert metadata["diagnosis"] == "pytest_no_report"
    assert metadata["statistics"]["ordinary_eligible"] is False


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


def test_verify_persists_terminal_receipt_when_outer_deadline_sends_sigterm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    history: dict[str, Any] = {}

    def interrupt(
        label: str,
        command: list[str],
        *,
        run: VerifyRun,
    ) -> tuple[int, float, dict[str, object]]:
        run.start_step(label=label, cmd=command)
        raise verify.VerificationInterrupted(signal.SIGTERM)

    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify, "assert_polylogue_matches_checkout", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(verify, "git_head", lambda _root: "head")
    monkeypatch.setattr(verify, "build_verify_steps", lambda **_kwargs: [("pytest native parallel (all)", ["pytest"])])
    monkeypatch.setattr(verify, "_run", interrupt)
    monkeypatch.setattr(verify, "append_verify_history", lambda payload: history.update(payload))

    assert verify._main(["--quick"]) == 143

    run_payload = json.loads((tmp_path / str(history["artifact_dir"]) / "run.json").read_text())
    current_payload = json.loads((tmp_path / CURRENT_RUN_PATH).read_text())
    for payload in (history, run_payload, current_payload):
        assert payload["status"] == "failed"
        assert payload["diagnosis"] == "verification_interrupted"
        assert payload["exit_code"] == 143
        assert payload["pytest_aggregate"]["termination_reason"] == "sigterm"
        assert payload["steps"][0]["status"] == "failed"
        assert payload["steps"][0]["termination_reason"] == "sigterm"


@pytest.mark.parametrize("early_exit", ("preparation", "bootstrap"))
def test_verify_records_early_native_terminal_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    early_exit: str,
) -> None:
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify, "assert_polylogue_matches_checkout", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(verify, "git_head", lambda _root: "head")
    monkeypatch.setattr(verify, "_git_commit", lambda _ref: "base")
    monkeypatch.setattr(verify, "_changed_paths", lambda _base, _head: ())
    monkeypatch.setattr(
        verify,
        "classify_native_testmon_changes",
        lambda *_args: SimpleNamespace(executable_paths=(), runtime_data_paths=()),
    )
    if early_exit == "preparation":

        def fail_preparation(*_args: object, **_kwargs: object) -> object:
            raise NativeTestmonRepairError("synthetic preparation failure")

        monkeypatch.setattr(verify, "prepare_native_testmon_environment", fail_preparation)
        expected_code = 125
    else:
        monkeypatch.setattr(
            verify,
            "prepare_native_testmon_environment",
            lambda *_args, **_kwargs: SimpleNamespace(
                selection_mode="bootstrap",
                environment_name="testmon",
                local_state=SimpleNamespace(
                    status="absent",
                    reason="synthetic bootstrap",
                    missing_executable_paths=(),
                ),
            ),
        )
        expected_code = 2

    assert verify._main([]) == expected_code
    history = tmp_path / ".cache" / "verify" / "history.jsonl"
    rows = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["exit_code"] == expected_code


def test_verify_emits_shared_workload_receipt_for_step_timing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    history: dict[str, Any] = {}

    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify, "assert_polylogue_matches_checkout", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(verify, "git_head", lambda _root: "head")
    monkeypatch.setattr(verify, "build_verify_steps", lambda **_kwargs: [("ruff check", ["ruff", "check"])])
    monkeypatch.setattr(verify, "_run", lambda *_args, **_kwargs: (0, 0.25, {"diagnosis": "gate_passed"}))
    monkeypatch.setattr(verify, "append_verify_history", lambda payload: history.update(payload))

    assert verify._main(["--quick"]) == 0

    receipt = history["workload_receipt"]
    assert receipt["spec"]["workload_id"] == "devtools:verify:quick"
    assert receipt["spec"]["measurement_scope"] == "process-tree"
    assert receipt["phases"] == [
        {
            "name": "ruff check",
            "measurement_scope": None,
            "wall_ms": 250.0,
            "cleanup_complete": None,
            "quiescent": False,
            "unavailable": list(verify._UNMEASURED_WORKLOAD_DIMENSIONS),
        }
    ]
