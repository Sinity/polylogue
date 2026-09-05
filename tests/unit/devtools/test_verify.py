"""Semantic verifier contracts independent of execution-host lifecycle."""

from __future__ import annotations

import io
import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import tomllib

from devtools import (
    agent_env,
    gate,
    required_gate,
    verify,
    verify_runs,
    why,
)
from devtools.testmon_provision import TestmonGraphStatus
from devtools.verification_result import declared_verification_result
from devtools.verify_runs import (
    CURRENT_RUN_PATH,
    VerifyRun,
    aggregate_pytest_statistics,
    env_for_pytest_step,
)

#: `_rerun_failed_once` and `_run` acquire the host's single pytest slot before
#: executing. These tests drive that code inline through its documented escape
#: rather than requiring a live pueue queue.
_SLOT_HELD_ENV = {"POLYLOGUE_PYTEST_SLOT": "held"}


def test_corpus_workers_default_to_the_corpus_width(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset means the corpus width; an explicit value, ``0`` included, wins.

    Anti-vacuity: reading the variable with a ``"0"`` default (the previous
    code) makes the unset case yield ``-n 0`` and fails the first assertion;
    ignoring the variable makes the override cases fail.
    """
    monkeypatch.delenv("POLYLOGUE_PYTEST_WORKERS", raising=False)
    assert verify._pytest_worker_args(maximum=verify.CORPUS_MAX_WORKERS)[-1] == str(verify.CORPUS_MAX_WORKERS)

    monkeypatch.setenv("POLYLOGUE_PYTEST_WORKERS", "0")
    assert verify._pytest_worker_args(maximum=verify.CORPUS_MAX_WORKERS)[-1] == "0"

    monkeypatch.setenv("POLYLOGUE_PYTEST_WORKERS", "1")
    assert verify._pytest_worker_args(maximum=verify.CORPUS_MAX_WORKERS)[-1] == "1"

    monkeypatch.setenv("POLYLOGUE_PYTEST_WORKERS", "64")
    assert verify._pytest_worker_args(maximum=verify.CORPUS_MAX_WORKERS)[-1] == str(verify.CORPUS_MAX_WORKERS)


def test_quick_steps_are_static_gates() -> None:
    labels = [label for label, _command in verify.build_verify_steps(quick=True)]

    assert "gate lint" in labels
    assert "gate oracle-integrity" in labels
    assert not any(label.startswith("pytest") for label in labels)


def test_verification_tools_are_absolute_paths_in_checkout_venv() -> None:
    steps = verify.build_verify_steps(quick=True)
    commands = dict(steps)

    assert commands["gate format"][0] == str(verify.ROOT / ".venv/bin/ruff")
    assert commands["gate lint"][0] == str(verify.ROOT / ".venv/bin/ruff")
    assert commands["gate mypy"][0].startswith(str(verify.ROOT / ".venv/bin/"))
    assert commands["gate generated-surfaces"][0] == str(verify.ROOT / ".venv/bin/python")
    assert commands["gate schema-privacy"][0] == str(verify.ROOT / ".venv/bin/python")


def test_missing_checkout_venv_tool_is_a_typed_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    history: dict[str, Any] = {}
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify, "assert_polylogue_matches_checkout", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(verify, "git_head", lambda _root: "head")
    monkeypatch.setattr(
        verify, "build_verify_steps", lambda **_kwargs: [("gate lint", [str(tmp_path / ".venv/bin/ruff")])]
    )
    monkeypatch.setattr(verify, "append_verify_history", lambda payload: history.update(payload))

    assert verify._main(["--quick"]) == 127
    assert history["diagnosis"] == "gate_missing_executable"
    assert history["steps"][0]["required_gate"]["executable"] == str(tmp_path / ".venv/bin/ruff")


def test_broken_venv_script_shebang_is_typed_as_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    script = tmp_path / ".venv/bin/mypy"
    script.parent.mkdir(parents=True)
    script.write_text("#!/missing/interpreter\n", encoding="utf-8")
    script.chmod(0o755)

    result = required_gate.executable_gate_result([str(script)], gate="mypy")

    assert result.diagnosis == "gate_missing_executable"


def test_mypy_gate_uses_a_foreground_checkout_local_process() -> None:
    assert gate.mypy_command() == [str(verify.ROOT / ".venv/bin/mypy")]


def test_removed_lab_mode_is_not_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ordinary caller reaches argparse for the removed option.

    Anti-vacuity: if managed-agent detection remains active, ``_main`` returns
    the agent-tier refusal before argparse and this assertion fails to protect
    the removed-option contract.
    """
    monkeypatch.delenv(agent_env.AGENT_PRINCIPAL_ENV, raising=False)
    monkeypatch.setattr(agent_env, "_inside_agent_cgroup", lambda _reader: False)

    with pytest.raises(SystemExit) as raised:
        verify._main(["--lab"])

    assert raised.value.code == 2


def test_quick_missing_ruff_is_a_named_failed_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    history: dict[str, Any] = {}
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify, "assert_polylogue_matches_checkout", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(verify, "git_head", lambda _root: "head")
    monkeypatch.setattr(verify, "build_verify_steps", lambda **_kwargs: [("gate lint", ["ruff", "check"])])
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
    monkeypatch.setattr(verify, "build_verify_steps", lambda **_kwargs: [("gate lint", ["ruff", "check"])])
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


def test_actual_render_all_diagnosis_reaches_receipt_and_why(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    run = VerifyRun(tier="quick", argv=["--quick"], git_head="head", root=tmp_path)
    checkout = Path(__file__).resolve().parents[3]
    missing_input = tmp_path / "missing.py"
    script = f"""
import sys
sys.path.insert(0, {str(checkout)!r})
from devtools import render_all


class MissingSurface:
    name = "cli-reference"
    inputs = ({str(missing_input)!r},)

    @staticmethod
    def main(_argv):
        return 0


render_all.GENERATED_SURFACES = (MissingSurface(),)
raise SystemExit(render_all.main())
"""
    command = [
        sys.executable,
        "-c",
        f"exec({script!r})",
    ]

    exit_code, elapsed, metadata = verify._run("gate generated-surfaces", command, run=run)

    assert exit_code == 1
    assert metadata["diagnosis"] == "render_input_missing"
    output = (tmp_path / str(metadata["output_path"])).read_text(encoding="utf-8")
    assert "diagnosis: render_input_missing " in output
    assert "render_input_missing;" not in output

    payload = run.finish(exit_code=exit_code, duration_s=elapsed, diagnosis=metadata["diagnosis"])
    assert payload["steps"][0]["diagnosis"] == "render_input_missing"
    stream = io.StringIO()
    why._render(payload, stream)
    rendered = stream.getvalue()
    assert "diagnosis: render_input_missing" in rendered
    assert "Restore the declared input" in rendered


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


def test_every_managed_pytest_run_traces_under_the_full_hypothesis_profile() -> None:
    """Graph writers share one profile; affected runs write edges too.

    Anti-vacuity: leaving a shell's HYPOTHESIS_PROFILE=ci in place lets a run
    report green under a reduced property-test budget.
    """
    env = {"HYPOTHESIS_PROFILE": "ci", "POLYLOGUE_CI": "1"}
    verify._normalize_managed_pytest_environment(env)
    assert env["HYPOTHESIS_PROFILE"] == "default"
    assert "POLYLOGUE_CI" not in env


def test_full_corpus_aggregate_sums_disjoint_lanes() -> None:
    aggregate = verify._aggregate_pytest_results(
        [
            {
                "name": "pytest parallel (all)",
                "statistics": {
                    "selected_count": 27,
                    "terminal_count": 27,
                    "outcomes": {"passed": 23, "skipped": 1, "xfailed": 1},
                },
            },
            {
                "name": "pytest serial (all)",
                "statistics": {"selected_count": 2, "terminal_count": 2, "outcomes": {"passed": 2}},
            },
            {
                "name": "pytest storage-scale (all)",
                "statistics": {"selected_count": 1, "terminal_count": 1, "outcomes": {"passed": 1}},
            },
        ],
        expected_step_count=3,
        mode="all",
        exit_code=0,
    )

    assert aggregate == {
        "selection_mode": "all",
        "selected_union_count": 30,
        "terminal_union_count": 30,
        "outcomes": {"passed": 26, "skipped": 1, "xfailed": 1},
        "terminal_green": True,
        "complete_corpus_covered": True,
    }


def test_declared_operation_requires_the_fixed_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SINNIXD_OPERATION", "verify_quick")

    assert verify._declared_agentctl_operation(["--quick"]) == "verify_quick"
    assert verify._declared_agentctl_operation([]) is None


def test_verify_quick_descriptor_accepts_the_declared_json_projection() -> None:
    descriptor = tomllib.loads((verify.ROOT / ".agentctl/project.toml").read_text(encoding="utf-8"))

    operation = descriptor["operations"]["verify_quick"]
    affected = descriptor["operations"]["verify_affected"]
    complete = descriptor["operations"]["verify_all"]
    projection = declared_verification_result(
        {"exit_code": 0, "status": "success", "verification_scope": "non-test"},
        operation="verify_quick",
    )

    assert operation["exec"] == ["devtools", "verify", "--quick"]
    assert operation["result"] == "json"
    assert affected["exec"] == ["env", "POLYLOGUE_PYTEST_WORKERS=2", "devtools", "verify"]
    assert affected["pool"] == "pytest"
    assert affected["result"] == "pytest"
    assert complete["exec"] == ["env", "POLYLOGUE_PYTEST_WORKERS=2", "devtools", "verify", "--all"]
    assert complete["checkout"] == "default"
    assert complete["schedule"] == "*-*-* 03:17:00"
    assert complete["pool"] == "pytest"
    assert projection["kind"] == "polylogue.verification-result"
    assert projection["operation"] == "verify_quick"


def test_descriptor_only_changes_use_contract_tests_and_python_changes_use_testmon() -> None:
    """A descriptor-only diff is bounded to descriptor contracts.

    Anti-vacuity: selecting the affected mode for the descriptor or selecting
    descriptor contracts for a Python diff makes one of the boundary checks
    below fail.
    """
    assert verify._selection_for_changes(frozenset({".agentctl/project.toml"})) == "descriptor"
    assert verify._selection_for_changes(frozenset({"polylogue/example.py"})) == "affected"
    assert verify._selection_for_changes(frozenset({".agentctl/project.toml", "polylogue/example.py"})) == "affected"
    assert verify._selection_for_changes(None) == "affected"

    descriptor_command = verify._pytest_steps(selection="descriptor", worker_args=[])[0][1]
    assert "--testmon" not in descriptor_command
    assert "tests" not in descriptor_command
    assert descriptor_command[-len(verify.DESCRIPTOR_CONTRACT_TESTS) :] == list(verify.DESCRIPTOR_CONTRACT_TESTS)

    affected_command = verify._pytest_steps(selection="affected", worker_args=[])[0][1]
    assert "--testmon" in affected_command
    assert "--testmon-forceselect" in affected_command
    assert not any(nodeid in affected_command for nodeid in verify.DESCRIPTOR_CONTRACT_TESTS)


def test_verify_main_routes_descriptor_diff_to_bounded_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The default verifier route applies the descriptor boundary."""
    from devtools.agent_env import AGENT_PRINCIPAL, AGENT_PRINCIPAL_ENV

    captured: dict[str, Any] = {}
    history: dict[str, Any] = {}
    monkeypatch.setenv(AGENT_PRINCIPAL_ENV, AGENT_PRINCIPAL)
    monkeypatch.setattr(verify, "refuse_verify_tier", lambda _argv, _env: None)
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify, "_git_changed_paths", lambda _root: frozenset({".agentctl/project.toml"}))
    monkeypatch.setattr(verify, "sync_testmon_graph", lambda _root: False)
    monkeypatch.setattr(
        verify,
        "inspect_testmon_graph",
        lambda _root: SimpleNamespace(
            status=TestmonGraphStatus.USABLE,
            reason="testmon datafile present",
            full_rerun_cause="the installed packages changed",
        ),
    )
    monkeypatch.setattr(verify, "assert_polylogue_matches_checkout", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(verify, "git_head", lambda _root: "head")

    def capture_steps(**kwargs: Any) -> list[tuple[str, list[str]]]:
        captured.update(kwargs)
        return [("gate lint", ["true"])]

    monkeypatch.setattr(
        verify,
        "build_verify_steps",
        capture_steps,
    )
    monkeypatch.setattr(verify, "_run", lambda *_args, **_kwargs: (0, 0.1, {"diagnosis": "gate_passed"}))
    monkeypatch.setattr(verify, "append_verify_history", lambda payload: history.update(payload))
    monkeypatch.setattr(verify, "append_verification_evidence", lambda _payload: None)
    monkeypatch.setattr(verify, "prune_successful_verify_runs", lambda **_kwargs: None)

    assert verify._main([]) == 0
    assert captured["selection"] == "descriptor"
    assert history["pytest_aggregate"]["selection_mode"] == "descriptor"


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


def test_zero_exit_without_a_report_is_a_failed_pytest_step(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("POLYLOGUE_PYTEST_SLOT", "held")
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify, "_clear_pytest_report", lambda _command: None)
    monkeypatch.setattr(
        "devtools.verify.subprocess.run", lambda *_args, **_kwargs: subprocess.CompletedProcess(["pytest"], 0)
    )
    run = VerifyRun(tier="test", argv=[], git_head="head", root=tmp_path)

    exit_code, _elapsed, metadata = verify._run("pytest serial (all)", ["pytest"], run=run)

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
    monkeypatch.setattr(verify, "build_verify_steps", lambda **_kwargs: [("pytest parallel (all)", ["pytest"])])
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


def test_verify_emits_shared_workload_receipt_for_step_timing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    history: dict[str, Any] = {}

    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify, "assert_polylogue_matches_checkout", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(verify, "git_head", lambda _root: "head")
    monkeypatch.setattr(verify, "build_verify_steps", lambda **_kwargs: [("gate lint", ["ruff", "check"])])
    monkeypatch.setattr(verify, "_run", lambda *_args, **_kwargs: (0, 0.25, {"diagnosis": "gate_passed"}))
    monkeypatch.setattr(verify, "append_verify_history", lambda payload: history.update(payload))

    assert verify._main(["--quick"]) == 0

    receipt = history["workload_receipt"]
    assert receipt["spec"]["workload_id"] == "devtools:verify:quick"
    assert receipt["spec"]["measurement_scope"] == "process-tree"
    assert receipt["phases"] == [
        {
            "name": "gate lint",
            "measurement_scope": None,
            "wall_ms": 250.0,
            "cleanup_complete": None,
            "quiescent": False,
            "unavailable": list(verify._UNMEASURED_WORKLOAD_DIMENSIONS),
        }
    ]


def test_git_dirty_fails_closed_when_status_cannot_be_read(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from devtools import verify_runs

    def broken(*_a: Any, **_k: Any) -> SimpleNamespace:
        return SimpleNamespace(returncode=128, stdout="", stderr="fatal: index file corrupt")

    monkeypatch.setattr(subprocess, "run", broken)

    assert verify_runs.git_dirty() is True


def test_git_dirty_sees_untracked_files_regardless_of_config(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from devtools import verify_runs

    seen: list[list[str]] = []

    def record(command: list[str], **_k: Any) -> SimpleNamespace:
        seen.append(command)
        return SimpleNamespace(returncode=0, stdout="?? tests/new_test.py\n")

    monkeypatch.setattr(subprocess, "run", record)
    assert verify_runs.git_dirty() is True
    assert "--untracked-files=all" in seen[0]


def test_failed_tests_are_rerun_once_and_flakes_are_named(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Anti-vacuity: without the rerun a load-induced failure is a red step;
    without the still-failed check a real failure passes as flaky."""
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.setattr(verify, "venv_python", lambda root: "python")
    report_path = tmp_path / ".cache" / "verify" / "last-pytest.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "tests": [
                    {"nodeid": "tests/test_a.py::test_flaky", "outcome": "failed"},
                    {"nodeid": "tests/test_a.py::test_red", "outcome": "failed"},
                    {"nodeid": "tests/test_a.py::test_ok", "outcome": "passed"},
                ],
                "summary": {"failed": 2, "passed": 1, "exitstatus": 1},
            }
        )
    )
    step_dir = tmp_path / "step"
    step_dir.mkdir()
    (step_dir / "summary.json").write_text(json.dumps({"exitstatus": 1, "failed": 2}))
    reruns: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        reruns.append(command)
        rerun_report = Path(next(a for a in command if a.startswith("--json-report-file=")).split("=", 1)[1])
        rerun_report.write_text(
            json.dumps(
                {
                    "tests": [
                        {"nodeid": "tests/test_a.py::test_flaky", "outcome": "passed"},
                        {"nodeid": "tests/test_a.py::test_red", "outcome": "failed"},
                    ]
                }
            )
        )
        return SimpleNamespace(returncode=1)

    import subprocess

    monkeypatch.setattr(subprocess, "run", fake_run)
    command = ["python", "-m", "pytest", f"--json-report-file={report_path}"]

    result = verify._rerun_failed_once(command, env=_SLOT_HELD_ENV, artifacts=SimpleNamespace(step_dir=step_dir))

    assert result is not None
    assert result["flaky"] == ["tests/test_a.py::test_flaky"]
    assert result["still_failed"] == ["tests/test_a.py::test_red"]
    assert "-p" in reruns[0] and "no:testmon" in reruns[0]
    # Exactly the failed tests are selected, and nothing that passed.
    assert [arg for arg in reruns[0] if arg.startswith("tests/")] == [
        "tests/test_a.py::test_flaky",
        "tests/test_a.py::test_red",
    ]
    patched = json.loads(report_path.read_text())
    by_id = {t["nodeid"]: t for t in patched["tests"]}
    assert by_id["tests/test_a.py::test_flaky"]["outcome"] == "passed"
    assert by_id["tests/test_a.py::test_flaky"]["flaky"] is True
    assert by_id["tests/test_a.py::test_red"]["outcome"] == "failed"
    assert patched["summary"]["failed"] == 1 and patched["summary"]["flaky"] == 1
    assert patched["summary"]["exitstatus"] == 1, "a real failure keeps the step red"
    assert json.loads((step_dir / "summary.json").read_text()) == {"exitstatus": 1, "failed": 2, "flaky": 1}

    # Every failure passes alone: both summaries agree the step is green.
    report_path.write_text(
        json.dumps(
            {
                "tests": [{"nodeid": "tests/test_a.py::test_flaky", "outcome": "failed"}],
                "summary": {"failed": 1, "exitstatus": 1},
            }
        )
    )
    (step_dir / "summary.json").write_text(json.dumps({"exitstatus": 1}))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_k: (
            Path(next(a for a in command if a.startswith("--json-report-file=")).split("=", 1)[1]).write_text(
                json.dumps({"tests": [{"nodeid": "tests/test_a.py::test_flaky", "outcome": "passed"}]})
            ),
            SimpleNamespace(returncode=0),
        )[1],
    )
    result = verify._rerun_failed_once(command, env=_SLOT_HELD_ENV, artifacts=SimpleNamespace(step_dir=step_dir))
    assert result is not None and result["still_failed"] == []
    assert json.loads((step_dir / "summary.json").read_text())["exitstatus"] == 0
    assert json.loads(report_path.read_text())["summary"]["exitstatus"] == 0

    # Every node passed but pytest exited 3 (internal error): nothing is cleared.
    report_path.write_text(
        json.dumps(
            {
                "tests": [{"nodeid": "tests/test_a.py::test_flaky", "outcome": "failed"}],
                "summary": {"failed": 1, "exitstatus": 1},
            }
        )
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_k: (
            Path(next(a for a in command if a.startswith("--json-report-file=")).split("=", 1)[1]).write_text(
                json.dumps({"tests": [{"nodeid": "tests/test_a.py::test_flaky", "outcome": "passed"}]})
            ),
            SimpleNamespace(returncode=3),
        )[1],
    )
    result = verify._rerun_failed_once(command, env=_SLOT_HELD_ENV, artifacts=SimpleNamespace(step_dir=step_dir))
    assert result is not None and result["still_failed"] == ["tests/test_a.py::test_flaky"] and result["flaky"] == []


def _flake_rerun_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    rerun_outcome: str = "passed",
) -> tuple[Path, list[list[str]]]:
    """Seed a first-run report with one failure and stub the rerun subprocess."""

    report_path = tmp_path / "pytest.json"
    report_path.write_text(
        json.dumps(
            {
                "exitcode": 1,
                "summary": {"failed": 1, "passed": 2},
                "tests": [
                    {"nodeid": "tests/test_a.py::test_flaky", "outcome": "failed"},
                    {"nodeid": "tests/test_a.py::test_ok", "outcome": "passed"},
                ],
            }
        ),
        encoding="utf-8",
    )
    step_dir = tmp_path / "step"
    step_dir.mkdir()
    launched: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        launched.append(list(command))
        rerun_file = next(
            Path(argument.split("=", 1)[1]) for argument in command if argument.startswith("--json-report-file=")
        )
        rerun_file.write_text(
            json.dumps(
                {
                    "exitcode": 0 if rerun_outcome == "passed" else 1,
                    "tests": [{"nodeid": "tests/test_a.py::test_flaky", "outcome": rerun_outcome}],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0 if rerun_outcome == "passed" else 1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return report_path, launched


def test_accepted_flake_clears_both_recorded_exit_statuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anti-vacuity: drop the ``patched["exitcode"] = 0`` line and this fails.

    The json report states its exit status twice. Patching only ``summary``
    left the top-level field naming the pre-rerun failure, so a consumer
    reading it saw a red run the verifier had already accepted as flaky.
    """

    report_path, _ = _flake_rerun_fixture(tmp_path, monkeypatch)
    command = ["pytest", f"--json-report-file={report_path}"]
    artifacts = SimpleNamespace(step_dir=tmp_path / "step")

    rerun = verify._rerun_failed_once(command, env=_SLOT_HELD_ENV, artifacts=artifacts)

    assert rerun is not None
    assert rerun["flaky"] == ["tests/test_a.py::test_flaky"]
    assert rerun["still_failed"] == []
    patched = json.loads(report_path.read_text(encoding="utf-8"))
    assert patched["exitcode"] == 0
    assert patched["summary"]["exitstatus"] == 0
    assert patched["flaky_nodeids"] == ["tests/test_a.py::test_flaky"]


def test_receipt_names_the_tests_that_only_passed_on_rerun() -> None:
    """Anti-vacuity: remove the rerun block from canonical_verification_receipt
    and the step below carries no flake evidence at all.

    A step green only because its failures passed alone is weaker evidence
    than one that never failed; the durable receipt must let a reader tell
    them apart.
    """

    entry = {
        "run_id": "run-1",
        "status": "passed",
        "steps": [
            {
                "step_id": "s1",
                "name": "pytest",
                "exit": 0,
                "diagnosis": "gate_passed",
                "rerun": {
                    "attempted": ["tests/test_a.py::test_flaky"],
                    "still_failed": [],
                    "flaky": ["tests/test_a.py::test_flaky"],
                },
            }
        ],
    }

    receipt = verify_runs.canonical_verification_receipt(entry)
    step = receipt["steps"][0]

    assert step["status"] == "passed"
    assert step["flaky"] == ["tests/test_a.py::test_flaky"]
    assert step["flaky_count"] == 1


def test_receipt_omits_flake_fields_when_no_test_flaked() -> None:
    """Anti-vacuity: emit the keys unconditionally and this fails.

    An empty flake list must not appear, or every clean step would look like
    it carried rerun evidence.
    """

    entry = {
        "run_id": "run-2",
        "status": "passed",
        "steps": [
            {
                "step_id": "s1",
                "name": "pytest",
                "exit": 0,
                "rerun": {"attempted": ["t"], "still_failed": ["t"], "flaky": []},
            }
        ],
    }

    step = verify_runs.canonical_verification_receipt(entry)["steps"][0]

    assert "flaky" not in step
    assert "flaky_count" not in step


def test_agent_job_caps_an_explicit_worker_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: return `selection` unchanged from _capped_selection and
    the explicit -n 32 below survives into the command.

    The cap protects a shared host. Honoring it only when the caller stayed
    silent let any explicit request claim the whole machine.
    """

    from devtools import agent_env, run_tests

    monkeypatch.setenv(agent_env.AGENT_PRINCIPAL_ENV, agent_env.AGENT_PRINCIPAL)
    command = run_tests.build_pytest_cmd(["tests/unit/foo.py", "-n", "32"])

    assert verify_runs.pytest_command_worker_request(command) == str(agent_env.AGENT_MAX_PYTEST_WORKERS)
    assert "32" not in command


def test_agent_job_leaves_a_request_within_the_cap_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: cap unconditionally and this run gets the ceiling instead
    of the single worker it asked for.
    """

    from devtools import agent_env, run_tests

    monkeypatch.setenv(agent_env.AGENT_PRINCIPAL_ENV, agent_env.AGENT_PRINCIPAL)
    command = run_tests.build_pytest_cmd(["tests/unit/foo.py", "-n", "1"])

    assert verify_runs.pytest_command_worker_request(command) == "1"


def test_explicit_zero_workers_survives_the_agent_cap() -> None:
    """Anti-vacuity: restore `requested is None or requested < 1` and an
    explicit zero becomes the ceiling.

    Zero asks for no xdist at all. Treating it as "unset" silently turned a
    deliberately serial run into a parallel one.
    """

    from devtools.agent_env import AGENT_MAX_PYTEST_WORKERS, AGENT_PRINCIPAL, AGENT_PRINCIPAL_ENV, agent_worker_cap

    env = {AGENT_PRINCIPAL_ENV: AGENT_PRINCIPAL}

    assert agent_worker_cap(0, env) == 0
    assert agent_worker_cap(None, env) == AGENT_MAX_PYTEST_WORKERS
    assert agent_worker_cap(1000, env) == AGENT_MAX_PYTEST_WORKERS
    assert agent_worker_cap(0, {}) == 0


def test_focused_managed_runs_also_force_the_full_hypothesis_profile() -> None:
    """Anti-vacuity: drop the HYPOTHESIS_PROFILE assignment from run_tests and
    a shell's `ci` profile survives into a run that writes graph edges.

    `devtools test` traces into the same datafile `devtools verify` selects
    from. An edge recorded under a reduced property budget would let a later
    selected green stand for less coverage than it claims.
    """

    from devtools import run_tests

    env = {"HYPOTHESIS_PROFILE": "ci", "PATH": "/usr/bin"}
    run_tests._normalize_managed_pytest_environment(env)

    assert env["HYPOTHESIS_PROFILE"] == "default"


def test_agent_tier_refusal_honors_the_json_contract(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Anti-vacuity: restore the unconditional stderr write and stdout is empty
    so the json.loads below raises.

    `--json` is the machine-readable contract. A refusal that answers in prose
    leaves a hole in it exactly where an automated caller needs a verdict.
    """

    from devtools import agent_env

    monkeypatch.setenv(agent_env.AGENT_PRINCIPAL_ENV, agent_env.AGENT_PRINCIPAL)

    exit_code = verify._main(["--json"])

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "refused"
    assert payload["diagnosis"] == "agent_tier_refused"
    assert payload["exit_code"] == 2
    assert "devtools test" in payload["message"]


def test_schema_promotion_audits_the_tree_it_writes_to(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: restore the "polylogue/schemas" literal and this resolves
    to a bare relative path that only exists from the checkout root.

    Promotion writes to the installed schema package; auditing a relative
    literal audits whatever happens to sit under the caller's cwd.
    """

    from devtools import schema_promote

    root = schema_promote._schema_registry_root()

    assert root.is_absolute()
    assert root.is_dir()
    assert root.name == "schemas"


def test_schema_promotion_json_stays_one_document(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: drop capture_output and the audit's own stdout lands
    after the JSON document, so stdout no longer parses as one value.
    """

    from devtools import schema_promote

    recorded: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        recorded.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="audit noise\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        schema_promote,
        "promote_schema_cluster",
        lambda request: SimpleNamespace(ok=True),
    )
    monkeypatch.setattr(schema_promote, "build_schema_privacy_config", lambda **kwargs: None)
    monkeypatch.setattr(schema_promote, "get_config", lambda: SimpleNamespace(db_path=Path("x")))
    monkeypatch.setattr(schema_promote, "render_schema_promote_result", lambda **kwargs: None)

    exit_code = schema_promote.main(["--provider", "p", "--cluster", "c", "--json"])

    assert exit_code == 0
    assert recorded["capture_output"] is True
