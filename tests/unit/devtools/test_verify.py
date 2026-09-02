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

from devtools import required_gate, verify, verify_runs, why
from devtools.testmon_bootstrap import NativeTestmonRepairError
from devtools.verify_runs import (
    CURRENT_RUN_PATH,
    VerifyRun,
    aggregate_pytest_statistics,
    env_for_pytest_step,
)


def test_quick_steps_are_static_gates() -> None:
    labels = [label for label, _command in verify.build_verify_steps(quick=True)]

    assert "ruff check" in labels
    assert "verify oracle-integrity" in labels
    assert not any(label.startswith("pytest") for label in labels)


def test_failed_pytest_nodes_are_rerun_without_testmon_selection() -> None:
    command = [
        "python",
        "-m",
        "pytest",
        "--testmon",
        "--testmon-env=env",
        "--testmon-forceselect",
        "-m",
        "parallel",
        "-p",
        "no:randomly",
        "-n",
        "2",
    ]

    rerun = verify._pytest_rerun_command(command, ["tests/test_b.py::test_b", "tests/test_a.py::test_a"])

    assert "--testmon" not in rerun
    assert not any(argument.startswith("--testmon-env=") for argument in rerun)
    assert "--testmon-forceselect" not in rerun
    assert "-m" not in rerun
    assert rerun[-2:] == ["tests/test_b.py::test_b", "tests/test_a.py::test_a"]


def test_verification_tools_are_absolute_paths_in_checkout_venv() -> None:
    steps = verify.build_verify_steps(quick=True)
    commands = dict(steps)

    assert commands["ruff format"][0] == str(verify.ROOT / ".venv/bin/ruff")
    assert commands["ruff check"][0] == str(verify.ROOT / ".venv/bin/ruff")
    assert commands["mypy"][0].startswith(str(verify.ROOT / ".venv/bin/"))
    assert commands["render all"][0] == str(verify.ROOT / ".venv/bin/python")
    assert commands["schema privacy registry"][0] == str(verify.ROOT / ".venv/bin/python")


def test_missing_checkout_venv_tool_is_a_typed_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    history: dict[str, Any] = {}
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify, "assert_polylogue_matches_checkout", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(verify, "git_head", lambda _root: "head")
    monkeypatch.setattr(
        verify, "build_verify_steps", lambda **_kwargs: [("ruff check", [str(tmp_path / ".venv/bin/ruff")])]
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


def test_mypy_starts_a_worktree_dmypy_when_no_daemon_is_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    dmypy = str(verify.ROOT / ".venv/bin/dmypy")

    def run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=1 if command[-1] == "status" else 0)

    monkeypatch.setattr(subprocess, "run", run)

    assert verify._mypy_cmd() == [
        dmypy,
        "run",
        f"--timeout={verify.DMYPY_IDLE_TIMEOUT_SECONDS}",
        "--",
        "--no-error-summary",
    ]
    assert calls == [
        [dmypy, "status"],
        [dmypy, "start", f"--timeout={verify.DMYPY_IDLE_TIMEOUT_SECONDS}", "--", "--no-error-summary"],
    ]


def test_removed_lab_mode_is_not_accepted() -> None:
    with pytest.raises(SystemExit) as raised:
        verify._main(["--lab"])

    assert raised.value.code == 2


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

    exit_code, elapsed, metadata = verify._run("render all", command, run=run)

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


def test_native_selection_partitions_semantic_lanes() -> None:
    steps = verify.build_verify_steps(
        quick=False,
        testmon_mode="affected",
        testmon_environment="polylogue-test",
    )

    labels = [label for label, _command in steps]
    assert labels[-3:] == [
        "pytest native parallel (affected)",
        "pytest native serial (affected)",
        "pytest native storage-scale (affected)",
    ]


def test_full_corpus_traces_without_selecting_and_without_partitioning() -> None:
    """The complete corpus publishes the graph affected selection needs.

    Anti-vacuity: partitioning the parallel lane keeps only the last
    partition's tests in the graph (testmon deletes what a run did not
    collect), and dropping --testmon leaves the graph untouched.
    """
    steps = verify.build_verify_steps(
        quick=False,
        testmon_mode="all",
        testmon_environment="polylogue-test",
    )

    assert [label for label, _command in steps[-3:]] == [
        "pytest native parallel (all)",
        "pytest native serial (all)",
        "pytest native storage-scale (all)",
    ]
    for _label, command in steps[-3:]:
        assert "--testmon" in command
        assert "--testmon-noselect" in command
        assert "--testmon-forceselect" not in command
        assert "--testmon-env=polylogue-test" in command
        assert not any(item.startswith("--polylogue-file-batch") for item in command)


def test_every_managed_pytest_run_traces_under_the_full_hypothesis_profile() -> None:
    """Graph writers share one profile; affected runs write edges too.

    Anti-vacuity: leaving a shell's HYPOTHESIS_PROFILE=ci in place lets an
    affected run replace a property test's edges with the reduced budget's,
    and a later change reachable only beyond that budget is never selected.
    """
    env = {"HYPOTHESIS_PROFILE": "ci", "POLYLOGUE_CI": "1"}
    verify._normalize_managed_pytest_environment(env)
    assert env["HYPOTHESIS_PROFILE"] == "default"
    assert "POLYLOGUE_CI" not in env


def test_full_corpus_aggregate_sums_disjoint_lanes() -> None:
    aggregate = verify._aggregate_pytest_results(
        [
            {
                "name": "pytest native parallel (all)",
                "statistics": {
                    "selected_count": 27,
                    "terminal_count": 27,
                    "outcomes": {"passed": 23, "skipped": 1, "xfailed": 1},
                },
            },
            {
                "name": "pytest native serial (all)",
                "statistics": {"selected_count": 2, "terminal_count": 2, "outcomes": {"passed": 2}},
            },
            {
                "name": "pytest native storage-scale (all)",
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
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify, "_clear_pytest_report", lambda _command: None)
    monkeypatch.setattr(
        "devtools.verify.subprocess.run", lambda *_args, **_kwargs: subprocess.CompletedProcess(["pytest"], 0)
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
    assert Path(env["TESTMON_DATAFILE"]) == tmp_path / ".cache" / "testmon" / "testmondata"


def test_production_testmon_selection_writes_only_to_owned_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A selected pytest step must not create testmon state at checkout root."""
    test_file = tmp_path / "test_sample.py"
    test_file.write_text("def test_sample():\n    assert True\n", encoding="utf-8")
    (tmp_path / ".cache" / "testmon").mkdir(parents=True)
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    run = VerifyRun(tier="affected", argv=[], git_head="head", root=tmp_path, mirror_current=False)

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--override-ini=addopts=",
        "-p",
        "testmon.pytest_testmon",
        "-p",
        "pytest_jsonreport.plugin",
        "--json-report",
        "--json-report-file=.cache/verify/last-pytest.json",
        "--testmon",
        "--testmon-forceselect",
        "--testmon-env=selection-test",
        "test_sample.py",
    ]
    artifacts = run.start_step(label="pytest native serial (affected)", cmd=command)
    env = verify._subprocess_env()
    verify._normalize_managed_pytest_environment(env)
    env = env_for_pytest_step(env, run=run, artifacts=artifacts)
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / ".cache" / "testmon" / "testmondata").is_file()
    assert not (tmp_path / ".testmondata").exists()


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


def test_verify_reports_native_testmon_refusal_to_the_operator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing graph must be visible at the production command boundary."""
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
    monkeypatch.setattr(
        verify,
        "prepare_native_testmon_environment",
        lambda *_args, **_kwargs: SimpleNamespace(
            selection_mode="bootstrap",
            environment_name="expected-environment",
            local_state=SimpleNamespace(
                status="absent",
                reason="native environment 'expected-environment' is absent",
                missing_executable_paths=(),
            ),
        ),
    )

    assert verify._main([]) == 2

    output = capsys.readouterr().err
    assert "refusing to measure affected verification" in output
    assert "selection: affected" in output
    assert "environment: 'expected-environment' (absent)" in output
    assert "native environment 'expected-environment' is absent" in output
    assert "devtools verify --all" in output


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


def test_complete_corpus_is_not_recomputed_for_a_verified_head(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Anti-vacuity: without the check the nightly reran the whole corpus on an
    unchanged master every night."""
    history: dict[str, Any] = {}
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".cache" / "verify" / "runs" / "run-prior").mkdir(parents=True)
    monkeypatch.setattr(verify, "assert_polylogue_matches_checkout", lambda *_a, **_k: None)
    monkeypatch.setattr(verify, "git_head", lambda _root: "head")
    monkeypatch.setattr(verify, "git_dirty", lambda _root: False)
    monkeypatch.setattr(verify, "_git_commit", lambda _ref: "base")
    monkeypatch.setattr(verify, "_changed_paths", lambda _b, _h: ())
    monkeypatch.setattr(
        verify,
        "classify_native_testmon_changes",
        lambda *_a: SimpleNamespace(executable_paths=(), runtime_data_paths=()),
    )
    monkeypatch.setattr(
        verify,
        "prepare_native_testmon_environment",
        lambda *_a, **_k: SimpleNamespace(
            selection_mode="affected",
            environment_name="env-1",
            local_state=SimpleNamespace(status="valid", reason="current", missing_executable_paths=(), valid=True),
        ),
    )
    monkeypatch.setattr(
        verify,
        "read_verify_history",
        lambda _path: [
            {
                "tier": "all",
                "status": "success",
                "run_id": "run-prior",
                "artifact_dir": ".cache/verify/runs/run-prior",
                "git_dirty": False,
                "git_head": "head",
                "semantic_receipt": {"source_revision": "head"},
                "testmon_selection": {
                    "environment_digest": "env-1",
                    "packages_digest": "pkgs-1",
                    "plan_digest": "plan-1",
                },
                "pytest_aggregate": {"complete_corpus_covered": True},
            }
        ],
    )
    monkeypatch.setattr(verify, "installed_packages_digest", lambda: "pkgs-1")
    monkeypatch.setattr(verify, "_execution_plan_digest", lambda: "plan-1")
    monkeypatch.setattr(verify, "append_verify_history", lambda payload: history.update(payload))

    def no_steps(**_kwargs: Any) -> list[tuple[str, list[str]]]:
        raise AssertionError("steps were built for an already-verified corpus")

    monkeypatch.setattr(verify, "build_verify_steps", no_steps)

    assert verify._main(["--all"]) == 0
    assert history["diagnosis"] == "corpus_already_verified"
    assert history["pytest_aggregate"]["covered_by_run"] == "run-prior"


def test_a_head_that_moves_during_preparation_is_not_reused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Anti-vacuity: without the recheck a clean tree at a new commit inherits
    the old commit's coverage."""
    history: dict[str, Any] = {}
    heads = iter(["head", "head-moved", "head-moved", "head-moved"])
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".cache" / "verify" / "runs" / "run-prior").mkdir(parents=True)
    monkeypatch.setattr(verify, "assert_polylogue_matches_checkout", lambda *_a, **_k: None)
    monkeypatch.setattr(verify, "git_head", lambda _root: next(heads, "head-moved"))
    monkeypatch.setattr(verify, "git_dirty", lambda _root: False)
    monkeypatch.setattr(verify, "_git_commit", lambda _ref: "base")
    monkeypatch.setattr(verify, "_changed_paths", lambda _b, _h: ())
    monkeypatch.setattr(
        verify,
        "classify_native_testmon_changes",
        lambda *_a: SimpleNamespace(executable_paths=(), runtime_data_paths=()),
    )
    monkeypatch.setattr(
        verify,
        "prepare_native_testmon_environment",
        lambda *_a, **_k: SimpleNamespace(
            selection_mode="affected",
            environment_name="env-1",
            local_state=SimpleNamespace(status="valid", reason="current", missing_executable_paths=(), valid=True),
        ),
    )
    monkeypatch.setattr(
        verify,
        "read_verify_history",
        lambda _path: [
            {
                "tier": "all",
                "status": "success",
                "run_id": "run-prior",
                "artifact_dir": ".cache/verify/runs/run-prior",
                "git_dirty": False,
                "git_head": "head",
                "semantic_receipt": {"source_revision": "head"},
                "testmon_selection": {
                    "environment_digest": "env-1",
                    "packages_digest": "pkgs-1",
                    "plan_digest": "plan-1",
                },
                "pytest_aggregate": {"complete_corpus_covered": True},
            }
        ],
    )
    monkeypatch.setattr(verify, "installed_packages_digest", lambda: "pkgs-1")
    monkeypatch.setattr(verify, "_execution_plan_digest", lambda: "plan-1")
    monkeypatch.setattr(verify, "append_verify_history", lambda payload: history.update(payload))
    built: list[str] = []

    def no_steps(**_kwargs: Any) -> list[tuple[str, list[str]]]:
        built.append("built")
        return []

    monkeypatch.setattr(verify, "build_verify_steps", no_steps)

    verify._main(["--all"])

    assert built == ["built"], "a moved head must run, not inherit"
    assert history.get("diagnosis") != "corpus_already_verified"


def test_execution_plan_digest_sees_gate_executables(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Anti-vacuity: a ruff that vanished after the green run must change the plan."""
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    binaries = tmp_path / ".venv" / "bin"
    binaries.mkdir(parents=True)
    for name in ("ruff", "mypy", "dmypy"):
        target = binaries / name
        target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        target.chmod(0o755)
    monkeypatch.setattr(verify, "venv_bin", lambda name, root: str(root / ".venv" / "bin" / name))

    with_ruff = verify._execution_plan_digest()
    (binaries / "ruff").unlink()
    without_ruff = verify._execution_plan_digest()

    assert with_ruff != without_ruff


def test_execution_plan_digest_sees_installed_js_trees(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Anti-vacuity: a node_modules tree deleted after the green run must change the plan."""
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.setattr(verify, "venv_bin", lambda name, root: str(root / ".venv" / "bin" / name))
    from devtools.verify_js_tests import _STAMP_NAME

    stamp = tmp_path / "webui" / "node_modules" / _STAMP_NAME
    stamp.parent.mkdir(parents=True)
    stamp.write_text("fingerprint-1", encoding="utf-8")

    installed = verify._execution_plan_digest()
    stamp.unlink()
    removed = verify._execution_plan_digest()

    assert installed != removed


def test_git_dirty_fails_closed_when_status_cannot_be_read(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from devtools import verify_runs

    def broken(*_a: Any, **_k: Any) -> SimpleNamespace:
        return SimpleNamespace(returncode=128, stdout="", stderr="fatal: index file corrupt")

    monkeypatch.setattr(subprocess, "run", broken)

    assert verify_runs.git_dirty() is True


def test_reuse_requires_a_clean_newest_attempt_on_the_same_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Anti-vacuity, one clause each: a moved package set or plan runs; a dirty
    source run is not coverage; a newer failed recompute is not hidden behind
    an older green; a prior skip is not an attempt."""
    monkeypatch.setattr(verify, "ROOT", tmp_path)

    def row(
        run_id: str,
        *,
        status: str = "success",
        dirty: bool = False,
        packages: str = "pkgs-1",
        plan: str = "plan-1",
        covered_by: str | None = None,
        start_head: str = "head",
        evidence: bool = True,
    ) -> dict[str, Any]:
        aggregate: dict[str, Any] = {"complete_corpus_covered": covered_by is None}
        if covered_by:
            aggregate["covered_by_run"] = covered_by
        if evidence:
            (tmp_path / ".cache" / "verify" / "runs" / run_id).mkdir(parents=True, exist_ok=True)
        return {
            "tier": "all",
            "status": status,
            "run_id": run_id,
            "artifact_dir": f".cache/verify/runs/{run_id}",
            "git_dirty": dirty,
            "git_head": start_head,
            "semantic_receipt": {"source_revision": start_head if run_id == "elsewhere" else "head"},
            "testmon_selection": {"environment_digest": "env-1", "packages_digest": packages, "plan_digest": plan},
            "pytest_aggregate": aggregate,
        }

    def covered(rows: list[dict[str, Any]], **inputs: str) -> str | None:
        monkeypatch.setattr(verify, "read_verify_history", lambda _p: rows)
        return verify._corpus_already_verified(
            head="head",
            environment="env-1",
            packages=inputs.get("packages", "pkgs-1"),
            plan=inputs.get("plan", "plan-1"),
        )

    assert covered([row("green")]) == "green"
    # A newer complete run at another head rewrote the shared graph.
    assert covered([row("green"), row("elsewhere", start_head="other")]) is None
    assert covered([row("green")], packages="pkgs-2") is None
    assert covered([row("green")], plan="plan-2") is None
    # A newer run under another package set rewrote the graph; an older
    # green under this set is not searched past it.
    assert covered([row("green"), row("other-packages", packages="pkgs-2")]) is None
    assert covered([row("dirty", dirty=True)]) is None
    assert covered([row("green"), row("red-later", status="failed")]) is None
    assert covered([row("green"), row("skip", covered_by="green")]) == "green"
    # A run whose HEAD advanced while it ran verified a mixture, not this head.
    assert covered([row("moved", start_head="older")]) is None
    # Pruned evidence is history, not coverage.
    assert covered([row("pruned", evidence=False)]) is None
    # A traced affected run since, at another head, rewrote the graph.
    affected = {**row("later-affected"), "tier": "affected"}
    assert covered([row("green"), affected]) is None


def test_execution_plan_digest_sees_the_js_worker_budget(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.setattr(verify, "venv_bin", lambda name, root: str(root / ".venv" / "bin" / name))
    monkeypatch.setenv("POLYLOGUE_EXTENSION_TEST_WORKERS", "2")
    two = verify._execution_plan_digest()
    monkeypatch.setenv("POLYLOGUE_EXTENSION_TEST_WORKERS", "4")
    four = verify._execution_plan_digest()

    assert two != four


def test_execution_plan_digest_sees_run_time_tools_and_installed_js_binaries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Anti-vacuity: an ast-grep that vanished from PATH, or a deleted
    node_modules/.bin/vitest, must change the plan."""
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.setattr(verify, "venv_bin", lambda name, root: str(root / ".venv" / "bin" / name))
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("ast-grep", "node", "npm"):
        target = tools / name
        target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        target.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools))
    binaries = tmp_path / "webui" / "node_modules" / ".bin"
    binaries.mkdir(parents=True)
    (binaries / "vitest").write_text("run", encoding="utf-8")

    baseline = verify._execution_plan_digest()
    (tools / "ast-grep").unlink()
    without_ast_grep = verify._execution_plan_digest()
    (binaries / "vitest").unlink()
    without_vitest = verify._execution_plan_digest()

    assert len({baseline, without_ast_grep, without_vitest}) == 3


def test_execution_plan_digest_sees_consumer_reachability_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.setattr(verify, "venv_bin", lambda name, root: str(root / ".venv" / "bin" / name))
    monkeypatch.delenv("CONSUMER_REACHABILITY_BASE", raising=False)
    plain = verify._execution_plan_digest()
    monkeypatch.setenv("CONSUMER_REACHABILITY_BASE", "HEAD")
    overridden = verify._execution_plan_digest()

    assert plain != overridden


def test_recompute_requires_the_complete_corpus(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as raised:
        verify._main(["--recompute"])
    assert raised.value.code == 2


def test_execution_plan_digest_sees_master_and_hypothesis_examples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Anti-vacuity: a moved origin/master or a stored counterexample must change the plan."""
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.setattr(verify, "venv_bin", lambda name, root: str(root / ".venv" / "bin" / name))
    monkeypatch.setattr(verify, "_git_commit", lambda ref: "m1")
    monkeypatch.setattr(verify, "_merge_base_with_master", lambda: "b1")
    examples = tmp_path / ".cache" / "hypothesis" / "examples"
    examples.mkdir(parents=True)
    first = verify._execution_plan_digest()
    monkeypatch.setattr(verify, "_git_commit", lambda ref: "m2")
    moved = verify._execution_plan_digest()
    (examples / "deadbeef").write_bytes(b"counterexample")
    stored = verify._execution_plan_digest()

    assert len({first, moved, stored}) == 3


def test_reuse_inputs_are_recomputed_at_the_decision(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Anti-vacuity: a plan captured before preparation must not be trusted
    after an environment sync changed it."""
    history: dict[str, Any] = {}
    plans = iter(["plan-1", "plan-2"])
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".cache" / "verify" / "runs" / "run-prior").mkdir(parents=True)
    monkeypatch.setattr(verify, "assert_polylogue_matches_checkout", lambda *_a, **_k: None)
    monkeypatch.setattr(verify, "git_head", lambda _root: "head")
    monkeypatch.setattr(verify, "git_dirty", lambda _root: False)
    monkeypatch.setattr(verify, "_git_commit", lambda _ref: "base")
    monkeypatch.setattr(verify, "_changed_paths", lambda _b, _h: ())
    monkeypatch.setattr(
        verify,
        "classify_native_testmon_changes",
        lambda *_a: SimpleNamespace(executable_paths=(), runtime_data_paths=()),
    )
    monkeypatch.setattr(
        verify,
        "prepare_native_testmon_environment",
        lambda *_a, **_k: SimpleNamespace(
            selection_mode="affected",
            environment_name="env-1",
            local_state=SimpleNamespace(status="valid", reason="current", missing_executable_paths=(), valid=True),
        ),
    )
    monkeypatch.setattr(
        verify,
        "read_verify_history",
        lambda _path: [
            {
                "tier": "all",
                "status": "success",
                "run_id": "run-prior",
                "artifact_dir": ".cache/verify/runs/run-prior",
                "git_dirty": False,
                "git_head": "head",
                "semantic_receipt": {"source_revision": "head"},
                "testmon_selection": {
                    "environment_digest": "env-1",
                    "packages_digest": "pkgs-1",
                    "plan_digest": "plan-1",
                },
                "pytest_aggregate": {"complete_corpus_covered": True},
            }
        ],
    )
    monkeypatch.setattr(verify, "installed_packages_digest", lambda: "pkgs-1")
    monkeypatch.setattr(verify, "_execution_plan_digest", lambda: next(plans, "plan-2"))
    monkeypatch.setattr(verify, "append_verify_history", lambda payload: history.update(payload))
    built: list[str] = []

    def no_steps(**_kwargs: Any) -> list[tuple[str, list[str]]]:
        built.append("built")
        return []

    monkeypatch.setattr(verify, "build_verify_steps", no_steps)

    verify._main(["--all"])

    assert built == ["built"]
    assert history.get("diagnosis") != "corpus_already_verified"


def test_unreadable_reuse_inputs_run_the_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    from devtools.testmon_bootstrap import NativeTestmonRepairError

    def broken() -> str:
        raise NativeTestmonRepairError("metadata malformed")

    monkeypatch.setattr(verify, "installed_packages_digest", broken)
    assert verify._reuse_inputs_still_hold("pkgs-1", "plan-1") is False


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
