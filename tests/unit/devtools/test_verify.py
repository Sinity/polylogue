"""Semantic verifier contracts independent of execution-host lifecycle."""

from __future__ import annotations

import io
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import tomllib

from devtools import (
    agent_env,
    gate,
    pytest_slot,
    required_gate,
    verify,
    verify_runs,
    why,
    worker_memory,
)
from devtools.pytest_stream_report import REPORT_FILE_OPTION, report_file_argument
from devtools.testmon_provision import TestmonGraphStatus
from devtools.testmon_provision import testmon_datafile as _testmon_datafile
from devtools.testmon_provision import testmon_environment as _testmon_environment
from devtools.verification_admission import AFFECTED_MAX_SELECTED_TESTS, AFFECTED_MAX_UNRECORDED_FILES
from devtools.verification_contracts import VerificationScope
from devtools.verification_result import declared_verification_result
from devtools.verify_runs import (
    CURRENT_RUN_PATH,
    VerifyRun,
    aggregate_pytest_statistics,
    env_for_pytest_step,
)

#: `_run` acquires the host's single pytest slot before executing. These tests drive that code inline through its
#: documented escape rather than requiring a live pueue queue.
_SLOT_HELD_ENV = {"POLYLOGUE_PYTEST_SLOT": "held"}

#: The prefix of the argument naming the report a managed step is judged from.
_REPORT_PREFIX = f"{REPORT_FILE_OPTION}="


def _stub_held_pytest(monkeypatch: pytest.MonkeyPatch, fake_run: Any) -> None:
    """Stand in for the pytest process the held slot launches (a process group, waited on)."""

    class FakeProcess:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            self.pid = os.getpid()
            self.returncode = int(fake_run(argv, **kwargs).returncode)

        def poll(self) -> int:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode

    from devtools import execution_source

    monkeypatch.setattr(execution_source, "start_execution", lambda *_args: None)
    monkeypatch.setattr(subprocess, "Popen", FakeProcess)
    # This fixture exercises adjudication after a synthetic launch. Physical
    # admission has independent actual-route controls and must not depend on
    # the live host or acquire a real reservation for this FakeProcess.
    monkeypatch.setattr(pytest_slot, "admission_ledger", lambda _env: None)
    monkeypatch.setattr(pytest_slot, "resize_worker_argument", lambda argv, **_kwargs: (list(argv), None))


def _outside_the_pytest_pool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An agent-pool job: the test itself runs inside the pytest pool, which owns the slot."""
    for name in agent_env.runtime_env_names(agent_env.QUEUE_POOL_ENV):
        monkeypatch.delenv(name, raising=False)
    cgroup = tmp_path / "cgroup"
    cgroup.write_text("0::/user.slice/user-1000.slice/user@1000.service/agent.slice/run-1.scope\n", encoding="utf-8")
    monkeypatch.setattr(agent_env, "_CGROUP_PATH", cgroup)


def test_corpus_workers_default_to_the_corpus_width(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset means the corpus width; an explicit value, ``0`` included, wins.

    Anti-vacuity: reading the variable with a ``"0"`` default (the previous
    code) makes the unset case yield ``-n 0`` and fails the first assertion;
    ignoring the variable makes the override cases fail.
    """
    monkeypatch.delenv("POLYLOGUE_PYTEST_WORKERS", raising=False)
    assert verify._pytest_worker_args(maximum=worker_memory.CORPUS_MAX_WORKERS)[-1] == str(
        worker_memory.CORPUS_MAX_WORKERS
    )

    monkeypatch.setenv("POLYLOGUE_PYTEST_WORKERS", "0")
    assert verify._pytest_worker_args(maximum=worker_memory.CORPUS_MAX_WORKERS)[-1] == "0"

    monkeypatch.setenv("POLYLOGUE_PYTEST_WORKERS", "1")
    assert verify._pytest_worker_args(maximum=worker_memory.CORPUS_MAX_WORKERS)[-1] == "1"

    monkeypatch.setenv("POLYLOGUE_PYTEST_WORKERS", "64")
    assert verify._pytest_worker_args(maximum=worker_memory.CORPUS_MAX_WORKERS)[-1] == str(
        worker_memory.CORPUS_MAX_WORKERS
    )


def test_quick_steps_are_static_gates() -> None:
    labels = [label for label, _command in verify.build_verify_steps(quick=True)]

    assert "gate lint" in labels
    assert "gate layering" in labels
    assert not any(label.startswith("pytest") for label in labels)


def test_static_gates_run_side_by_side_and_report_in_declared_order(tmp_path: Path) -> None:
    """Gates overlap in time, and their outcomes keep the declared order.

    Anti-vacuity: run the gates one after another and the rendezvous below
    times out; return outcomes in completion order and the first gate, which
    finishes last, is reported last.
    """
    rendezvous = threading.Barrier(2, timeout=10)
    finished: list[str] = []

    def fake_run(label: str, command: list[str], *, run: Any, runner: str) -> tuple[int, float, dict[str, Any]]:
        del command, run, runner
        rendezvous.wait()
        if label == "gate first":
            time.sleep(0.05)
        finished.append(label)
        return 0, 0.0, {"diagnosis": "gate_passed"}

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(verify, "_run", fake_run)
        patch.setattr(verify, "GATE_PARALLELISM", 2)
        outcomes = verify._run_steps([("gate first", ["a"]), ("gate second", ["b"])], run=None, runner="managed")  # type: ignore[arg-type]

    assert finished == ["gate second", "gate first"]
    assert [label for label, _outcome in outcomes] == ["gate first", "gate second"]


def test_no_gate_started_around_the_interruption_runs_to_completion(tmp_path: Path) -> None:
    """With more gates than workers, every gate process is stopped, none awaited.

    A queued gate is either cancelled before it starts or, if a freed worker
    picks it up first, registered before the interruption's snapshot of live
    processes, so it is terminated rather than waited for.

    Anti-vacuity: snapshot the live processes before cancelling queued gates,
    or register a process without checking the interruption under the same
    lock, and a gate launched after the snapshot runs its ``sleep`` to a
    natural exit, so a return code is 0 instead of a signal.
    """
    spawned: list[subprocess.Popen[str]] = []
    real_popen = subprocess.Popen

    def tracking_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[str]:
        process = real_popen(*args, **kwargs)
        spawned.append(process)
        return process

    def run_gate(label: str, command: list[str], *, run: Any, runner: str) -> tuple[int, float, dict[str, Any]]:
        del run, runner
        if label == "gate interrupting":
            for _ in range(1000):
                if verify._LIVE_GATE_PROCESSES:
                    break
                time.sleep(0.01)
            raise verify.VerificationInterrupted(signal.SIGTERM)
        completed = verify._run_gate_process(command, env=dict(os.environ))
        return completed.returncode, 0.0, {}

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(verify, "ROOT", tmp_path)
        patch.setattr(verify, "GATE_PARALLELISM", 2)
        patch.setattr(subprocess, "Popen", tracking_popen)
        patch.setattr(verify, "_run", run_gate)
        with pytest.raises(verify.VerificationInterrupted):
            verify._run_steps(
                [("gate running", ["sleep", "3"]), ("gate interrupting", ["true"]), ("gate queued", ["sleep", "3"])],
                run=None,  # type: ignore[arg-type]
                runner="managed",
            )

    assert spawned
    assert all(process.returncode is not None and process.returncode < 0 for process in spawned)


def _process_alive(pid: int) -> bool:
    """Whether *pid* is running; a zombie awaiting its reaper counts as dead."""
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except OSError:
        return False
    return "\nState:\tZ" not in status


def test_an_interruption_kills_a_gate_child_that_ignores_sigterm(tmp_path: Path) -> None:
    """The group is killed even when its leader exits on SIGTERM and a child does not.

    Anti-vacuity: send SIGKILL only when the leader outlives the grace period
    and the TERM-ignoring child keeps running (and keeps the gate's pipes open).
    """
    child_pid = tmp_path / "child.pid"
    ignoring_child = f'sh -c \'trap "" TERM; echo $$ > "{child_pid}"; exec sleep 30\' & wait'

    def interrupting_run(label: str, command: list[str], *, run: Any, runner: str) -> tuple[int, float, dict[str, Any]]:
        del run, runner
        if label == "gate slow":
            completed = verify._run_gate_process(command, env=dict(os.environ))
            return completed.returncode, 0.0, {}
        for _ in range(1000):
            if verify._LIVE_GATE_PROCESSES and child_pid.exists() and child_pid.read_text().strip():
                break
            time.sleep(0.01)
        raise verify.VerificationInterrupted(signal.SIGTERM)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(verify, "ROOT", tmp_path)
        patch.setattr(verify, "_run", interrupting_run)
        with pytest.raises(verify.VerificationInterrupted):
            verify._run_steps(
                [("gate slow", ["sh", "-c", ignoring_child]), ("gate interrupted", ["true"])],
                run=None,  # type: ignore[arg-type]
                runner="managed",
            )

    pid = int(child_pid.read_text(encoding="utf-8"))
    for _ in range(500):
        if not _process_alive(pid):
            break
        time.sleep(0.01)
    else:
        os.kill(pid, signal.SIGKILL)
        pytest.fail("a TERM-ignoring gate child survived the interruption")


def test_an_interrupted_run_stops_and_joins_its_running_gates(tmp_path: Path) -> None:
    """An interruption terminates live gates and returns only after their workers.

    Anti-vacuity: drop ``_stop_gate_processes`` from the interruption path and
    the sleeping gate outlives the run; shut the pool down without waiting and
    the interrupted gate's worker is still running when ``_run_steps`` raises;
    let a stopped gate return normally and it is recorded as an ordinary result;
    signal only the gate process, not its group, and the checker it started
    (as ``devtools.mypy_gate`` starts ``mypy``) keeps running.
    """
    grandchild_pid = tmp_path / "grandchild.pid"
    spawned: list[subprocess.Popen[str]] = []
    worker_outcomes: list[str] = []
    real_popen = subprocess.Popen

    def tracking_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[str]:
        process = real_popen(*args, **kwargs)
        spawned.append(process)
        return process

    def interrupting_run(label: str, command: list[str], *, run: Any, runner: str) -> tuple[int, float, dict[str, Any]]:
        del run, runner
        if label == "gate slow":
            try:
                completed = verify._run_gate_process(command, env=dict(os.environ))
            except verify._GateInterruptedError:
                time.sleep(0.2)
                worker_outcomes.append("interrupted")
                raise
            worker_outcomes.append("recorded")
            return completed.returncode, 0.0, {}
        for _ in range(1000):
            if verify._LIVE_GATE_PROCESSES and grandchild_pid.exists() and grandchild_pid.read_text().strip():
                break
            time.sleep(0.01)
        raise verify.VerificationInterrupted(signal.SIGTERM)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(verify, "ROOT", tmp_path)
        patch.setattr(subprocess, "Popen", tracking_popen)
        patch.setattr(verify, "_run", interrupting_run)
        with pytest.raises(verify.VerificationInterrupted):
            verify._run_steps(
                [
                    ("gate slow", ["sh", "-c", f'sleep 30 & echo $! > "{grandchild_pid}"; wait']),
                    ("gate interrupted", ["true"]),
                ],
                run=None,  # type: ignore[arg-type]
                runner="managed",
            )

    assert len(spawned) == 1
    pid = int(grandchild_pid.read_text(encoding="utf-8"))
    for _ in range(500):
        if not _process_alive(pid):
            break
        time.sleep(0.01)
    else:
        os.kill(pid, signal.SIGKILL)
        pytest.fail("the gate's child process outlived the interruption")
    # Terminated by the interruption, not left to sleep out its 30 seconds.
    assert spawned[0].poll() is not None
    # Joined before the interruption propagated, and never recorded as a result.
    assert worker_outcomes == ["interrupted"]


def test_verification_tools_are_absolute_paths_in_checkout_venv() -> None:
    steps = verify.build_verify_steps(quick=True)
    commands = dict(steps)

    assert commands["gate format"][0] == str(verify.ROOT / ".venv/bin/ruff")
    assert commands["gate lint"][0] == str(verify.ROOT / ".venv/bin/ruff")
    assert commands["gate mypy"][0].startswith(str(verify.ROOT / ".venv/bin/"))
    assert commands["gate generated-surfaces"][0] == str(verify.ROOT / ".venv/bin/python")
    assert commands["gate schema-closure"][0] == str(verify.ROOT / ".venv/bin/python")


@pytest.mark.parametrize(
    ("provisioned", "expected_diagnosis"),
    [
        # An entirely absent .venv is an unprovisioned checkout, not a tool
        # that happens to be missing: nothing is installed and the remedy is
        # different.
        (False, "gate_unprovisioned_environment"),
        (True, "gate_missing_executable"),
    ],
)
def test_missing_checkout_venv_tool_is_a_typed_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, provisioned: bool, expected_diagnosis: str
) -> None:
    history: dict[str, Any] = {}
    if provisioned:
        (tmp_path / ".venv" / "bin").mkdir(parents=True)
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify, "assert_polylogue_matches_checkout", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(verify, "git_head", lambda _root: "head")
    monkeypatch.setattr(
        verify, "build_verify_steps", lambda **_kwargs: [("gate lint", [str(tmp_path / ".venv/bin/ruff")])]
    )
    monkeypatch.setattr(verify, "append_verify_history", lambda payload, **_kwargs: history.update(payload))

    assert verify._main(["--quick"]) == 127
    assert history["diagnosis"] == expected_diagnosis
    assert history["steps"][0]["required_gate"]["executable"] == str(tmp_path / ".venv/bin/ruff")


def test_broken_venv_script_shebang_is_typed_as_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    script = tmp_path / ".venv/bin/mypy"
    script.parent.mkdir(parents=True)
    script.write_text("#!/missing/interpreter\n", encoding="utf-8")
    script.chmod(0o755)

    result = required_gate.executable_gate_result([str(script)], gate="mypy")

    assert result.diagnosis == "gate_missing_executable"


def test_mypy_gate_uses_a_foreground_checkout_local_process() -> None:
    assert gate.mypy_command() == [str(verify.ROOT / ".venv/bin/python"), "-m", "devtools.mypy_gate"]


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
    monkeypatch.setattr(verify, "append_verify_history", lambda payload, **_kwargs: history.update(payload))

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
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("ruff")),
    )
    monkeypatch.setattr(verify, "append_verify_history", lambda payload, **_kwargs: history.update(payload))

    assert verify._main(["--quick"]) == 127
    assert history["diagnosis"] == "gate_subprocess_launch_failed"
    assert history["steps"][0]["error"] == "ruff"


def test_actual_render_all_diagnosis_reaches_receipt_and_why(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    verify._GATES_INTERRUPTED.clear()
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


def test_descriptor_selection_does_not_enable_archive_prewarm() -> None:
    """Anti-vacuity: descriptor-only tests must not construct shared archives."""
    env = {"POLYLOGUE_BROAD_PREWARM": "1"}
    verify._normalize_managed_pytest_environment(env, verify.DESCRIPTOR_CONTRACT_TESTS)
    assert "POLYLOGUE_BROAD_PREWARM" not in env


def test_optimized_python_is_refused_before_running_verification() -> None:
    """Anti-vacuity: a -O child must report the preflight refusal, not run gates."""
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            "from devtools.verify import _main; raise SystemExit(_main(['--quick', '--json']))",
        ],
        cwd=verify.ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 125
    assert json.loads(result.stdout)["diagnosis"] == "optimized_python"


def test_incomplete_dependency_sync_is_refused_before_running_verification(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A devshell whose ``uv sync --frozen`` failed must not produce a verdict.

    The hook keeps the previous ``.venv`` and exports
    ``POLYLOGUE_DEVSHELL_DEPENDENCY_SYNC=incomplete``; verification of that
    environment would be evidence about another lockfile.

    Anti-vacuity: drop the refusal and ``_main`` proceeds to anchor paths and
    record a run, which the sentinel below turns into a failure.
    """

    monkeypatch.setenv(verify.DEPENDENCY_SYNC_ENV, verify.DEPENDENCY_SYNC_INCOMPLETE)

    def ran_past_preflight() -> None:
        raise AssertionError("verification started on an unsynced environment")

    monkeypatch.setattr(verify, "_anchor_verification_paths", ran_past_preflight)

    exit_code = verify._main(["--quick", "--json"])

    assert exit_code == 125
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "refused"
    assert payload["diagnosis"] == "dependency_sync_incomplete"


def test_interrupted_aggregate_keeps_completed_lane_outcomes() -> None:
    aggregate = verify._aggregate_pytest_results(
        [{"name": "pytest (parallel)", "statistics": {"outcomes": {"passed": 4}}}],
        expected_step_count=3,
        mode="all",
        exit_code=130,
    )
    assert aggregate["outcomes"] == {"passed": 4}
    assert aggregate["terminal_green"] is False
    assert aggregate["complete_corpus_covered"] is False


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
    assert result["exit"] == result["process_exit"] == 1
    assert result["diagnosis"] == "pytest_evidence_unavailable"
    assert result["evidence_error"]["type"] == "OSError"


def test_broad_managed_pytest_profile_honors_environment_then_defaults() -> None:
    env = {"HYPOTHESIS_PROFILE": "ci", "POLYLOGUE_CI": "1"}
    verify._normalize_managed_pytest_environment(env)
    assert env["HYPOTHESIS_PROFILE"] == "ci"
    assert "POLYLOGUE_CI" not in env
    default: dict[str, str] = {}
    verify._normalize_managed_pytest_environment(default)
    assert default["HYPOTHESIS_PROFILE"] == "default"


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


@pytest.mark.parametrize("variable", ["AGENTCTL_OPERATION", "SINNIXD_OPERATION"])
def test_declared_operation_requires_the_fixed_route(monkeypatch: pytest.MonkeyPatch, variable: str) -> None:
    monkeypatch.delenv("AGENTCTL_OPERATION", raising=False)
    monkeypatch.delenv("SINNIXD_OPERATION", raising=False)
    monkeypatch.setenv(variable, "verify_quick")

    assert verify._declared_agentctl_operation(["--quick"]) == "verify_quick"
    assert verify._declared_agentctl_operation([]) is None


def test_focused_profile_requires_a_behavioral_pytest_selection() -> None:
    descriptor = tomllib.loads((verify.ROOT / ".agentctl/project.toml").read_text(encoding="utf-8"))

    focused = descriptor["operations"]["pytest_focused"]
    operation = descriptor["operations"]["verify_quick"]
    affected = descriptor["operations"]["verify_affected"]
    complete = descriptor["operations"]["verify_all"]
    projection = declared_verification_result(
        {"exit_code": 0, "status": "success", "verification_scope": "non-test"},
        operation="verify_quick",
    )

    assert descriptor["workspace"]["verify"] == {
        "focused": "pytest_focused",
        "candidate": "hosted:ci/circleci: quick-gate",
    }
    assert descriptor["workspace"]["publish"] == "pr"
    assert focused["exec"] == ["python", "-m", "devtools.pytest_slot"]
    assert focused["arguments"] == "required"
    assert focused["result"] == "pytest"
    assert operation["exec"] == ["devtools", "verify", "--quick"]
    assert operation["result"] == "json"
    assert operation["timeout_seconds"] == 2400
    assert affected["exec"] == ["devtools", "verify"]
    assert affected["pool"] == "pytest"
    assert affected["result"] == "pytest"
    assert affected["cache"] == "tree+environment"
    assert affected["timeout_seconds"] == 7200
    # Anti-vacuity: deleting either operation's descriptor deadline makes
    # this fail, even if AgentCTL applies a host default.
    assert complete["exec"] == ["devtools", "verify", "--all"]
    # polylogue-p2mbi AC4 (#5405): `checkout = "candidate"`. The unset default
    # does not select a tree, it REFUSES every workspace but the project root
    # -- which is the operator's working checkout and deliberately divergent,
    # so the corpus run could only ever qualify that branch and a coordinator
    # could not point it at an integrated candidate. Anti-vacuity: removing the
    # key makes this red.
    assert complete["checkout"] == "candidate"
    assert complete["pool"] == "pytest-heavy"
    assert complete["result"] == "pytest"
    assert complete["cache"] == "tree+environment"
    assert complete["timeout_seconds"] == 14400
    assert projection["kind"] == "polylogue.verification-result"
    assert projection["operation"] == "verify_quick"


def test_verification_docs_distinguish_local_receipts_and_sidecars() -> None:
    """Anti-vacuity: docs must name real mutable DB files and local run evidence."""
    sidecars = (verify.ROOT / "docs/sidecars.md").read_text(encoding="utf-8")
    authority = (verify.ROOT / "docs/verification-authority.md").read_text(encoding="utf-8")

    assert ".cache/testmon/testmondata` plus `-wal`, `-shm`, and `-journal" in sidecars
    assert "`.cache/verify/graph/**`" in sidecars
    assert "`.testmondata.bound-*`" in sidecars
    assert "append them to the checkout-local run" in authority
    assert "AgentCTL-managed job evidence" in authority
    assert "no AgentCTL run record" not in authority


def test_agentctl_parser_preserves_distinct_verification_pools() -> None:
    """The production descriptor parser sees affected and corpus pool policy."""
    repository_root = Path(__file__).resolve().parents[3]
    sinnix_root = Path("/realm/project/sinnix")
    package_root = sinnix_root / "pkgs" / "agentctl"
    parser_program = """
import json
import sys
from pathlib import Path

from agentctl.projects import load_project_adapter

adapter = load_project_adapter(Path(sys.argv[1]))
affected = adapter.operation("verify_affected")
complete = adapter.operation("verify_all")
print(json.dumps({
    "affected": {
        "command": affected.command,
        "pool": affected.pool,
        "result": affected.result,
        "timeout": affected.timeout_seconds,
    },
    "complete": {
        "command": complete.command,
        "pool": complete.pool,
        "result": complete.result,
        "timeout": complete.timeout_seconds,
        "checkout": complete.checkout,
    },
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", parser_program, str(repository_root)],
        cwd=sinnix_root,
        env=os.environ | {"PYTHONPATH": str(package_root)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "affected": {
            "command": ["devtools", "verify"],
            "pool": "pytest",
            "result": "pytest",
            "timeout": 7200,
        },
        "complete": {
            "command": ["devtools", "verify", "--all"],
            "pool": "pytest-heavy",
            "result": "pytest",
            "timeout": 14400,
            # Read back through the production parser rather than the TOML, so
            # what "candidate" resolves to is agentctl's answer and not this
            # test's guess at it (polylogue-p2mbi AC4, #5405).
            "checkout": "candidate",
        },
    }


def test_descriptor_only_changes_use_contract_tests_and_python_changes_use_testmon() -> None:
    """A descriptor-only diff is bounded to descriptor contracts.

    Anti-vacuity: selecting the affected mode for the descriptor or selecting
    descriptor contracts for a Python diff makes one of the boundary checks
    below fail.
    """
    assert verify._selection_for_changes(frozenset({".agentctl/project.toml"})) == "descriptor"
    assert verify._selection_for_changes(frozenset({".agentctl/project.toml", "README.md"})) == "descriptor"
    assert verify._selection_for_changes(frozenset({"polylogue/example.py"})) == "affected"
    assert verify._selection_for_changes(frozenset({".agentctl/project.toml", "polylogue/example.py"})) == "affected"
    assert verify._selection_for_changes(None) == "affected"
    assert verify._selection_for_changes(frozenset()) == "affected"

    descriptor_command = verify._pytest_steps(selection="descriptor", worker_args=[])[0][1]
    assert "--testmon" not in descriptor_command
    assert "tests" not in descriptor_command
    contract_slice = [*verify.DESCRIPTOR_CONTRACT_TESTS, *verify.CONTRACT_DOCUMENT_TESTS]
    assert descriptor_command[-len(contract_slice) :] == contract_slice

    affected_command = verify._pytest_steps(selection="affected", worker_args=[])[0][1]
    assert "--testmon" in affected_command
    assert "--testmon-forceselect" in affected_command
    assert not any(nodeid in affected_command for nodeid in verify.DESCRIPTOR_CONTRACT_TESTS)


@pytest.mark.parametrize(
    "changed",
    [
        frozenset({"docs/devtools.md"}),
        frozenset({".github/workflows/verify.yml"}),
        frozenset({".agentctl/README.md", "README.md", ".github/CODEOWNERS"}),
    ],
)
def test_metadata_only_changes_select_no_pytest_step(changed: frozenset[str]) -> None:
    """Orchestration metadata, documentation and workflows are exercised by no test.

    Anti-vacuity: dropping the path-class rule routes these through the testmon
    graph, which selects most of the corpus for a change outside Python.
    """
    assert verify._selection_for_changes(changed) == "none"
    assert verify._selection_reason("none")
    assert verify._selection_reason("affected") is None
    assert verify._scope(quick=False, selection="none") is VerificationScope.NON_TEST
    labels = [label for label, _command in verify.build_verify_steps(quick=False, selection="none")]
    assert labels and not any(label.startswith("pytest") for label in labels)


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        (frozenset({"docs/devtools.md", "polylogue/example.py"}), "affected"),
        (frozenset({"README.md", "tests/unit/test_example.py"}), "affected"),
        (frozenset({"pyproject.toml"}), "affected"),
    ],
)
def test_one_code_path_makes_the_change_set_affected(changed: frozenset[str], expected: str) -> None:
    assert verify._selection_for_changes(changed) == expected


def test_an_agents_only_change_runs_the_tracked_reference_ratchet() -> None:
    """AGENTS.md is read by a contract test, so it is not no-test documentation.

    Anti-vacuity: drop ``_CONTRACT_READ_DOCUMENTS`` and an AGENTS-only change
    selects ``none``, so the retired-name ratchet never runs on it.
    """
    assert verify._selection_for_changes(frozenset({"AGENTS.md"})) == "descriptor"
    commands = [command for label, command in verify.build_verify_steps(quick=False, selection="descriptor")]
    ratchet = (
        "tests/unit/architecture/test_retired_analysis_modules.py::test_no_tracked_reference_to_a_retired_analysis_name"
    )
    assert any(ratchet in command for command in commands)


def test_a_mixed_change_touching_agents_still_runs_the_ratchet() -> None:
    """AGENTS.md plus a source edit keeps the affected selection and adds the ratchet.

    Anti-vacuity: without the contract-document step, the affected step is the
    only pytest step, and testmon never selects a test that reads AGENTS.md
    through ``git grep``, so a retired name added there passes the verifier.
    """
    changed = frozenset({"AGENTS.md", "polylogue/example.py"})
    assert verify._selection_for_changes(changed) == "affected"
    steps = [
        (label, command)
        for label, command in verify.build_verify_steps(quick=False, selection="affected", changed_paths=changed)
        if label.startswith("pytest")
    ]
    assert [label for label, _command in steps] == ["pytest (affected)", "pytest (contract documents)"]
    affected_command, contract_command = (command for _label, command in steps)
    assert "--testmon-forceselect" in affected_command
    assert not any(nodeid in affected_command for nodeid in verify.CONTRACT_DOCUMENT_TESTS)
    assert "--testmon" not in contract_command
    assert contract_command[-len(verify.CONTRACT_DOCUMENT_TESTS) :] == list(verify.CONTRACT_DOCUMENT_TESTS)

    source_only = frozenset({"polylogue/example.py"})
    labels = [
        label
        for label, _command in verify.build_verify_steps(quick=False, selection="affected", changed_paths=source_only)
    ]
    assert "pytest (contract documents)" not in labels


class _StubTestmonData:
    """Just enough of ``TestmonData`` for the estimator to reach its arithmetic."""

    system_packages_change = False

    def __init__(self, selected: tuple[str, ...], recorded: tuple[str, ...] = ()) -> None:
        self.unstable_test_names = list(selected)
        self.failing_tests: list[str] = []
        self.all_tests = {name: {"duration": 0.5} for name in (*selected, *recorded)}

    @classmethod
    def factory(cls, selected: tuple[str, ...], recorded: tuple[str, ...] = ()) -> Any:
        return lambda **_kwargs: cls(selected, recorded)

    def determine_stable(self) -> None:
        return None


def _stub_affected_graph(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    selected: tuple[str, ...],
    unrecorded_files: tuple[str, ...],
    unrecorded_tests: int | None,
    recorded: tuple[str, ...] = (),
) -> None:
    """Point the estimator at a stub graph with a known selection and unknown set."""
    import testmon.db
    import testmon.testmon_core

    datafile = _testmon_datafile(tmp_path)
    datafile.parent.mkdir(parents=True, exist_ok=True)
    sqlite3.connect(datafile).close()
    monkeypatch.setattr(verify, "snapshot_testmon_graph", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        verify, "inspect_testmon_graph", lambda *_args, **_kwargs: SimpleNamespace(usable=True, full_rerun_cause=None)
    )
    monkeypatch.setattr(testmon.db, "DB", lambda *_a, **_k: SimpleNamespace(con=SimpleNamespace(close=lambda: None)))
    monkeypatch.setattr(
        testmon.testmon_core.TestmonData, "for_local_run", _StubTestmonData.factory(selected, recorded), raising=False
    )
    from devtools.verify_test_collection import CollectedSelection

    monkeypatch.setattr(verify, "declared_test_files", lambda _root: frozenset(unrecorded_files))

    def collect(**kwargs: Any) -> CollectedSelection | None:
        if unrecorded_tests is None:
            return None
        paths = kwargs.get("paths")
        if paths:
            nodes = tuple(
                name
                for name in (*selected, *recorded)
                if any(name == path or name.startswith(path + "[") for path in paths)
            )
            return CollectedSelection(len(nodes), nodes, 0) if nodes else None
        unknown = tuple(f"tests/unit/a.py::test_new[{index}]" for index in range(unrecorded_tests))
        nodes = (*selected, *unknown)
        return CollectedSelection(len(nodes), nodes, 0)

    monkeypatch.setattr(verify, "collect_selection", collect)


def test_the_estimate_counts_forced_contract_tests(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Forced contract tests count toward the admitted plan, every parametrization.

    Anti-vacuity: drop ``forced_tests`` from the estimate and the count stays
    at the one testmon-selected test while the run launches three; union the
    forced tests into the selection and the overlap case counts three of the
    four launches.
    """
    graph = SimpleNamespace(status=TestmonGraphStatus.USABLE, full_rerun_cause=None)
    (forced,) = verify.CONTRACT_DOCUMENT_TESTS
    _stub_affected_graph(
        monkeypatch,
        tmp_path,
        selected=("tests/unit/a.py::test_one",),
        unrecorded_files=(),
        unrecorded_tests=0,
        recorded=(f"{forced}[alpha]", f"{forced}[beta]", "tests/unit/b.py::test_unrelated"),
    )

    count, seconds, error, _unrecorded = verify._estimate_affected_selection(tmp_path, graph, (forced,))
    assert (count, seconds, error) == (3, 1.5, None)

    count, _seconds, error, _unrecorded = verify._estimate_affected_selection(
        tmp_path, graph, ("tests/unit/never.py::test_missing",)
    )
    assert count is None and error is not None

    # A forced test the affected step also selected runs twice and counts twice.
    _stub_affected_graph(
        monkeypatch,
        tmp_path,
        selected=("tests/unit/a.py::test_one", f"{forced}[alpha]"),
        unrecorded_files=(),
        unrecorded_tests=0,
        recorded=(f"{forced}[beta]",),
    )
    count, seconds, error, _unrecorded = verify._estimate_affected_selection(tmp_path, graph, (forced,))
    assert (count, seconds, error) == (4, 2.0, None)


def test_an_agents_only_selection_reason_names_the_contract_document() -> None:
    """Anti-vacuity: reuse the descriptor reason and an AGENTS-only receipt
    claims the change included the AgentCTL descriptor."""
    reason = verify._selection_reason("descriptor", frozenset({"AGENTS.md"}))
    assert reason is not None and "AGENTS.md" in reason and "descriptor" not in reason
    descriptor_reason = verify._selection_reason("descriptor", frozenset({".agentctl/project.toml", "AGENTS.md"}))
    assert descriptor_reason is not None and "AgentCTL descriptor" in descriptor_reason


def test_the_estimate_counts_tests_the_graph_never_recorded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Unknown nodes are included in the actual launch count and receipt.

    Removing the unknown selected identities understates the first plan and
    admits the oversized second plan. New nodes share a recorded filename in
    the stub, so filename subtraction cannot recover their count.
    """
    graph = SimpleNamespace(status=TestmonGraphStatus.USABLE, full_rerun_cause=None)
    _stub_affected_graph(
        monkeypatch,
        tmp_path,
        selected=("tests/unit/a.py::test_one", "tests/unit/a.py::test_two"),
        unrecorded_files=("tests/unit/new.py",),
        unrecorded_tests=68,
    )

    count, seconds, error, unrecorded = verify._estimate_affected_selection(tmp_path, graph)

    assert (count, unrecorded, error) == (70, 68, None)
    assert seconds == 1.0

    _stub_affected_graph(
        monkeypatch,
        tmp_path,
        selected=("tests/unit/a.py::test_one", "tests/unit/a.py::test_two"),
        unrecorded_files=("tests/unit/new.py",),
        unrecorded_tests=AFFECTED_MAX_SELECTED_TESTS + 1,
    )
    decision = verify._affected_admission(root=tmp_path, graph=graph)
    assert decision.status == "refused"
    assert decision.unrecorded_tests == AFFECTED_MAX_SELECTED_TESTS + 1


@pytest.mark.parametrize(
    ("unrecorded_files", "unrecorded_tests", "expected"),
    [
        (tuple(f"tests/unit/test_{index}.py" for index in range(AFFECTED_MAX_UNRECORDED_FILES + 1)), 0, "more than"),
        (("tests/unit/new.py",), None, "could not be collected"),
    ],
)
def test_an_unpriceable_unknown_set_refuses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    unrecorded_files: tuple[str, ...],
    unrecorded_tests: int | None,
    expected: str,
) -> None:
    _stub_affected_graph(
        monkeypatch, tmp_path, selected=(), unrecorded_files=unrecorded_files, unrecorded_tests=unrecorded_tests
    )
    graph = SimpleNamespace(status=TestmonGraphStatus.USABLE, full_rerun_cause=None)
    count, _seconds, reason, _unknown = verify._estimate_affected_selection(tmp_path, graph)
    assert count is None
    assert reason is not None and expected in reason


def test_a_markdown_test_fixture_still_selects_tests() -> None:
    """Markdown under ``tests/`` is fixture content, not documentation.

    ``tests/data/golden/chatgpt-simple.md`` is read and compared byte-for-byte
    by ``tests/unit/ui/test_ui_visual.py``'s
    ``TestGoldenMarkdownRendering::test_chatgpt_simple_session``. The blanket
    ``.md`` suffix exemption made a change set containing only that fixture
    select no pytest at all, and the hosted check accepted the resulting
    no-test receipt while reporting that no test exercises the path.

    Anti-vacuity: drop the ``tests/`` carve-out from ``_no_test_path`` and the
    first two assertions go red. The last two pin the opposite direction, so
    retiring the exemption wholesale -- which would route every README edit
    through the testmon graph -- does not pass either.
    """
    fixture = "tests/data/golden/chatgpt-simple.md"
    checkout = Path(__file__).resolve().parents[3]
    assert (checkout / fixture).is_file(), "the fixture this case is anchored to moved"
    assert verify._selection_for_changes(frozenset({fixture})) == "affected"
    assert verify._no_test_path(fixture) is False
    assert verify._no_test_path("docs/devtools.md") is True
    assert verify._selection_for_changes(frozenset({"docs/devtools.md"})) == "none"


def test_verify_main_records_why_no_pytest_step_ran(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A run with no pytest step carries its reason in the receipt, where the hosted check reads it."""
    from devtools.agent_env import AGENT_PRINCIPAL, AGENT_PRINCIPAL_ENV

    history: dict[str, Any] = {}
    steps_seen: dict[str, Any] = {}
    monkeypatch.setenv(AGENT_PRINCIPAL_ENV, AGENT_PRINCIPAL)
    monkeypatch.setattr(verify, "refuse_verify_tier", lambda _argv, _env: None)
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify, "_git_changed_paths", lambda _root: frozenset({"docs/devtools.md"}))
    monkeypatch.setattr(verify, "sync_testmon_graph", lambda _root: False)
    monkeypatch.setattr(
        verify,
        "inspect_testmon_graph",
        lambda _root: SimpleNamespace(status=TestmonGraphStatus.UNUSABLE, reason="corrupt", full_rerun_cause=None),
    )
    monkeypatch.setattr(verify, "assert_polylogue_matches_checkout", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(verify, "git_head", lambda _root: "head")

    def capture_steps(**kwargs: Any) -> list[tuple[str, list[str]]]:
        steps_seen.update(kwargs)
        return [("gate lint", ["true"])]

    monkeypatch.setattr(verify, "build_verify_steps", capture_steps)
    monkeypatch.setattr(verify, "_run", lambda *_args, **_kwargs: (0, 0.1, {"diagnosis": "gate_passed"}))
    monkeypatch.setattr(verify, "append_verify_history", lambda payload, **_kwargs: history.update(payload))
    monkeypatch.setattr(verify, "append_verification_evidence", lambda _payload: None)
    monkeypatch.setattr(verify, "prune_successful_verify_runs", lambda **_kwargs: None)

    assert verify._main([]) == 0, "an unusable graph is irrelevant when no selection consults it"
    assert steps_seen["selection"] == "none"
    assert history["testmon_selection"]["selection_mode"] == "none"
    assert "no test exercises them" in history["testmon_selection"]["selection_reason"]
    assert history["verification_scope"] == VerificationScope.NON_TEST.value
    run_payload = json.loads((tmp_path / str(history["artifact_dir"]) / "run.json").read_text())
    assert run_payload["testmon_selection"]["selection_reason"] == history["testmon_selection"]["selection_reason"]


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
    monkeypatch.setattr(verify, "append_verify_history", lambda payload, **_kwargs: history.update(payload))
    monkeypatch.setattr(verify, "append_verification_evidence", lambda _payload: None)
    monkeypatch.setattr(verify, "prune_successful_verify_runs", lambda **_kwargs: None)

    assert verify._main([]) == 0
    assert captured["selection"] == "descriptor"
    assert history["pytest_aggregate"]["selection_mode"] == "descriptor"


@pytest.mark.parametrize(
    ("graph_status", "selected_count", "expected_status"),
    [
        (TestmonGraphStatus.USABLE, AFFECTED_MAX_SELECTED_TESTS + 1, "refused"),
        (TestmonGraphStatus.UNUSABLE, None, "unknown"),
    ],
)
def test_affected_admission_refuses_without_launching_pytest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    graph_status: TestmonGraphStatus,
    selected_count: int | None,
    expected_status: str,
) -> None:
    """An oversized or unknown plan is terminal before any pytest step exists."""
    from devtools.agent_env import AGENT_PRINCIPAL, AGENT_PRINCIPAL_ENV

    history: dict[str, Any] = {}
    launched: list[str] = []
    monkeypatch.setenv(AGENT_PRINCIPAL_ENV, AGENT_PRINCIPAL)
    monkeypatch.setattr(verify, "refuse_verify_tier", lambda _argv, _env: None)
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify, "_git_changed_paths", lambda _root: frozenset({"polylogue/example.py"}))
    monkeypatch.setattr(verify, "sync_testmon_graph", lambda _root: False)
    graph = SimpleNamespace(status=graph_status, reason="synthetic graph state", full_rerun_cause=None)
    monkeypatch.setattr(verify, "inspect_testmon_graph", lambda _root: graph)
    monkeypatch.setattr(
        verify,
        "_estimate_affected_selection",
        lambda _root, _graph, _forced=(), **_kwargs: (selected_count, 1.0, None, 0),
    )
    monkeypatch.setattr(verify, "assert_polylogue_matches_checkout", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(verify, "git_head", lambda _root: "head")

    def capture_run(label: str, _command: list[str], **_kwargs: Any) -> tuple[int, float, dict[str, Any]]:
        launched.append(label)
        return 0, 0.1, {}

    monkeypatch.setattr(verify, "_run", capture_run)
    monkeypatch.setattr(verify, "append_verify_history", lambda payload, **_kwargs: history.update(payload))
    monkeypatch.setattr(verify, "append_verification_evidence", lambda _payload: None)
    monkeypatch.setattr(verify, "prune_successful_verify_runs", lambda **_kwargs: None)

    assert verify._main([]) == 2
    assert launched, "an affected admission refusal must still run the static gates"
    assert not any(label.startswith("pytest") for label in launched)
    assert history["testmon_selection"]["admission"]["status"] == expected_status
    assert history["pytest_aggregate"]["selected_union_count"] == selected_count
    assert history["pytest_aggregate"]["terminal_union_count"] == 0
    # The refusal withholds pytest from execution, not from the declared plan.
    receipt = history["workload_receipt"]
    assert "pytest (affected)" in receipt["spec"]["phases"]
    # Gates finish in any order; the receipt observes them in declared order.
    assert [phase["name"] for phase in receipt["phases"]] == [
        phase for phase in receipt["spec"]["phases"] if not phase.startswith("pytest")
    ]
    assert sorted(launched) == sorted(phase["name"] for phase in receipt["phases"])
    assert receipt["status"] == "failed"
    output = capsys.readouterr().err
    assert "refused before pytest launch" in output
    assert "next boundary" in output


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


@pytest.mark.parametrize("runner", ["managed", "isolated"])
def test_verify_retains_first_failure_without_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runner: str
) -> None:
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify, "executable_gate_result", lambda *_args, **_kwargs: SimpleNamespace(ok=True))
    calls: list[list[str]] = []

    def execute(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(command)
        report = verify._pytest_report_path(command)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                {
                    "exitcode": 1,
                    "summary": {"failed": 1},
                    "tests": [{"nodeid": "tests/test_a.py::test_red", "outcome": "failed"}],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=1, slot="held", receipt=None, termination=None)

    monkeypatch.setattr(verify, "run_pytest", execute)
    monkeypatch.setattr(verify, "run_pytest_isolated", execute)
    run = VerifyRun(tier="test", argv=[], git_head="head", root=tmp_path)
    exit_code, _elapsed, metadata = verify._run("pytest selected", ["pytest"], run=run, runner=runner)
    assert exit_code == 1
    assert len(calls) == 1
    assert "rerun" not in metadata
    assert (
        json.loads(verify._pytest_report_path(calls[0]).read_text(encoding="utf-8"))["tests"][0]["outcome"] == "failed"
    )


def test_zero_exit_without_a_report_is_a_failed_pytest_step(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify, "_clear_pytest_report", lambda _command: None)
    monkeypatch.setattr(verify, "executable_gate_result", lambda *_args, **_kwargs: SimpleNamespace(ok=True))
    # The executor boundary: the managed runner reports exit 0 and writes no
    # report. The held runner launches pytest with ``Popen``, so the former
    # ``subprocess.run`` stub never reached it: a real ``pytest`` ran in the
    # empty directory and exited 5, which read as ``pytest_failed``.
    monkeypatch.setattr(
        verify,
        "run_pytest",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, slot="held", receipt=None, termination=None),
    )
    run = VerifyRun(tier="test", argv=[], git_head="head", root=tmp_path)

    exit_code, _elapsed, metadata = verify._run("pytest serial (all)", ["pytest"], run=run)

    assert exit_code != 0
    assert metadata["diagnosis"] == "pytest_no_report"
    assert metadata["statistics"]["ordinary_eligible"] is False


@pytest.mark.parametrize("runner", ["managed", "isolated"])
def test_verify_pytest_step_uses_the_explicit_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runner: str
) -> None:
    """CI isolation is opt-in; the normal verifier remains pool-managed."""
    called: list[str] = []
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify, "_clear_pytest_report", lambda _command: None)
    monkeypatch.setattr(verify, "executable_gate_result", lambda *_args, **_kwargs: SimpleNamespace(ok=True))

    def managed(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        called.append("managed")
        return SimpleNamespace(returncode=0, slot="managed", receipt=None, termination=None)

    def isolated(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        called.append("isolated")
        return SimpleNamespace(returncode=0, slot="isolated", receipt=None, termination=None)

    monkeypatch.setattr(verify, "run_pytest", managed)
    monkeypatch.setattr(verify, "run_pytest_isolated", isolated)
    run = VerifyRun(tier="test", argv=[], git_head="head", root=tmp_path)

    _exit_code, _elapsed, metadata = verify._run("pytest selected", ["pytest"], run=run, runner=runner)

    assert called == [runner]
    assert metadata["runner"] == runner
    assert metadata["pytest_slot"] == runner


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
        runner: str = "managed",
    ) -> tuple[int, float, dict[str, object]]:
        run.start_step(label=label, cmd=command)
        raise verify.VerificationInterrupted(signal.SIGTERM)

    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify, "assert_polylogue_matches_checkout", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(verify, "git_head", lambda _root: "head")
    monkeypatch.setattr(verify, "build_verify_steps", lambda **_kwargs: [("pytest parallel (all)", ["pytest"])])
    monkeypatch.setattr(verify, "_run", interrupt)
    monkeypatch.setattr(verify, "append_verify_history", lambda payload, **_kwargs: history.update(payload))

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
    monkeypatch.setattr(verify, "append_verify_history", lambda payload, **_kwargs: history.update(payload))

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


_PLANNED_STEPS: list[tuple[str, list[str]]] = [
    ("gate a", ["a"]),
    ("gate b", ["b"]),
    ("pytest (all)", ["pytest", "--dist=loadgroup", "-n", "8"]),
]


def _drive_planned_verification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    interrupt_pytest: bool,
    dirty: bool = False,
    operation: str | None = None,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """Run ``_main`` over ``_PLANNED_STEPS``; return exit, history row, and the spec persisted before any step."""
    history: dict[str, Any] = {}
    declared: dict[str, Any] = {}

    def run_step(
        label: str, command: list[str], *, run: VerifyRun, runner: str = "managed"
    ) -> tuple[int, float, dict[str, object]]:
        del runner
        declared.setdefault("spec", json.loads((run.run_dir / "run.json").read_text()).get("workload_spec"))
        artifacts = run.start_step(label=label, cmd=command)
        if interrupt_pytest and label.startswith("pytest"):
            raise verify.VerificationInterrupted(signal.SIGTERM)
        run.finish_step(step_id=artifacts.step_id, result={"duration_s": 0.5, "exit": 0})
        return 0, 0.5, {}

    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify, "assert_polylogue_matches_checkout", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(verify, "git_head", lambda _root: "head")
    monkeypatch.setattr(verify, "git_worktree_content_sha256", lambda _root: "content")
    monkeypatch.setattr(verify_runs, "git_dirty", lambda *_args, **_kwargs: dirty)
    monkeypatch.setattr(verify, "build_verify_steps", lambda **_kwargs: list(_PLANNED_STEPS))
    monkeypatch.setattr(verify, "_run", run_step)
    monkeypatch.setattr(verify, "append_verify_history", lambda payload, **_kwargs: history.update(payload))
    monkeypatch.setattr(verify, "append_verification_evidence", lambda _payload: None)
    monkeypatch.setattr(verify, "prune_successful_verify_runs", lambda **_kwargs: None)

    exit_code = verify._main(["--quick"], agentctl_operation=operation)
    return exit_code, history, declared["spec"]


def test_interrupted_verification_receipt_keeps_the_plan_declared_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One workload identity names one plan, however far the run got.

    Anti-vacuity: derive the spec's phases from the executed results again and
    the interrupted receipt's spec loses ``pytest (all)``, so its ``spec_id``
    differs from the completed run's and from the spec persisted before the
    first step.
    """
    exit_code, interrupted, declared = _drive_planned_verification(monkeypatch, tmp_path, interrupt_pytest=True)
    assert exit_code == 143
    receipt = interrupted["workload_receipt"]
    assert declared == receipt["spec"]
    assert receipt["spec"]["phases"] == [label for label, _command in _PLANNED_STEPS]
    assert receipt["status"] == "interrupted"
    assert [phase["name"] for phase in receipt["phases"]] == ["gate a", "gate b"]

    exit_code, completed, _declared = _drive_planned_verification(monkeypatch, tmp_path, interrupt_pytest=False)
    assert exit_code == 0
    assert completed["workload_receipt"]["spec_id"] == receipt["spec_id"]
    assert completed["workload_receipt"]["status"] == "succeeded"
    assert [phase["name"] for phase in completed["workload_receipt"]["phases"]] == receipt["spec"]["phases"]


@pytest.mark.parametrize(
    ("dirty", "expected_build"),
    [(False, "git:head"), (True, "worktree-sha256:content")],
)
def test_workload_receipt_names_a_dirty_tree_by_content_not_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dirty: bool,
    expected_build: str,
) -> None:
    """A dirty run executed its working tree, not the immutable HEAD build.

    Anti-vacuity: build the identity from ``git_head`` alone and the dirty case
    claims ``git:head``.
    """
    _exit, history, _declared = _drive_planned_verification(monkeypatch, tmp_path, interrupt_pytest=False, dirty=dirty)
    receipt = history["workload_receipt"]
    assert receipt["build_id"] == expected_build
    assert [ref["input_id"] for ref in receipt["spec"]["inputs"]] == [expected_build]


def test_agentctl_result_carries_the_persisted_workload_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The operation's stdout cites the same workload receipt run.json holds.

    Anti-vacuity: drop the ``workload_receipt`` projection from ``_emit`` and
    the AgentCTL result has no receipt to compare.
    """
    _exit, history, _declared = _drive_planned_verification(
        monkeypatch, tmp_path, interrupt_pytest=False, operation="verify_quick"
    )
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    persisted = json.loads((tmp_path / str(history["artifact_dir"]) / "run.json").read_text())
    assert result["kind"] == "polylogue.verification-result"
    assert result["workload_receipt"] == persisted["workload_receipt"]
    assert result["workload_receipt"]["receipt_id"] == history["workload_receipt"]["receipt_id"]


def test_workload_spec_concurrency_is_the_planned_worker_width(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The declared concurrency is the plan's widest fan-out, not a constant 1.

    Anti-vacuity: leave ``WorkloadEnvelopeSpec.concurrency`` at its default and
    both the xdist plan and the gate-pool plan declare 1.
    """
    monkeypatch.setattr(verify, "GATE_PARALLELISM", 4)
    xdist = verify._verification_workload_spec(tier="all", steps=_PLANNED_STEPS, build_id=None)
    assert xdist.concurrency == 8
    gates_only = verify._verification_workload_spec(
        tier="quick",
        steps=[("gate a", ["a"]), ("gate b", ["b"]), ("pytest (all)", ["pytest", "-n", "0"])],
        build_id=None,
    )
    assert gates_only.concurrency == 2

    # The production plan: the capped xdist width its pytest command carries.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POLYLOGUE_PYTEST_WORKERS", "3")
    planned = verify.build_verify_steps(quick=False, selection="all")
    (pytest_command,) = [command for label, command in planned if label.startswith("pytest")]
    width = int(verify_runs.pytest_command_worker_request(pytest_command) or 0)
    assert width == min(3, worker_memory.CORPUS_MAX_WORKERS)
    assert verify._verification_workload_spec(tier="all", steps=planned, build_id=None).concurrency >= max(1, width)


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


def test_focused_explicit_workers_are_sized_inside_the_admitted_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command keeps intent; the slot measures the live cgroup on start.

    Anti-vacuity: restoring the old agent-identity cap would rewrite the
    explicit request before the pool can apply its actual memory bound.
    ``test_corpus_worker_memory_bound`` exercises that slot-side bound.
    """

    from devtools import agent_env, run_tests

    monkeypatch.setenv(agent_env.AGENT_PRINCIPAL_ENV, agent_env.AGENT_PRINCIPAL)
    command = run_tests.build_pytest_cmd(["tests/unit/foo.py", "-n", "32"])

    assert verify_runs.pytest_command_worker_request(command) == "32"


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


def test_focused_managed_runs_default_to_the_verify_hypothesis_profile() -> None:

    from devtools import run_tests

    env = {"HYPOTHESIS_PROFILE": "ci", "PATH": "/usr/bin"}
    run_tests._normalize_managed_pytest_environment(env)

    assert env["HYPOTHESIS_PROFILE"] == "ci"
    default: dict[str, str] = {}
    run_tests._normalize_managed_pytest_environment(default)
    assert default["HYPOTHESIS_PROFILE"] == "verify"


def test_hypothesis_profile_prefers_cli_then_environment_then_default() -> None:
    from devtools.pytest_invocation import effective_hypothesis_profile

    assert effective_hypothesis_profile(
        ["--hypothesis-profile", "cli"], {"HYPOTHESIS_PROFILE": "env"}, default="fallback"
    ) == (
        "cli",
        "cli",
    )
    assert effective_hypothesis_profile(
        ["--hypothesis-profile=first", "--hypothesis-profile", "final"],
        {"HYPOTHESIS_PROFILE": "env"},
        default="fallback",
    ) == ("final", "cli")
    assert effective_hypothesis_profile([], {"HYPOTHESIS_PROFILE": "env"}, default="fallback") == ("env", "environment")
    assert effective_hypothesis_profile([], {}, default="fallback") == ("fallback", "default")


def test_agent_tier_refusal_honors_the_json_contract(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Anti-vacuity: restore the unconditional stderr write and stdout is empty
    so the json.loads below raises.

    `--json` is the machine-readable contract. A refusal that answers in prose
    leaves a hole in it exactly where an automated caller needs a verdict.
    """

    _outside_the_pytest_pool(monkeypatch, tmp_path)
    monkeypatch.setenv(agent_env.AGENT_PRINCIPAL_ENV, agent_env.AGENT_PRINCIPAL)

    exit_code = verify._main(["--json"])

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "refused"
    assert payload["diagnosis"] == "agent_tier_refused"
    assert payload["exit_code"] == 2
    assert "devtools test" in payload["message"]


def test_schema_promotion_audits_the_tree_it_writes_to(monkeypatch: pytest.MonkeyPatch) -> None:
    """The audited root is the registry storage root promotion writes to.

    This case previously asserted only that the root is an absolute existing
    directory named ``schemas``, which the installed ``polylogue/schemas``
    package satisfies -- and that is NOT where promotion writes.
    ``promote_schema_cluster`` goes through
    ``polylogue.schemas.operator.registry.schema_registry()``, whose
    ``storage_root`` is ``data_home()/schemas``, so the audit was inspecting
    bundled artifacts promotion never touched.

    Anti-vacuity: restore ``Path(next(iter(polylogue.schemas.__path__)))`` and
    the equality below fails, because that path is inside the checkout and
    does not move with ``XDG_DATA_HOME``.
    """

    from devtools import schema_promote
    from polylogue.schemas.registry import SchemaRegistry

    root = schema_promote._schema_registry_root()

    assert root.is_absolute()
    assert root.name == "schemas"
    assert root == SchemaRegistry().storage_root


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


def test_report_selector_strips_the_xdist_group_suffix() -> None:
    """Anti-vacuity: passing the report node id through unchanged makes every
    grouped test unselectable ("not found") when a report id is run again."""
    from devtools.pytest_stream_report import report_nodeid_to_selector

    assert report_nodeid_to_selector("tests/a.py::test_x@web-reader") == "tests/a.py::test_x"
    assert report_nodeid_to_selector("tests/a.py::T::test_x[p]@grp") == "tests/a.py::T::test_x[p]"
    assert report_nodeid_to_selector("tests/a.py::test_x[a@b]") == "tests/a.py::test_x[a@b]"
    assert report_nodeid_to_selector("tests/a.py::test_x") == "tests/a.py::test_x"
    # A long id is shortened after xdist names its group, which leaves the
    # group between the name and the shortened label (baseline job 3431).
    assert (
        report_nodeid_to_selector("tests/a.py::test_x@web-reader[param-7494436ed9e73275]")
        == "tests/a.py::test_x[param-7494436ed9e73275]"
    )


def test_unavailable_pytest_counts_remain_absent() -> None:
    """Anti-vacuity: an empty list of executed pytest steps is not a measured zero."""
    empty = verify._aggregate_pytest_results([], expected_step_count=0, mode="quick", exit_code=0)
    assert empty["selected_union_count"] is None
    assert empty["terminal_union_count"] is None


def test_complete_corpus_tier_traces_and_deselects_nothing() -> None:
    """The ``all`` tier must load testmon and select every collected test."""
    command = verify.build_verify_steps(quick=False, selection="all")[-1][1]

    assert "pytest-testmon" in command
    assert "--testmon" in command
    assert f"--testmon-env={_testmon_environment(verify.ROOT)}" in command
    assert "--testmon-noselect" in command
    assert "--testmon-forceselect" not in command


def _verify_shaped_argv(*, selection: str, target: Path, tmp_path: Path) -> list[str]:
    """The real verifier argv, redirected at one temporary test file."""
    command = list(verify.build_verify_steps(quick=False, selection=selection)[-1][1])
    command[command.index("tests")] = str(target)
    command[command.index("-n") + 1] = "2"
    return [
        argument for argument in command if not argument.startswith(("--junitxml=", _REPORT_PREFIX, "--ignore="))
    ] + [report_file_argument(tmp_path / "report.json")]


@pytest.mark.slow
def test_complete_corpus_run_records_a_usable_testmon_graph(tmp_path: Path) -> None:
    """A verify-shaped ``all`` session leaves fingerprints behind.

    Anti-vacuity: dropping ``pytest-testmon`` from the tier's plugin profile, or
    replacing ``--testmon-noselect`` with no testmon flags at all -- the state
    this checkout shipped, under which a 58-minute corpus run wrote nothing --
    leaves ``test_execution`` empty and makes this red.
    """
    target = tmp_path / "test_traced.py"
    target.write_text("def test_one():\n    assert True\n\n\ndef test_two():\n    assert True\n")
    datafile = tmp_path / "testmondata"
    checkout_root = Path(__file__).resolve().parents[3]

    env = dict(os.environ)
    env.update(
        {
            "TESTMON_DATAFILE": str(datafile),
            "POLYLOGUE_PYTEST_RUN_ID": "testmon-graph-regression",
            "POLYLOGUE_PYTEST_EVENTS_DIR": str(tmp_path / "events"),
            "POLYLOGUE_PYTEST_SELECTION_PATH": str(tmp_path / "selection.json"),
            "POLYLOGUE_PYTEST_SUMMARY_PATH": str(tmp_path / "summary.json"),
        }
    )
    env.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)

    result = subprocess.run(
        _verify_shaped_argv(selection="all", target=target, tmp_path=tmp_path),
        cwd=checkout_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert datafile.exists(), result.stdout + result.stderr
    with sqlite3.connect(datafile) as connection:
        recorded = connection.execute("SELECT count(*) FROM test_execution").fetchone()[0]
        environments = [row[0] for row in connection.execute("SELECT environment_name FROM environment")]
    assert recorded == 2
    assert environments == [_testmon_environment(verify.ROOT)]


def test_two_verify_runs_in_one_checkout_do_not_share_a_report_spool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preparing a second run must not clear the first run's open report spool.

    Anti-vacuity: drop the per-step rebinding and both commands resolve to the
    shared last-pytest.json, so the second run's preparation unlinks the first
    run's spool and the first assembles an incomplete report for tests that
    passed.
    """
    # The shared "current-*" projections are relative to the checkout; keep
    # this test's clearing inside tmp_path so it cannot touch the run
    # executing it.
    for name in (
        "PYTEST_PROGRESS_PATH",
        "PYTEST_EVENTS_PATH",
        "PYTEST_EVENTS_DIR",
        "PYTEST_SELECTION_PATH",
        "PYTEST_SUMMARY_PATH",
    ):
        monkeypatch.setattr(verify, name, tmp_path / "current" / name.lower())

    first_step = tmp_path / "runs" / "first" / "steps" / "01-pytest"
    second_step = tmp_path / "runs" / "second" / "steps" / "01-pytest"
    for step in (first_step, second_step):
        step.mkdir(parents=True)

    from devtools.pytest_stream_report import report_file_argument

    command = [
        "python",
        "-m",
        "pytest",
        f"--junitxml={verify.PYTEST_JUNIT_REPORT_DIR}/verify-latest.xml",
        report_file_argument(verify.PYTEST_REPORT_PATH),
    ]
    first = verify._bind_pytest_reports_to_step(command, SimpleNamespace(step_dir=first_step))
    second = verify._bind_pytest_reports_to_step(command, SimpleNamespace(step_dir=second_step))

    first_report = verify._pytest_report_path(first)
    second_report = verify._pytest_report_path(second)
    assert first_report != second_report
    assert first_report.parent == first_step
    assert second_report.parent == second_step

    # The first run is mid-flight with an open spool; preparing the second must
    # leave it alone.
    spool = first_report.parent / f"{first_report.name}.4242.parts"
    spool.write_text("{}", encoding="utf-8")
    verify._clear_pytest_report(second)

    assert spool.exists()
    assert all(not argument.endswith("verify-latest.xml") for argument in first)


def test_a_focused_run_does_not_inherit_the_broad_archive_prewarm() -> None:
    """polylogue-62j1f: a focused selection builds only what it asked for.

    ``tests/conftest.py``'s ``pytest_sessionstart`` warms all seven shared
    archives when ``POLYLOGUE_BROAD_PREWARM`` is set, which is right for the
    broad verifier and wrong for a named selection. Measured on one head with
    a warm artifact cache, interleaved off/on/off/on over a 12-test
    devtools-only selection: 13.37 s / 26.29 s / 10.36 s / 25.65 s, and 24 vs
    36 archive-tier initializations. The selection needs none of it.

    The invariant is fragile by construction: ``run_tests`` and ``verify``
    each define a ``_normalize_managed_pytest_environment``, and verify's SETS
    the variable that run_tests' caller just popped. Routing the focused
    runner through the wrong one of two identically-named functions would
    reinstate the prewarm silently.

    Anti-vacuity: call ``verify._normalize_managed_pytest_environment`` below
    instead of ``run_tests``', or delete the pop in ``run_tests.run_focused``,
    and the focused assertion goes red while the broad one still passes.
    """
    from devtools import run_tests

    ambient = {"POLYLOGUE_BROAD_PREWARM": "1", "PATH": "/usr/bin"}

    focused = dict(ambient)
    focused.pop("POLYLOGUE_BROAD_PREWARM", None)
    run_tests._normalize_managed_pytest_environment(focused)
    assert "POLYLOGUE_BROAD_PREWARM" not in focused

    # The broad verifier opts in deliberately, and must keep doing so.
    broad = dict(ambient)
    verify._normalize_managed_pytest_environment(broad)
    assert broad["POLYLOGUE_BROAD_PREWARM"] == "1"


def _verify_payload(exit_code: int, diagnosis: str | None) -> dict[str, Any]:
    """A terminal verification payload, shaped as ``VerifyRun.finish`` writes one."""
    return {
        "run_id": "verify-quick-20260922",
        "artifact_dir": ".cache/verify/runs/verify-quick-20260922",
        "exit_code": exit_code,
        "diagnosis": diagnosis,
        "status": "passed" if exit_code == 0 else "failed",
    }


def test_a_failing_run_states_its_verdict_after_the_last_gate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A truncating reader sees the run's verdict, not the last gate's ``ok``.

    ``devtools verify | tail`` exits with tail's status, and the per-gate lines
    do not close that gap: a run whose third gate failed still ends its output
    with a later gate's ``ok``. Three separate reports of "devtools exits 0
    while failing" were this. The verdict has to be the last line.

    Anti-vacuity: drop the ``_write_verdict_line`` call from ``_emit`` and the
    last line here is the simulated ``... ok`` gate line, for a run that exited
    1. Emit the verdict only when ``exit_code`` is zero and this goes red;
    emitting it unconditionally as ``PASSED`` fails the verdict assertion,
    while emitting it only on failure fails the passing case below.
    """
    # The gate line a real run writes just before its terminal emit.
    sys.stderr.write("  gate population-coverage ... ok (1.3s)\n")

    verify._emit(_verify_payload(1, "gate_failed"), use_json=False, operation=None)

    final = capsys.readouterr().err.strip().splitlines()[-1]
    assert final == (
        f"verify: FAILED exit=1 diagnosis=gate_failed receipt={(verify.ROOT / '.cache/verify/runs/verify-quick-20260922/run.json').resolve()}"
    )


def test_a_passing_run_states_its_verdict_too(capsys: pytest.CaptureFixture[str]) -> None:
    """Both directions are stated, so silence never means success.

    A verdict written only for failures makes "this run passed" and "this
    runner said nothing" the same output, which is the ambiguity the line
    exists to remove. A green verification carries no diagnosis -- the payload
    field is ``None``, which is an absence, not a determination that failed --
    so the clause is omitted rather than reported as ``unknown``.

    Anti-vacuity: guard the write with ``if exit_code:`` and this goes red
    while the failing case above stays green. Restore ``payload.get(
    "diagnosis") or "unknown"`` and the ``unknown`` assertion goes red while
    the failing case, which has a real diagnosis, still passes.
    """
    verify._emit(_verify_payload(0, None), use_json=False, operation=None)

    final = capsys.readouterr().err.strip().splitlines()[-1]
    assert final == (
        f"verify: PASSED exit=0 receipt={(verify.ROOT / '.cache/verify/runs/verify-quick-20260922/run.json').resolve()}"
    )
    assert "unknown" not in final


def test_a_json_verdict_line_stays_off_the_machine_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--json`` stdout stays exactly one document; the verdict is stderr.

    Anti-vacuity: write the verdict to ``sys.stdout`` and ``json.loads`` below
    raises on the trailing prose.
    """
    verify._emit(_verify_payload(1, "gate_failed"), use_json=True, operation=None)

    captured = capsys.readouterr()
    assert json.loads(captured.out)["exit_code"] == 1
    assert captured.err.strip().splitlines()[-1].startswith("verify: FAILED exit=1")


def test_interruption_cleanup_defers_a_second_signal() -> None:
    """A repeated SIGTERM during gate cleanup does not escape it.

    Anti-vacuity: run the cleanup without ``_signals_deferred`` and the
    SIGTERM raised inside it reaches the installed handler, which raises.
    """

    class _RaisedError(Exception):
        pass

    def raising(_signum: int, _frame: object) -> None:
        raise _RaisedError

    previous = signal.signal(signal.SIGTERM, raising)
    try:
        with verify._signals_deferred():
            os.kill(os.getpid(), signal.SIGTERM)
        assert signal.getsignal(signal.SIGTERM) is raising
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_stopping_gates_shares_one_grace_period(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every stuck gate gets the same deadline, not ten seconds each in turn.

    Anti-vacuity: wait ``timeout=10`` per process and the second wait is
    asked for the full ten seconds again.
    """
    clock = [0.0]
    waits: list[float] = []

    class _Stuck:
        pid = 0

        def wait(self, timeout: float) -> int:
            waits.append(timeout)
            clock[0] += timeout
            raise subprocess.TimeoutExpired("gate", timeout)

    monkeypatch.setattr("devtools.verify.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("devtools.verify.os.killpg", lambda *_args: None)
    monkeypatch.setattr(verify, "_LIVE_GATE_PROCESSES", {_Stuck(), _Stuck()})

    verify._stop_gate_processes()

    assert waits == [10.0, 0.0]


def test_verify_names_an_oomd_killed_pytest_step(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Anti-vacuity: drop the termination merge in ``verify._run`` and the
    step reads ``pytest_failed`` with no killer or unit."""
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify, "_clear_pytest_report", lambda _command: None)
    monkeypatch.setattr(verify, "executable_gate_result", lambda *_args, **_kwargs: SimpleNamespace(ok=True))
    killed = SimpleNamespace(
        returncode=137,
        slot="agentctl job 9",
        receipt=None,
        termination={"killer": "oom-kill", "unit": "unit.service"},
    )
    monkeypatch.setattr(verify, "run_pytest", lambda *_args, **_kwargs: killed)
    run = VerifyRun(tier="test", argv=[], git_head="head", root=tmp_path)

    exit_code, _elapsed, metadata = verify._run("pytest selected", ["pytest"], run=run)

    assert exit_code == 137
    assert metadata["diagnosis"] == "oom_killed"
    assert metadata["termination_killer"] == "oom-kill"
    assert metadata["termination_unit"] == "unit.service"


def test_actual_admission_counts_new_parametrized_nodes_in_a_recorded_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filename coverage cannot hide newly collected nodes or a second launch."""
    from devtools import verify_test_collection
    from devtools.testmon_provision import inspect_testmon_graph
    from devtools.toolchain import venv_python
    from tests.infra.devtools_admission_fixture import seed_admission_graph

    checkout = Path(__file__).resolve().parents[3]
    testfile = seed_admission_graph(tmp_path, checkout=checkout)
    testfile.write_text(
        testfile.read_text() + '\n@pytest.mark.parametrize("label", ["alpha@value", "beta"])\n'
        "def test_new(label):\n    assert label\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_test_collection, "venv_python", lambda **_kwargs: venv_python(root=checkout))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(tmp_path), str(checkout))))
    graph = inspect_testmon_graph(tmp_path)
    assert graph.status is TestmonGraphStatus.USABLE and graph.full_rerun_cause is None
    connection = sqlite3.connect(_testmon_datafile(tmp_path))
    cursor = connection.execute(
        "SELECT duration FROM test_execution WHERE test_name=?", ("tests/test_nodes.py::test_old",)
    )
    try:
        original_duration = cursor.fetchone()[0]
    finally:
        cursor.close()
        connection.close()
    before = _testmon_datafile(tmp_path).read_bytes()
    count, seconds, error, unknown = verify._estimate_affected_selection(tmp_path, graph)
    assert (count, error, unknown) == (3, None, 2), (count, error, unknown)
    assert seconds == original_duration, (seconds, original_duration)
    forced = "tests/test_nodes.py::test_new"
    count, seconds, error, unknown = verify._estimate_affected_selection(tmp_path, graph, (forced, forced))
    assert (count, error, unknown) == (5, None, 4), (count, error, unknown)
    assert seconds == original_duration, (seconds, original_duration)
    assert _testmon_datafile(tmp_path).read_bytes() == before
    decision = verify._affected_admission(root=tmp_path, graph=graph, forced_tests=(forced,))
    assert decision.to_payload()["selected_count"] == 5
    assert decision.to_payload()["unrecorded_tests"] == 4


@pytest.mark.parametrize("state", [TestmonGraphStatus.UNUSABLE, TestmonGraphStatus.ABSENT])
def test_unavailable_graph_never_collects_or_claims_a_zero_selection(
    state: TestmonGraphStatus,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(**_kwargs: Any) -> None:
        raise AssertionError("unavailable graph dispatched collection")

    monkeypatch.setattr(verify, "collect_selection", forbidden)
    graph = SimpleNamespace(status=state, full_rerun_cause=None, reason="unavailable")
    decision = verify._affected_admission(root=tmp_path, graph=graph)
    assert decision.status == "unknown"
    assert decision.selected_count is None


@pytest.mark.parametrize("corruption", ["version", "bytes"])
def test_actual_changed_graph_refuses_before_selector_dispatch(
    corruption: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from devtools.testmon_provision import inspect_testmon_graph
    from tests.infra.devtools_admission_fixture import seed_admission_graph

    checkout = Path(__file__).resolve().parents[3]
    seed_admission_graph(tmp_path, checkout=checkout)
    graph = inspect_testmon_graph(tmp_path)
    assert graph.usable
    path = _testmon_datafile(tmp_path)
    if corruption == "version":
        connection = sqlite3.connect(path)
        try:
            connection.execute("PRAGMA user_version=999")
            connection.commit()
        finally:
            connection.close()
    else:
        path.write_bytes(b"not a sqlite graph")

    def forbidden(**_kwargs: Any) -> None:
        raise AssertionError("changed graph dispatched collection")

    monkeypatch.setattr(verify, "collect_selection", forbidden)
    decision = verify._affected_admission(root=tmp_path, graph=graph)
    assert decision.status == "unknown"
    assert decision.selected_count is None


def test_declared_testmon_environment_follows_the_verify_command() -> None:
    """The execution-source guard compares this with the admitted environment.

    Anti-vacuity: report ``default`` for an untraced command and a descriptor
    run would be refused for an environment it never writes.
    """
    from devtools.pytest_options import declared_testmon_environment

    traced = verify._pytest_command(selection="affected", worker_args=(), hypothesis_profile=None, explicit_tests=())
    untraced = verify._pytest_command(
        selection="descriptor", worker_args=(), hypothesis_profile=None, explicit_tests=()
    )

    assert declared_testmon_environment(traced) == _testmon_environment(verify.ROOT)
    assert declared_testmon_environment(untraced) is None
    assert declared_testmon_environment(["python", "-m", "pytest", "--testmon", "--no-testmon"]) is None
    assert declared_testmon_environment(["pytest", "--testmon", "--testmon-env", "x"]) == "x"
