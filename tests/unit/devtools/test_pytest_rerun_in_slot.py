"""A queued focused run adjudicates its failures inside the slot it holds."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from devtools import pytest_rerun, pytest_slot
from devtools.pytest_rerun import RERUN_IN_SLOT_ENV, RERUN_IN_SLOT_RESULT, rerun_failed_once


def _failed_report(path: Path, nodeid: str) -> None:
    path.write_text(
        json.dumps({"tests": [{"nodeid": nodeid, "outcome": "failed"}], "summary": {"failed": 1}}),
        encoding="utf-8",
    )


def test_slot_job_reruns_failures_and_records_the_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The job writes the rerun record the client adjudicates from.

    Anti-vacuity: drop the ``_rerun_failures_in_slot`` call from the launch
    path's exit-1 branch (or this helper's write) and no record appears, so
    the client queues a second job for its rerun.
    """
    step = tmp_path / "step"
    step.mkdir()
    report = step / "pytest-report.json"
    _failed_report(report, "tests/test_x.py::test_flaky")
    rerun_report = step / "pytest-rerun.json"
    script = (
        "import json, pathlib\n"
        f"pathlib.Path({str(rerun_report)!r}).write_text(json.dumps("
        "{'tests': [{'nodeid': 'tests/test_x.py::test_flaky', 'outcome': 'passed'}]}))\n"
    )
    monkeypatch.setattr(
        pytest_rerun,
        "build_rerun",
        lambda **_kwargs: (["tests/test_x.py::test_flaky"], [sys.executable, "-c", script], rerun_report),
    )
    environment = {
        RERUN_IN_SLOT_ENV: json.dumps({"report_path": str(report), "step_dir": str(step), "root": str(tmp_path)}),
        "PATH": "/usr/bin:/bin",
    }

    started: list[object] = []
    with (tmp_path / "slot.log").open("wb") as log:
        pytest_slot._rerun_failures_in_slot(environment, cwd=str(tmp_path), log=log, on_start=started.append)

    # The rerun is registered with the launch before it is waited on, so the
    # launch's signal handling can stop it.
    assert len(started) == 1

    record = json.loads((step / RERUN_IN_SLOT_RESULT).read_text(encoding="utf-8"))
    assert record["attempted"] == ["tests/test_x.py::test_flaky"]
    assert record["rerun_exit"] == 0
    # Provenance is recorded only when the launch asked for it; the client
    # compares it with the first run's before clearing anything.
    assert "worktree_provenance" in record


def test_client_adjudicates_from_the_slot_record_without_requeueing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded in-slot rerun is folded in; the pytest slot is not requested again.

    Anti-vacuity: remove the record branch from ``rerun_failed_once`` and the
    fake slot below is reached, failing the test.
    """
    step = tmp_path / "step"
    step.mkdir()
    report = step / "pytest-report.json"
    _failed_report(report, "tests/test_x.py::test_flaky")
    (step / "pytest-rerun.json").write_text(
        json.dumps({"tests": [{"nodeid": "tests/test_x.py::test_flaky", "outcome": "passed"}]}), encoding="utf-8"
    )
    (step / RERUN_IN_SLOT_RESULT).write_text(
        json.dumps({"attempted": ["tests/test_x.py::test_flaky"], "rerun_exit": 0}), encoding="utf-8"
    )

    def must_not_queue(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the client queued a second job for a rerun the slot already ran")

    monkeypatch.setattr(pytest_rerun, "run_pytest", must_not_queue)

    verdict = rerun_failed_once(report_path=report, step_dir=step, env={}, root=tmp_path)

    assert verdict is not None
    assert verdict["still_failed"] == []
    assert verdict["flaky"] == ["tests/test_x.py::test_flaky"]
    assert json.loads(report.read_text(encoding="utf-8"))["exitcode"] == 0


def test_a_rerun_that_fails_again_stays_red(tmp_path: Path) -> None:
    step = tmp_path / "step"
    step.mkdir()
    report = step / "pytest-report.json"
    _failed_report(report, "tests/test_x.py::test_real")
    (step / "pytest-rerun.json").write_text(
        json.dumps({"tests": [{"nodeid": "tests/test_x.py::test_real", "outcome": "failed"}]}), encoding="utf-8"
    )
    (step / RERUN_IN_SLOT_RESULT).write_text(
        json.dumps({"attempted": ["tests/test_x.py::test_real"], "rerun_exit": 1}), encoding="utf-8"
    )

    verdict = rerun_failed_once(report_path=report, step_dir=step, env={}, root=tmp_path)

    assert verdict is not None
    assert verdict["still_failed"] == ["tests/test_x.py::test_real"]


@pytest.mark.parametrize(
    ("rerun_exit", "outcome", "cleared"),
    [(0, "passed", True), (1, "failed", False), (0, "skipped", False)],
)
def test_scratch_follows_the_adjudicated_outcome(tmp_path: Path, rerun_exit: int, outcome: str, cleared: bool) -> None:
    """A queued run whose in-slot rerun cleared every failure disposes of its scratch.

    Anti-vacuity: ignore the rerun record in ``run_pytest`` and a cleared run
    keeps its scratch tree, which only a later sweep would remove.
    """
    step = tmp_path / "step"
    step.mkdir()
    (step / RERUN_IN_SLOT_RESULT).write_text(
        json.dumps({"attempted": ["t"], "rerun_exit": rerun_exit}), encoding="utf-8"
    )
    # A skipped rerun exits 0 yet clears nothing; the report decides.
    (step / "pytest-rerun.json").write_text(
        json.dumps({"tests": [{"nodeid": "t", "outcome": outcome}]}), encoding="utf-8"
    )
    env = {RERUN_IN_SLOT_ENV: json.dumps({"report_path": "r", "step_dir": str(step), "root": str(tmp_path)})}

    assert pytest_slot._in_slot_rerun_cleared(env) is cleared
    assert pytest_slot._in_slot_rerun_cleared({}) is False


def test_an_unattributable_rerun_is_not_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Provenance that cannot be captured means no rerun, and the failures stand.

    Anti-vacuity: suppress the capture error and the rerun starts, so a pass
    over unverified content could clear the failure.
    """
    step = tmp_path / "step"
    step.mkdir()
    report = step / "pytest-report.json"
    _failed_report(report, "tests/test_x.py::test_real")
    monkeypatch.setattr(
        pytest_rerun,
        "build_rerun",
        lambda **_kwargs: (["tests/test_x.py::test_real"], [sys.executable, "-c", "pass"], step / "pytest-rerun.json"),
    )

    def unavailable(*_args: Any, **_kwargs: Any) -> Any:
        raise pytest_slot.PytestSlotUnavailableError("focused worktree content could not be identified")

    monkeypatch.setattr(pytest_slot, "_focused_worktree_provenance", unavailable)
    environment = {
        RERUN_IN_SLOT_ENV: json.dumps({"report_path": str(report), "step_dir": str(step), "root": str(tmp_path)}),
    }
    started: list[object] = []
    with (tmp_path / "slot.log").open("wb") as log:
        pytest_slot._rerun_failures_in_slot(environment, cwd=str(tmp_path), log=log, on_start=started.append)

    record = json.loads((step / RERUN_IN_SLOT_RESULT).read_text(encoding="utf-8"))
    assert started == []
    assert record["rerun_exit"] == 125
    verdict = rerun_failed_once(report_path=report, step_dir=step, env={}, root=tmp_path)
    assert verdict is not None and verdict["still_failed"] == ["tests/test_x.py::test_real"]


def test_a_grouped_node_that_passes_alone_is_patched_in_the_report(tmp_path: Path) -> None:
    """``--dist=loadgroup`` reports ``node@group``; the rerun names the plain node.

    Anti-vacuity: compare the raw report node id and the row stays failed while
    the verdict says green, leaving a green receipt with a failed outcome.
    """
    step = tmp_path / "step"
    step.mkdir()
    report = step / "pytest-report.json"
    report.write_text(
        json.dumps(
            {
                "tests": [{"nodeid": "tests/test_x.py::test_web@web-reader", "outcome": "failed"}],
                "summary": {"failed": 1},
            }
        ),
        encoding="utf-8",
    )
    (step / "pytest-rerun.json").write_text(
        json.dumps({"tests": [{"nodeid": "tests/test_x.py::test_web", "outcome": "passed"}]}), encoding="utf-8"
    )
    (step / RERUN_IN_SLOT_RESULT).write_text(
        json.dumps({"attempted": ["tests/test_x.py::test_web"], "rerun_exit": 0}), encoding="utf-8"
    )

    verdict = rerun_failed_once(report_path=report, step_dir=step, env={}, root=tmp_path)

    patched = json.loads(report.read_text(encoding="utf-8"))
    assert verdict is not None and verdict["still_failed"] == []
    assert patched["tests"][0]["outcome"] == "passed"
    assert patched["summary"]["failed"] == 0


def test_the_in_slot_rerun_gets_fresh_temporary_scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: reuse the first attempt's TMPDIR and a test's own sentinel
    from that attempt makes it pass on rerun."""
    step = tmp_path / "step"
    step.mkdir()
    report = step / "pytest-report.json"
    _failed_report(report, "tests/test_x.py::test_sentinel")
    first_scratch = tmp_path / "scratch"
    first_scratch.mkdir()
    seen = tmp_path / "seen-tmpdir"
    script = f"import os, pathlib\npathlib.Path({str(seen)!r}).write_text(os.environ['TMPDIR'])\n"
    monkeypatch.setattr(
        pytest_rerun,
        "build_rerun",
        lambda **_kwargs: (
            ["tests/test_x.py::test_sentinel"],
            [sys.executable, "-c", script],
            step / "pytest-rerun.json",
        ),
    )
    monkeypatch.setattr(pytest_slot, "_focused_worktree_provenance", lambda *_a, **_k: None)
    environment = {
        RERUN_IN_SLOT_ENV: json.dumps({"report_path": str(report), "step_dir": str(step), "root": str(tmp_path)}),
        "TMPDIR": str(first_scratch),
        "PATH": "/usr/bin:/bin",
    }
    with (tmp_path / "slot.log").open("wb") as log:
        pytest_slot._rerun_failures_in_slot(environment, cwd=str(tmp_path), log=log, on_start=lambda _p: None)

    rerun_tmp = Path(seen.read_text(encoding="utf-8"))
    assert rerun_tmp != first_scratch
    assert rerun_tmp.parent == first_scratch
    assert list(rerun_tmp.iterdir()) == []


def test_no_rerun_without_fresh_scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: fall back to the first attempt's TMPDIR and the rerun starts on contaminated scratch."""
    step = tmp_path / "step"
    step.mkdir()
    report = step / "pytest-report.json"
    _failed_report(report, "tests/test_x.py::test_sentinel")
    monkeypatch.setattr(
        pytest_rerun,
        "build_rerun",
        lambda **_kwargs: (
            ["tests/test_x.py::test_sentinel"],
            [sys.executable, "-c", "pass"],
            step / "pytest-rerun.json",
        ),
    )
    monkeypatch.setattr(pytest_slot, "_focused_worktree_provenance", lambda *_a, **_k: None)

    def no_space(*_args: Any, **_kwargs: Any) -> str:
        raise OSError("read-only scratch")

    monkeypatch.setattr("devtools.pytest_slot.tempfile.mkdtemp", no_space)
    environment = {
        RERUN_IN_SLOT_ENV: json.dumps({"report_path": str(report), "step_dir": str(step), "root": str(tmp_path)}),
        "TMPDIR": str(tmp_path),
    }
    started: list[object] = []
    with (tmp_path / "slot.log").open("wb") as log:
        pytest_slot._rerun_failures_in_slot(environment, cwd=str(tmp_path), log=log, on_start=started.append)

    assert started == []
    assert json.loads((step / RERUN_IN_SLOT_RESULT).read_text(encoding="utf-8"))["rerun_exit"] == 125


def test_a_rerun_keeps_the_first_runs_execution_options() -> None:
    """Anti-vacuity: drop the kept options and ``-W error`` is lost, so a warning
    failure passes on rerun under the default policy."""
    command = [
        "python",
        "-m",
        "pytest",
        "-p",
        "devtools.pytest_progress_plugin",
        "--override-ini=addopts=",
        "--polylogue-report-file=r.json",
        "tests/test_w.py",
        "-W",
        "error",
        "-k",
        "slow",
        "-n",
        "4",
        "--dist=loadgroup",
        "-p",
        "no:randomly",
        "-o",
        "xfail_strict=true",
        "-x",
    ]
    assert pytest_rerun.semantic_rerun_options(command) == [
        "-W",
        "error",
        "-p",
        "no:randomly",
        "-o",
        "xfail_strict=true",
    ]


def test_attached_short_option_values_are_classified_by_their_option() -> None:
    """Anti-vacuity: classify ``-n8`` by its whole spelling and it is kept, so
    the adjudicating rerun fans out to eight workers instead of running alone."""
    command = [
        "python",
        "-m",
        "pytest",
        "tests/test_w.py",
        "-n8",
        "-kslow",
        "-rf",
        "-pxdist",
        "-pno:randomly",
        "-Werror",
    ]
    assert pytest_rerun.semantic_rerun_options(command) == ["-pno:randomly", "-Werror"]


def test_a_callers_plugin_load_survives_into_the_rerun() -> None:
    """Anti-vacuity: drop every positive ``-p`` and a failure a caller plugin
    causes passes on the plugin-free rerun, so it is called flaky."""
    command = ["python", "-m", "pytest", "-p", "xdist", "-p", "custom_plugin", "-pother", "tests/test_w.py"]
    assert pytest_rerun.semantic_rerun_options(command) == ["-p", "custom_plugin", "-pother"]


def test_the_suite_file_batch_is_dropped_with_its_value() -> None:
    """Anti-vacuity: read the option table without the suite's conftest and
    ``1/2`` is dropped as an operand, so the rerun gets a value-less option."""
    command = ["python", "-m", "pytest", "--polylogue-file-batch", "1/2", "-W", "error", "tests/test_w.py"]
    assert pytest_rerun.semantic_rerun_options(command) == ["-W", "error"]


@pytest.mark.parametrize(("option", "value"), [("--assert", "plain"), ("--show-capture", "no"), ("--durations", "5")])
def test_an_option_pytest_reads_a_value_for_keeps_it_in_the_rerun(option: str, value: str) -> None:
    """Anti-vacuity: classify arity from a hand-kept list that omits ``option``
    and its value is dropped as an operand, so the rerun passes the failed node
    id as the option's value and pytest exits with a usage error."""
    command = ["python", "-m", "pytest", option, value, "tests/test_w.py"]
    assert pytest_rerun.semantic_rerun_options(command) == [option, value]


def test_the_first_attempts_descendants_are_reaped_before_a_rerun() -> None:
    """Anti-vacuity: skip ``_group_reaped`` and the backgrounded ``sleep`` the
    failed attempt left behind is still in its group when the rerun starts."""
    leader = subprocess.Popen(["sh", "-c", "sleep 60 & exit 1"], start_new_session=True)
    assert leader.wait() == 1
    assert pytest_slot._group_alive(leader.pid)

    assert pytest_slot._group_reaped(leader.pid)
    assert not pytest_slot._group_alive(leader.pid)


def test_scratch_is_kept_when_the_rerun_ran_other_content(tmp_path: Path) -> None:
    """Anti-vacuity: ignore provenance in the cleanup decision and a rejected
    rerun deletes the red run's diagnostic scratch."""
    step = tmp_path / "step"
    step.mkdir()
    (step / RERUN_IN_SLOT_RESULT).write_text(
        json.dumps({"attempted": ["t"], "rerun_exit": 0, "worktree_provenance": {"git_head": "new"}}),
        encoding="utf-8",
    )
    (step / "pytest-rerun.json").write_text(
        json.dumps({"tests": [{"nodeid": "t", "outcome": "passed"}]}), encoding="utf-8"
    )
    env = {RERUN_IN_SLOT_ENV: json.dumps({"report_path": "r", "step_dir": str(step), "root": str(tmp_path)})}

    assert pytest_slot._in_slot_rerun_cleared(env, first_provenance={"git_head": "old"}) is False
    assert pytest_slot._in_slot_rerun_cleared(env, first_provenance={"git_head": "new"}) is True


def test_a_caller_plugins_value_option_keeps_its_value_in_the_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity (Codex P2, #5708): probe the option table without the
    caller's ``-p`` plugin and its ``--mode`` has no known arity, so ``strict``
    is dropped as a path operand and the rerun exits with a usage error."""
    plugin = "polylogue_synthetic_mode_plugin"
    (tmp_path / f"{plugin}.py").write_text(
        "def pytest_addoption(parser):\n    parser.addoption('--mode', action='store')\n", encoding="utf-8"
    )
    monkeypatch.setenv("PYTHONPATH", f"{tmp_path}:{os.environ.get('PYTHONPATH', '')}")
    command = ["python", "-m", "pytest", "tests/test_w.py", "-p", plugin, "--mode", "strict"]
    assert pytest_rerun.semantic_rerun_options(command) == ["-p", plugin, "--mode", "strict"]


@pytest.mark.parametrize(("cluster", "kept"), [("-ln8", ["-l"]), ("-xl", ["-l"]), ("-lW", ["-lW", "error"])])
def test_clustered_dropped_options_leave_the_rerun(cluster: str, kept: list[str]) -> None:
    """Anti-vacuity (Codex P1, #5708): judge a cluster by its leading flag and
    ``-ln8`` rides into the one-process rerun whole, starting eight workers
    (``-l`` is kept on its own; ``-n`` and ``-x`` are dropped)."""
    command = ["python", "-m", "pytest", "tests/test_w.py", cluster, *(["error"] if cluster == "-lW" else [])]
    assert pytest_rerun.semantic_rerun_options(command) == kept


def test_coverage_options_stay_with_the_first_attempt() -> None:
    """Anti-vacuity (Codex P2, #5708): keep ``--cov-fail-under`` on the rerun
    of the failed subset and it fails the threshold although every node passed."""
    command = ["python", "-m", "pytest", "tests/test_w.py", "--cov=polylogue", "--cov-fail-under=90", "-l"]
    assert pytest_rerun.semantic_rerun_options(command) == ["-l"]
    # A separate value goes with its option, never read as a path operand.
    command = ["python", "-m", "pytest", "--cov", "polylogue", "tests/test_w.py", "-l"]
    assert pytest_rerun.semantic_rerun_options(command) == ["-l"]


def test_an_unpublishable_rerun_result_leaves_a_typed_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A rerun whose result cannot be written is not mistaken for no rerun at all.

    Anti-vacuity (Codex P2, #5708): write the record only after the rerun and
    a failed write leaves nothing, so the client runs the failures a third
    time and can call a twice-failed test flaky.
    """
    step = tmp_path / "step"
    step.mkdir()
    report = step / "pytest-report.json"
    _failed_report(report, "tests/test_x.py::test_twice")
    monkeypatch.setattr(
        pytest_rerun,
        "build_rerun",
        lambda **_kwargs: (
            ["tests/test_x.py::test_twice"],
            [sys.executable, "-c", "raise SystemExit(1)"],
            step / "r.json",
        ),
    )
    real_write_text = Path.write_text
    writes: list[str] = []

    def failing_after_the_rerun(self: Path, data: str, *args: Any, **kwargs: Any) -> int:
        if self.name == RERUN_IN_SLOT_RESULT:
            writes.append(data)
            if len(writes) > 1:
                raise OSError("device full")
        return real_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", failing_after_the_rerun)
    environment = {
        RERUN_IN_SLOT_ENV: json.dumps({"report_path": str(report), "step_dir": str(step), "root": str(tmp_path)}),
        "PATH": "/usr/bin:/bin",
    }
    started: list[object] = []
    with (tmp_path / "slot.log").open("wb") as log:
        pytest_slot._rerun_failures_in_slot(environment, cwd=str(tmp_path), log=log, on_start=started.append)

    assert len(started) == 1
    record = json.loads(real_read(step / RERUN_IN_SLOT_RESULT))
    assert record["rerun_exit"] == 125 and record["result_unpublished"] is True


def real_read(path: Path) -> str:
    with path.open(encoding="utf-8") as handle:
        return handle.read()
