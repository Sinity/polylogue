"""The host's single pytest slot is acquired before any managed pytest runs.

Anti-vacuity: deleting the queueing branch in ``devtools.pytest_slot.run_pytest``
makes ``test_outside_a_task_the_run_is_queued`` red — the command executes here
and the marker file appears. Widening ``INHERITED_ENVIRONMENT_KEYS`` makes
``test_the_adder_environment_carries_only_the_allowed_keys`` red. Dropping
either half of the temporary-directory containment (the ``--basetemp``
argument or the exported TMPDIR) makes
``test_a_queued_run_contains_its_temporary_trees`` red.

Every queueing test here resolves ``pueue`` and ``agentctl-run`` from
fakes that are the whole PATH, so a green run says nothing about what the
workstation has deployed; the cgroup a slot decision depends on is stubbed for
the same reason.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from devtools import cloud_sentinels, pytest_slot
from devtools.pytest_slot import (
    BASETEMP_ROOT_ENV,
    PytestSlotUnavailableError,
    basetemp_root,
    holds_pytest_slot,
    run_pytest,
)
from tests.unit.devtools.cgroups import DEPLOYED_PYTEST_CGROUP, stub_cgroup

#: ``pueue add`` resolves the runner and records it; the fake ``pueue`` never
#: executes what it was given, so an executable that exits is the whole fake.
FAKE_QUEUE_RUNNER = "raise SystemExit(0)\n"

FAKE_PUEUE = """import json, os, sys
import shutil

# The record path is derived from this script's own location: a queued run's
# adder environment is scrubbed, so nothing can be passed through it.
with open(sys.argv[0] + ".calls.jsonl", "a", encoding="utf-8") as handle:
    handle.write(json.dumps({{"argv": sys.argv[1:], "env": dict(os.environ)}}) + "\\n")

command = sys.argv[1] if len(sys.argv) > 1 else ""
if command == "add":
    shutil.copyfile(sys.argv[-1], {launch_snapshot!r})
    print("{task_id}")
elif command == "status":
    print(json.dumps({{"tasks": {{"{task_id}": {{"status": {{"Done": {{"result": {result}}}}}}}}}}}))
sys.exit(0)
"""


def _install_executable(directory: Path, name: str, source: str) -> Path:
    """Install ``source`` as an executable that needs nothing else on PATH.

    The shebang names the interpreter absolutely: these fakes are the entire
    PATH of the run under test, so ``/usr/bin/env python3`` would not resolve.
    """
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / name
    script.write_text(f"#!{sys.executable}\n{source}", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _install_fake_queue_runner(directory: Path) -> Path:
    """The runner ``pueue add`` is handed, so queueing never needs the workstation's."""
    return _install_executable(directory, pytest_slot.QUEUE_RUNNERS[0], FAKE_QUEUE_RUNNER)


def _install_fake_pueue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    task_id: str = "7",
    result: str = '"Success"',
) -> Path:
    directory = tmp_path / "fakebin"
    _install_fake_queue_runner(directory)
    script = _install_executable(
        directory,
        "pueue",
        FAKE_PUEUE.format(
            task_id=task_id,
            result=result,
            launch_snapshot=str(tmp_path / "queued-launch.json"),
        ),
    )
    # The fakes are the whole PATH: queueing must resolve its tools from what
    # the test installed, never from whatever the workstation has deployed.
    monkeypatch.setenv("PATH", str(directory))
    return Path(str(script) + ".calls.jsonl")


def _calls(record: Path) -> list[dict[str, Any]]:
    if not record.exists():
        return []
    return [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines() if line]


def _marker_command(marker: Path) -> list[str]:
    """A command that proves it ran by creating ``marker``."""
    return [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"]


def _environment(**extra: str) -> dict[str, str]:
    base = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", "/home/nobody"),
        "XDG_RUNTIME_DIR": "/run/user/1000",
        "XDG_DATA_HOME": "/home/nobody/.local/share",
        "POLYLOGUE_ROOT": "/somewhere",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    base.update(extra)
    return base


def test_outside_a_task_the_run_is_queued(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = _install_fake_pueue(tmp_path, monkeypatch)
    marker = tmp_path / "pytest-ran"

    outcome = run_pytest(
        _marker_command(marker),
        cwd=str(tmp_path),
        env=_environment(PATH=os.environ["PATH"]),
        root=tmp_path,
        label="polylogue:test:1",
    )

    assert not marker.exists(), "pytest ran directly instead of through the queue"
    assert outcome.slot == "pueue task 7"
    assert outcome.returncode == 0
    add = next(call for call in _calls(record) if call["argv"][0] == "add")
    assert "--group" in add["argv"] and add["argv"][add["argv"].index("--group") + 1] == "pytest"
    assert add["argv"][add["argv"].index("--label") + 1] == "polylogue:test:1"
    assert "--print-task-id" in add["argv"] and "--escape" in add["argv"]
    assert add["argv"][-2:] == [
        str(tmp_path / "fakebin" / pytest_slot.QUEUE_RUNNERS[0]),
        str(tmp_path / ".cache" / "verify" / f"pytest-slot-{os.getpid()}.json"),
    ]
    launch = json.loads((tmp_path / "queued-launch.json").read_text(encoding="utf-8"))
    assert launch["project_id"] == "polylogue"
    assert launch["operation"] == "test"
    assert launch["pool"] == "pytest"
    assert launch["result_kind"] == "exit"
    assert launch["timeout_seconds"] == 3600
    assert [call["argv"][0] for call in _calls(record)] == ["add", "wait", "status"]


def test_the_adder_environment_carries_only_the_allowed_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = _install_fake_pueue(tmp_path, monkeypatch)
    secret_environment = _environment(
        PATH=os.environ["PATH"],
        ANTHROPIC_API_KEY="secret",
        AGENTCTL_PRINCIPAL="agent-control",
        SINNIXD_PRINCIPAL="agent-control",
        POLYLOGUE_ARCHIVE_ROOT="/realm/state/polylogue",
    )

    run_pytest(
        _marker_command(tmp_path / "unused"),
        cwd=str(tmp_path),
        env=secret_environment,
        root=tmp_path,
        label="polylogue:test:1",
    )

    allowed = set(pytest_slot.INHERITED_ENVIRONMENT_KEYS)
    for call in _calls(record):
        recorded = {key for key in call["env"] if not key.startswith(("PYTHON", "LC_", "LANG"))}
        assert recorded <= allowed, f"pueue add inherited {sorted(recorded - allowed)}"


def test_lane_job_is_queued_even_with_generic_job_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = _install_fake_pueue(tmp_path, monkeypatch)
    marker = tmp_path / "pytest-ran"
    lane_environment = _environment(AGENTCTL_JOB_ID="job-1", AGENTCTL_OPERATION="lane")

    assert not holds_pytest_slot(lane_environment)
    outcome = run_pytest(
        _marker_command(marker),
        cwd=str(tmp_path),
        env=lane_environment,
        root=tmp_path,
        label="polylogue:test:1",
    )

    assert not marker.exists()
    assert outcome.slot == "pueue task 7"
    assert outcome.returncode == 0
    assert _calls(record)[0]["argv"][0] == "add"


def test_declared_pytest_worker_runs_directly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Queue identity alone is not the slot: the pytest slice is what binds it.

    Anti-vacuity: pointing the stub at any other slice makes this red, because
    the worker queues instead of running.
    """
    record = _install_fake_pueue(tmp_path, monkeypatch)
    stub_cgroup(DEPLOYED_PYTEST_CGROUP, tmp_path=tmp_path, monkeypatch=monkeypatch)
    marker = tmp_path / "pytest-ran"
    holder = {
        "AGENTCTL_JOB_ID": "job-1",
        "AGENTCTL_OPERATION": "verify_affected",
        "AGENTCTL_QUEUE_WORKER": "1",
        "AGENTCTL_POOL": "pytest",
    }

    assert holds_pytest_slot(holder)
    outcome = run_pytest(
        _marker_command(marker),
        cwd=str(tmp_path),
        env=_environment(**holder),
        root=tmp_path,
        label="polylogue:test:1",
    )

    assert marker.exists()
    assert outcome.slot == "held"
    assert outcome.returncode == 0
    assert _calls(record) == [], "a declared pytest worker must not recursively queue"


def test_explicit_slot_holder_runs_directly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = _install_fake_pueue(tmp_path, monkeypatch)
    marker = tmp_path / "pytest-ran"
    holder = {"POLYLOGUE_PYTEST_SLOT": "held"}

    assert holds_pytest_slot(holder)
    outcome = run_pytest(
        _marker_command(marker),
        cwd=str(tmp_path),
        env=_environment(**holder),
        root=tmp_path,
        label="polylogue:test:1",
    )

    assert marker.exists()
    assert outcome.slot == "held"
    assert outcome.returncode == 0
    assert _calls(record) == [], "a slot holder must not talk to pueue"


def test_an_agent_pool_worker_queues_its_focused_run_in_the_pytest_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _install_fake_pueue(tmp_path, monkeypatch)
    marker = tmp_path / "pytest-ran"

    outcome = run_pytest(
        _marker_command(marker),
        cwd=str(tmp_path),
        env=_environment(
            AGENTCTL_JOB_ID="agent-job",
            AGENTCTL_QUEUE_WORKER="1",
            AGENTCTL_POOL="agent",
        ),
        root=tmp_path,
        label="polylogue:test:agent",
    )

    assert not marker.exists()
    assert outcome.slot == "pueue task 7"
    add = next(call for call in _calls(record) if call["argv"][0] == "add")
    assert add["argv"][add["argv"].index("--group") + 1] == "pytest"


def test_missing_scoped_queue_runner_refuses_before_queueing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pueue(tmp_path, monkeypatch)
    original_which = shutil.which
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name, path=None: None if name in pytest_slot.QUEUE_RUNNERS else original_which(name, path),
    )

    with pytest.raises(PytestSlotUnavailableError, match="agentctl-run, sinnixd-queue-run"):
        run_pytest(
            _marker_command(tmp_path / "pytest-ran"),
            cwd=str(tmp_path),
            env=_environment(),
            root=tmp_path,
            label="polylogue:test:missing-runner",
        )


def test_an_unreachable_queue_refuses_rather_than_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The runner resolves and only the queue is missing, so this names pueue itself.

    Anti-vacuity: dropping the fake runner refuses on the runner instead, which
    is ``test_missing_scoped_queue_runner_refuses_before_queueing``; asserting
    only the shared advice line would pass either way.
    """
    marker = tmp_path / "pytest-ran"
    runner_only = tmp_path / "runner-only"
    _install_fake_queue_runner(runner_only)
    monkeypatch.setenv("PATH", str(runner_only))

    with pytest.raises(PytestSlotUnavailableError) as failure:
        run_pytest(
            _marker_command(marker),
            cwd=str(tmp_path),
            env=_environment(PATH=str(runner_only)),
            root=tmp_path,
            label="polylogue:test:1",
        )

    assert "`pueue` is not on PATH" in str(failure.value)
    assert "systemctl --user start pueued" in str(failure.value)
    assert not marker.exists()


def test_a_failed_task_reports_its_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pueue(tmp_path, monkeypatch, task_id="12", result='{"Failed": 1}')

    outcome = run_pytest(
        _marker_command(tmp_path / "unused"),
        cwd=str(tmp_path),
        env=_environment(PATH=os.environ["PATH"]),
        root=tmp_path,
        label="polylogue:test:1",
    )

    assert (outcome.returncode, outcome.slot) == (1, "pueue task 12")


def test_the_slot_runner_executes_the_launch_file(tmp_path: Path) -> None:
    marker = tmp_path / "pytest-ran"
    log_path = tmp_path / "slot.log"
    launch_path = tmp_path / "launch.json"
    launch_path.write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-c",
                    f"import os; open({str(marker)!r},'w').write(os.environ['ONLY_THIS']); print('hi')",
                ],
                "working_directory": str(tmp_path),
                "environment": {"PATH": os.environ["PATH"], "ONLY_THIS": "value"},
                "log_path": str(log_path),
            }
        ),
        encoding="utf-8",
    )

    assert pytest_slot.main([str(launch_path)]) == 0

    assert marker.read_text(encoding="utf-8") == "value"
    assert "hi" in log_path.read_text(encoding="utf-8")
    assert not launch_path.exists(), "the launch file carries a resolved environment and must not persist"


@pytest.mark.uses_real_clock("exercises a real signal deadline and process-group reap")
def test_slot_timeout_writes_typed_receipt_and_reaps_child_group(tmp_path: Path) -> None:
    """An external deadline retains progress even when pytest cannot finalize reports.

    Anti-vacuity: removing the signal handler loses the sibling receipt; omitting
    ``start_new_session`` leaves the sleeping descendant alive after the runner
    exits.
    """
    log_path = tmp_path / "pytest-slot-1.log"
    launch_path = tmp_path / "launch.json"
    events_path = tmp_path / "events.jsonl"
    started = tmp_path / "started"
    survivor = tmp_path / "survivor"
    events_path.write_text(json.dumps({"event": "test_report", "outcome": "passed"}) + "\n", encoding="utf-8")
    child = f"touch {started}; (sleep 2; touch {survivor}) & sleep 30"
    launch_path.write_text(
        json.dumps(
            {
                "argv": ["sh", "-c", child],
                "working_directory": str(tmp_path),
                "environment": {
                    "PATH": os.environ["PATH"],
                    "POLYLOGUE_PYTEST_EVENTS_PATH": str(events_path),
                },
                "log_path": str(log_path),
            }
        ),
        encoding="utf-8",
    )
    process = subprocess.Popen([sys.executable, "-m", "devtools.pytest_slot", str(launch_path)])
    try:
        deadline = time.monotonic() + 5
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 128 + signal.SIGTERM
    finally:
        if process.poll() is None:
            process.kill()

    receipt = json.loads(log_path.with_suffix(".result.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "timed_out"
    assert receipt["diagnosis"] == "pytest_deadline"
    assert receipt["elapsed_s"] >= 0
    assert receipt["progress"]["terminal_count"] == 1
    time.sleep(0.1)
    assert not survivor.exists()


def _launch_document(root: Path) -> dict[str, Any]:
    """The launch file captured by the fake pueue adder before consumption."""
    document: dict[str, Any] = json.loads((root / "queued-launch.json").read_text(encoding="utf-8"))
    return document


def test_a_queued_run_contains_its_temporary_trees(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither pytest nor the code under test may write to the ambient TMPDIR.

    ``nix develop`` points TMPDIR at a small tmpfs; a corpus run left there
    fills the mount and dies on exhausted file descriptors.
    """
    _install_fake_pueue(tmp_path, monkeypatch)
    scratch = tmp_path / ".cache" / "verify"

    run_pytest(
        _marker_command(tmp_path / "unused"),
        cwd=str(tmp_path),
        env=_environment(TMPDIR="/tmp/nix-shell.L3brFS"),
        root=tmp_path,
        label="polylogue:test:1",
    )

    launch = _launch_document(tmp_path)
    assert Path(launch["environment"]["TMPDIR"]).is_relative_to(scratch)
    argv = launch["argv"]
    basetemp = Path(argv[argv.index("--basetemp") + 1])
    assert basetemp.is_relative_to(scratch)


def test_a_declared_basetemp_is_kept_and_still_anchors_tmpdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``devtools test`` names its own basetemp so it can dispose of it."""
    _install_fake_pueue(tmp_path, monkeypatch)
    declared = tmp_path / ".cache" / "verify" / "tmp-chosen"

    run_pytest(
        [*_marker_command(tmp_path / "unused"), "--basetemp", str(declared)],
        cwd=str(tmp_path),
        env=_environment(TMPDIR="/tmp/nix-shell.L3brFS"),
        root=tmp_path,
        label="polylogue:test:1",
    )

    launch = _launch_document(tmp_path)
    assert launch["argv"].count("--basetemp") == 1
    assert launch["argv"][launch["argv"].index("--basetemp") + 1] == str(declared)
    # pytest empties its own basetemp as it starts, so TMPDIR must be beside it.
    tmpdir = Path(launch["environment"]["TMPDIR"])
    assert tmpdir.parent == declared.parent and tmpdir != declared


def test_a_run_holding_the_slot_sees_the_contained_tmpdir(tmp_path: Path) -> None:
    recorded = tmp_path / "tmpdir-seen"
    command = [sys.executable, "-c", f"import os; open({str(recorded)!r},'w').write(os.environ['TMPDIR'])"]

    run_pytest(
        command,
        cwd=str(tmp_path),
        env=_environment(POLYLOGUE_PYTEST_SLOT="held", TMPDIR="/tmp/nix-shell.L3brFS"),
        root=tmp_path,
        label="polylogue:test:1",
    )

    seen = Path(recorded.read_text(encoding="utf-8"))
    assert seen.is_relative_to(tmp_path / ".cache" / "verify")


def test_a_configured_basetemp_root_is_honoured(tmp_path: Path) -> None:
    """A sandbox with no checkout-local scratch names its own root."""
    elsewhere = tmp_path / "sandbox-scratch"

    assert basetemp_root({BASETEMP_ROOT_ENV: str(elsewhere)}, root=tmp_path) == elsewhere


def test_the_leaked_cloud_basetemp_sentinel_is_declined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`.claude/settings.json` exports the cloud value into workstation sessions."""
    monkeypatch.setattr(cloud_sentinels, "_WORKSTATION_SCRATCH_MOUNT", tmp_path)
    sentinel = cloud_sentinels.CLOUD_SENTINELS[BASETEMP_ROOT_ENV]

    assert basetemp_root({BASETEMP_ROOT_ENV: sentinel}, root=tmp_path) == tmp_path / ".cache" / "verify"


#: A ``pueue`` whose ``wait`` kills the process waiting on it, the way a
#: session or a wrapper being killed leaves a queued task with no waiter.
FAKE_PUEUE_KILLS_ITS_WAITER = """import json, os, signal, sys, time

with open(sys.argv[0] + ".calls.jsonl", "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"argv": sys.argv[1:]}) + "\\n")

command = sys.argv[1] if len(sys.argv) > 1 else ""
if command == "add":
    print("11")
elif command == "wait":
    os.kill(os.getppid(), signal.SIGTERM)
    time.sleep(2)
sys.exit(0)
"""

#: The cancellation AgentCTL owns, recorded the way the ``pueue`` fake records.
FAKE_AGENTCTL = """import json, sys

with open(sys.argv[0] + ".calls.jsonl", "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"argv": sys.argv[1:]}) + "\\n")
sys.exit(0)
"""

#: An AgentCTL that refuses, the way a job id pueue no longer holds refuses.
FAKE_AGENTCTL_REFUSES = """import json, sys

with open(sys.argv[0] + ".calls.jsonl", "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"argv": sys.argv[1:]}) + "\\n")
sys.stderr.write("agentctl: no such job\\n")
sys.exit(1)
"""

_WAITER = """
import os, sys
sys.path.insert(0, {repo!r})
from devtools.pytest_slot import run_pytest

run_pytest(
    [sys.executable, "-c", "pass"],
    cwd={cwd!r},
    env={{"PATH": os.environ["PATH"], "HOME": os.environ["HOME"]}},
    root={root!r},
    label="polylogue:test:signalled",
)
"""


def test_a_killed_waiter_reaps_the_task_it_queued(tmp_path: Path) -> None:
    """A task outlives its waiter, and the slot's parallelism is one.

    Anti-vacuity: dropping the ``_reaping`` context records no cancellation at
    all -- the task stays queued with nothing left to wait on it, which is
    exactly the starvation this reap exists to prevent. Reaping through
    ``pueue kill`` instead leaves the workload running in its scope, and shows
    up here as a ``kill`` among the pueue verbs.
    """
    import subprocess

    directory = tmp_path / "fakebin"
    _install_fake_queue_runner(directory)
    script = _install_executable(directory, "pueue", FAKE_PUEUE_KILLS_ITS_WAITER)
    agentctl = _install_executable(directory, "agentctl", FAKE_AGENTCTL)
    record = Path(str(script) + ".calls.jsonl")
    cancellation_record = Path(str(agentctl) + ".calls.jsonl")
    repo = str(Path(pytest_slot.__file__).resolve().parents[1])

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _WAITER.format(repo=repo, cwd=str(tmp_path), root=str(tmp_path)),
        ],
        env={
            "PATH": str(directory),
            "HOME": os.environ.get("HOME", "/home/nobody"),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == -int(signal.SIGTERM), completed.stderr
    verbs = [call["argv"][0] for call in _calls(record)]
    assert verbs == ["add", "wait"], verbs
    assert _calls(cancellation_record) == [{"argv": ["job", "cancel", "11"]}]
    leftover = list((tmp_path / "verify").glob("pytest-slot-*.json")) + list(
        (tmp_path / ".cache" / "verify").glob("pytest-slot-*.json")
    )
    assert leftover == [], "the launch file carries a resolved environment and must not survive the reap"


def test_a_refused_cancellation_leaves_the_launch_file_for_the_task(tmp_path: Path) -> None:
    """A task AgentCTL would not stop still reads its launch file when it starts.

    Anti-vacuity: unlinking the launch file regardless of the cancellation's
    outcome empties the glob below, and the surviving task then starts with no
    resolved environment to read.
    """
    import subprocess

    directory = tmp_path / "fakebin"
    _install_fake_queue_runner(directory)
    _install_executable(directory, "pueue", FAKE_PUEUE_KILLS_ITS_WAITER)
    agentctl = _install_executable(directory, "agentctl", FAKE_AGENTCTL_REFUSES)
    repo = str(Path(pytest_slot.__file__).resolve().parents[1])

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _WAITER.format(repo=repo, cwd=str(tmp_path), root=str(tmp_path)),
        ],
        env={
            "PATH": str(directory),
            "HOME": os.environ.get("HOME", "/home/nobody"),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == -int(signal.SIGTERM), completed.stderr
    assert _calls(Path(str(agentctl) + ".calls.jsonl")) == [{"argv": ["job", "cancel", "11"]}]
    surviving = list((tmp_path / ".cache" / "verify").glob("pytest-slot-*.json"))
    assert len(surviving) == 1, surviving
