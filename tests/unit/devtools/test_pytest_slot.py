"""The host's single pytest slot is the pytest pool, entered through agentctl.

Anti-vacuity: deleting the submitting branch in ``devtools.pytest_slot.run_pytest``
makes ``test_outside_the_pool_the_run_is_submitted`` red — the command executes
here and the marker file appears. Widening ``INHERITED_ENVIRONMENT_KEYS`` makes
``test_the_client_environment_carries_only_the_allowed_keys`` red. Dropping
either half of the temporary-directory containment (the ``--basetemp``
argument or the exported TMPDIR) makes
``test_a_submitted_run_contains_its_temporary_trees`` red. Treating a job id
as ownership makes ``test_a_job_id_is_never_slot_ownership`` red.

Every submitting test here resolves ``agentctl`` from a fake that is the whole
PATH, so a green run says nothing about what the workstation has deployed; the
cgroup a slot decision depends on is stubbed for the same reason.
"""

from __future__ import annotations

import json
import os
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
from tests.unit.devtools.cgroups import (
    AGENT_CGROUPS,
    OUTSIDE_CGROUP,
    PYTEST_CGROUPS,
    stub_cgroup,
)

#: An ``agentctl`` whose ``job start`` snapshots the launch file it was handed
#: and whose ``job get`` reports one terminal job. It records every call with
#: the environment it ran in; the record path is derived from the script's own
#: location because a submitting client's environment is scrubbed.
FAKE_AGENTCTL = """import json, os, sys, shutil

with open(sys.argv[0] + ".calls.jsonl", "a", encoding="utf-8") as handle:
    handle.write(json.dumps({{"argv": sys.argv[1:], "env": dict(os.environ)}}) + "\\n")

words = [word for word in sys.argv[1:] if word != "--json"]
verb = " ".join(words[:2])
if verb == "job start":
    shutil.copyfile(sys.argv[-1], {launch_snapshot!r})
    print(json.dumps({{"job_id": {job_id}, "phase": "queued", "terminal": False}}))
elif verb == "job get":
    print(json.dumps({{"job_id": {job_id}, "phase": {phase!r}, "terminal": True, "exit_code": {exit_code}}}))
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


def _install_fake_agentctl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    job_id: int = 7,
    phase: str = "succeeded",
    exit_code: int = 0,
) -> Path:
    directory = tmp_path / "fakebin"
    script = _install_executable(
        directory,
        "agentctl",
        FAKE_AGENTCTL.format(
            job_id=job_id,
            phase=phase,
            exit_code=exit_code,
            launch_snapshot=str(tmp_path / "submitted-launch.json"),
        ),
    )
    # The fake is the whole PATH: submitting must resolve its tool from what
    # the test installed, never from whatever the workstation has deployed.
    monkeypatch.setenv("PATH", str(directory))
    monkeypatch.setattr(pytest_slot, "POLL_INTERVAL_S", 0.0)
    return Path(str(script) + ".calls.jsonl")


def _calls(record: Path) -> list[dict[str, Any]]:
    if not record.exists():
        return []
    return [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines() if line]


def _verbs(record: Path) -> list[str]:
    return [" ".join(word for word in call["argv"][:3] if word != "--json") for call in _calls(record)]


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


def _launch_document(root: Path) -> dict[str, Any]:
    """The launch file captured by the fake agentctl before consumption."""
    document: dict[str, Any] = json.loads((root / "submitted-launch.json").read_text(encoding="utf-8"))
    return document


@pytest.fixture(autouse=True)
def _outside_the_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The session running these tests may itself hold the slot; the decision under test must not."""
    stub_cgroup(OUTSIDE_CGROUP, tmp_path=tmp_path, monkeypatch=monkeypatch)


def test_outside_the_pool_the_run_is_submitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = _install_fake_agentctl(tmp_path, monkeypatch)
    marker = tmp_path / "pytest-ran"

    outcome = run_pytest(_marker_command(marker), cwd=str(tmp_path), env=_environment(), root=tmp_path)

    assert not marker.exists(), "pytest ran directly instead of through the pytest pool"
    assert outcome.slot == "agentctl job 7"
    assert outcome.returncode == 0
    assert outcome.log_path == tmp_path / ".cache" / "verify" / f"pytest-slot-{os.getpid()}.log"
    start = next(call for call in _calls(record) if "start" in call["argv"])
    assert start["argv"] == [
        "--json",
        "job",
        "start",
        str(tmp_path),
        "pytest_focused",
        "--workspace",
        str(tmp_path),
        "--",
        str(tmp_path / ".cache" / "verify" / f"pytest-slot-{os.getpid()}.json"),
    ]
    launch = _launch_document(tmp_path)
    assert launch["kind"] == "polylogue.pytest-slot-launch"
    assert launch["argv"][:3] == _marker_command(marker)
    assert launch["working_directory"] == str(tmp_path)
    assert launch["log_path"] == str(outcome.log_path)
    assert _verbs(record) == ["job start", "job get 7"]
    assert not list((tmp_path / ".cache" / "verify").glob("pytest-slot-*.json")), "the launch file outlived its run"


def test_the_client_environment_carries_only_the_allowed_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = _install_fake_agentctl(tmp_path, monkeypatch)
    secret_environment = _environment(
        ANTHROPIC_API_KEY="secret",
        AGENTCTL_PRINCIPAL="agent-control",
        SINNIXD_PRINCIPAL="agent-control",
        POLYLOGUE_ARCHIVE_ROOT="/realm/state/polylogue",
    )

    run_pytest(_marker_command(tmp_path / "unused"), cwd=str(tmp_path), env=secret_environment, root=tmp_path)

    allowed = set(pytest_slot.INHERITED_ENVIRONMENT_KEYS)
    for call in _calls(record):
        recorded = {key for key in call["env"] if not key.startswith(("PYTHON", "LC_", "LANG"))}
        assert recorded <= allowed, f"agentctl inherited {sorted(recorded - allowed)}"
    launch = _launch_document(tmp_path)
    assert launch["environment"]["ANTHROPIC_API_KEY"] == "secret", "pytest's own environment travels in the launch file"


@pytest.mark.parametrize("prefix", ["AGENTCTL_", "SINNIXD_"])
def test_a_job_id_is_never_slot_ownership(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefix: str) -> None:
    """A lane inherits job identity, a principal and an operation name; none is the slot."""
    record = _install_fake_agentctl(tmp_path, monkeypatch)
    marker = tmp_path / "pytest-ran"
    lane_environment = _environment(
        **{
            f"{prefix}JOB_ID": "job-1",
            f"{prefix}OPERATION": "verify_affected",
            f"{prefix}QUEUE_WORKER": "1",
            f"{prefix}PRINCIPAL": "agent-control",
        }
    )

    assert not holds_pytest_slot(lane_environment)
    outcome = run_pytest(_marker_command(marker), cwd=str(tmp_path), env=lane_environment, root=tmp_path)

    assert not marker.exists()
    assert outcome.slot == "agentctl job 7"
    assert _verbs(record)[0] == "job start"


@pytest.mark.parametrize("cgroup", PYTEST_CGROUPS)
def test_the_pytest_pool_cgroup_holds_the_slot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cgroup: str) -> None:
    """The slice the runtime placed the job in is ownership, with no environment at all.

    Anti-vacuity: pointing the stub at any other slice makes this red, because
    the run submits instead of executing.
    """
    record = _install_fake_agentctl(tmp_path, monkeypatch)
    stub_cgroup(cgroup, tmp_path=tmp_path, monkeypatch=monkeypatch)
    marker = tmp_path / "pytest-ran"

    assert holds_pytest_slot({})
    outcome = run_pytest(_marker_command(marker), cwd=str(tmp_path), env=_environment(), root=tmp_path)

    assert marker.exists()
    assert outcome.slot == "held"
    assert outcome.returncode == 0
    assert _calls(record) == [], "a pytest-pool job must not recursively submit"


@pytest.mark.parametrize("pool_variable", ["AGENTCTL_POOL", "SINNIXD_QUEUE_POOL"])
def test_the_declared_pytest_pool_holds_the_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pool_variable: str
) -> None:
    record = _install_fake_agentctl(tmp_path, monkeypatch)
    marker = tmp_path / "pytest-ran"
    holder = {pool_variable: "pytest"}

    assert holds_pytest_slot(holder)
    outcome = run_pytest(_marker_command(marker), cwd=str(tmp_path), env=_environment(**holder), root=tmp_path)

    assert marker.exists()
    assert outcome.slot == "held"
    assert _calls(record) == []


def test_explicit_slot_holder_runs_directly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = _install_fake_agentctl(tmp_path, monkeypatch)
    marker = tmp_path / "pytest-ran"
    holder = {"POLYLOGUE_PYTEST_SLOT": "held"}

    assert holds_pytest_slot(holder)
    outcome = run_pytest(_marker_command(marker), cwd=str(tmp_path), env=_environment(**holder), root=tmp_path)

    assert marker.exists()
    assert outcome.slot == "held"
    assert outcome.returncode == 0
    assert _calls(record) == [], "a slot holder must not talk to agentctl"


@pytest.mark.parametrize("cgroup", AGENT_CGROUPS)
def test_an_agent_pool_job_submits_its_focused_run_to_the_pytest_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cgroup: str
) -> None:
    record = _install_fake_agentctl(tmp_path, monkeypatch)
    stub_cgroup(cgroup, tmp_path=tmp_path, monkeypatch=monkeypatch)
    marker = tmp_path / "pytest-ran"

    outcome = run_pytest(
        _marker_command(marker),
        cwd=str(tmp_path),
        env=_environment(AGENTCTL_JOB_ID="agent-job", AGENTCTL_POOL="agent"),
        root=tmp_path,
    )

    assert not marker.exists()
    assert outcome.slot == "agentctl job 7"
    start = next(call for call in _calls(record) if "start" in call["argv"])
    assert "pytest_focused" in start["argv"]


def test_a_missing_runtime_refuses_rather_than_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    marker = tmp_path / "pytest-ran"
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))

    with pytest.raises(PytestSlotUnavailableError) as failure:
        run_pytest(_marker_command(marker), cwd=str(tmp_path), env=_environment(PATH=str(empty)), root=tmp_path)

    assert "`agentctl` is not on PATH" in str(failure.value)
    assert "systemctl --user start pueued" in str(failure.value)
    assert not marker.exists()
    assert not list((tmp_path / ".cache" / "verify").glob("tmp-*")), "a refused run leaves no scratch"


def test_a_failed_job_reports_its_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_agentctl(tmp_path, monkeypatch, job_id=12, phase="failed", exit_code=1)

    outcome = run_pytest(_marker_command(tmp_path / "unused"), cwd=str(tmp_path), env=_environment(), root=tmp_path)

    assert (outcome.returncode, outcome.slot) == (1, "agentctl job 12")


@pytest.mark.parametrize("phase", ["cancelled", "refused", "slot-occupied", "vanished"])
def test_a_job_that_did_not_run_is_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    _install_fake_agentctl(tmp_path, monkeypatch, job_id=12, phase=phase, exit_code=130)

    with pytest.raises(PytestSlotUnavailableError, match=f"agentctl job 12 ended '{phase}'"):
        run_pytest(_marker_command(tmp_path / "unused"), cwd=str(tmp_path), env=_environment(), root=tmp_path)


def test_a_timed_out_job_reports_the_typed_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_agentctl(tmp_path, monkeypatch, job_id=12, phase="timed-out", exit_code=124)
    receipt_path = tmp_path / ".cache" / "verify" / f"pytest-slot-{os.getpid()}.result.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps({"status": "timed_out", "diagnosis": "pytest_deadline"}), encoding="utf-8")

    outcome = run_pytest(_marker_command(tmp_path / "unused"), cwd=str(tmp_path), env=_environment(), root=tmp_path)

    assert outcome.returncode == 124
    assert outcome.receipt == {"status": "timed_out", "diagnosis": "pytest_deadline"}


def test_the_slot_runner_executes_the_launch_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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
    result = json.loads(capsys.readouterr().out)
    assert result["kind"] == "polylogue.pytest-slot-result"
    assert (result["status"], result["exit_code"]) == ("success", 0)


def test_the_slot_runner_runs_a_selection_through_devtools_test(monkeypatch: pytest.MonkeyPatch) -> None:
    """``agentctl job start polylogue pytest_focused -- <selection>`` is ``devtools test`` in the pool."""
    from devtools import run_tests

    seen: list[list[str]] = []

    def fake_main(argv: list[str]) -> int:
        seen.append(list(argv))
        return 3

    monkeypatch.setattr(run_tests, "main", fake_main)

    assert pytest_slot.main(["tests/unit/devtools/test_agent_env.py", "-n", "0"]) == 3
    assert seen == [["tests/unit/devtools/test_agent_env.py", "-n", "0"]]
    assert pytest_slot.main([]) == 2


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


def test_a_submitted_run_contains_its_temporary_trees(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither pytest nor the code under test may write to the ambient TMPDIR.

    ``nix develop`` points TMPDIR at a small tmpfs; a corpus run left there
    fills the mount and dies on exhausted file descriptors.
    """
    _install_fake_agentctl(tmp_path, monkeypatch)
    scratch = tmp_path / ".cache" / "verify"

    run_pytest(
        _marker_command(tmp_path / "unused"),
        cwd=str(tmp_path),
        env=_environment(TMPDIR="/tmp/nix-shell.L3brFS"),
        root=tmp_path,
    )

    launch = _launch_document(tmp_path)
    assert Path(launch["environment"]["TMPDIR"]).is_relative_to(scratch)
    argv = launch["argv"]
    basetemp = Path(argv[argv.index("--basetemp") + 1])
    assert basetemp.is_relative_to(scratch)


def test_a_declared_basetemp_is_kept_and_still_anchors_tmpdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``devtools test`` names its own basetemp so it can dispose of it."""
    _install_fake_agentctl(tmp_path, monkeypatch)
    declared = tmp_path / ".cache" / "verify" / "tmp-chosen"

    run_pytest(
        [*_marker_command(tmp_path / "unused"), "--basetemp", str(declared)],
        cwd=str(tmp_path),
        env=_environment(TMPDIR="/tmp/nix-shell.L3brFS"),
        root=tmp_path,
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
    )

    seen = Path(recorded.read_text(encoding="utf-8"))
    assert seen.is_relative_to(tmp_path / ".cache" / "verify")


def _scratch_trees(root: Path) -> list[Path]:
    return sorted((root / ".cache" / "verify").glob("tmp-*"))


def test_a_successful_run_removes_both_temporary_trees(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        "import os, sys; os.makedirs(os.path.join(os.environ['TMPDIR'], 'used')); os.makedirs(sys.argv[2])",
    ]

    outcome = run_pytest(command, cwd=str(tmp_path), env=_environment(POLYLOGUE_PYTEST_SLOT="held"), root=tmp_path)

    assert outcome.returncode == 0
    assert _scratch_trees(tmp_path) == []


def test_a_failed_run_keeps_its_temporary_trees_for_reading(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        "import os, sys; os.makedirs(os.path.join(os.environ['TMPDIR'], 'used')); os.makedirs(sys.argv[2]); sys.exit(1)",
    ]

    outcome = run_pytest(command, cwd=str(tmp_path), env=_environment(POLYLOGUE_PYTEST_SLOT="held"), root=tmp_path)

    assert outcome.returncode == 1
    assert [tree.name.endswith(".tmpdir") for tree in _scratch_trees(tmp_path)] == [False, True]


def test_an_interrupted_held_run_removes_its_temporary_trees(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: disposing only on a normal return leaves the trees this test finds."""

    def interrupted(*_args: Any, **_kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "Popen", interrupted)

    with pytest.raises(KeyboardInterrupt):
        run_pytest(
            _marker_command(tmp_path / "unused"),
            cwd=str(tmp_path),
            env=_environment(POLYLOGUE_PYTEST_SLOT="held"),
            root=tmp_path,
        )

    assert _scratch_trees(tmp_path) == []


def test_a_configured_basetemp_root_is_honoured(tmp_path: Path) -> None:
    """A sandbox with no checkout-local scratch names its own root."""
    elsewhere = tmp_path / "sandbox-scratch"

    assert basetemp_root({BASETEMP_ROOT_ENV: str(elsewhere)}, root=tmp_path) == elsewhere


def test_the_leaked_cloud_basetemp_sentinel_is_declined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`.claude/settings.json` exports the cloud value into workstation sessions."""
    monkeypatch.setattr(cloud_sentinels, "_WORKSTATION_SCRATCH_MOUNT", tmp_path)
    sentinel = cloud_sentinels.CLOUD_SENTINELS[BASETEMP_ROOT_ENV]

    assert basetemp_root({BASETEMP_ROOT_ENV: sentinel}, root=tmp_path) == tmp_path / ".cache" / "verify"


#: An ``agentctl`` whose ``job get`` kills the process waiting on it, the way a
#: session or a wrapper being killed leaves a queued job with no waiter. The
#: cancellation is recorded like every other call; the refusing variant
#: answers ``job cancel`` the way a job id pueue no longer holds is answered.
FAKE_AGENTCTL_KILLS_ITS_WAITER = """import json, os, signal, sys, time

with open(sys.argv[0] + ".calls.jsonl", "a", encoding="utf-8") as handle:
    handle.write(json.dumps({{"argv": sys.argv[1:]}}) + "\\n")

words = [word for word in sys.argv[1:] if word != "--json"]
verb = " ".join(words[:2])
if verb == "job start":
    print(json.dumps({{"job_id": 11, "phase": "queued", "terminal": False}}))
elif verb == "job get":
    os.kill(os.getppid(), signal.SIGTERM)
    time.sleep(2)
    print(json.dumps({{"job_id": 11, "phase": "queued", "terminal": False}}))
elif verb == "job cancel":
    if {refuses}:
        sys.stderr.write("agentctl: no such job\\n")
        sys.exit(1)
sys.exit(0)
"""

_WAITER = """
import os, sys
sys.path.insert(0, {repo!r})
from devtools import agent_env
agent_env._CGROUP_PATH = type(agent_env._CGROUP_PATH)({cgroup!r})
from devtools.pytest_slot import run_pytest

run_pytest(
    [sys.executable, "-c", "pass"],
    cwd={cwd!r},
    env={{"PATH": os.environ["PATH"], "HOME": os.environ["HOME"]}},
    root={root!r},
)
"""


def _killed_waiter(tmp_path: Path, *, refuses: bool) -> tuple[subprocess.CompletedProcess[str], Path]:
    directory = tmp_path / "fakebin"
    agentctl = _install_executable(directory, "agentctl", FAKE_AGENTCTL_KILLS_ITS_WAITER.format(refuses=refuses))
    repo = str(Path(pytest_slot.__file__).resolve().parents[1])
    cgroup = tmp_path / "cgroup"
    cgroup.write_text(OUTSIDE_CGROUP, encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _WAITER.format(repo=repo, cwd=str(tmp_path), root=str(tmp_path), cgroup=str(cgroup)),
        ],
        env={
            "PATH": str(directory),
            "HOME": os.environ.get("HOME", "/home/nobody"),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    return completed, Path(str(agentctl) + ".calls.jsonl")


def test_a_killed_waiter_reaps_the_job_it_submitted(tmp_path: Path) -> None:
    """A job outlives its waiter, and the pool's parallelism is one.

    Anti-vacuity: dropping the reaping action records no cancellation at all --
    the job stays queued with nothing left to wait on it, which is exactly the
    starvation this reap exists to prevent. Disposing of the scratch trees
    only on a normal return leaves the ``tmp-*`` directories this test finds.
    """
    completed, record = _killed_waiter(tmp_path, refuses=False)

    assert completed.returncode == -int(signal.SIGTERM), completed.stderr
    assert _verbs(record) == ["job start", "job get 11", "job cancel 11"], _verbs(record)
    assert not list((tmp_path / ".cache" / "verify").glob("pytest-slot-*.json")), (
        "the launch file carries a resolved environment and must not survive the reap"
    )
    assert _scratch_trees(tmp_path) == [], "a killed waiter leaves no scratch behind"


def test_a_refused_cancellation_leaves_the_launch_file_for_the_job(tmp_path: Path) -> None:
    """A job agentctl would not stop still reads its launch file when it starts.

    Anti-vacuity: unlinking the launch file regardless of the cancellation's
    outcome empties the glob below, and the surviving job then starts with no
    resolved environment to read.
    """
    completed, record = _killed_waiter(tmp_path, refuses=True)

    assert completed.returncode == -int(signal.SIGTERM), completed.stderr
    assert _verbs(record)[-1] == "job cancel 11"
    surviving = list((tmp_path / ".cache" / "verify").glob("pytest-slot-*.json"))
    assert len(surviving) == 1, surviving
