"""The host's single pytest slot is acquired before any managed pytest runs.

Anti-vacuity: deleting the queueing branch in ``devtools.pytest_slot.run_pytest``
makes ``test_outside_a_task_the_run_is_queued`` red — the command executes here
and the marker file appears. Widening ``INHERITED_ENVIRONMENT_KEYS`` makes
``test_the_adder_environment_carries_only_the_allowed_keys`` red. Dropping
either half of the temporary-directory containment (the ``--basetemp``
argument or the exported TMPDIR) makes
``test_a_queued_run_contains_its_temporary_trees`` red.
"""

from __future__ import annotations

import json
import os
import signal
import stat
import sys
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

FAKE_PUEUE = """#!/usr/bin/env python3
import json, os, sys

# The record path is derived from this script's own location: a queued run's
# adder environment is scrubbed, so nothing can be passed through it.
with open(sys.argv[0] + ".calls.jsonl", "a", encoding="utf-8") as handle:
    handle.write(json.dumps({{"argv": sys.argv[1:], "env": dict(os.environ)}}) + "\\n")

command = sys.argv[1] if len(sys.argv) > 1 else ""
if command == "add":
    print("{task_id}")
elif command == "status":
    print(json.dumps({{"tasks": {{"{task_id}": {{"status": {{"Done": {{"result": {result}}}}}}}}}}}))
sys.exit(0)
"""


def _install_fake_pueue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    task_id: str = "7",
    result: str = '"Success"',
) -> Path:
    directory = tmp_path / "fakebin"
    directory.mkdir(exist_ok=True)
    script = directory / "pueue"
    script.write_text(FAKE_PUEUE.format(task_id=task_id, result=result), encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{directory}{os.pathsep}{os.environ['PATH']}")
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
    assert "devtools.pytest_slot" in add["argv"]
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


@pytest.mark.parametrize(
    "holder", [{"AGENTCTL_JOB_ID": "job-1"}, {"SINNIXD_JOB_ID": "job-1"}, {"POLYLOGUE_PYTEST_SLOT": "held"}]
)
def test_inside_the_slot_the_run_is_direct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, holder: dict[str, str]
) -> None:
    record = _install_fake_pueue(tmp_path, monkeypatch)
    marker = tmp_path / "pytest-ran"

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


def test_an_unreachable_queue_refuses_rather_than_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    marker = tmp_path / "pytest-ran"
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    with pytest.raises(PytestSlotUnavailableError) as failure:
        run_pytest(
            _marker_command(marker),
            cwd=str(tmp_path),
            env=_environment(PATH=str(tmp_path / "empty")),
            root=tmp_path,
            label="polylogue:test:1",
        )

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


def _launch_document(root: Path) -> dict[str, Any]:
    """The launch file the (faked) adder left behind, unconsumed."""
    launches = list((root / pytest_slot.LAUNCH_DIR).glob("pytest-slot-*.json"))
    assert len(launches) == 1, f"expected one launch file, found {launches}"
    document: dict[str, Any] = json.loads(launches[0].read_text(encoding="utf-8"))
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
FAKE_PUEUE_KILLS_ITS_WAITER = """#!/usr/bin/env python3
import json, os, signal, sys, time

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

    Anti-vacuity: dropping the ``_reaping`` context leaves the recorded calls
    at ``add``/``wait`` -- the task stays queued with nothing left to wait on
    it, which is exactly the starvation this reap exists to prevent.
    """
    import subprocess

    directory = tmp_path / "fakebin"
    directory.mkdir()
    script = directory / "pueue"
    script.write_text(FAKE_PUEUE_KILLS_ITS_WAITER, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    record = Path(str(script) + ".calls.jsonl")
    repo = str(Path(pytest_slot.__file__).resolve().parents[1])

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _WAITER.format(repo=repo, cwd=str(tmp_path), root=str(tmp_path)),
        ],
        env={
            "PATH": f"{directory}{os.pathsep}{os.environ['PATH']}",
            "HOME": os.environ.get("HOME", "/home/nobody"),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == -int(signal.SIGTERM), completed.stderr
    verbs = [call["argv"][0] for call in _calls(record)]
    assert verbs == ["add", "wait", "kill", "remove"], verbs
    leftover = list((tmp_path / "verify").glob("pytest-slot-*.json")) + list(
        (tmp_path / ".cache" / "verify").glob("pytest-slot-*.json")
    )
    assert leftover == [], "the launch file carries a resolved environment and must not survive the reap"
