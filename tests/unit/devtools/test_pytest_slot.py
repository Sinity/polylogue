"""The host's single pytest slot is acquired before any managed pytest runs.

Anti-vacuity: deleting the queueing branch in ``devtools.pytest_slot.run_pytest``
makes ``test_outside_a_task_the_run_is_queued`` red — the command executes here
and the marker file appears. Widening ``INHERITED_ENVIRONMENT_KEYS`` makes
``test_the_adder_environment_carries_only_the_allowed_keys`` red.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from devtools import pytest_slot
from devtools.pytest_slot import PytestSlotUnavailableError, holds_pytest_slot, run_pytest

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


@pytest.mark.parametrize("holder", [{"SINNIXD_JOB_ID": "job-1"}, {"POLYLOGUE_PYTEST_SLOT": "held"}])
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
