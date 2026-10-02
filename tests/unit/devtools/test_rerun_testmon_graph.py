"""An accepted rerun clears the failure the testmon graph recorded.

testmon reselects every recorded failure on each later run, so a flaky test
whose passing rerun was not recorded keeps being selected by every affected
``devtools verify``, whatever the change.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

import devtools
from devtools import pytest_rerun, verify
from devtools.pytest_invocation import MANAGED_PLUGIN_ARGS
from devtools.testmon_provision import TESTMON_ENVIRONMENT
from devtools.testmon_provision import testmon_environment as _testmon_environment

_NODEID = "tests/test_flaky.py::test_passes_the_second_time"


def _recorded_failures(datafile: Path) -> set[str]:
    with closing(sqlite3.connect(datafile)) as conn:
        return {str(row[0]) for row in conn.execute("SELECT test_name FROM test_execution WHERE failed")}


def test_a_passing_rerun_replaces_the_recorded_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: rerun with ``-p no:testmon`` and the node stays recorded
    as failed, so the final assertion goes red."""
    monkeypatch.setattr(pytest_rerun, "venv_python", lambda root: sys.executable)
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    sentinel = tmp_path / "first-attempt-ran"
    (tests_dir / "test_flaky.py").write_text(
        "from pathlib import Path\n\n\n"
        "def test_passes_the_second_time():\n"
        f"    sentinel = Path({str(sentinel)!r})\n"
        "    first = not sentinel.exists()\n"
        "    sentinel.touch()\n"
        "    assert not first\n",
        encoding="utf-8",
    )
    datafile = tmp_path / ".testmondata"
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("TESTMON", "PYTEST_", "POLYLOGUE_PYTEST", "COV_"))
    }
    env.update(
        {
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "TESTMON_DATAFILE": str(datafile),
            "PYTHONPATH": str(Path(devtools.__file__).resolve().parents[1]),
        }
    )
    first_command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *MANAGED_PLUGIN_ARGS,
        "--testmon",
        f"--testmon-env={TESTMON_ENVIRONMENT}",
        "--testmon-noselect",
        "-p",
        "no:randomly",
        "--override-ini=addopts=",
        "tests/test_flaky.py",
    ]
    first = subprocess.run(first_command, cwd=tmp_path, env=env, capture_output=True, text=True, check=False)
    assert first.returncode == 1, first.stdout + first.stderr
    assert _recorded_failures(datafile) == {_NODEID}

    step_dir = tmp_path / "step"
    step_dir.mkdir()
    report_path = step_dir / "pytest-report.json"
    report_path.write_text(json.dumps({"tests": [{"nodeid": _NODEID, "outcome": "failed"}]}), encoding="utf-8")
    plan = pytest_rerun.build_rerun(
        report_path=report_path,
        step_dir=step_dir,
        root=tmp_path,
        testmon_env=pytest_rerun.testmon_rerun_environment(first_command),
    )
    assert plan is not None
    failed, rerun_command, _rerun_report = plan
    assert failed == [_NODEID]

    rerun = subprocess.run(rerun_command, cwd=tmp_path, env=env, capture_output=True, text=True, check=False)

    assert rerun.returncode == 0, rerun.stdout + rerun.stderr
    assert _recorded_failures(datafile) == set()


def test_the_rerun_records_into_the_environment_the_first_run_traced() -> None:
    """The verify lane traces with testmon on corpus and affected runs, and a
    descriptor run writes no fingerprints, so its rerun must not either."""
    traced = verify._pytest_command(selection="affected", worker_args=(), hypothesis_profile=None, explicit_tests=())
    untraced = verify._pytest_command(
        selection="descriptor", worker_args=(), hypothesis_profile=None, explicit_tests=()
    )

    assert pytest_rerun.testmon_rerun_environment(traced) == _testmon_environment(verify.ROOT)
    assert pytest_rerun.testmon_rerun_environment(untraced) is None
    assert not [option for option in pytest_rerun.semantic_rerun_options(traced) if "testmon" in option]
