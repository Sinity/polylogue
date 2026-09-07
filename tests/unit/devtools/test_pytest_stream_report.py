"""The managed run's JSON report is written from a spool, not from memory.

The report is what ``devtools why`` and every managed step's success predicate
read, so replacing the accumulating writer has to preserve the shape those
consumers depend on while removing the accumulation.

Anti-vacuity:
- give ``StreamingReport`` a list it appends every finished test to and
  ``test_retention_does_not_grow_with_the_number_of_tests`` goes red -- that is
  the accumulation the plugin exists to remove;
- stop flushing on teardown and ``test_a_finished_test_is_not_held_in_memory``
  goes red;
- take the phase outcome instead of the test's and
  ``test_the_report_states_each_test_outcome_consumers_read`` goes red on the
  erroring fixture, which a phase reading calls ``passed``;
- drop the ``workerinput`` guard and
  ``test_an_xdist_worker_does_not_write_the_controller_report`` goes red -- each
  worker would truncate the report to its own share of the session.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from devtools import run_tests
from devtools.pytest_evidence import evaluate_pytest_evidence
from devtools.pytest_invocation import CLEAR_CONFIGURED_ADDOPTS
from devtools.pytest_stream_report import (
    PLUGIN_NAME,
    StreamingReport,
    report_file_argument,
    spool_paths,
)
from devtools.toolchain import venv_python

ROOT = Path(__file__).resolve().parents[3]

#: One file exercising every outcome a consumer distinguishes: a plain pass, a
#: failure, a skip, an expected failure, and a fixture that errors during setup
#: while its teardown still reports ``passed``.
_SUITE = """
import pytest


@pytest.fixture
def broken():
    raise RuntimeError("fixture is broken")


def test_passes():
    assert True


def test_fails():
    assert 1 == 2


@pytest.mark.skip(reason="declared")
def test_skipped():
    pass


@pytest.mark.xfail(reason="declared")
def test_xfails():
    assert False


def test_errors(broken):
    pass
"""


class _FakeReport(SimpleNamespace):
    """A phase report shaped like the attributes the plugin reads."""

    def __init__(self, nodeid: str, when: str, outcome: str = "passed", duration: float = 0.01) -> None:
        super().__init__(
            nodeid=nodeid,
            when=when,
            outcome=outcome,
            duration=duration,
            location=("tests/test_a.py", 3, nodeid),
            longrepr=None,
            longreprtext="",
        )


def _plugin(tmp_path: Path) -> StreamingReport:
    """A ``StreamingReport`` whose hook lookups answer with the phase category."""

    def report_teststatus(*, report: Any, config: Any) -> tuple[str, str, str]:
        del config
        outcome = "" if report.when != "call" and report.outcome == "passed" else str(report.outcome)
        return (outcome, "", outcome.upper())

    config = SimpleNamespace(
        hook=SimpleNamespace(pytest_report_teststatus=report_teststatus),
        rootpath=tmp_path,
    )
    plugin = StreamingReport(config, tmp_path / "report.json")
    plugin.pytest_sessionstart(object())
    return plugin


def _drive(plugin: StreamingReport, nodeid: str, *, call_outcome: str = "passed") -> None:
    plugin.pytest_runtest_logreport(_FakeReport(nodeid, "setup"))
    plugin.pytest_runtest_logreport(_FakeReport(nodeid, "call", call_outcome))
    plugin.pytest_runtest_logreport(_FakeReport(nodeid, "teardown"))


def test_a_finished_test_is_not_held_in_memory(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path)

    plugin.pytest_runtest_logreport(_FakeReport("t::one", "setup"))
    assert "t::one" in plugin._open, "a started test is in flight until its teardown"
    plugin.pytest_runtest_logreport(_FakeReport("t::one", "call"))
    plugin.pytest_runtest_logreport(_FakeReport("t::one", "teardown"))

    assert plugin._open == {}


def test_retention_does_not_grow_with_the_number_of_tests(tmp_path: Path) -> None:
    """The structure the controller keeps is bounded by the tests in flight.

    A 20,000-test corpus is the case that matters; the invariant is that
    nothing the plugin keeps is indexed by test, so it is checked directly
    rather than by running one.
    """
    plugin = _plugin(tmp_path)

    for index in range(5000):
        _drive(plugin, f"tests/test_a.py::test_{index}")
        assert plugin._open == {}

    # Counted outcomes, not tests: the only per-test bytes are on disk.
    assert set(plugin._outcomes) == {"passed"}
    assert plugin._outcomes["passed"] == 5000
    oversized = {
        name: len(value)
        for name, value in vars(plugin).items()
        if isinstance(value, (dict, list, set, tuple)) and len(value) > 16
    }
    assert oversized == {}, "the controller keeps a structure indexed by test"


def test_the_spool_is_removed_once_the_report_is_assembled(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path)
    _drive(plugin, "tests/test_a.py::test_one")
    spool = plugin._spool_path
    assert spool.exists()

    plugin.pytest_sessionfinish(SimpleNamespace(testscollected=1), 0)

    assert not spool.exists()
    assert spool_paths(tmp_path / "report.json") == ()
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert [test["nodeid"] for test in report["tests"]] == ["tests/test_a.py::test_one"]


def test_a_started_test_without_a_teardown_still_reaches_the_report(tmp_path: Path) -> None:
    """A worker lost mid-test executed that test; the report must say so."""
    plugin = _plugin(tmp_path)
    plugin.pytest_runtest_logreport(_FakeReport("tests/test_a.py::test_lost", "setup"))
    plugin.pytest_runtest_logreport(_FakeReport("tests/test_a.py::test_lost", "call", "failed"))

    plugin.pytest_sessionfinish(SimpleNamespace(testscollected=1), 1)

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert [test["nodeid"] for test in report["tests"]] == ["tests/test_a.py::test_lost"]
    assert report["tests"][0]["outcome"] == "failed"


def test_a_leftover_spool_is_named_for_removal(tmp_path: Path) -> None:
    """A killed run leaves its spool; the next run of the same report clears it."""
    report = tmp_path / "report.json"
    leftover = tmp_path / "report.json.4242.parts"
    leftover.write_text("{}\n", encoding="utf-8")
    (tmp_path / "unrelated.json").write_text("{}", encoding="utf-8")

    assert spool_paths(report) == (leftover,)


def test_an_xdist_worker_does_not_write_the_controller_report(tmp_path: Path) -> None:
    from devtools import pytest_stream_report

    registered: list[str] = []
    manager = SimpleNamespace(register=lambda _plugin, name: registered.append(name))
    worker = SimpleNamespace(
        getoption=lambda _name, _default=None: str(tmp_path / "report.json"),
        pluginmanager=manager,
        workerinput={"workerid": "gw0"},
    )

    pytest_stream_report.pytest_configure(worker)

    assert registered == []

    controller = SimpleNamespace(
        getoption=lambda _name, _default=None: str(tmp_path / "report.json"),
        pluginmanager=manager,
    )
    pytest_stream_report.pytest_configure(controller)
    assert registered == [PLUGIN_NAME]


@pytest.mark.slow
def test_the_report_states_each_test_outcome_consumers_read(tmp_path: Path) -> None:
    """A real pytest session writes the shape ``devtools`` reads back.

    Driven as a subprocess because the outcome of a phase is pytest's own
    ``pytest_report_teststatus`` decision, which is what a hand-built report
    would have to assume rather than exercise.
    """
    suite = tmp_path / "test_outcomes.py"
    suite.write_text(_SUITE, encoding="utf-8")
    report_path = tmp_path / "report.json"
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONPATH": str(ROOT)}
    env.pop("PYTEST_ADDOPTS", None)

    completed = subprocess.run(
        [
            venv_python(root=ROOT),
            "-m",
            "pytest",
            "-q",
            CLEAR_CONFIGURED_ADDOPTS,
            "-p",
            "no:cacheprovider",
            "-p",
            "no:randomly",
            "-p",
            "devtools.pytest_stream_report",
            report_file_argument(report_path),
            str(suite),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert report_path.exists(), completed.stdout + completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    outcomes = {test["nodeid"].split("::")[-1]: test["outcome"] for test in report["tests"]}
    assert outcomes == {
        "test_passes": "passed",
        "test_fails": "failed",
        "test_skipped": "skipped",
        "test_xfails": "xfailed",
        "test_errors": "error",
    }
    assert report["exitcode"] == completed.returncode == 1
    assert report["summary"]["total"] == 5
    assert report["summary"]["passed"] == 1
    # ``root`` is pytest's rootdir, which the checkout's inifile decides.
    assert report["root"] == str(ROOT)
    assert not spool_paths(report_path)

    # The two things devtools computes from a test entry.
    assert all(run_tests._phase_duration(test) > 0 for test in report["tests"])
    evidence = evaluate_pytest_evidence(
        report=report,
        selection={"selected_count": 5},
        summary={},
        events=[{"event": "collection_finished"}],
        exit_code=1,
    )
    assert evidence["diagnosis"] == "pytest_failed"


def test_a_managed_run_directs_its_report_through_this_plugin() -> None:
    """The option the runner passes is the one the plugin declares."""
    command = run_tests.build_pytest_cmd(["tests/unit/pipeline"])

    assert report_file_argument(run_tests.PYTEST_REPORT_PATH) in command
    assert "devtools.pytest_stream_report" in command
    assert not any(argument.startswith("--json-report") for argument in command)
