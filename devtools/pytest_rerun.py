"""Adjudicate a failed pytest run by rerunning its failures alone, once.

One mechanism, two callers. ``devtools verify`` has had this since the
workstation pytest pool made contention normal: a run beside six sibling jobs
can fail on load alone, and that is not a verdict on the head. ``devtools
test`` -- the command every parallel worker actually uses for its inner loop
-- had no equivalent, so the mitigation existed and was bypassed by the
command that needed it most.

The rule is the same on both routes: a test that fails twice is red and
decides the run. A test that passes alone is flaky, recorded with both
outcomes in the canonical report, where it counts as passed. Nothing here can
turn a run that pytest itself could not finish (exit 2/3/4, a signal) into a
pass -- only exit 1, "tests failed", is adjudicable.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from devtools.pytest_invocation import (
    CLEAR_CONFIGURED_ADDOPTS,
    MANAGED_PLUGIN_ARGS,
    REPORT_PLUGIN_ARGS,
)
from devtools.pytest_slot import PytestSlotUnavailableError, run_pytest, run_pytest_isolated
from devtools.pytest_stream_report import report_file_argument
from devtools.toolchain import venv_python

__all__ = [
    "MAX_RERUN_NODEIDS",
    "read_json",
    "report_nodeid_to_selector",
    "rerun_failed_once",
]

#: Above this the failure is systemic, not contention, and rerunning it one
#: node at a time costs more than it can ever clear.
MAX_RERUN_NODEIDS = 300


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def report_nodeid_to_selector(nodeid: str) -> str:
    """Strip xdist's ``@<group>`` suffix so a report node id selects again.

    ``--dist=loadgroup`` reports ``path::test[param]@group``; pytest cannot
    collect that literal, so a rerun built from it errors before running.
    A parametrization id may itself contain ``@``, so only a suffix after the
    closing bracket (or after the bare test name) is removed.
    """
    head, sep, tail = nodeid.rpartition("@")
    if not sep or "::" not in head:
        return nodeid
    if "[" in tail or "]" in tail or "/" in tail or "::" in tail:
        return nodeid
    if head.endswith("]") or "[" not in head.rsplit("::", 1)[-1]:
        return head
    return nodeid


def _path_for_receipt(path: Path, *, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def rerun_failed_once(
    *,
    report_path: Path,
    step_dir: Path,
    env: Mapping[str, str],
    root: Path,
    runner: str = "managed",
) -> dict[str, Any] | None:
    """Rerun exactly the failed tests once, alone and unselected.

    ``report_path`` is the just-finished run's JSON report; it is patched in
    place when a failure clears, so the caller's downstream statistics read
    the adjudicated outcome. Returns ``None`` when there is nothing to
    adjudicate (no readable report, no failures, or more failures than
    :data:`MAX_RERUN_NODEIDS`).
    """
    report = read_json(report_path)
    if not isinstance(report, Mapping):
        return None
    failed = [
        report_nodeid_to_selector(str(test["nodeid"]))
        for test in report.get("tests", [])
        if isinstance(test, Mapping) and test.get("outcome") in {"failed", "error"} and test.get("nodeid")
    ]
    if not failed or len(failed) > MAX_RERUN_NODEIDS:
        return None
    rerun_report = step_dir / "pytest-rerun.json"
    rerun_command = [
        venv_python(root=root),
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        report_file_argument(rerun_report),
        *REPORT_PLUGIN_ARGS,
        *MANAGED_PLUGIN_ARGS,
        "-p",
        "no:testmon",
        "-p",
        "no:randomly",
        CLEAR_CONFIGURED_ADDOPTS,
        *failed,
    ]
    sys.stderr.write(f"\n  rerun {len(failed)} failed test(s) alone ... ")
    sys.stderr.flush()
    rerun_env = {key: value for key, value in env.items() if not key.startswith("PYTEST_XDIST")}
    # The rerun is pytest too, so it holds the host's pytest slot like the run
    # it is adjudicating.
    try:
        executor = run_pytest if runner == "managed" else run_pytest_isolated
        rerun_completed = executor(rerun_command, cwd=str(root), env=rerun_env, root=root, stdout=sys.stderr)
    except PytestSlotUnavailableError as exc:
        sys.stderr.write(f"\n  rerun could not acquire the pytest slot: {exc}\n")
        return {"attempted": failed, "still_failed": failed, "flaky": [], "rerun_report": None, "rerun_exit": 125}
    second = read_json(rerun_report)
    if not isinstance(second, Mapping) or rerun_completed.returncode not in (0, 1):
        # No report, or pytest itself did not finish cleanly (exit 3 is an
        # internal error): nothing here clears a failure.
        return {
            "attempted": failed,
            "still_failed": failed,
            "flaky": [],
            "rerun_report": None,
            "rerun_exit": rerun_completed.returncode,
        }
    second_outcome = {
        str(test["nodeid"]): str(test.get("outcome"))
        for test in second.get("tests", [])
        if isinstance(test, Mapping) and test.get("nodeid")
    }
    still_failed = [nodeid for nodeid in failed if second_outcome.get(nodeid) != "passed"]
    flaky = [nodeid for nodeid in failed if second_outcome.get(nodeid) == "passed"]
    if not still_failed and rerun_completed.returncode != 0:
        # Every node passed but the process did not: the run is not green.
        still_failed, flaky = failed, []
    if flaky:
        patched = dict(report)
        tests = []
        for test in report.get("tests", []):
            if isinstance(test, Mapping) and test.get("nodeid") in flaky:
                test = {**test, "first_outcome": test.get("outcome"), "outcome": "passed", "flaky": True}
            tests.append(test)
        patched["tests"] = tests
        summary = dict(report.get("summary") or {})
        for key in ("failed", "error"):
            if key in summary:
                summary[key] = max(
                    0,
                    int(summary[key])
                    - sum(
                        1
                        for t in report.get("tests", [])
                        if isinstance(t, Mapping) and t.get("nodeid") in flaky and t.get("outcome") == key
                    ),
                )
        summary["passed"] = int(summary.get("passed", 0)) + len(flaky)
        summary["flaky"] = len(flaky)
        if not still_failed:
            summary["exitstatus"] = 0
            # The report carries its exit status twice. A consumer reading the
            # top-level field would still see the pre-rerun failure.
            patched["exitcode"] = 0
        patched["summary"] = summary
        patched["flaky_nodeids"] = list(flaky)
        report_path.write_text(json.dumps(patched), encoding="utf-8")
        # The progress plugin's own summary carries the first exit status;
        # evidence evaluation compares the two, so both tell the same story.
        plugin_summary_path = step_dir / "summary.json"
        plugin_summary = read_json(plugin_summary_path)
        if isinstance(plugin_summary, Mapping):
            updated = dict(plugin_summary)
            updated["flaky"] = len(flaky)
            if not still_failed:
                updated["exitstatus"] = 0
            plugin_summary_path.write_text(json.dumps(updated), encoding="utf-8")
    return {
        "attempted": failed,
        "still_failed": still_failed,
        "flaky": flaky,
        # Reported relative to the checkout when it lies inside one, absolute
        # otherwise. A step directory can sit outside the root -- a configured
        # basetemp root, or a caller that supplies its own artifact directory --
        # and a rerun that adjudicated flakes correctly must not then die
        # formatting its own path.
        "rerun_report": str(_path_for_receipt(rerun_report, root=root)),
    }
