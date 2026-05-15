from __future__ import annotations

import subprocess
import sys
from unittest.mock import patch

from devtools.verify import _format_completion_notification, _parse_pytest_test_count, _run, build_verify_steps


def test_quick_verify_omits_pytest() -> None:
    steps = build_verify_steps(quick=True, lab=False, skip_slow=False)

    labels = [label for label, _command in steps]
    assert labels == [
        "ruff format",
        "ruff check",
        "mypy",
        "render-all",
        "verify-topology",
        "verify-layering",
        "verify-file-budgets",
        "verify-test-ownership",
        "verify-schema-roundtrip",
        "verify-cross-cuts",
        "verify-suppressions",
        "verify-manifests",
        "verify-witness-lifecycle",
        "verify-lane-assertions",
        "proof-pack check",
    ]


def test_full_verify_includes_pytest() -> None:
    steps = build_verify_steps(quick=False, lab=False, skip_slow=False)

    labels = [label for label, _command in steps]
    assert labels[-1] == "pytest"


def test_lab_verify_delegates_to_lab_scenario() -> None:
    steps = build_verify_steps(quick=True, lab=True, skip_slow=False)

    labels = [label for label, _command in steps]
    assert "lab scenario" in labels
    assert "verify-slos" in labels
    lab_step = next(step for step in steps if step[0] == "lab scenario")
    assert lab_step == (
        "lab scenario",
        [sys.executable, "-m", "devtools", "lab-scenario", "run", "archive-smoke", "--tier", "0"],
    )


def test_parse_pytest_test_count_from_summary() -> None:
    output = "bringing up nodes...\n\n6 passed, 2 skipped, 1 xfailed in 8.49s\n"

    assert _parse_pytest_test_count(output) == 9


def test_parse_pytest_test_count_handles_no_tests() -> None:
    assert _parse_pytest_test_count("no tests ran in 0.02s\n") == 0


def test_run_records_pytest_count_metadata() -> None:
    completed = subprocess.CompletedProcess(
        args=["pytest"],
        returncode=0,
        stdout="....\n4 passed in 1.23s\n",
        stderr="",
    )

    with patch("devtools.verify.subprocess.run", return_value=completed):
        rc, _elapsed, metadata = _run("pytest affected", ["pytest"])

    assert rc == 0
    assert metadata == {"count": 4}


def test_completion_notification_uses_pytest_count() -> None:
    summary = _format_completion_notification(
        exit_code=0,
        total_duration=118.2,
        step_results=[
            {"name": "ruff check", "duration_s": 0.1, "exit": 0},
            {"name": "pytest affected", "duration_s": 100.0, "exit": 0, "count": 12},
        ],
    )

    assert summary == "PASS (118s), 12 tests"


def test_completion_notification_omits_unknown_pytest_count() -> None:
    summary = _format_completion_notification(
        exit_code=0,
        total_duration=118.2,
        step_results=[{"name": "pytest", "duration_s": 100.0, "exit": 0}],
    )

    assert summary == "PASS (118s)"
