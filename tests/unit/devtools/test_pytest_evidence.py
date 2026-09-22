from __future__ import annotations

from pathlib import Path

import pytest

from devtools.pytest_evidence import evaluate_pytest_evidence


def _evidence(
    tmp_path: Path,
    *,
    selected_count: int = 1,
    tests: list[dict[str, object]] | None = None,
    events: list[dict[str, object]] | None = None,
    exit_code: int = 0,
) -> dict[str, object]:
    report = {"tests": tests if tests is not None else [{"nodeid": "tests/test_ok.py::test_ok", "outcome": "passed"}]}
    selection = {"selected_count": selected_count}
    summary = {"exitstatus": exit_code, "selected_count": selected_count}
    return evaluate_pytest_evidence(
        report=report,
        selection=selection,
        summary=summary,
        events=events if events is not None else [{"event": "collection_finished", "selected_count": selected_count}],
        exit_code=exit_code,
    )


def test_clean_selection_is_evidence_bearing() -> None:
    result = _evidence(Path("."))

    assert result["ok"] is True
    assert result["diagnosis"] == "pytest_passed"
    assert result["selected_count"] == 1
    assert result["terminal_count"] == 1


@pytest.mark.parametrize(
    ("kwargs", "diagnosis"),
    [
        ({"report": None}, "pytest_no_report"),
        ({"selected_count": 0}, "pytest_no_tests_selected"),
        (
            {"tests": [{"nodeid": "tests/test_ok.py::test_ok", "outcome": "passed"}], "selected_count": 2},
            "pytest_report_incomplete",
        ),
        ({"events": []}, "pytest_collection_incomplete"),
    ],
)
def test_exit_zero_does_not_override_missing_pytest_evidence(
    tmp_path: Path, kwargs: dict[str, object], diagnosis: str
) -> None:
    if kwargs.get("report", object()) is None:
        result = evaluate_pytest_evidence(
            report=None,
            selection={"selected_count": 1},
            summary={"exitstatus": 0},
            events=[{"event": "collection_finished", "selected_count": 1}],
            exit_code=0,
        )
    else:
        result = _evidence(tmp_path, **kwargs)  # type: ignore[arg-type]

    assert result["ok"] is False
    assert result["diagnosis"] == diagnosis


def test_worker_loss_is_typed() -> None:
    result = _evidence(Path("."), exit_code=3)

    assert result["ok"] is False
    assert result["diagnosis"] == "pytest_worker_loss"


def test_missing_report_mutation_probe_is_red() -> None:
    result = _evidence(Path("."))
    mutated = evaluate_pytest_evidence(
        report=None,
        selection={"selected_count": result["selected_count"]},
        summary={"exitstatus": 0},
        events=[{"event": "collection_finished", "selected_count": 1}],
        exit_code=0,
    )

    assert mutated["ok"] is False


def test_positive_execution_mutation_probe_is_red() -> None:
    mutated = _evidence(Path("."), selected_count=0, tests=[])

    assert mutated["ok"] is False


def test_collection_only_is_explicit_success_before_terminal_completeness() -> None:
    result = evaluate_pytest_evidence(
        report={"tests": []},
        selection={"selected_count": 2},
        summary={"exitstatus": 0},
        events=[{"event": "collection_finished", "selected_count": 2}],
        exit_code=0,
        collection_only=True,
    )

    assert result["ok"] is True
    assert result["ordinary_eligible"] is False
    assert result["diagnosis"] == "pytest_collection_only"


def test_maxfail_is_an_ordinary_pytest_failure_not_incomplete_evidence() -> None:
    result = _evidence(
        Path("."),
        selected_count=2,
        tests=[{"nodeid": "test_a", "outcome": "failed"}],
        exit_code=1,
    )

    assert result["ok"] is False
    assert result["diagnosis"] == "pytest_failed"


def test_unpublished_terminal_summary_is_refused() -> None:
    """Exit zero plus every earlier artifact is not a terminal verdict.

    ``pytest_progress_plugin._write_summary`` writes ``summary.json`` from
    ``pytest_sessionfinish`` under ``contextlib.suppress(OSError)``: a full
    verification volume loses the terminal artifact while pytest still exits
    zero with its report, selection and collection event already on disk.

    Anti-vacuity: restoring ``summary.get("exitstatus") not in (None,
    exit_code)`` accepts the absent summary and this test reports
    ``pytest_passed``, which is the defect.
    """
    result = evaluate_pytest_evidence(
        report={"tests": [{"nodeid": "tests/test_ok.py::test_ok", "outcome": "passed"}]},
        selection={"selected_count": 1},
        summary=None,
        events=[{"event": "collection_finished", "selected_count": 1}],
        exit_code=0,
    )

    assert result["ok"] is False
    assert result["ordinary_eligible"] is False
    assert result["diagnosis"] == "pytest_summary_missing"
    assert result["summary_status"] == "missing"


def test_summary_without_exitstatus_is_inconsistent() -> None:
    """A summary present but silent about its exit is not agreement either."""
    result = evaluate_pytest_evidence(
        report={"tests": [{"nodeid": "tests/test_ok.py::test_ok", "outcome": "passed"}]},
        selection={"selected_count": 1},
        summary={"selected_count": 1},
        events=[{"event": "collection_finished", "selected_count": 1}],
        exit_code=0,
    )

    assert result["ok"] is False
    assert result["diagnosis"] == "pytest_summary_inconsistent"
    assert result["summary_status"] == "present"


def test_agreeing_summary_still_passes() -> None:
    """The opposite direction: the tightened check must not refuse a real run."""
    result = _evidence(Path("."))

    assert result["diagnosis"] == "pytest_passed"
    assert result["summary_status"] == "present"
