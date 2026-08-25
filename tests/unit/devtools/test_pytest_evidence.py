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


@pytest.mark.parametrize(
    ("termination_reason", "diagnosis"),
    [("stall", "pytest_stall_terminated"), ("sigterm", "pytest_interrupted")],
)
def test_terminal_interruption_is_never_green(termination_reason: str, diagnosis: str) -> None:
    result = evaluate_pytest_evidence(
        report={"tests": [{"nodeid": "test_ok", "outcome": "passed"}]},
        selection={"selected_count": 1},
        summary={"exitstatus": 0},
        events=[{"event": "collection_finished", "selected_count": 1}],
        exit_code=0,
        termination_reason=termination_reason,
    )

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
