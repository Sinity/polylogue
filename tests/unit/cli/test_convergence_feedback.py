from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.cli.convergence_feedback import convergence_warning_line


def test_convergence_warning_line_reports_actionable_raw_debt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("polylogue.paths.archive_root", lambda: Path("/archive"))
    monkeypatch.setattr(
        "polylogue.storage.archive_readiness.raw_materialization_readiness_snapshot",
        lambda _root: {
            "available": True,
            "actionable": 1,
            "blocked": 0,
            "affected_actionable": 4,
            "affected_blocked": 0,
            "affected_open": 0,
            "category_counts": {"parse_failed": 4},
        },
    )

    warning = convergence_warning_line()

    assert warning == (
        "Archive has raw materialization debt: 1 issue group(s); 4 parse-failed raw artifact(s); "
        "results may be partial for affected source artifacts."
    )


def test_convergence_warning_line_omits_classified_raw_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("polylogue.paths.archive_root", lambda: Path("/archive"))
    monkeypatch.setattr(
        "polylogue.storage.archive_readiness.raw_materialization_readiness_snapshot",
        lambda _root: {
            "available": True,
            "total": 1,
            "classified": 1,
            "affected_classified": 372,
            "actionable": 0,
            "blocked": 0,
            "affected_actionable": 0,
            "affected_blocked": 0,
            "affected_open": 0,
        },
    )

    assert convergence_warning_line() is None


def test_convergence_warning_line_reports_unclassified_join_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("polylogue.paths.archive_root", lambda: Path("/archive"))
    monkeypatch.setattr(
        "polylogue.storage.archive_readiness.raw_materialization_readiness_snapshot",
        lambda _root: {
            "available": True,
            "classification": "not_run",
            "raw_artifact_count": 10,
            "materialized_raw_artifact_count": 7,
            "unchecked": 3,
            "affected_unchecked": 3,
            "actionable": 0,
            "blocked": 0,
            "affected_actionable": 0,
            "affected_blocked": 0,
            "affected_open": 0,
        },
    )

    warning = convergence_warning_line()

    assert warning == (
        "Archive materialization needs classification: 7/10 raw artifact(s) materialized; "
        "3 raw/index join gap(s) found; "
        "results may be partial until daemon convergence classifies them."
    )


def test_convergence_warning_line_reports_undetermined_when_probe_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unanswerable readiness check must not render as a healthy archive.

    ``None`` means "checked, and results are complete". A probe failure that
    returned ``None`` would present partial results as complete.
    """

    def _raise(_root: Path) -> dict[str, object]:
        raise sqlite3.OperationalError("no such table: raw_materialization_status")

    monkeypatch.setattr("polylogue.paths.archive_root", lambda: Path("/archive"))
    monkeypatch.setattr("polylogue.storage.archive_readiness.raw_materialization_readiness_snapshot", _raise)

    warning = convergence_warning_line()

    assert warning == "Archive convergence state could not be determined; results may be partial."
