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
            "raw_artifact_count": 4,
            "raw_authority_parser_census": {"available": True},
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
            "raw_artifact_count": 1,
            "raw_authority_parser_census": {"available": True},
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
            "raw_authority_parser_census": {"available": True},
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


def test_convergence_warning_line_reports_zero_denominator_as_undetermined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("polylogue.paths.archive_root", lambda: Path("/archive"))
    monkeypatch.setattr(
        "polylogue.storage.archive_readiness.raw_materialization_readiness_snapshot",
        lambda _root: {
            "available": True,
            "raw_artifact_count": 0,
            "materialized_raw_artifact_count": 0,
            "raw_authority_parser_census": {"available": True},
        },
    )

    assert convergence_warning_line() == (
        "Archive convergence state could not be determined; results may be partial. "
        "(no raw artifacts: raw materialization is undefined at a zero denominator, not converged)"
    )


def test_convergence_warning_line_reports_invalid_counter_as_undetermined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("polylogue.paths.archive_root", lambda: Path("/archive"))
    monkeypatch.setattr(
        "polylogue.storage.archive_readiness.raw_materialization_readiness_snapshot",
        lambda _root: {
            "available": True,
            "raw_artifact_count": 1,
            "materialized_raw_artifact_count": 1,
            "raw_authority_parser_census": {"available": True},
            "critical": "not-an-int",
        },
    )

    warning = convergence_warning_line()

    assert warning is not None
    assert warning.startswith("Archive convergence state could not be determined; results may be partial.")
    assert "critical" in warning


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


def test_convergence_warning_line_reports_undetermined_when_snapshot_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ``available: False`` snapshot renders the caveat, carrying its own reason.

    The probe signals "I could not check" by RETURNING ``{"available": False,
    "error": ...}``, not by raising -- a missing source/index tier, an
    unreadable ops tier, a schema drift. The existing ``except`` arm therefore
    never fires for it, and ``_raw_materialization_warning`` returned ``None``,
    which the function's own contract defines as "checked, and results are
    complete". Every caller then printed nothing and presented a possibly
    incomplete page as a complete one, discarding the snapshot's reason.

    Anti-vacuity: restore the bare ``return None`` for an unavailable snapshot
    and ``warning`` is ``None``, so both assertions go red.
    """
    monkeypatch.setattr("polylogue.paths.archive_root", lambda: Path("/archive"))
    monkeypatch.setattr(
        "polylogue.storage.archive_readiness.raw_materialization_readiness_snapshot",
        lambda _root: {"available": False, "error": "source.db or index.db missing"},
    )

    warning = convergence_warning_line()

    assert warning is not None
    assert warning == (
        "Archive convergence state could not be determined; results may be partial. (source.db or index.db missing)"
    )
