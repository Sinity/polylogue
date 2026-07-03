from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.cli.convergence_feedback import convergence_warning_line


def test_convergence_warning_line_prefers_active_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("polylogue.paths.archive_root", lambda: Path("/archive"))
    monkeypatch.setattr(
        "polylogue.storage.archive_readiness.active_rebuild_index_attempts",
        lambda _ops_db: [{"parsed_raw_count": 12, "materialized_count": 3}],
    )

    warning = convergence_warning_line()

    assert warning == (
        "Archive is converging: 1 index rebuild attempt(s) active "
        "(3 sessions materialized from 12 parsed raw rows); results may be partial."
    )


def test_convergence_warning_line_reports_actionable_raw_debt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("polylogue.paths.archive_root", lambda: Path("/archive"))
    monkeypatch.setattr("polylogue.storage.archive_readiness.active_rebuild_index_attempts", lambda _ops_db: [])
    monkeypatch.setattr(
        "polylogue.storage.archive_readiness.raw_materialization_readiness_snapshot",
        lambda _root: {
            "available": True,
            "actionable": 1,
            "blocked": 0,
            "affected_actionable": 4,
            "affected_blocked": 0,
            "affected_open": 0,
        },
    )

    warning = convergence_warning_line()

    assert warning == "Archive is converging: 4 raw artifact(s) are not materialized; results may be partial."


def test_convergence_warning_line_omits_classified_raw_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("polylogue.paths.archive_root", lambda: Path("/archive"))
    monkeypatch.setattr("polylogue.storage.archive_readiness.active_rebuild_index_attempts", lambda _ops_db: [])
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
