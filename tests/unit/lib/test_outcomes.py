"""Outcome report contract tests."""

from __future__ import annotations

from polylogue.core.outcomes import OutcomeCheck, OutcomeReport, OutcomeStatus


def test_outcome_report_counts_are_typed_and_json_ready() -> None:
    report = OutcomeReport(
        checks=[
            OutcomeCheck(name="ok", status=OutcomeStatus.OK),
            OutcomeCheck(name="warn", status=OutcomeStatus.WARNING),
            OutcomeCheck(name="error", status=OutcomeStatus.ERROR),
            OutcomeCheck(name="skip", status=OutcomeStatus.SKIP),
        ]
    )

    counts = report.counts()

    assert counts.ok == 1
    assert counts.warning == 1
    assert counts.error == 1
    assert counts.skip == 1
    assert counts.to_json(include_skip=True) == {
        "ok": 1,
        "warning": 1,
        "error": 1,
        "skip": 1,
    }
    assert report.summary_counts(include_skip=True) == {
        "ok": 1,
        "warning": 1,
        "error": 1,
        "skip": 1,
    }
