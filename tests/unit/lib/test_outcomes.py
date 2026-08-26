"""Outcome report contract tests."""

from __future__ import annotations

from polylogue.core.outcomes import (
    BoundOutcomeOwner,
    OutcomeCheck,
    OutcomeCompositionFailure,
    OutcomeCompositionFailureKind,
    OutcomeReport,
    OutcomeStatus,
    compose_outcome_checks,
)


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


def test_composition_keeps_siblings_when_owner_is_omitted_or_raises() -> None:
    """Anti-vacuity: removing exception isolation or sibling iteration fails."""

    def _raising_check() -> OutcomeCheck:
        raise ZeroDivisionError("owner blew up")

    report = compose_outcome_checks(
        (
            BoundOutcomeOwner(name="omitted", check=None),
            BoundOutcomeOwner(name="raises", check=_raising_check),
            BoundOutcomeOwner(
                name="survivor",
                check=lambda: OutcomeCheck(name="survivor", status=OutcomeStatus.OK),
            ),
        )
    )

    assert [check.name for check in report.checks] == ["omitted", "raises", "survivor"]
    assert isinstance(report.checks[0], OutcomeCompositionFailure)
    assert report.checks[0].failure_kind is OutcomeCompositionFailureKind.OWNER_OMITTED
    assert isinstance(report.checks[1], OutcomeCompositionFailure)
    assert report.checks[1].failure_kind is OutcomeCompositionFailureKind.OWNER_RAISED
    assert report.checks[2].status is OutcomeStatus.OK
