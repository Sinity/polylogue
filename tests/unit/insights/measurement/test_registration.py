from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from polylogue.insights.measurement.registration import (
    PreRegistration,
    RegistrationEvaluation,
    registration_status,
    render_badge,
)

_T0 = datetime(2026, 7, 1, tzinfo=timezone.utc)


def _registration(**overrides: object) -> PreRegistration:
    fields: dict[str, object] = {
        "registration_id": "reg-1",
        "hypothesis": "silent_proceed_rate drops after the retry-cap change",
        "expected": "rate < 0.05",
        "metric_ref": "metric:" + "a" * 64,
        "query_ref": "query:" + "b" * 64,
        "registered_at": _T0,
        "registered_epoch": 100,
    }
    fields.update(overrides)
    return PreRegistration(**fields)  # type: ignore[arg-type]


def _evaluation(**overrides: object) -> RegistrationEvaluation:
    fields: dict[str, object] = {
        "registration_id": "reg-1",
        "metric_ref": "metric:" + "a" * 64,
        "query_ref": "query:" + "b" * 64,
        "run_at": _T0 + timedelta(days=1),
        "run_epoch": 101,
        "actual": "rate = 0.03",
    }
    fields.update(overrides)
    return RegistrationEvaluation(**fields)  # type: ignore[arg-type]


def test_registered_when_run_strictly_after_timestamp_and_epoch() -> None:
    status = registration_status(_registration(), _evaluation())

    assert status == "registered"
    assert render_badge(status) == "confirmed (pre-registered)"


def test_exploratory_when_timestamps_reversed() -> None:
    status = registration_status(_registration(), _evaluation(run_at=_T0 - timedelta(hours=1)))

    assert status == "exploratory"
    assert render_badge(status) == "exploratory"


def test_exploratory_when_run_at_equals_registered_at() -> None:
    status = registration_status(_registration(), _evaluation(run_at=_T0))

    assert status == "exploratory"


def test_exploratory_post_hoc_when_data_already_observed_at_registration() -> None:
    status = registration_status(_registration(registered_epoch=200), _evaluation(run_epoch=150))

    assert status == "exploratory-post-hoc"
    assert "already observed" in render_badge(status)


def test_exploratory_definition_drift_when_metric_ref_changes_after_registration() -> None:
    status = registration_status(_registration(), _evaluation(metric_ref="metric:" + "c" * 64))

    assert status == "exploratory-definition-drift"
    assert "changed" in render_badge(status)


def test_exploratory_definition_drift_when_query_ref_changes_after_registration() -> None:
    status = registration_status(_registration(), _evaluation(query_ref="query:" + "d" * 64))

    assert status == "exploratory-definition-drift"


def test_evaluation_must_reference_the_registration_it_claims_to_evaluate() -> None:
    with pytest.raises(ValueError, match="does not reference"):
        registration_status(_registration(), _evaluation(registration_id="reg-other"))


def test_definition_drift_is_checked_before_timestamp_ordering() -> None:
    """A late-arriving, definition-swapped evaluation must not misreport as
    plain 'exploratory' (which would imply only the timing failed) -- the
    more specific drift reason must win."""

    status = registration_status(
        _registration(),
        _evaluation(metric_ref="metric:" + "c" * 64, run_at=_T0 - timedelta(hours=1)),
    )

    assert status == "exploratory-definition-drift"
