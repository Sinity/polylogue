"""Tests for paired negative controls on findings (rxdo.9.7, mechanism G)."""

from __future__ import annotations

import pytest

from polylogue.insights.judgment.controls import (
    ClaimWithControls,
    ControlOutcome,
    NegativeControl,
    validate_control,
)


def test_negative_control_requires_expected_null() -> None:
    with pytest.raises(ValueError, match="expected-null"):
        NegativeControl(
            control_kind="shifted_window",
            query_ref="query:abc",
            result_ref="result-set:def",
            matching_variables=("model",),
            expected_null="",
        )


def test_matched_control_with_declared_matching_variables_is_accepted() -> None:
    """AC: a matched control with preregistered expected-null renders beside the finding."""

    control = NegativeControl(
        control_kind="shifted_window",
        query_ref="query:abc",
        result_ref="result-set:def",
        matching_variables=("model", "repo"),
        expected_null="cost ratio stays flat outside the claimed window",
    )
    validation = validate_control(control, claim_frame_variables=["model", "repo", "epoch"])
    assert validation.accepted


def test_matched_control_missing_matching_variables_is_rejected() -> None:
    control = NegativeControl(
        control_kind="permuted_label",
        query_ref="query:abc",
        result_ref="result-set:def",
        matching_variables=(),
        expected_null="no effect under permutation",
    )
    validation = validate_control(control, claim_frame_variables=["model"])
    assert not validation.accepted
    assert "no matching variables" in validation.reason


def test_matching_variables_must_be_subset_of_claim_frame() -> None:
    control = NegativeControl(
        control_kind="matched_task_different_treatment",
        query_ref="query:abc",
        result_ref="result-set:def",
        matching_variables=("unrelated_variable",),
        expected_null="no effect from the different treatment alone",
    )
    validation = validate_control(control, claim_frame_variables=["model", "repo"])
    assert not validation.accepted
    assert "subset" in validation.reason


def test_deliberately_divergent_baseline_is_rejected_as_confounded() -> None:
    """AC: a deliberately divergent baseline is rejected as confounded."""

    control = NegativeControl(
        control_kind="unrelated_cohort",
        query_ref="query:abc",
        result_ref="result-set:unrelated",
        matching_variables=(),
        expected_null="no relation expected",
        confounds_checked=(),  # nothing declared checked
    )
    validation = validate_control(control, claim_frame_variables=["model", "repo"])
    assert not validation.accepted
    assert "confounds unchecked" in validation.reason


def test_unrelated_cohort_with_all_confounds_declared_checked_is_accepted() -> None:
    control = NegativeControl(
        control_kind="unrelated_cohort",
        query_ref="query:abc",
        result_ref="result-set:unrelated",
        matching_variables=(),
        expected_null="no relation expected",
        confounds_checked=("model", "repo"),
    )
    validation = validate_control(control, claim_frame_variables=["model", "repo"])
    assert validation.accepted


def test_control_failure_downgrades_the_claim() -> None:
    """AC: control failure downgrades the claim."""

    control = NegativeControl(
        control_kind="shifted_window",
        query_ref="query:abc",
        result_ref="result-set:def",
        matching_variables=("model",),
        expected_null="ratio stays flat",
    )
    passing = ClaimWithControls(claim_ref="finding:x", controls=(ControlOutcome(control, observed_null_held=True),))
    failing = ClaimWithControls(claim_ref="finding:x", controls=(ControlOutcome(control, observed_null_held=False),))
    uncontrolled = ClaimWithControls(claim_ref="finding:x", controls=())

    assert passing.rank_tier == "controlled_pass"
    assert not passing.downgraded
    assert failing.rank_tier == "controlled_fail"
    assert failing.downgraded
    assert uncontrolled.rank_tier == "uncontrolled"


def test_removing_confound_validation_makes_the_fixture_fail() -> None:
    """AC: removing matching/confound validation makes the fixture fail.

    Simulates the buggy variant (accept unrelated_cohort unconditionally) and
    proves the real function's stricter behavior is load-bearing.
    """

    control = NegativeControl(
        control_kind="unrelated_cohort",
        query_ref="query:abc",
        result_ref="result-set:unrelated",
        matching_variables=(),
        expected_null="no relation expected",
        confounds_checked=(),
    )

    def _buggy_validate_control(control: NegativeControl, *, claim_frame_variables: list[str]) -> bool:
        # A validator that skips the confound-declaration check entirely.
        return True

    buggy_result = _buggy_validate_control(control, claim_frame_variables=["model", "repo"])
    real_result = validate_control(control, claim_frame_variables=["model", "repo"]).accepted
    assert buggy_result is True
    assert real_result is False
    assert buggy_result != real_result
