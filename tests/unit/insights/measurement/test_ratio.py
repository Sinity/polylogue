from __future__ import annotations

import pytest

from polylogue.insights.measurement.metric import MetricDefinition
from polylogue.insights.measurement.ratio import (
    IncompatibleRatioError,
    derive_ratio_metric,
    evaluate_ratio,
)


def _metric(**overrides: object) -> MetricDefinition:
    fields: dict[str, object] = {
        "construct": "silent-proceed",
        "unit": "count",
        "unit_source": "tool_calls",
        "aggregation": "count",
    }
    fields.update(overrides)
    return MetricDefinition(**fields)  # type: ignore[arg-type]


def test_derive_ratio_metric_cites_both_component_refs() -> None:
    numerator = _metric(construct="silent-proceed-count")
    denominator = _metric(construct="tool-call-count")

    ratio = derive_ratio_metric(numerator, denominator, construct="silent_proceed_rate")

    assert ratio.formula_kind == "ratio"
    assert ratio.component_refs == (numerator.ref, denominator.ref)


def test_derive_ratio_metric_refuses_incompatible_grain() -> None:
    numerator = _metric(grain="logical")
    denominator = _metric(grain="physical")

    with pytest.raises(IncompatibleRatioError):
        derive_ratio_metric(numerator, denominator, construct="mismatched")


def test_derive_ratio_metric_refuses_incompatible_measurement_authority() -> None:
    numerator = _metric(measurement_authority=("provider-reported",))
    denominator = _metric(measurement_authority=("heuristic",))

    with pytest.raises(IncompatibleRatioError):
        derive_ratio_metric(numerator, denominator, construct="mismatched")


def test_derive_ratio_metric_two_independently_constructed_pairs_collapse_to_same_ref() -> None:
    """rxdo.9.2: ratios are DERIVED objects, not a second registry -- two call
    sites describing the same numerator/denominator pair must resolve to the
    same metric:<hash>, not mint two competing ratio identities."""

    numerator = _metric(construct="a")
    denominator = _metric(construct="b")

    first = derive_ratio_metric(numerator, denominator, construct="a_over_b")
    second = derive_ratio_metric(_metric(construct="a"), _metric(construct="b"), construct="a_over_b")

    assert first.ref == second.ref


def test_evaluate_ratio_rejects_non_ratio_metric() -> None:
    plain = _metric()

    with pytest.raises(IncompatibleRatioError):
        evaluate_ratio(plain, numerator_value=1, numerator_n=1, denominator_value=2, denominator_n=2)


def test_evaluate_ratio_preserves_unknown_bucket_regardless_of_null_policy() -> None:
    ratio_metric = derive_ratio_metric(_metric(construct="a"), _metric(construct="b"), construct="a_over_b")

    result = evaluate_ratio(
        ratio_metric,
        numerator_value=3,
        numerator_n=3,
        denominator_value=10,
        denominator_n=10,
        unknown_n=2,
    )

    assert result.unknown_n == 2
    assert "unknown=2" in result.render()


def test_evaluate_ratio_suppress_policy_never_renders_a_bare_percentage_with_unknowns() -> None:
    ratio_metric = derive_ratio_metric(
        _metric(construct="a"), _metric(construct="b"), construct="a_over_b", null_policy="suppress"
    )

    result = evaluate_ratio(
        ratio_metric,
        numerator_value=3,
        numerator_n=3,
        denominator_value=10,
        denominator_n=10,
        unknown_n=1,
    )

    assert result.is_suppressed is True
    assert result.value is None
    assert "suppressed" in result.render()


def test_evaluate_ratio_zero_denominator_never_renders_a_number() -> None:
    ratio_metric = derive_ratio_metric(_metric(construct="a"), _metric(construct="b"), construct="a_over_b")

    result = evaluate_ratio(ratio_metric, numerator_value=0, numerator_n=0, denominator_value=0, denominator_n=0)

    assert result.value is None
    assert "n/a" in result.render()
