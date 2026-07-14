from __future__ import annotations

import pytest

from polylogue.insights.measurement.uncertainty import (
    UncertaintyRefusalError,
    bootstrap_interval,
    resolve_uncertainty,
)


def test_exact_complete_structural_carries_no_uncertainty_at_all() -> None:
    rendering = resolve_uncertainty(enumeration="exact", point_estimate=42.0, n=42)

    assert rendering.sampling_interval is None
    assert rendering.frame_note is None
    assert rendering.measurement_note is None
    assert rendering.uncertainty_sources == ()


def test_exact_frame_incomplete_retains_coverage_note_without_sampling_ci() -> None:
    rendering = resolve_uncertainty(
        enumeration="exact",
        point_estimate=42.0,
        n=42,
        frame_complete=False,
        frame_note="12% of sessions never captured",
    )

    assert rendering.sampling_interval is None
    assert rendering.frame_note == "12% of sessions never captured"
    assert rendering.measurement_note is None
    assert rendering.uncertainty_sources == ("frame",)


def test_exact_model_derived_retains_measurement_note_without_sampling_ci() -> None:
    rendering = resolve_uncertainty(
        enumeration="exact",
        point_estimate=0.87,
        n=500,
        measurement_authority="model-derived",
    )

    assert rendering.sampling_interval is None
    assert rendering.frame_note is None
    assert rendering.measurement_note == "measurement authority: model-derived"
    assert rendering.uncertainty_sources == ("measurement",)


def test_sampled_result_receives_a_named_sampling_interval() -> None:
    samples = [1.0, 2.0, 2.0, 3.0, 4.0, 2.5, 3.5, 1.5, 2.0, 3.0]

    rendering = resolve_uncertainty(
        enumeration="sampled",
        point_estimate=2.45,
        n=len(samples),
        samples=samples,
    )

    assert rendering.sampling_interval is not None
    lo, hi = rendering.sampling_interval
    assert lo <= hi
    assert rendering.sampling_method == "bootstrap-percentile"
    assert rendering.uncertainty_sources == ("sampling",)


def test_bootstrap_request_over_exact_enumeration_is_refused() -> None:
    """A caller may not paper over frame-incompleteness or parser
    disagreement by asking for a bootstrap CI on an exact count."""

    with pytest.raises(UncertaintyRefusalError, match="carries no sampling error"):
        resolve_uncertainty(
            enumeration="exact",
            point_estimate=42.0,
            n=42,
            samples=[1.0, 2.0, 3.0],
        )


def test_bootstrap_request_over_capped_enumeration_is_also_refused() -> None:
    with pytest.raises(UncertaintyRefusalError):
        resolve_uncertainty(enumeration="capped", point_estimate=10.0, n=10, samples=[1.0, 2.0])


def test_sampled_enumeration_without_samples_is_refused() -> None:
    with pytest.raises(UncertaintyRefusalError, match=">=2 sample members"):
        resolve_uncertainty(enumeration="sampled", point_estimate=1.0, n=1, samples=None)


def test_sampled_enumeration_with_single_sample_is_refused() -> None:
    with pytest.raises(UncertaintyRefusalError):
        resolve_uncertainty(enumeration="estimate", point_estimate=1.0, n=1, samples=[1.0])


def test_bootstrap_interval_is_deterministic_for_a_fixed_seed() -> None:
    samples = [1.0, 5.0, 3.0, 2.0, 8.0, 4.0]

    first = bootstrap_interval(samples, seed=7)
    second = bootstrap_interval(samples, seed=7)

    assert first == second


def test_bootstrap_interval_requires_at_least_two_samples() -> None:
    with pytest.raises(ValueError, match="at least two"):
        bootstrap_interval([1.0])


def test_bootstrap_interval_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_interval([1.0, 2.0, 3.0], confidence=1.5)
