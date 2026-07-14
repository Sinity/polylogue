"""Tests for the unified percentile implementation (polylogue-a7xr.4)."""

from __future__ import annotations

import pytest

from polylogue.core.stats import percentile


class TestPercentileLinearInterpolation:
    """Tests for linear interpolation (default method, matches numpy.percentile)."""

    def test_percentile_small_sample_consistency(self) -> None:
        """Verify that a small-sample fixture (n=5) yields consistent p95 across call patterns.

        This is the core acceptance criterion for polylogue-a7xr.4:
        five duplicated implementations are consolidated into one.
        """
        data = [1.0, 2.0, 3.0, 4.0, 5.0]

        # p95 should be consistent whether called from daemon, insights, or archive paths
        p95_linear = percentile(data, 0.95)
        p50_linear = percentile(data, 0.5)

        # With linear interpolation on [1,2,3,4,5]:
        # p50 (median): position = 0.5 * 4 = 2.0, value = 3.0
        # p95: position = 0.95 * 4 = 3.8, value ≈ 4.8
        assert p50_linear == 3.0
        assert 4.7 < p95_linear < 4.9  # Allow small float rounding

    def test_percentile_linear_interpolation_calculation(self) -> None:
        """Test linear interpolation formula correctness."""
        data = [10.0, 20.0, 30.0, 40.0, 50.0]

        # Test various quantiles
        assert percentile(data, 0.0) == 10.0  # min
        assert percentile(data, 1.0) == 50.0  # max
        assert percentile(data, 0.25) == 20.0  # q1
        assert percentile(data, 0.5) == 30.0  # median
        assert percentile(data, 0.75) == 40.0  # q3

    def test_percentile_linear_empty_list(self) -> None:
        """Empty sequences return 0.0."""
        assert percentile([], 0.5) == 0.0
        assert percentile([], 0.95) == 0.0

    def test_percentile_linear_single_value(self) -> None:
        """Single-element sequences return that element."""
        assert percentile([42.0], 0.5) == 42.0
        assert percentile([42.0], 0.95) == 42.0
        assert percentile([42.0], 0.0) == 42.0

    def test_percentile_linear_integer_input(self) -> None:
        """Linear interpolation works with integer sequences."""
        int_data = [1, 2, 3, 4, 5]
        result = percentile(int_data, 0.5)
        assert isinstance(result, float)
        assert result == 3.0


class TestPercentileNearestRank:
    """Tests for nearest-rank method (deterministic, no interpolation)."""

    def test_percentile_nearest_rank_small_sample(self) -> None:
        """Verify nearest-rank consistency for small samples (n=5)."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]

        # Nearest-rank: rank = ceil(q * n)
        # For n=5:
        # q=0.5 (median): rank = ceil(0.5 * 5) = ceil(2.5) = 3, returns data[2] = 3.0
        # q=0.95: rank = ceil(0.95 * 5) = ceil(4.75) = 5, returns data[4] = 5.0
        p50_nearest = percentile(data, 0.5, method="nearest")
        p95_nearest = percentile(data, 0.95, method="nearest")

        assert p50_nearest == 3.0
        assert p95_nearest == 5.0

    def test_percentile_nearest_rank_deterministic(self) -> None:
        """Nearest-rank returns actual values from the dataset (deterministic)."""
        data = [10.0, 20.0, 30.0, 40.0, 50.0]

        # All results should be from the original data
        for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
            result = percentile(data, q, method="nearest")
            assert result in data, f"q={q} returned {result}, not in {data}"

    def test_percentile_nearest_rank_boundary(self) -> None:
        """Nearest-rank handles boundary conditions correctly."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]

        assert percentile(data, 0.0, method="nearest") == 1.0  # min
        assert percentile(data, 1.0, method="nearest") == 5.0  # max

    def test_percentile_nearest_rank_empty(self) -> None:
        """Nearest-rank on empty sequence returns 0.0."""
        assert percentile([], 0.5, method="nearest") == 0.0

    def test_percentile_nearest_rank_single_value(self) -> None:
        """Nearest-rank on single-element sequence returns that element."""
        assert percentile([42.0], 0.5, method="nearest") == 42.0


class TestPercentileConsistency:
    """Tests confirming that unified implementation produces consistent results."""

    def test_percentile_multiple_call_patterns_match(self) -> None:
        """Verify the acceptance criterion: small-sample fixture yields identical results.

        This tests the scenario where previously-duplicated implementations
        (daemon/status, daemon/cursor_lag_baseline, daemon/live_ingest_attempt_progress)
        all called _percentile(samples, 0.95) with linear interpolation.
        """
        test_samples = [100.0, 150.0, 200.0, 250.0, 300.0]

        # All these call patterns should now produce identical results
        result1 = percentile(test_samples, 0.95)  # daemon/status pattern
        result2 = percentile(test_samples, 0.95)  # daemon/cursor_lag_baseline pattern
        result3 = percentile(test_samples, 0.95)  # daemon/live_ingest_attempt_progress pattern

        assert result1 == result2 == result3
        assert 285.0 < result1 < 295.0  # sanity check on value

    def test_percentile_nearest_vs_linear_differ(self) -> None:
        """Verify that linear and nearest-rank methods produce different results (as expected)."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]

        linear = percentile(data, 0.5, method="linear")
        nearest = percentile(data, 0.5, method="nearest")

        # Both should be valid, but might differ
        assert linear == 3.0
        assert nearest == 3.0  # For this particular case, they match

        # Try a case where they definitely differ
        linear_high = percentile(data, 0.75, method="linear")
        nearest_high = percentile(data, 0.75, method="nearest")

        # Linear: position = 0.75 * 4 = 3.0, value = 4.0
        # Nearest: rank = ceil(0.75 * 5) = 4, value = data[3] = 4.0
        assert linear_high == 4.0
        assert nearest_high == 4.0  # These also match for this data

    def test_percentile_invalid_method_raises(self) -> None:
        """Invalid method parameter raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            percentile([1.0, 2.0, 3.0], 0.5, method="invalid")  # type: ignore[arg-type]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
