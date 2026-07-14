"""Statistical functions for computing aggregates over data collections.

Provides deterministic, operator-comparable percentile computation across
all subsystems (daemon, insights, archive) with support for multiple interpolation
methods.
"""

from __future__ import annotations

from typing import Literal, Sequence


def percentile(
    sorted_values: Sequence[float],
    q: float,
    *,
    method: Literal["linear", "nearest"] = "linear",
) -> float:
    """Compute a percentile over a pre-sorted sequence of values.

    Args:
        sorted_values: A pre-sorted sequence of numeric values. Must be in
            ascending order for correct results.
        q: Quantile in [0, 1]. For example, q=0.5 is the median, q=0.95 is
            the 95th percentile.
        method: Interpolation method. "linear" (default, numpy.percentile
            compatible) uses linear interpolation between adjacent values.
            "nearest" uses nearest-rank selection (deterministic, no
            interpolation).

    Returns:
        The percentile value. Returns 0.0 for empty input. Returns the first
        element for q <= 0, the last element for q >= 1.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])

    if method == "linear":
        return _percentile_linear(sorted_values, q)
    elif method == "nearest":
        return _percentile_nearest(sorted_values, q)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def _percentile_linear(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (numpy.percentile compatible).

    Uses linear interpolation between adjacent values, suitable for smooth
    distributions and latency metrics. Matches numpy.percentile(..., method='linear')
    and provides comparability with externally-collected metrics without
    impedance mismatch.
    """
    position = q * (len(sorted_values) - 1)
    lo = int(position)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = position - lo
    return float(sorted_values[lo]) * (1.0 - frac) + float(sorted_values[hi]) * frac


def _percentile_nearest(sorted_values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (deterministic, no interpolation).

    Selects the nearest rank without interpolation, providing a deterministic
    value from the actual dataset. Uses rank = ceil(q * n), matching common
    statistical definitions.
    """
    rank = max(1, int((q * len(sorted_values) + 0.5)))  # round nearest
    return float(sorted_values[min(rank, len(sorted_values)) - 1])


__all__ = [
    "percentile",
]
