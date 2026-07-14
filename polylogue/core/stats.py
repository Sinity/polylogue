"""Statistical utilities for archive analysis."""

import math
from collections.abc import Sequence
from typing import Literal, TypeVar

T = TypeVar("T", float, int)


def percentile(
    sorted_values: Sequence[T],
    q: float,
    *,
    method: Literal["linear", "nearest"] = "linear",
) -> float:
    """Compute a percentile from a pre-sorted sequence of values.

    Parameters
    ----------
    sorted_values : Sequence[float | int]
        Pre-sorted (ascending) sequence of numeric values. Must be sorted.
    q : float
        Quantile in [0, 1]. For example, q=0.5 is the median, q=0.95 is the
        95th percentile.
    method : {"linear", "nearest"}, optional
        Interpolation method (default: "linear").
        - "linear": Linear interpolation (numpy.percentile's default).
          Matches externally-collected metrics without impedance mismatch.
        - "nearest": Nearest-rank method (deterministic, does not interpolate).

    Returns
    -------
    float
        The q-th percentile. Returns 0.0 for an empty sequence.

    Notes
    -----
    - Linear interpolation matches numpy.percentile(..., method='linear'),
      allowing baseline comparisons with external metrics.
    - Nearest-rank is useful for discrete distributions where interpolation
      is not meaningful.
    - Empty sequences return 0.0; single-element sequences return that element.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    if method == "linear":
        return _percentile_linear(sorted_values, q)
    elif method == "nearest":
        return _percentile_nearest(sorted_values, q)
    else:
        raise ValueError(f"Unknown method: {method}")


def _percentile_linear(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (numpy.percentile equivalent)."""
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])

    position = q * (len(sorted_values) - 1)
    lo = int(position)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = position - lo
    return float(sorted_values[lo]) * (1.0 - frac) + float(sorted_values[hi]) * frac


def _percentile_nearest(sorted_values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (deterministic, no interpolation)."""
    # Nearest-rank formula: rank = ceil(q * n), 1-indexed
    rank = max(1, math.ceil(q * len(sorted_values)))
    return float(sorted_values[min(rank, len(sorted_values)) - 1])


__all__ = ["percentile"]
