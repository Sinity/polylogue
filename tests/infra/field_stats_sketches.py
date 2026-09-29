"""Build FieldStats distribution sketches from literal observations for inference fixtures."""

from __future__ import annotations

from collections.abc import Iterable

from polylogue.schemas.field_stats.distributions import DistributionSketch


def sketch(values: Iterable[int | float]) -> DistributionSketch:
    """A distribution sketch that has observed exactly ``values``."""
    result = DistributionSketch()
    for value in values:
        result.observe(value)
    return result


__all__ = ["sketch"]
