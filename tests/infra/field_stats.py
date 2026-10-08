"""Helpers for constructing explicit FieldStats distribution summaries."""

from __future__ import annotations

from collections.abc import Iterable

from polylogue.schemas.field_stats.distributions import DistributionSketch


def distribution_sketch(values: Iterable[int | float]) -> DistributionSketch:
    """Build the current summary representation from concise test observations."""
    sketch = DistributionSketch()
    for value in values:
        sketch.observe(value)
    return sketch
