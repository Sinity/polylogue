from __future__ import annotations

from dataclasses import replace

import pytest

from polylogue.analysis.measurement.metric import MetricDefinition
from polylogue.analysis.measurement.registered_measures import DEFAULT_MEASURE_REGISTRY
from polylogue.analysis.measurement.registry import (
    MeasureRegistry,
    MeasureSpec,
    MeasureValidityError,
)


def _spec(*, name: str = "tool_calls") -> MeasureSpec:
    return MeasureSpec(
        name,
        MetricDefinition(
            construct="structural tool calls",
            unit="count",
            unit_source="actions",
            aggregation="count",
            required_frame="all captured sessions",
            measurement_authority=("structural",),
        ),
        "structural",
        "all captured sessions",
        ("capture completeness",),
    )


def test_registry_requires_construct_validity_metadata() -> None:
    with pytest.raises(MeasureValidityError, match="evidence_tier"):
        MeasureSpec("x", _spec().metric, "", "frame", ("capture",))
    with pytest.raises(MeasureValidityError, match="sample_frame"):
        MeasureSpec("x", _spec().metric, "structural", "", ("capture",))
    with pytest.raises(MeasureValidityError, match="confounds"):
        MeasureSpec("x", _spec().metric, "structural", "frame", ())


def test_equivalent_metric_resolves_to_one_measure_identity() -> None:
    registry = MeasureRegistry()
    first = _spec()
    assert registry.register(first) == registry.register(_spec())
    assert registry.get(first.ref) == first
    assert len(registry) == 1


def test_production_registry_contains_existing_measure_families() -> None:
    assert len(DEFAULT_MEASURE_REGISTRY) >= 4
    assert {spec.name for spec in DEFAULT_MEASURE_REGISTRY} >= {
        "session_cost_usd",
        "tool_calls",
        "message_count",
        "wall_duration_ms",
    }


def test_registry_rejects_rebinding_a_name_to_another_metric() -> None:
    registry = MeasureRegistry()
    original = _spec()
    registry.register(original)
    conflicting = replace(original, metric=replace(original.metric, aggregation="sum"))
    with pytest.raises(MeasureValidityError, match="already bound"):
        registry.register(conflicting)
    assert registry.require(original.name) is original
    assert registry.require(original.ref) is original


def test_registry_rejects_conflicting_evidence_metadata_for_one_identity() -> None:
    registry = MeasureRegistry()
    original = _spec()
    registry.register(original)
    conflicting = replace(original, sample_frame="only selected sessions")
    assert conflicting.ref == original.ref
    with pytest.raises(MeasureValidityError, match="different definition"):
        registry.register(conflicting)
    assert tuple(registry) == (original,)


def test_registry_refuses_unknown_names_without_fabricating_a_definition() -> None:
    registry = MeasureRegistry()
    assert registry.get("missing") is None
    with pytest.raises(MeasureValidityError, match="unknown measure"):
        registry.require("missing")
