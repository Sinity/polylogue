from __future__ import annotations

import pytest

from polylogue.insights.measurement.metric import MetricDefinition
from polylogue.insights.measurement.registered_measures import DEFAULT_MEASURE_REGISTRY
from polylogue.insights.measurement.registry import (
    MeasurePlan,
    MeasureRegistry,
    MeasureSpec,
    MeasureValidityError,
    compose_measure,
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


def test_composition_preserves_unknown_zero_and_renders_tier() -> None:
    registry = MeasureRegistry()
    spec = _spec()
    registry.register(spec)
    results = compose_measure(
        registry,
        MeasurePlan(spec.ref, group_by="origin"),
        [
            {"ref": "session:a", "origin": "codex", "value": 0},
            {"ref": "session:b", "origin": "codex", "value": None, "value_state": "unknown"},
        ],
        group_field="origin",
    )
    assert results[0].value == 1
    assert results[0].value_state == "known"
    assert "Evidence tier: structural" in results[0].render()
    assert results[0].member_refs == ("session:a", "session:b")


def test_empty_composition_has_no_fabricated_zero() -> None:
    registry = MeasureRegistry()
    spec = _spec()
    registry.register(spec)
    assert compose_measure(registry, MeasurePlan(spec.ref), []) == ()


def test_composition_checks_declared_coverage_preconditions() -> None:
    registry = MeasureRegistry()
    base = _spec()
    spec = MeasureSpec(
        base.name,
        base.metric,
        base.evidence_tier,
        base.sample_frame,
        base.confounds,
        coverage_preconditions=("coverage_complete",),
    )
    registry.register(spec)
    with pytest.raises(MeasureValidityError, match="complete frame coverage"):
        compose_measure(registry, MeasurePlan(spec.ref), [{"ref": "session:a", "value": 1}])
    result = compose_measure(
        registry,
        MeasurePlan(spec.ref),
        [{"ref": "session:a", "value": 1, "coverage_complete": True}],
    )
    assert result[0].value == 1


def test_composition_rejects_fake_evidence_payloads() -> None:
    registry = MeasureRegistry()
    spec = _spec()
    registry.register(spec)
    with pytest.raises(MeasureValidityError, match="must be an EvidenceValue"):
        compose_measure(registry, MeasurePlan(spec.ref), [{"ref": "session:a", "value": 1, "evidence": {}}])


def test_production_registry_contains_existing_measure_families() -> None:
    assert len(DEFAULT_MEASURE_REGISTRY) >= 5
    assert {spec.name for spec in DEFAULT_MEASURE_REGISTRY} >= {
        "session_cost_usd",
        "plan_completion_rate",
        "tool_calls",
        "message_count",
        "wall_duration_ms",
    }
