from __future__ import annotations

import pytest

from polylogue.insights.measurement.metric import (
    DuplicateMetricIdentityError,
    MetricDefinition,
    MetricRegistry,
)


def _definition(**overrides: object) -> MetricDefinition:
    fields: dict[str, object] = {
        "construct": "cost_per_session",
        "unit": "usd",
        "unit_source": "sessions where origin:codex-session",
        "aggregation": "sum",
        "required_frame": "codex-session",
        "measurement_authority": ("provider-reported",),
    }
    fields.update(overrides)
    return MetricDefinition(**fields)  # type: ignore[arg-type]


def test_identical_content_hashes_identically_regardless_of_construction_order() -> None:
    first = MetricDefinition(
        construct="cost",
        unit="usd",
        unit_source="sessions",
        aggregation="sum",
        confounds=frozenset({"b", "a"}),
    )
    second = MetricDefinition(
        construct="cost",
        unit="usd",
        unit_source="sessions",
        aggregation="sum",
        confounds=frozenset({"a", "b"}),
    )

    assert first.ref == second.ref
    assert first.ref.startswith("metric:")


def test_differing_denominator_or_null_policy_changes_the_hash() -> None:
    base = _definition()
    different_denominator = _definition(denominator_expr="count(*)")
    different_null_policy = _definition(null_policy="suppress")

    assert base.ref != different_denominator.ref
    assert base.ref != different_null_policy.ref
    assert different_denominator.ref != different_null_policy.ref


def test_this_is_the_mechanism_that_would_have_caught_the_cost_inflation_bug() -> None:
    """Two 'cost' metrics differing only in whether cached tokens are excluded
    from the input-token component must resolve to different refs -- the
    exact failure class rxdo.9.1 exists to make visible (7.69x Codex cost
    inflation, docs/design/analysis-rigor.md mechanism A)."""

    cost_including_cached = _definition(
        construct="cost", exclusions=frozenset(), confounds=frozenset({"cached-tokens-included"})
    )
    cost_excluding_cached = _definition(
        construct="cost", exclusions=frozenset({"cached-input-tokens"}), confounds=frozenset()
    )

    assert cost_including_cached.ref != cost_excluding_cached.ref


def test_is_compatible_with_rejects_grain_mismatch() -> None:
    logical = _definition(grain="logical")
    physical = _definition(grain="physical")

    assert not logical.is_compatible_with(physical)


def test_is_compatible_with_rejects_disjoint_single_authority_measurement() -> None:
    provider_reported = _definition(measurement_authority=("provider-reported",))
    heuristic = _definition(measurement_authority=("heuristic",))

    assert not provider_reported.is_compatible_with(heuristic)


def test_is_compatible_with_allows_mixed_declared_provenance() -> None:
    provider_reported = _definition(measurement_authority=("provider-reported",), provenance_mixing="mixed-declared")
    heuristic = _definition(measurement_authority=("heuristic",), provenance_mixing="mixed-declared")

    assert provider_reported.is_compatible_with(heuristic)


def test_registry_registration_is_idempotent_by_content_hash() -> None:
    registry = MetricRegistry()
    definition = _definition()

    first_ref = registry.register(definition)
    second_ref = registry.register(_definition())  # separately constructed, same content

    assert first_ref == second_ref
    assert len(registry) == 1


def test_registry_rejects_second_identity_for_same_friendly_name() -> None:
    registry = MetricRegistry()
    registry.register(_definition(), name="cost_per_session")

    with pytest.raises(DuplicateMetricIdentityError):
        registry.register(_definition(unit="eur"), name="cost_per_session")


def test_registry_resolves_by_friendly_name() -> None:
    registry = MetricRegistry()
    definition = _definition()
    registry.register(definition, name="cost_per_session")

    assert registry.resolve("cost_per_session") == definition
    assert registry.resolve("unknown-name") is None


def test_registry_get_returns_none_for_unknown_ref() -> None:
    registry = MetricRegistry()

    assert registry.get("metric:" + "0" * 64) is None
