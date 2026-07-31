"""The process-wide default metric registry has a real registered definition (rxdo.9.1)."""

from __future__ import annotations

from polylogue.insights.measurement.registered_metrics import DEFAULT_METRIC_REGISTRY, SESSION_COST_USD_METRIC


def test_default_registry_resolves_the_session_cost_metric_by_name() -> None:
    resolved = DEFAULT_METRIC_REGISTRY.resolve("session_cost_usd")
    assert resolved is not None
    assert resolved.ref == SESSION_COST_USD_METRIC.ref


def test_default_registry_resolves_the_session_cost_metric_by_hash() -> None:
    resolved = DEFAULT_METRIC_REGISTRY.get(SESSION_COST_USD_METRIC.ref)
    assert resolved is not None
    assert resolved.construct == SESSION_COST_USD_METRIC.construct


def test_session_cost_metric_declares_mixed_provenance_honestly() -> None:
    """Session cost blends provider-reported and catalog-estimated lanes -- must be declared, not silent."""
    assert SESSION_COST_USD_METRIC.provenance_mixing == "mixed-declared"
    assert set(SESSION_COST_USD_METRIC.measurement_authority) == {"provider-reported", "catalog-estimated"}
