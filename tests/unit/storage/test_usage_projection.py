from __future__ import annotations

import sqlite3

import pytest

from polylogue.archive.semantic import pricing as pricing_module
from polylogue.archive.semantic.pricing import estimate_cost
from polylogue.storage.usage import UsageProjectionModel, project_provider_usage_events, rollup_usage_projections


def test_projection_uses_latest_session_global_cumulative_once() -> None:
    events = [
        {
            "session_id": "s1",
            "position": 1,
            "provider_event_type": "token_count",
            "model_name": "gpt-4o",
            "total_input_tokens": 100,
            "total_output_tokens": 20,
            "total_cached_input_tokens": 40,
        },
        {
            "session_id": "s1",
            "position": 2,
            "provider_event_type": "token_count",
            "model_name": "gpt-4o",
            "total_input_tokens": 300,
            "total_output_tokens": 50,
            "total_cached_input_tokens": 0,
        },
    ]
    (projection,) = project_provider_usage_events(events, origin="codex")
    assert projection.input_tokens == 300
    assert projection.cache_read_tokens == 0
    assert projection.output_tokens == 50
    assert projection.cost_usd == estimate_cost(300, 50, "gpt-4o", 0, 0)


def test_projection_splits_models_and_marks_missing_cache_rate_incomplete() -> None:
    events = [
        {
            "session_id": "s1",
            "position": 1,
            "provider_event_type": "message_usage",
            "model_name": "gpt-4o",
            "last_input_tokens": 100,
            "last_output_tokens": 20,
        },
        {
            "session_id": "s1",
            "position": 2,
            "provider_event_type": "message_usage",
            "model_name": "unknown-model",
            "last_cached_input_tokens": 10,
        },
    ]
    projections = project_provider_usage_events(events, origin="test")
    assert {row.model_name for row in projections} == {"gpt-4o", "unknown-model"}
    incomplete = next(row for row in projections if row.model_name == "unknown-model")
    assert incomplete.cost_usd is None
    assert incomplete.state == "incomplete"
    assert incomplete.missing_reasons == ("missing_model_price",)


def test_rollup_keeps_cost_unknown_when_one_model_is_incomplete() -> None:
    projections = (
        UsageProjectionModel("s1", "gpt-4o", 100, 0, 0, 0, 0.001, "complete"),
        UsageProjectionModel("s2", "gpt-4o", 100, 0, 0, 0, None, "incomplete", ("missing_model_price",)),
    )
    (rollup,) = rollup_usage_projections(projections, origins={"s1": "test", "s2": "test"})
    assert rollup.cost_usd is None
    assert rollup.state == "incomplete"
    assert rollup.incomplete_session_count == 1


def test_paid_model_with_missing_cache_rate_is_not_persisted_as_priced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider rollup must not claim complete pricing for an omitted lane.

    Anti-vacuity: restoring the old ``estimate_cost``-only writer path makes
    this row ``priced`` with a zero-priced cache lane.
    """
    monkeypatch.setitem(
        pricing_module.PRICING,
        "paid-without-cache-rate",
        pricing_module.ModelPricing(
            source_name="test",
            input_usd_per_1m=1.0,
            output_usd_per_1m=2.0,
            cache_read_usd_per_1m=0.0,
            cache_write_usd_per_1m=0.0,
        ),
    )

    from polylogue.storage.sqlite.archive_tiers.write import _price_provider_usage_tokens

    conn = sqlite3.connect(":memory:")
    provenance, cost = _price_provider_usage_tokens(
        conn,
        "paid-without-cache-rate",
        input_tokens=100,
        output_tokens=10,
        cache_read_tokens=1_000,
        cache_write_tokens=0,
    )
    conn.close()
    assert provenance is None
    assert cost is None


def test_free_model_with_zero_cache_rate_remains_priced(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero is a valid cache rate for a genuinely free catalog model."""
    monkeypatch.setitem(
        pricing_module.PRICING,
        "free-with-zero-cache-rate",
        pricing_module.ModelPricing(
            source_name="test",
            input_usd_per_1m=0.0,
            output_usd_per_1m=0.0,
        ),
    )
    events = [
        {
            "session_id": "s1",
            "provider_event_type": "message_usage",
            "model_name": "free-with-zero-cache-rate",
            "last_input_tokens": 100,
            "last_cached_input_tokens": 1_000,
        }
    ]
    (projection,) = project_provider_usage_events(events, origin="test")
    assert projection.state == "complete"
    assert projection.cost_usd == 0.0
