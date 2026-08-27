from __future__ import annotations

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
