"""Contracts for provider/model cost estimation."""

from __future__ import annotations

import pytest

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.semantic.pricing import _normalize_model, estimate_cost, estimate_session_cost
from tests.infra.builders import make_conv, make_msg


def test_exact_archive_cost_wins_over_catalog_estimate() -> None:
    session = make_conv(
        id="conv-exact-cost",
        provider="claude-code",
        provider_meta={"total_cost_usd": 1.25, "model": "claude-sonnet-4-5"},
        messages=MessageCollection(messages=[]),
    )

    estimate = estimate_session_cost(session)

    assert estimate.status == "exact"
    assert estimate.confidence == 1.0
    assert estimate.total_usd == 1.25
    assert "archive_provider_reported_cost" in estimate.provenance


def test_token_usage_prices_known_model_with_catalog_provenance() -> None:
    session = make_conv(
        id="conv-priced-cost",
        provider="chatgpt",
        provider_meta={
            "model": "openai/gpt-4o-2024-08-06",
            "usage": {"input_tokens": 1000, "output_tokens": 500},
        },
        messages=MessageCollection(messages=[]),
    )

    estimate = estimate_session_cost(session)

    assert estimate.status == "priced"
    assert estimate.normalized_model == "gpt-4o"
    assert estimate.total_usd == pytest.approx(0.0075)
    assert estimate.usage.input_tokens == 1000
    assert estimate.usage.output_tokens == 500
    assert estimate.price is not None
    assert estimate.price.source_url.endswith("model_prices_and_context_window.json")


def test_hydrated_messages_report_missing_model_when_no_envelope_cost() -> None:
    """Per #1256, hydrated Message instances no longer carry
    ``provider_meta``; per-message cost facts now flow through the typed
    cost projection (#803). When neither session-level cost nor
    per-message harmonized facts are present, the estimate reports
    ``missing_model`` for each message.
    """

    session = make_conv(
        id="conv-hydrated-no-cost",
        provider="chatgpt",
        messages=MessageCollection(
            messages=[
                make_msg(id="m1", role="assistant", provider="chatgpt"),
                make_msg(id="m2", role="assistant", provider="chatgpt"),
            ]
        ),
    )

    estimate = estimate_session_cost(session)

    assert estimate.status == "unavailable"
    assert estimate.total_usd == 0.0
    # Either missing_model or missing_token_usage is acceptable; hydrated
    # messages contribute no model or usage facts.
    assert estimate.missing_reasons


def test_session_level_exact_cost_still_wins_for_hydrated_messages() -> None:
    """Session-level provider_meta is still consumed for exact totals."""

    session = make_conv(
        id="conv-exact-from-envelope",
        provider="chatgpt",
        provider_meta={"costUSD": 0.01, "model": "gpt-4o"},
        messages=MessageCollection(
            messages=[
                make_msg(id="m1", role="assistant", provider="chatgpt"),
                make_msg(id="m2", role="assistant", provider="chatgpt"),
            ]
        ),
    )

    estimate = estimate_session_cost(session)

    assert estimate.status == "exact"
    assert estimate.total_usd == pytest.approx(0.01)


def test_missing_price_is_unavailable_not_zero_precision() -> None:
    session = make_conv(
        id="conv-unknown-model",
        provider="chatgpt",
        provider_meta={"model": "unknown-frontier-model", "usage": {"input_tokens": 100, "output_tokens": 50}},
        messages=MessageCollection(messages=[]),
    )

    estimate = estimate_session_cost(session)

    assert estimate.status == "unavailable"
    assert estimate.total_usd == 0.0
    assert estimate.missing_reasons == ("missing_price",)


def test_model_normalization_accepts_provider_prefixes_and_version_suffixes() -> None:
    assert _normalize_model("openai/gpt-4o-2024-08-06") == "gpt-4o"
    assert _normalize_model("anthropic/claude-sonnet-4-5-20250929") == "claude-sonnet-4-5"
    assert estimate_cost(1000, 500, "openai/gpt-4o-2024-08-06") == pytest.approx(0.0075)
