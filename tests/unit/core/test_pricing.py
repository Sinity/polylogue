"""Contracts for provider/model cost estimation."""

from __future__ import annotations

import pytest

from polylogue.archive.message.messages import MessageCollection
from polylogue.lib.pricing import _normalize_model, estimate_conversation_cost, estimate_cost
from tests.infra.builders import make_conv, make_msg


def test_exact_archive_cost_wins_over_catalog_estimate() -> None:
    conversation = make_conv(
        id="conv-exact-cost",
        provider="claude-code",
        provider_meta={"total_cost_usd": 1.25, "model": "claude-sonnet-4-5"},
        messages=MessageCollection(messages=[]),
    )

    estimate = estimate_conversation_cost(conversation)

    assert estimate.status == "exact"
    assert estimate.confidence == 1.0
    assert estimate.total_usd == 1.25
    assert "archive_provider_reported_cost" in estimate.provenance


def test_token_usage_prices_known_model_with_catalog_provenance() -> None:
    conversation = make_conv(
        id="conv-priced-cost",
        provider="chatgpt",
        provider_meta={
            "model": "openai/gpt-4o-2024-08-06",
            "usage": {"input_tokens": 1000, "output_tokens": 500},
        },
        messages=MessageCollection(messages=[]),
    )

    estimate = estimate_conversation_cost(conversation)

    assert estimate.status == "priced"
    assert estimate.normalized_model == "gpt-4o"
    assert estimate.total_usd == pytest.approx(0.0075)
    assert estimate.usage.input_tokens == 1000
    assert estimate.usage.output_tokens == 500
    assert estimate.price is not None
    assert estimate.price.source_url.endswith("model_prices_and_context_window.json")


def test_partial_conversation_preserves_missing_reasons() -> None:
    conversation = make_conv(
        id="conv-partial-cost",
        provider="chatgpt",
        messages=MessageCollection(
            messages=[
                make_msg(
                    id="m-priced",
                    role="assistant",
                    provider="chatgpt",
                    provider_meta={"model": "gpt-4o-mini", "usage": {"input_tokens": 100, "output_tokens": 50}},
                ),
                make_msg(id="m-missing", role="assistant", provider="chatgpt", provider_meta={"model": "gpt-4o"}),
            ]
        ),
    )

    estimate = estimate_conversation_cost(conversation)

    assert estimate.status == "partial"
    assert estimate.total_usd == pytest.approx(0.000045)
    assert "missing_token_usage" in estimate.missing_reasons


def test_partial_conversation_with_exact_message_is_not_exact() -> None:
    conversation = make_conv(
        id="conv-exact-plus-missing",
        provider="chatgpt",
        messages=MessageCollection(
            messages=[
                make_msg(
                    id="m-exact",
                    role="assistant",
                    provider="chatgpt",
                    provider_meta={"costUSD": 0.01},
                ),
                make_msg(id="m-missing", role="assistant", provider="chatgpt"),
            ]
        ),
    )

    estimate = estimate_conversation_cost(conversation)

    assert estimate.status == "partial"
    assert estimate.total_usd == pytest.approx(0.01)


def test_missing_price_is_unavailable_not_zero_precision() -> None:
    conversation = make_conv(
        id="conv-unknown-model",
        provider="chatgpt",
        provider_meta={"model": "unknown-frontier-model", "usage": {"input_tokens": 100, "output_tokens": 50}},
        messages=MessageCollection(messages=[]),
    )

    estimate = estimate_conversation_cost(conversation)

    assert estimate.status == "unavailable"
    assert estimate.total_usd == 0.0
    assert estimate.missing_reasons == ("missing_price",)


def test_model_normalization_accepts_provider_prefixes_and_version_suffixes() -> None:
    assert _normalize_model("openai/gpt-4o-2024-08-06") == "gpt-4o"
    assert _normalize_model("anthropic/claude-sonnet-4-5-20250929") == "claude-sonnet-4-5"
    assert estimate_cost(1000, 500, "openai/gpt-4o-2024-08-06") == pytest.approx(0.0075)
