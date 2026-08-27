"""The catalogs must price the models actually in use."""

from __future__ import annotations

import pytest

from polylogue.archive.semantic.pricing import PRICING
from polylogue.archive.semantic.subscription_pricing import get_credit_rate

#: Anthropic list prices per million tokens: input, output, cache read,
#: 5-minute cache write.
_LIST_PRICES = {
    "claude-opus-5": (5.0, 25.0, 0.5, 6.25),
    "claude-opus-4-8": (5.0, 25.0, 0.5, 6.25),
    "claude-sonnet-5": (2.0, 10.0, 0.2, 2.5),
    "claude-sonnet-4-6": (3.0, 15.0, 0.3, 3.75),
    "claude-haiku-4-5": (1.0, 5.0, 0.1, 1.25),
    "claude-fable-5": (10.0, 50.0, 1.0, 12.5),
}


@pytest.mark.parametrize(("model", "expected"), sorted(_LIST_PRICES.items()))
def test_catalog_matches_the_published_list_price(model: str, expected: tuple[float, float, float, float]) -> None:
    """A priced model must carry its real rate, not a predecessor's.

    Anti-vacuity: the Opus entries previously carried the retired Opus 4.1
    rates, overstating every Opus session threefold, and nothing was red.
    """
    pricing = PRICING.get(model)
    assert pricing is not None, f"{model} is absent from the catalog"
    assert (
        pricing.input_usd_per_1m,
        pricing.output_usd_per_1m,
        pricing.cache_read_usd_per_1m,
        pricing.cache_write_usd_per_1m,
    ) == expected


@pytest.mark.parametrize("model", sorted(_LIST_PRICES))
def test_subscription_credits_track_the_dollar_price(model: str) -> None:
    """Credits bill at API rates, so they follow the price they are billed at.

    Anti-vacuity: adding a model to one catalog and not the other leaves a
    subscription session with no API-equivalent figure at all, which is how a
    real cost becomes a reported zero.
    """
    rate = get_credit_rate(model)
    assert rate is not None, f"{model} has no subscription credit rate"
    pricing = PRICING[model]
    assert rate.input_credits / rate.input_divisor == pytest.approx(pricing.input_usd_per_1m * 2 / 15)
    assert rate.output_credits / rate.output_divisor == pytest.approx(pricing.output_usd_per_1m * 2 / 15)
    assert rate.cache_read_credits == 0, "subscription cache reads are free"
