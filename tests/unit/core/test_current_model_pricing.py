"""The catalogs must price the models actually in use."""

from __future__ import annotations

import pytest

from polylogue.archive.semantic.pricing import PRICING
from polylogue.archive.semantic.subscription_pricing import get_credit_rate

#: Anthropic list prices per million tokens: input, output, cache read,
#: 5-minute cache write.
_LIST_PRICES = {
    "claude-opus-5": (5.0, 25.0, 0.5, 6.25),
    "claude-sonnet-5": (2.0, 10.0, 0.2, 2.5),
    "claude-opus-4-8": (5.0, 25.0, 0.5, 6.25),
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
    ) == pytest.approx(expected)


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
    assert rate.output_credits == rate.input_credits * 5, "output credits must retain the 5x rate"
    assert rate.cache_read_credits == 0, "subscription cache reads are free"
    assert rate.cache_write_credits == rate.input_credits


def test_every_credited_model_has_an_api_price() -> None:
    """A subscription rate without a catalog price has no API-equivalent figure.

    ``MODEL_CREDIT_RATES`` declares that a model is billed and at what credit
    rate. With no row in the sole vendored catalog, ``estimate_cost`` returns
    0.0 and ``compute_session_cost`` downgrades the breakdown to ``unknown``,
    so every session on that model reports an unknown API-equivalent total --
    which is what happened to the current Claude 5 family.

    Anti-vacuity: deleting ``claude-opus-5`` from the catalog names it here,
    and the per-model assertions above pin the rate, so restoring the row with
    a wrong price does not make this green.
    """
    from polylogue.archive.semantic.subscription_pricing import MODEL_CREDIT_RATES

    assert sorted(model for model in MODEL_CREDIT_RATES if PRICING.get(model) is None) == []


def test_declared_credit_rates_are_not_reported_as_a_gap() -> None:
    """polylogue-t83q: every model the catalog declares must pass the check.

    Anti-vacuity: dropping a model from ``MODEL_CREDIT_RATES`` -- the way the
    Claude 5 family was absent while it was the current model -- makes this
    name it.
    """
    from polylogue.archive.semantic.subscription_pricing import (
        MODEL_CREDIT_RATES,
        models_without_credit_rate,
    )

    assert models_without_credit_rate(sorted(MODEL_CREDIT_RATES)) == ()


def test_undeclared_claude_model_is_named_as_a_credit_gap() -> None:
    """An unpriced Claude model must be reported, not silently credited zero.

    ``compute_credit_cost`` returns 0 for an undeclared model, which is the
    correct refusal but indistinguishable from "this model cost nothing"; the
    check is what makes the blind spot visible.  Non-Claude models are not a
    gap: they have no Claude subscription rate by construction.

    Anti-vacuity: making ``models_without_credit_rate`` return ``()``
    unconditionally turns the first assertion red.
    """
    from polylogue.archive.semantic.subscription_pricing import (
        compute_credit_cost,
        models_without_credit_rate,
    )

    assert compute_credit_cost("claude-opus-9", 1_000_000, 1_000_000) == 0
    assert models_without_credit_rate(["claude-opus-9", "claude-opus-5"]) == ("claude-opus-9",)
    assert models_without_credit_rate(["gpt-5", "deepseek-v4-flash", ""]) == ()
