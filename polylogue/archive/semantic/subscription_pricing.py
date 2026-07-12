"""Claude Code subscription/credit pricing alongside API-equivalent pricing.

Pricing assumptions are dated and carry explicit provenance. Do not
hard-code current vendor prices without a documented source/update path.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

SUBSCRIPTION_CATALOG_PROVENANCE = "polylogue-curated-claude-code-subscription-v1"
SUBSCRIPTION_CATALOG_EFFECTIVE_DATE = "2026-05-07"
SUBSCRIPTION_SOURCE_URL = "https://docs.anthropic.com/en/docs/claude-code/pricing"


class SubscriptionTier(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    monthly_fee_usd: float
    credit_pool: int
    description: str = ""
    max_instances: int = 1


SUBSCRIPTION_TIERS: dict[str, SubscriptionTier] = {
    "pro": SubscriptionTier(
        name="pro",
        monthly_fee_usd=20.0,
        credit_pool=21_700_000,
        max_instances=1,
        description="Claude Code Pro ($20/mo, 21.7M credits)",
    ),
    "max_5x": SubscriptionTier(
        name="max_5x",
        monthly_fee_usd=100.0,
        credit_pool=180_600_000,
        max_instances=5,
        description="Claude Code Max 5x ($100/mo, 180.6M credits)",
    ),
    "max_20x": SubscriptionTier(
        name="max_20x",
        monthly_fee_usd=200.0,
        credit_pool=361_100_000,
        max_instances=20,
        description="Claude Code Max 20x ($200/mo, 361.1M credits)",
    ),
}


@dataclass(frozen=True, slots=True)
class ModelCreditRate:
    source_name: str
    normalized_model: str
    input_credits: int
    output_credits: int
    input_divisor: int = 15
    output_divisor: int = 15
    cache_read_credits: int = 0
    cache_read_divisor: int = 15
    cache_write_credits: int = 1
    cache_write_divisor: int = 15
    provenance: str = SUBSCRIPTION_CATALOG_PROVENANCE
    effective_date: str = SUBSCRIPTION_CATALOG_EFFECTIVE_DATE

    def credits_for(
        self, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0, cache_write_tokens: int = 0
    ) -> int:
        import math

        total = 0
        if input_tokens > 0:
            total += math.ceil(input_tokens * self.input_credits / self.input_divisor)
        if output_tokens > 0:
            total += math.ceil(output_tokens * self.output_credits / self.output_divisor)
        if cache_read_tokens > 0:
            total += math.ceil(cache_read_tokens * self.cache_read_credits / self.cache_read_divisor)
        if cache_write_tokens > 0:
            total += math.ceil(cache_write_tokens * self.cache_write_credits / self.cache_write_divisor)
        return total


MODEL_CREDIT_RATES: dict[str, ModelCreditRate] = {
    # Subscription credits are "billed at API rates"
    # (https://docs.anthropic.com/en/docs/claude-code/pricing), and every Claude
    # model prices output at 5x input ($15/$75 Opus, $3/$15 Sonnet, $0.80/$4
    # Haiku). So output_credits must be 5x input_credits; cache writes (5-min)
    # bill at the input rate, cache reads are free on subscription plans.
    "claude-opus-4-6": ModelCreditRate(
        "anthropic", "claude-opus-4-6", 10, 50, cache_read_credits=0, cache_write_credits=10
    ),
    "claude-opus-4-5": ModelCreditRate(
        "anthropic", "claude-opus-4-5", 10, 50, cache_read_credits=0, cache_write_credits=10
    ),
    "claude-sonnet-4-6": ModelCreditRate(
        "anthropic", "claude-sonnet-4-6", 6, 30, cache_read_credits=0, cache_write_credits=6
    ),
    "claude-sonnet-4-5": ModelCreditRate(
        "anthropic", "claude-sonnet-4-5", 6, 30, cache_read_credits=0, cache_write_credits=6
    ),
    "claude-haiku-4-5": ModelCreditRate(
        "anthropic", "claude-haiku-4-5", 2, 10, cache_read_credits=0, cache_write_credits=2
    ),
}


def get_credit_rate(normalized_model: str) -> ModelCreditRate | None:
    return MODEL_CREDIT_RATES.get(normalized_model)


def compute_credit_cost(
    normalized_model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> int:
    rate = get_credit_rate(normalized_model)
    if rate is None:
        return 0
    return rate.credits_for(input_tokens, output_tokens, cache_read_tokens, cache_write_tokens)


def credits_to_usd(credit_cost: float, *, tier: str = "pro") -> float:
    """Convert a subscription credit cost to a subscription-equivalent USD figure.

    The conversion is ``monthly_fee_usd / credit_pool`` for the named tier
    (default ``pro``, the cheapest/most conservative tier). This is a single
    shared conversion so every surface reporting a subscription-equivalent
    dollar figure (session cost, provider-usage ledger, ...) draws from the
    same tier assumption instead of re-deriving the ratio ad hoc (#f2qv.3 /
    #5hf). Non-authoritative: see docs/cost-model.md caveats.
    """
    if credit_cost <= 0:
        return 0.0
    plan = SUBSCRIPTION_TIERS.get(tier)
    if plan is None or plan.credit_pool <= 0:
        return 0.0
    return credit_cost / plan.credit_pool * plan.monthly_fee_usd


@dataclass(frozen=True, slots=True)
class SubscriptionCostEstimate:
    model: str | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    credit_cost: int = 0
    api_equivalent_usd: float = 0.0
    subscription_equivalent_usd: float = 0.0
    tier_name: str | None = None
    confidence: str = "unknown"
    provenance: str = SUBSCRIPTION_CATALOG_PROVENANCE


__all__ = [
    "MODEL_CREDIT_RATES",
    "SUBSCRIPTION_CATALOG_EFFECTIVE_DATE",
    "SUBSCRIPTION_CATALOG_PROVENANCE",
    "SUBSCRIPTION_SOURCE_URL",
    "SUBSCRIPTION_TIERS",
    "ModelCreditRate",
    "SubscriptionCostEstimate",
    "SubscriptionTier",
    "compute_credit_cost",
    "credits_to_usd",
    "get_credit_rate",
]
