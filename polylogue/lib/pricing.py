"""Provider/model pricing table for cost estimation across all AI providers.

Polylogue directly captures actual cost for Claude Code sessions as
conversation-level totals. For all other providers (ChatGPT, Claude web, Gemini)
cost must be estimated from token counts using public pricing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation


@dataclass(frozen=True)
class ModelPricing:
    """Per-million-token pricing for a specific model."""
    input_usd_per_1m: float
    output_usd_per_1m: float
    cache_read_usd_per_1m: float = 0.0


# Public pricing as of early 2026 (USD per 1M tokens)
PRICING: dict[str, ModelPricing] = {
    # Anthropic Claude 4 family
    "claude-opus-4-6": ModelPricing(15.0, 75.0, 1.5),
    "claude-opus-4-5": ModelPricing(15.0, 75.0, 1.5),
    "claude-sonnet-4-6": ModelPricing(3.0, 15.0, 0.3),
    "claude-sonnet-4-5": ModelPricing(3.0, 15.0, 0.3),
    "claude-haiku-4-5": ModelPricing(0.8, 4.0, 0.08),
    # Anthropic Claude 3.x
    "claude-3-5-sonnet-20241022": ModelPricing(3.0, 15.0, 0.3),
    "claude-3-5-sonnet-20240620": ModelPricing(3.0, 15.0, 0.3),
    "claude-3-5-haiku-20241022": ModelPricing(0.8, 4.0, 0.08),
    "claude-3-opus-20240229": ModelPricing(15.0, 75.0, 1.5),
    "claude-3-sonnet-20240229": ModelPricing(3.0, 15.0, 0.3),
    "claude-3-haiku-20240307": ModelPricing(0.25, 1.25, 0.03),
    # OpenAI GPT-4
    "gpt-4o": ModelPricing(2.5, 10.0),
    "gpt-4o-mini": ModelPricing(0.15, 0.6),
    "gpt-4-turbo": ModelPricing(10.0, 30.0),
    "gpt-4": ModelPricing(30.0, 60.0),
    "gpt-3.5-turbo": ModelPricing(0.5, 1.5),
    "o1": ModelPricing(15.0, 60.0),
    "o1-mini": ModelPricing(3.0, 12.0),
    "o3": ModelPricing(10.0, 40.0),
    "o3-mini": ModelPricing(1.1, 4.4),
    # Google Gemini
    "gemini-1.5-pro": ModelPricing(3.5, 10.5),
    "gemini-1.5-flash": ModelPricing(0.075, 0.3),
    "gemini-2.0-flash": ModelPricing(0.1, 0.4),
    "gemini-2.5-pro": ModelPricing(1.25, 10.0),
}


def _normalize_model(model: str) -> str:
    """Normalize a model name for lookup, trying prefix matches."""
    if model in PRICING:
        return model
    # Try prefix: "claude-sonnet-4-6-20250101" → "claude-sonnet-4-6"
    for key in PRICING:
        if model.startswith(key):
            return key
    return model


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
    cache_read_tokens: int = 0,
) -> float:
    """Estimate cost from token counts using public pricing table.

    Returns 0.0 if the model is unknown.
    """
    pricing = PRICING.get(_normalize_model(model))
    if not pricing:
        return 0.0
    return (
        input_tokens * pricing.input_usd_per_1m / 1_000_000
        + output_tokens * pricing.output_usd_per_1m / 1_000_000
        + cache_read_tokens * pricing.cache_read_usd_per_1m / 1_000_000
    )


def harmonize_session_cost(conversation: "Conversation") -> tuple[float, bool]:
    """Return (cost_usd, is_estimated) for a conversation.

    - If actual session cost metadata is present, use it directly (is_estimated=False).
    - Otherwise, sum token counts from messages and look up pricing (is_estimated=True).
    - Returns (0.0, True) if no cost data is available.
    """
    # Ingest persists actual Claude Code cost as conversation-level totals.
    actual = conversation.total_cost_usd
    if actual and actual > 0.0:
        return actual, False

    # Try to estimate from token counts
    total_estimated = 0.0
    found_tokens = False
    for msg in conversation.messages:
        harmonized = msg.harmonized
        if harmonized is None:
            continue
        tokens = getattr(harmonized, "tokens", None)
        if tokens is None:
            continue
        meta = getattr(harmonized, "meta", None)
        model_name = getattr(meta, "model", None) if meta else None
        if not model_name:
            continue
        found_tokens = True
        input_tok = getattr(tokens, "input_tokens", 0) or 0
        output_tok = getattr(tokens, "output_tokens", 0) or 0
        cache_tok = getattr(tokens, "cache_read_tokens", 0) or 0
        total_estimated += estimate_cost(input_tok, output_tok, model_name, cache_tok)

    return total_estimated, True  # is_estimated=True even if 0 (unknown model)
