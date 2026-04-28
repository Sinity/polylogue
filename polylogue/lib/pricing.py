"""Typed provider/model cost estimation for archive conversations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from polylogue.lib.json import json_document
from polylogue.lib.semantic.facts import message_model_name, message_tokens
from polylogue.lib.viewport.viewports import TokenUsage

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation, Message

CostEstimateStatus = Literal["exact", "priced", "partial", "unavailable"]

LITELLM_PRICE_MAP_URL = "https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json"
CATALOG_PROVENANCE = "polylogue-curated-litellm-shaped-seed"
CATALOG_EFFECTIVE_DATE = "2026-04-24"


class PricingModel(BaseModel):
    """Base model for immutable cost-estimation payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CostUsagePayload(PricingModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0

    @property
    def billable_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens

    def plus(self, other: CostUsagePayload) -> CostUsagePayload:
        return CostUsagePayload(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class CostPricePayload(PricingModel):
    provider_name: str
    model_name: str
    normalized_model: str
    currency: str = "USD"
    input_usd_per_1m: float = 0.0
    output_usd_per_1m: float = 0.0
    cache_read_usd_per_1m: float = 0.0
    cache_write_usd_per_1m: float = 0.0
    provenance: str = CATALOG_PROVENANCE
    source_url: str = LITELLM_PRICE_MAP_URL
    effective_date: str = CATALOG_EFFECTIVE_DATE
    historical_exact: bool = False


class CostComponentPayload(PricingModel):
    name: str
    tokens: int = 0
    usd: float = 0.0


class CostEstimatePayload(PricingModel):
    provider_name: str
    conversation_id: str | None = None
    message_id: str | None = None
    model_name: str | None = None
    normalized_model: str | None = None
    status: CostEstimateStatus
    confidence: float = 0.0
    currency: str = "USD"
    total_usd: float = 0.0
    usage: CostUsagePayload = Field(default_factory=CostUsagePayload)
    price: CostPricePayload | None = None
    components: tuple[CostComponentPayload, ...] = ()
    missing_reasons: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ()

    @property
    def priced(self) -> bool:
        return self.status in {"exact", "priced", "partial"}


@dataclass(frozen=True)
class ModelPricing:
    """Per-million-token pricing for a normalized model identifier."""

    provider_name: str
    input_usd_per_1m: float
    output_usd_per_1m: float
    cache_read_usd_per_1m: float = 0.0
    cache_write_usd_per_1m: float = 0.0
    currency: str = "USD"
    source_url: str = LITELLM_PRICE_MAP_URL
    provenance: str = CATALOG_PROVENANCE
    effective_date: str = CATALOG_EFFECTIVE_DATE

    def to_payload(self, *, model_name: str, normalized_model: str) -> CostPricePayload:
        return CostPricePayload(
            provider_name=self.provider_name,
            model_name=model_name,
            normalized_model=normalized_model,
            currency=self.currency,
            input_usd_per_1m=self.input_usd_per_1m,
            output_usd_per_1m=self.output_usd_per_1m,
            cache_read_usd_per_1m=self.cache_read_usd_per_1m,
            cache_write_usd_per_1m=self.cache_write_usd_per_1m,
            provenance=self.provenance,
            source_url=self.source_url,
            effective_date=self.effective_date,
        )


# Curated seed shaped after LiteLLM's model price map. Costs are USD per
# 1M tokens. This is intentionally small and provenance-marked; provider-
# reported exact archive costs still take precedence.
PRICING: dict[str, ModelPricing] = {
    "claude-opus-4-6": ModelPricing("anthropic", 15.0, 75.0, 1.5, 18.75),
    "claude-opus-4-5": ModelPricing("anthropic", 15.0, 75.0, 1.5, 18.75),
    "claude-sonnet-4-6": ModelPricing("anthropic", 3.0, 15.0, 0.3, 3.75),
    "claude-sonnet-4-5": ModelPricing("anthropic", 3.0, 15.0, 0.3, 3.75),
    "claude-haiku-4-5": ModelPricing("anthropic", 0.8, 4.0, 0.08, 1.0),
    "claude-3-5-sonnet-20241022": ModelPricing("anthropic", 3.0, 15.0, 0.3, 3.75),
    "claude-3-5-sonnet-20240620": ModelPricing("anthropic", 3.0, 15.0, 0.3, 3.75),
    "claude-3-5-haiku-20241022": ModelPricing("anthropic", 0.8, 4.0, 0.08, 1.0),
    "claude-3-opus-20240229": ModelPricing("anthropic", 15.0, 75.0, 1.5, 18.75),
    "claude-3-sonnet-20240229": ModelPricing("anthropic", 3.0, 15.0, 0.3, 3.75),
    "claude-3-haiku-20240307": ModelPricing("anthropic", 0.25, 1.25, 0.03, 0.3),
    "gpt-4o": ModelPricing("openai", 2.5, 10.0),
    "gpt-4o-mini": ModelPricing("openai", 0.15, 0.6),
    "gpt-4-turbo": ModelPricing("openai", 10.0, 30.0),
    "gpt-4": ModelPricing("openai", 30.0, 60.0),
    "gpt-3.5-turbo": ModelPricing("openai", 0.5, 1.5),
    "o1": ModelPricing("openai", 15.0, 60.0),
    "o1-mini": ModelPricing("openai", 3.0, 12.0),
    "o3": ModelPricing("openai", 10.0, 40.0),
    "o3-mini": ModelPricing("openai", 1.1, 4.4),
    "gemini-1.5-pro": ModelPricing("google", 3.5, 10.5),
    "gemini-1.5-flash": ModelPricing("google", 0.075, 0.3),
    "gemini-2.0-flash": ModelPricing("google", 0.1, 0.4),
    "gemini-2.5-pro": ModelPricing("google", 1.25, 10.0),
}


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_int(value: object) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str):
        try:
            return max(int(float(value)), 0)
        except ValueError:
            return 0
    return 0


def _record(value: object) -> Mapping[str, object]:
    document = json_document(value)
    return document or {}


def _provider_text(value: object) -> str:
    return str(value) if value is not None else "unknown"


def _normalize_model(model: str) -> str:
    """Normalize a provider model name for catalog lookup."""

    candidate = model.strip()
    if not candidate:
        return candidate
    lowered = candidate.casefold()
    lowered = lowered.removeprefix("openai/")
    lowered = lowered.removeprefix("anthropic/")
    lowered = lowered.removeprefix("google/")
    lowered = lowered.removeprefix("gemini/")
    if lowered in PRICING:
        return lowered
    for key in sorted(PRICING, key=len, reverse=True):
        if lowered.startswith(key):
            return key
    return lowered


def _usage_payload(value: object) -> CostUsagePayload:
    usage = _record(value)
    input_tokens = _coerce_int(
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or usage.get("inputTokenCount")
        or usage.get("promptTokenCount")
    )
    output_tokens = _coerce_int(
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or usage.get("outputTokenCount")
        or usage.get("candidatesTokenCount")
    )
    cache_read_tokens = _coerce_int(
        usage.get("cache_read_tokens")
        or usage.get("cache_read_input_tokens")
        or usage.get("cached_tokens")
        or usage.get("cachedContentTokenCount")
    )
    cache_write_tokens = _coerce_int(
        usage.get("cache_write_tokens")
        or usage.get("cache_creation_input_tokens")
        or usage.get("cache_creation_tokens")
    )
    explicit_total = _coerce_int(usage.get("total_tokens") or usage.get("totalTokenCount"))
    total = explicit_total or input_tokens + output_tokens + cache_read_tokens + cache_write_tokens
    return CostUsagePayload(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        total_tokens=total,
    )


def _token_usage_payload(tokens: TokenUsage | None) -> CostUsagePayload:
    if tokens is None:
        return CostUsagePayload()
    input_tokens = tokens.input_tokens or 0
    output_tokens = tokens.output_tokens or 0
    cache_read_tokens = tokens.cache_read_tokens or 0
    cache_write_tokens = tokens.cache_write_tokens or 0
    total = tokens.total_tokens or input_tokens + output_tokens + cache_read_tokens + cache_write_tokens
    return CostUsagePayload(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        total_tokens=total,
    )


def _cost_components(usage: CostUsagePayload, pricing: ModelPricing) -> tuple[CostComponentPayload, ...]:
    return (
        CostComponentPayload(
            name="input",
            tokens=usage.input_tokens,
            usd=usage.input_tokens * pricing.input_usd_per_1m / 1_000_000,
        ),
        CostComponentPayload(
            name="output",
            tokens=usage.output_tokens,
            usd=usage.output_tokens * pricing.output_usd_per_1m / 1_000_000,
        ),
        CostComponentPayload(
            name="cache_read",
            tokens=usage.cache_read_tokens,
            usd=usage.cache_read_tokens * pricing.cache_read_usd_per_1m / 1_000_000,
        ),
        CostComponentPayload(
            name="cache_write",
            tokens=usage.cache_write_tokens,
            usd=usage.cache_write_tokens * pricing.cache_write_usd_per_1m / 1_000_000,
        ),
    )


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Estimate cost from token counts using the curated price catalog."""

    normalized_model = _normalize_model(model)
    pricing = PRICING.get(normalized_model)
    if pricing is None:
        return 0.0
    usage = CostUsagePayload(
        input_tokens=max(input_tokens, 0),
        output_tokens=max(output_tokens, 0),
        cache_read_tokens=max(cache_read_tokens, 0),
        cache_write_tokens=max(cache_write_tokens, 0),
    )
    return sum(component.usd for component in _cost_components(usage, pricing))


def _exact_estimate(
    *,
    provider_name: str,
    total_usd: float,
    conversation_id: str | None = None,
    message_id: str | None = None,
    model_name: str | None = None,
    usage: CostUsagePayload | None = None,
) -> CostEstimatePayload:
    normalized_model = _normalize_model(model_name) if model_name else None
    return CostEstimatePayload(
        provider_name=provider_name,
        conversation_id=conversation_id,
        message_id=message_id,
        model_name=model_name,
        normalized_model=normalized_model,
        status="exact",
        confidence=1.0,
        total_usd=total_usd,
        usage=usage or CostUsagePayload(),
        components=(CostComponentPayload(name="provider_reported_total", usd=total_usd),),
        provenance=("archive_provider_reported_cost",),
    )


def _estimate_from_usage(
    *,
    provider_name: str,
    model_name: str | None,
    usage: CostUsagePayload,
    conversation_id: str | None = None,
    message_id: str | None = None,
    provenance: tuple[str, ...],
) -> CostEstimatePayload:
    if model_name is None or not model_name.strip():
        return CostEstimatePayload(
            provider_name=provider_name,
            conversation_id=conversation_id,
            message_id=message_id,
            status="unavailable",
            confidence=0.0,
            usage=usage,
            missing_reasons=("missing_model",),
            provenance=provenance,
        )
    if usage.billable_tokens <= 0:
        return CostEstimatePayload(
            provider_name=provider_name,
            conversation_id=conversation_id,
            message_id=message_id,
            model_name=model_name,
            normalized_model=_normalize_model(model_name),
            status="unavailable",
            confidence=0.0,
            usage=usage,
            missing_reasons=("missing_token_usage",),
            provenance=provenance,
        )
    normalized_model = _normalize_model(model_name)
    pricing = PRICING.get(normalized_model)
    if pricing is None:
        return CostEstimatePayload(
            provider_name=provider_name,
            conversation_id=conversation_id,
            message_id=message_id,
            model_name=model_name,
            normalized_model=normalized_model,
            status="unavailable",
            confidence=0.0,
            usage=usage,
            missing_reasons=("missing_price",),
            provenance=provenance,
        )
    components = _cost_components(usage, pricing)
    return CostEstimatePayload(
        provider_name=provider_name,
        conversation_id=conversation_id,
        message_id=message_id,
        model_name=model_name,
        normalized_model=normalized_model,
        status="priced",
        confidence=0.85,
        total_usd=sum(component.usd for component in components),
        usage=usage,
        price=pricing.to_payload(model_name=model_name, normalized_model=normalized_model),
        components=components,
        provenance=(*provenance, CATALOG_PROVENANCE),
    )


def estimate_message_cost(
    message: Message,
    *,
    provider_name: str,
    conversation_id: str | None = None,
    fallback_model: str | None = None,
) -> CostEstimatePayload:
    """Estimate one message cost with explicit uncertainty."""

    exact = message.cost_usd
    provider_meta = _record(message.provider_meta)
    raw = _record(provider_meta.get("raw")) or provider_meta
    model_name = message_model_name(message) or str(raw.get("model") or fallback_model or "").strip() or None
    usage = _token_usage_payload(message_tokens(message))
    if usage.billable_tokens <= 0:
        usage = _usage_payload(raw.get("usage") or raw.get("tokens") or raw)
    if exact is not None and exact > 0.0:
        return _exact_estimate(
            provider_name=provider_name,
            conversation_id=conversation_id,
            message_id=str(message.id),
            model_name=model_name,
            usage=usage,
            total_usd=exact,
        )
    return _estimate_from_usage(
        provider_name=provider_name,
        conversation_id=conversation_id,
        message_id=str(message.id),
        model_name=model_name,
        usage=usage,
        provenance=("message_token_usage",),
    )


def _conversation_level_estimate(conversation: Conversation) -> CostEstimatePayload | None:
    provider_name = _provider_text(conversation.provider)
    provider_meta = _record(conversation.provider_meta)
    raw = _record(provider_meta.get("raw")) or provider_meta
    exact = (
        _coerce_float(raw.get("total_cost_usd"))
        or _coerce_float(raw.get("costUSD"))
        or _coerce_float(raw.get("cost_usd"))
    )
    model_name = str(raw.get("model") or raw.get("model_slug") or "").strip() or None
    usage = _usage_payload(raw.get("usage") or raw.get("tokens") or raw)
    if exact is not None and exact > 0.0:
        return _exact_estimate(
            provider_name=provider_name,
            conversation_id=str(conversation.id),
            model_name=model_name,
            usage=usage,
            total_usd=exact,
        )
    if usage.billable_tokens > 0 or model_name:
        return _estimate_from_usage(
            provider_name=provider_name,
            conversation_id=str(conversation.id),
            model_name=model_name,
            usage=usage,
            provenance=("conversation_provider_meta",),
        )
    return None


def _dominant_model(estimates: Iterable[CostEstimatePayload]) -> tuple[str | None, str | None]:
    counts: Counter[tuple[str | None, str | None]] = Counter()
    for estimate in estimates:
        if estimate.model_name or estimate.normalized_model:
            counts[(estimate.model_name, estimate.normalized_model)] += 1
    if not counts:
        return None, None
    return counts.most_common(1)[0][0]


def estimate_conversation_cost(conversation: Conversation) -> CostEstimatePayload:
    """Estimate cost for a conversation/session with confidence metadata."""

    provider_name = _provider_text(conversation.provider)
    conversation_estimate = _conversation_level_estimate(conversation)
    if conversation_estimate is not None and conversation_estimate.status == "exact":
        return conversation_estimate

    message_estimates = [
        estimate_message_cost(
            message,
            provider_name=provider_name,
            conversation_id=str(conversation.id),
            fallback_model=conversation_estimate.model_name if conversation_estimate else None,
        )
        for message in conversation.messages
    ]
    priced = [estimate for estimate in message_estimates if estimate.priced]
    if not message_estimates and conversation_estimate is not None:
        return conversation_estimate
    if not priced and conversation_estimate is not None:
        return conversation_estimate
    if not priced:
        missing = ("missing_token_usage",) if message_estimates else ("no_messages",)
        return CostEstimatePayload(
            provider_name=provider_name,
            conversation_id=str(conversation.id),
            status="unavailable",
            confidence=0.0,
            missing_reasons=missing,
            provenance=("conversation_messages",),
        )

    usage = CostUsagePayload()
    total_usd = 0.0
    components: list[CostComponentPayload] = []
    missing_reasons: list[str] = []
    provenance: list[str] = []
    for estimate in message_estimates:
        usage = usage.plus(estimate.usage)
        total_usd += estimate.total_usd
        components.extend(estimate.components)
        missing_reasons.extend(estimate.missing_reasons)
        provenance.extend(estimate.provenance)
    model_name, normalized_model = _dominant_model(message_estimates)
    missing_count = len(message_estimates) - len(priced)
    all_exact = bool(priced) and all(e.status == "exact" for e in priced)
    if all_exact:
        status: CostEstimateStatus = "exact"
    elif missing_count == 0:
        status = "priced"
    else:
        status = "partial"
    confidence = 0.95 if status == "exact" else 0.85 if status == "priced" else 0.55
    if conversation_estimate is not None and conversation_estimate.status == "priced":
        # Prefer conversation-level usage when present; it avoids double-counting
        # per-message fallbacks from providers that report only session totals.
        return conversation_estimate
    return CostEstimatePayload(
        provider_name=provider_name,
        conversation_id=str(conversation.id),
        model_name=model_name,
        normalized_model=normalized_model,
        status=status,
        confidence=confidence,
        total_usd=total_usd,
        usage=usage,
        components=tuple(components),
        missing_reasons=tuple(sorted(set(missing_reasons))),
        provenance=tuple(sorted(set(provenance))),
    )


def harmonize_session_cost(conversation: Conversation) -> tuple[float, bool]:
    """Return the legacy ``(cost_usd, is_estimated)`` session-cost tuple."""

    estimate = estimate_conversation_cost(conversation)
    return estimate.total_usd, estimate.status != "exact"


def generated_at() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "CATALOG_EFFECTIVE_DATE",
    "CATALOG_PROVENANCE",
    "LITELLM_PRICE_MAP_URL",
    "CostComponentPayload",
    "CostEstimatePayload",
    "CostEstimateStatus",
    "CostPricePayload",
    "CostUsagePayload",
    "ModelPricing",
    "PRICING",
    "_normalize_model",
    "estimate_conversation_cost",
    "estimate_cost",
    "estimate_message_cost",
    "generated_at",
    "harmonize_session_cost",
]
