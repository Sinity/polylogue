"""Typed provider/model cost estimation for archive sessions."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from polylogue.archive.semantic.facts import message_model_name, message_tokens
from polylogue.archive.viewport.viewports import TokenUsage
from polylogue.core.json import json_document

if TYPE_CHECKING:
    from polylogue.archive.models import Message, Session

CostEstimateStatus = Literal["exact", "priced", "partial", "unavailable"]

# Discrete reasons a session/message lacks a priced estimate. Surfaces in the
# typed cost read model so consumers can show "why" instead of an opaque
# `total_usd = 0.0`. See #1136.
CostUnavailableReason = Literal[
    "no_model",
    "no_price",
    "no_tokens",
    "provider_zero",
    "subscription_unconfigured",
    "no_messages",
]

# Discrete cost basis labels (#1136). A single estimate may carry non-zero
# values on more than one axis (e.g. provider-reported total AND a parallel
# catalog-priced reconciliation).
CostBasis = Literal[
    "provider_reported",
    "api_equivalent",
    "subscription_equivalent",
    "catalog_priced",
    "tool_surcharge",
]

LITELLM_PRICE_MAP_URL = "https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json"
CATALOG_PROVENANCE = "litellm-model-prices-vendored+polylogue-curated-overrides"
CATALOG_EFFECTIVE_DATE = "2026-06-27"


class PricingModel(BaseModel):
    """Base model for immutable cost-estimation payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


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
    source_name: str
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


class CostBasisPayload(PricingModel):
    """Cost split across the five basis axes (#1136).

    Each basis is independent; values do not sum to ``total_usd`` because the
    same usage may be expressed on multiple axes (e.g. a provider-reported
    total AND a parallel catalog-priced reconciliation for the same tokens).
    Consumers pick the basis that matches their question.
    """

    provider_reported_usd: float = 0.0
    api_equivalent_usd: float = 0.0
    subscription_equivalent_usd: float = 0.0
    catalog_priced_usd: float = 0.0
    tool_surcharge_usd: float = 0.0

    def plus(self, other: CostBasisPayload) -> CostBasisPayload:
        return CostBasisPayload(
            provider_reported_usd=self.provider_reported_usd + other.provider_reported_usd,
            api_equivalent_usd=self.api_equivalent_usd + other.api_equivalent_usd,
            subscription_equivalent_usd=self.subscription_equivalent_usd + other.subscription_equivalent_usd,
            catalog_priced_usd=self.catalog_priced_usd + other.catalog_priced_usd,
            tool_surcharge_usd=self.tool_surcharge_usd + other.tool_surcharge_usd,
        )


class CostModelBreakdown(PricingModel):
    """Per-model cost roll-up within a session or cross-session rollup (#1136).

    Keyed by ``normalized_model`` when available; falls back to
    ``model_name``. Sessions that mix models surface one row per model;
    nothing is collapsed.
    """

    model_name: str | None = None
    normalized_model: str | None = None
    usage: CostUsagePayload = Field(default_factory=CostUsagePayload)
    basis: CostBasisPayload = Field(default_factory=CostBasisPayload)
    total_usd: float = 0.0
    session_count: int = 0


class CostEstimatePayload(PricingModel):
    source_name: str
    session_id: str | None = None
    message_id: str | None = None
    model_name: str | None = None
    normalized_model: str | None = None
    status: CostEstimateStatus
    confidence: float = 0.0
    currency: str = "USD"
    # ``total_usd`` remains the legacy single-number summary for
    # backwards-compat. It draws from ``basis.provider_reported_usd`` when an
    # exact provider total is present, otherwise from ``basis.catalog_priced_usd``.
    total_usd: float = 0.0
    basis: CostBasisPayload = Field(default_factory=CostBasisPayload)
    usage: CostUsagePayload = Field(default_factory=CostUsagePayload)
    price: CostPricePayload | None = None
    components: tuple[CostComponentPayload, ...] = ()
    per_model_breakdown: tuple[CostModelBreakdown, ...] = ()
    missing_reasons: tuple[str, ...] = ()
    unavailable_reason: CostUnavailableReason | None = None
    provenance: tuple[str, ...] = ()

    @property
    def priced(self) -> bool:
        return self.status in {"exact", "priced", "partial"}


@dataclass(frozen=True)
class ModelPricing:
    """Per-million-token pricing for a normalized model identifier."""

    source_name: str
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
            source_name=self.source_name,
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


# Hand-verified overrides (USD per 1M tokens). These win over the vendored
# LiteLLM catalog (built in `_load_litellm_catalog`), so Anthropic cache rates
# we've checked stay exact even if upstream drifts. Provider-reported exact
# archive costs still take precedence over both.
_CURATED_PRICING: dict[str, ModelPricing] = {
    "claude-opus-4-8": ModelPricing("anthropic", 15.0, 75.0, 1.5, 18.75),
    "claude-opus-4-7": ModelPricing("anthropic", 15.0, 75.0, 1.5, 18.75),
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


def _load_litellm_catalog() -> dict[str, ModelPricing]:
    """Build a ModelPricing catalog from the vendored LiteLLM price map.

    LiteLLM's ``model_prices_and_context_window.json`` covers ~all current
    models (incl. gpt-5.x / codex / deepseek) with per-token input/output and
    cache costs. We vendor it at data/litellm_model_prices.json and convert
    per-token to per-1M. Refresh with::

        curl -sL https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json \
          -o polylogue/archive/semantic/data/litellm_model_prices.json

    Keys are stored both fully-qualified (``openai/gpt-5.4``) and bare
    (``gpt-5.4``); on a bare-name collision a non-Azure provider wins so
    generic lookups get list pricing.
    """
    path = Path(__file__).parent / "data" / "litellm_model_prices.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    catalog: dict[str, ModelPricing] = {}
    for key, entry in raw.items():
        # Skip the meta spec row and any blank key — an empty key would be a
        # prefix of every model name and silently price unknown models.
        if not key or not key.strip() or key == "sample_spec":
            continue
        if not isinstance(entry, dict):
            continue
        ic = entry.get("input_cost_per_token")
        oc = entry.get("output_cost_per_token")
        if not isinstance(ic, (int, float)) or not isinstance(oc, (int, float)):
            continue
        provider = str(entry.get("litellm_provider") or "litellm")
        pricing = ModelPricing(
            source_name=provider,
            input_usd_per_1m=float(ic) * 1_000_000,
            output_usd_per_1m=float(oc) * 1_000_000,
            cache_read_usd_per_1m=float(entry.get("cache_read_input_token_cost") or 0.0) * 1_000_000,
            cache_write_usd_per_1m=float(entry.get("cache_creation_input_token_cost") or 0.0) * 1_000_000,
            source_url=LITELLM_PRICE_MAP_URL,
            provenance="litellm-model-prices-vendored",
        )
        catalog[key] = pricing
        bare = key.split("/")[-1]
        if bare and (bare not in catalog or not provider.startswith("azure")):
            catalog[bare] = pricing
    return catalog


# Full catalog: vendored LiteLLM map as the base, hand-verified entries override.
PRICING: dict[str, ModelPricing] = {**_load_litellm_catalog(), **_CURATED_PRICING}

_PRICING_KEYS_DESC = tuple(sorted(PRICING, key=len, reverse=True))


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
    # Canonicalize trailing date snapshots to the base model so cost rollups
    # don't fragment by release date (e.g. gpt-4o-2024-08-06 -> gpt-4o,
    # claude-opus-4-8-20260101 -> claude-opus-4-8). Done before the exact-match
    # lookup because the vendored LiteLLM catalog carries dated keys too.
    lowered = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", lowered)
    lowered = re.sub(r"-\d{8}$", "", lowered)
    if lowered in PRICING:
        return lowered
    for key in _PRICING_KEYS_DESC:
        if lowered.startswith(key):
            return key
    return lowered


# polylogue-4c27: pure model-name pattern matching for the semantic vendor
# family. Deliberately independent of PRICING/ModelPricing.source_name --
# that field is *pricing-catalog provenance* (for the vendored LiteLLM
# catalog it is the literal `litellm_provider` routing tag, e.g.
# "vertex_ai-anthropic_models", "bedrock_converse", "openrouter",
# "azure_ai" -- whichever catalog row happened to win a bare-name
# collision), not a stable vendor identity. Reusing it as "family" was the
# defect: `canonical_model_family("claude-fable-5")` returned
# "vertex_ai-anthropic_models" and `canonical_model_family("gemini-2.0-flash-001")`
# returned "openrouter" -- neither is Anthropic/Google, they are catalog
# routing tags for models that plainly belong to those vendors. Ordered
# most-specific-first; matched via substring on the normalized model name so
# vendor-prefixed or routed forms (e.g. "vertex_ai/claude-haiku-4-5",
# "openrouter/anthropic/claude-3-haiku") still resolve.
_VENDOR_MARKERS: tuple[tuple[str, str], ...] = (
    ("claude", "anthropic"),
    ("gemini", "google"),
    ("palm-", "google"),
    ("bard", "google"),
    ("deepseek", "deepseek"),
    ("grok", "xai"),
    ("codestral", "mistral"),
    ("mixtral", "mistral"),
    ("mistral", "mistral"),
    ("llama", "meta"),
    ("qwen", "alibaba"),
    ("kimi", "moonshot"),
    ("moonshot", "moonshot"),
    ("glm-", "zhipu"),
    ("command-", "cohere"),
    ("chatgpt", "openai"),
    ("codex", "openai"),
    ("gpt-", "openai"),
    ("gpt5", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
)


def semantic_model_vendor(model_name: str | None) -> str | None:
    """Map a raw provider model name to its semantic vendor family
    (anthropic, openai, google, deepseek, ...) via pure model-name pattern
    matching -- never via which pricing-catalog entry happens to match (see
    the module-level note above `_VENDOR_MARKERS`). Unknown/unrecognized
    names return None rather than a guess."""
    if not model_name:
        return None
    normalized = _normalize_model(model_name)
    if not normalized:
        return None
    for marker, vendor in _VENDOR_MARKERS:
        if marker in normalized:
            return vendor
    return None


# polylogue-4c27: coarse model-line grouping within a vendor (e.g. Anthropic's
# opus/sonnet/haiku/fable tiers, OpenAI's gpt-5/gpt-4/o-series/codex lines).
# Best-effort and deliberately conservative: returns None rather than
# fabricating a line for a name it doesn't recognize.
_MODEL_LINE_MARKERS: tuple[tuple[str, str], ...] = (
    ("claude-opus", "opus"),
    ("claude-sonnet", "sonnet"),
    ("claude-haiku", "haiku"),
    ("claude-fable", "fable"),
    ("codex", "codex"),
    ("gpt-5", "gpt-5"),
    ("gpt-4", "gpt-4"),
    ("gpt-3", "gpt-3"),
    ("o1", "o1"),
    ("o3", "o3"),
    ("o4", "o4"),
    ("gemini-2.5", "gemini-2.5"),
    ("gemini-2.0", "gemini-2.0"),
    ("gemini-1.5", "gemini-1.5"),
    ("deepseek-chat", "deepseek-chat"),
    ("deepseek-reasoner", "deepseek-reasoner"),
)


def semantic_model_line(model_name: str | None) -> str | None:
    """Best-effort coarse model-line grouping within a vendor family (e.g.
    "opus"/"sonnet"/"haiku"/"fable" within Anthropic). Unknown stays None."""
    if not model_name:
        return None
    normalized = _normalize_model(model_name)
    if not normalized:
        return None
    for marker, line in _MODEL_LINE_MARKERS:
        if marker in normalized:
            return line
    return None


def pricing_catalog_source(model_name: str | None) -> str | None:
    """The pricing catalog's own provenance tag for a model (the vendored
    LiteLLM `litellm_provider` field, or a curated override's source label).
    This is catalog/routing provenance, NOT a semantic vendor family -- see
    `semantic_model_vendor`. Named explicitly so a cost-lookup caller that
    genuinely wants "which catalog entry priced this" has an honest entry
    point instead of overloading `canonical_model_family`."""
    if not model_name:
        return None
    normalized = _normalize_model(model_name)
    pricing = PRICING.get(normalized)
    return pricing.source_name if pricing is not None else None


def canonical_model_family(model_name: str | None) -> str | None:
    """Map a raw provider model name to its canonical semantic family
    (anthropic, openai, deepseek, ...). Delegates to `semantic_model_vendor`
    -- pure model-name pattern matching, independent of the pricing catalog
    (1vpm.1 named this the enabling primitive for the `delegations` view's
    model identity; polylogue-4c27 fixed it to stop returning the pricing
    catalog's routing-tag provenance, which drifted from vendor identity on
    every bare-name catalog collision)."""
    return semantic_model_vendor(model_name)


ModelAttributionSource = Literal[
    "dispatch_turn",
    "requested",
    "child_observed",
    "session_dominant_fallback",
    "unknown",
]

ModelIdentityConfidence = Literal["exact", "normalized", "unknown"]


@dataclass(frozen=True)
class ModelIdentity:
    """polylogue-4c27: one shared model identity projection over a single raw
    provider model-name value, tagged with WHERE that value was attributed
    from. Callers combine several of these (one per raw column) instead of
    the archive collapsing dispatch-turn authorship, requested routing,
    observed child execution, and session-dominant fallback into a single
    "orchestrator model" -- construct-validity requires keeping them
    separate even when they happen to agree."""

    raw_model_name: str | None
    attribution_source: ModelAttributionSource
    normalized_model: str | None = None
    vendor: str | None = None
    model_line: str | None = None
    pricing_source: str | None = None
    confidence: ModelIdentityConfidence = "unknown"


def resolve_model_identity(
    model_name: str | None,
    *,
    attribution_source: ModelAttributionSource,
) -> ModelIdentity:
    """Build a `ModelIdentity` for one raw model-name value, explicit about
    which turn/scope it was attributed from. `model_name=None` (unsupported
    attribution, e.g. a provider that never records a dispatch-turn model)
    stays unknown rather than falling back to a different scope's value --
    callers that want a fallback must resolve a *different* raw column
    (e.g. the session-dominant model) and label it accordingly."""
    if not model_name:
        return ModelIdentity(raw_model_name=None, attribution_source=attribution_source, confidence="unknown")
    normalized = _normalize_model(model_name)
    vendor = semantic_model_vendor(model_name)
    return ModelIdentity(
        raw_model_name=model_name,
        attribution_source=attribution_source,
        normalized_model=normalized or None,
        vendor=vendor,
        model_line=semantic_model_line(model_name),
        pricing_source=pricing_catalog_source(model_name),
        confidence="exact" if normalized in PRICING else ("normalized" if vendor is not None else "unknown"),
    )


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
    source_name: str,
    total_usd: float,
    session_id: str | None = None,
    message_id: str | None = None,
    model_name: str | None = None,
    usage: CostUsagePayload | None = None,
) -> CostEstimatePayload:
    normalized_model = _normalize_model(model_name) if model_name else None
    effective_usage = usage or CostUsagePayload()
    # Parallel catalog estimate when a price exists, so consumers can compare
    # provider-reported cost against catalog-estimated cost on identical usage.
    catalog_usd = 0.0
    if normalized_model is not None:
        pricing = PRICING.get(normalized_model)
        if pricing is not None:
            catalog_usd = sum(component.usd for component in _cost_components(effective_usage, pricing))
    basis = CostBasisPayload(
        provider_reported_usd=total_usd,
        api_equivalent_usd=total_usd,
        catalog_priced_usd=catalog_usd,
    )
    return CostEstimatePayload(
        source_name=source_name,
        session_id=session_id,
        message_id=message_id,
        model_name=model_name,
        normalized_model=normalized_model,
        status="exact",
        confidence=1.0,
        total_usd=total_usd,
        basis=basis,
        usage=effective_usage,
        components=(CostComponentPayload(name="provider_reported_total", usd=total_usd),),
        provenance=("archive_provider_reported_cost",),
    )


def _estimate_from_usage(
    *,
    source_name: str,
    model_name: str | None,
    usage: CostUsagePayload,
    session_id: str | None = None,
    message_id: str | None = None,
    provenance: tuple[str, ...],
) -> CostEstimatePayload:
    if model_name is None or not model_name.strip():
        return CostEstimatePayload(
            source_name=source_name,
            session_id=session_id,
            message_id=message_id,
            status="unavailable",
            confidence=0.0,
            usage=usage,
            missing_reasons=("missing_model",),
            unavailable_reason="no_model",
            provenance=provenance,
        )
    if usage.billable_tokens <= 0:
        return CostEstimatePayload(
            source_name=source_name,
            session_id=session_id,
            message_id=message_id,
            model_name=model_name,
            normalized_model=_normalize_model(model_name),
            status="unavailable",
            confidence=0.0,
            usage=usage,
            missing_reasons=("missing_token_usage",),
            unavailable_reason="no_tokens",
            provenance=provenance,
        )
    normalized_model = _normalize_model(model_name)
    pricing = PRICING.get(normalized_model)
    if pricing is None:
        return CostEstimatePayload(
            source_name=source_name,
            session_id=session_id,
            message_id=message_id,
            model_name=model_name,
            normalized_model=normalized_model,
            status="unavailable",
            confidence=0.0,
            usage=usage,
            missing_reasons=("missing_price",),
            unavailable_reason="no_price",
            provenance=provenance,
        )
    components = _cost_components(usage, pricing)
    total = sum(component.usd for component in components)
    # Flag a paid model that carries cache tokens but has no catalog cache rate:
    # _cost_components priced that lane at $0 silently, which understates cost.
    # The likely cause is a model new enough that the vendored LiteLLM snapshot
    # lacks its cache pricing. Surface it as a data-quality reason rather than a
    # hidden understatement. Gated on the model being paid so genuinely-free
    # models (e.g. local-llama, all lanes $0) are not flagged.
    missing_reasons: tuple[str, ...] = ()
    if pricing.input_usd_per_1m > 0 or pricing.output_usd_per_1m > 0:
        unpriced: list[str] = []
        if usage.cache_read_tokens > 0 and pricing.cache_read_usd_per_1m == 0.0:
            unpriced.append("missing_cache_read_price")
        if usage.cache_write_tokens > 0 and pricing.cache_write_usd_per_1m == 0.0:
            unpriced.append("missing_cache_write_price")
        missing_reasons = tuple(unpriced)
    return CostEstimatePayload(
        source_name=source_name,
        session_id=session_id,
        message_id=message_id,
        model_name=model_name,
        normalized_model=normalized_model,
        status="priced",
        confidence=0.85,
        total_usd=total,
        basis=CostBasisPayload(
            api_equivalent_usd=total,
            catalog_priced_usd=total,
        ),
        usage=usage,
        price=pricing.to_payload(model_name=model_name, normalized_model=normalized_model),
        components=components,
        provenance=(*provenance, CATALOG_PROVENANCE),
        missing_reasons=missing_reasons,
    )


def estimate_message_cost(
    message: Message,
    *,
    source_name: str,
    session_id: str | None = None,
    fallback_model: str | None = None,
) -> CostEstimatePayload:
    """Estimate one message cost with explicit uncertainty.

    Typed cost/usage facts come from message columns and the
    ``message_token_usage`` provenance. Provider-reported session totals are
    stored in typed archive cost rows and are not read from hydrated sessions.
    """

    model_name = (
        message_model_name(message)
        or str(getattr(message, "model_name", "") or "").strip()
        or (fallback_model.strip() if fallback_model else None)
        or None
    )
    usage = _token_usage_payload(message_tokens(message))
    if usage.billable_tokens <= 0:
        usage = CostUsagePayload(
            input_tokens=_coerce_int(getattr(message, "input_tokens", 0)),
            output_tokens=_coerce_int(getattr(message, "output_tokens", 0)),
            cache_read_tokens=_coerce_int(getattr(message, "cache_read_tokens", 0)),
            cache_write_tokens=_coerce_int(getattr(message, "cache_write_tokens", 0)),
        )
    return _estimate_from_usage(
        source_name=source_name,
        session_id=session_id,
        message_id=str(message.id),
        model_name=model_name,
        usage=usage,
        provenance=("message_token_usage",),
    )


def _session_level_estimate(session: Session) -> CostEstimatePayload | None:
    del session
    return None


def _dominant_model(estimates: Iterable[CostEstimatePayload]) -> tuple[str | None, str | None]:
    counts: Counter[tuple[str | None, str | None]] = Counter()
    for estimate in estimates:
        if estimate.model_name or estimate.normalized_model:
            counts[(estimate.model_name, estimate.normalized_model)] += 1
    if not counts:
        return None, None
    return counts.most_common(1)[0][0]


def estimate_session_cost(session: Session) -> CostEstimatePayload:
    """Estimate cost for a session/session with confidence metadata."""

    source_name = _provider_text(session.origin)
    session_estimate = _session_level_estimate(session)
    if session_estimate is not None and session_estimate.status == "exact":
        return session_estimate

    message_estimates = [
        estimate_message_cost(
            message,
            source_name=source_name,
            session_id=str(session.id),
            fallback_model=session_estimate.model_name if session_estimate else None,
        )
        for message in session.messages
    ]
    priced = [estimate for estimate in message_estimates if estimate.priced]
    if not message_estimates and session_estimate is not None:
        return session_estimate
    if not priced and session_estimate is not None:
        return session_estimate
    if not priced:
        missing = ("missing_token_usage",) if message_estimates else ("no_messages",)
        unavailable: CostUnavailableReason = "no_tokens" if message_estimates else "no_messages"
        return CostEstimatePayload(
            source_name=source_name,
            session_id=str(session.id),
            status="unavailable",
            confidence=0.0,
            missing_reasons=missing,
            unavailable_reason=unavailable,
            provenance=("session_messages",),
        )

    usage = CostUsagePayload()
    total_usd = 0.0
    basis = CostBasisPayload()
    components: list[CostComponentPayload] = []
    missing_reasons: list[str] = []
    provenance: list[str] = []
    per_model_map: dict[tuple[str | None, str | None], CostModelBreakdown] = {}
    for estimate in message_estimates:
        usage = usage.plus(estimate.usage)
        total_usd += estimate.total_usd
        basis = basis.plus(estimate.basis)
        components.extend(estimate.components)
        missing_reasons.extend(estimate.missing_reasons)
        provenance.extend(estimate.provenance)
        key = (estimate.model_name, estimate.normalized_model)
        existing = per_model_map.get(key)
        if existing is None:
            per_model_map[key] = CostModelBreakdown(
                model_name=estimate.model_name,
                normalized_model=estimate.normalized_model,
                usage=estimate.usage,
                basis=estimate.basis,
                total_usd=estimate.total_usd,
                session_count=1,
            )
        else:
            per_model_map[key] = CostModelBreakdown(
                model_name=existing.model_name,
                normalized_model=existing.normalized_model,
                usage=existing.usage.plus(estimate.usage),
                basis=existing.basis.plus(estimate.basis),
                total_usd=existing.total_usd + estimate.total_usd,
                session_count=existing.session_count,
            )
    per_model_breakdown = tuple(
        sorted(
            per_model_map.values(),
            key=lambda entry: entry.total_usd,
            reverse=True,
        )
    )
    model_name, normalized_model = _dominant_model(message_estimates)
    missing_count = len(message_estimates) - len(priced)
    all_exact = missing_count == 0 and bool(priced) and all(e.status == "exact" for e in priced)
    if all_exact:
        status: CostEstimateStatus = "exact"
    elif missing_count == 0:
        status = "priced"
    else:
        status = "partial"
    confidence = 0.95 if status == "exact" else 0.85 if status == "priced" else 0.55
    if session_estimate is not None and session_estimate.status == "priced":
        # Prefer session-level usage when present; it avoids double-counting
        # per-message fallbacks from providers that report only session totals.
        return session_estimate
    return CostEstimatePayload(
        source_name=source_name,
        session_id=str(session.id),
        model_name=model_name,
        normalized_model=normalized_model,
        status=status,
        confidence=confidence,
        total_usd=total_usd,
        basis=basis,
        usage=usage,
        components=tuple(components),
        per_model_breakdown=per_model_breakdown,
        missing_reasons=tuple(sorted(set(missing_reasons))),
        provenance=tuple(sorted(set(provenance))),
    )


def harmonize_session_cost(session: Session) -> tuple[float, bool]:
    """Return the legacy ``(cost_usd, is_estimated)`` session-cost tuple."""

    estimate = estimate_session_cost(session)
    return estimate.total_usd, estimate.status != "exact"


def generated_at() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "CATALOG_EFFECTIVE_DATE",
    "CATALOG_PROVENANCE",
    "LITELLM_PRICE_MAP_URL",
    "CostBasis",
    "CostBasisPayload",
    "CostComponentPayload",
    "CostEstimatePayload",
    "CostEstimateStatus",
    "CostModelBreakdown",
    "CostPricePayload",
    "CostUnavailableReason",
    "CostUsagePayload",
    "ModelPricing",
    "PRICING",
    "_normalize_model",
    "estimate_session_cost",
    "estimate_cost",
    "estimate_message_cost",
    "generated_at",
    "harmonize_session_cost",
]
