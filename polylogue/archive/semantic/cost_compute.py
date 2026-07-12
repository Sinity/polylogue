"""Cost computation for session profiles from session data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.archive.semantic.cost_records import SessionCostBreakdown, SessionCostSummary
from polylogue.archive.semantic.pricing import (
    CATALOG_EFFECTIVE_DATE,
    CATALOG_PROVENANCE,
    _normalize_model,
    estimate_cost,
    estimate_session_cost,
)
from polylogue.archive.semantic.subscription_pricing import compute_credit_cost, credits_to_usd, get_credit_rate
from polylogue.archive.semantic.tokenizer import TOKENIZER_VERSION, estimate_tokens_from_words

if TYPE_CHECKING:
    from polylogue.archive.models import Session
    from polylogue.archive.semantic.pricing import CostEstimatePayload

# Stamped onto computed cost records so re-priced rows are distinguishable from
# rows priced under an earlier catalog. Derived from the canonical catalog
# constants in pricing.py (single source of truth) rather than hardcoded.
_PRICE_SNAPSHOT_VERSION = f"{CATALOG_PROVENANCE}-{CATALOG_EFFECTIVE_DATE}"


def compute_session_cost(
    session: Session,
    *,
    session_estimate: CostEstimatePayload | None = None,
    estimate_if_missing: bool = True,
) -> SessionCostSummary:
    """Compute per-model cost breakdown and aggregate cost summary."""
    estimate = session_estimate or (estimate_session_cost(session) if estimate_if_missing else None)
    if estimate is not None and estimate.status == "exact":
        return SessionCostSummary(
            total_input_tokens=estimate.usage.input_tokens,
            total_output_tokens=estimate.usage.output_tokens,
            total_cache_read_tokens=estimate.usage.cache_read_tokens,
            total_cache_write_tokens=estimate.usage.cache_write_tokens,
            total_api_cost_usd=round(estimate.total_usd, 6),
            total_credit_cost=0.0,
            total_subscription_equivalent_usd=round(
                estimate.basis.subscription_equivalent_usd,
                6,
            ),
            cost_provenance="provider_reported",
            cost_confidence="reported",
            tokenizer_version=TOKENIZER_VERSION,
            price_snapshot_version=_PRICE_SNAPSHOT_VERSION,
            per_model=(
                SessionCostBreakdown(
                    normalized_model=estimate.normalized_model,
                    provider_model_name=estimate.model_name,
                    input_tokens=estimate.usage.input_tokens,
                    output_tokens=estimate.usage.output_tokens,
                    cache_read_tokens=estimate.usage.cache_read_tokens,
                    cache_write_tokens=estimate.usage.cache_write_tokens,
                    total_tokens=estimate.usage.total_tokens,
                    api_cost_usd=round(estimate.total_usd, 6),
                    subscription_equivalent_usd=round(
                        estimate.basis.subscription_equivalent_usd,
                        6,
                    ),
                    confidence="reported",
                    provenance="provider_reported",
                ),
            ),
        )
    per_model: dict[str, SessionCostBreakdown] = {}

    for message in session.messages:
        model_name = _get_message_model_name(message)
        norm_model = _normalize_model(model_name) if model_name else None
        key = norm_model or "unknown"

        if key not in per_model:
            per_model[key] = SessionCostBreakdown(
                normalized_model=norm_model,
                provider_model_name=model_name,
            )

        tokens = _get_message_token_counts(message)
        word_count: int = getattr(message, "word_count", 0) or 0

        if tokens is not None and getattr(tokens, "billable_tokens", 0) > 0:
            per_model[key] = _add_provider_reported_tokens(per_model[key], tokens, model_name)
        elif word_count > 0:
            est = estimate_tokens_from_words(word_count)
            per_model[key] = SessionCostBreakdown(
                normalized_model=per_model[key].normalized_model,
                provider_model_name=per_model[key].provider_model_name,
                input_tokens=per_model[key].input_tokens + est.input_tokens,
                total_tokens=per_model[key].total_tokens + est.input_tokens,
                confidence="estimated",
                provenance="heuristic_estimated",
            )

    breakdowns: list[SessionCostBreakdown] = []
    total_api = 0.0
    total_credit = 0.0
    total_sub = 0.0
    agg_confidence = "reported"
    has_estimates = False

    for _key, breakdown in sorted(per_model.items()):
        norm = breakdown.normalized_model
        api_cost = 0.0
        credit_cost = 0.0

        if norm:
            api_cost = estimate_cost(
                input_tokens=breakdown.input_tokens,
                output_tokens=breakdown.output_tokens,
                model=norm,
                cache_read_tokens=breakdown.cache_read_tokens,
                cache_write_tokens=breakdown.cache_write_tokens,
            )
            credit_cost = float(
                compute_credit_cost(
                    norm,
                    breakdown.input_tokens,
                    breakdown.output_tokens,
                    breakdown.cache_read_tokens,
                    breakdown.cache_write_tokens,
                )
            )

        sub_equivalent = 0.0
        credit_rate = get_credit_rate(norm) if norm else None
        if credit_rate and credit_cost > 0:
            sub_equivalent = round(credits_to_usd(credit_cost), 6)

        updated = SessionCostBreakdown(
            normalized_model=norm,
            provider_model_name=breakdown.provider_model_name,
            input_tokens=breakdown.input_tokens,
            output_tokens=breakdown.output_tokens,
            cache_read_tokens=breakdown.cache_read_tokens,
            cache_write_tokens=breakdown.cache_write_tokens,
            total_tokens=breakdown.total_tokens,
            api_cost_usd=round(api_cost, 6),
            credit_cost=credit_cost,
            subscription_equivalent_usd=sub_equivalent,
            confidence=breakdown.confidence,
            provenance=breakdown.provenance,
        )
        breakdowns.append(updated)
        total_api += api_cost
        total_credit += credit_cost
        total_sub += sub_equivalent
        if updated.confidence == "estimated":
            has_estimates = True

    if not breakdowns:
        agg_confidence = "unknown"
    elif has_estimates:
        agg_confidence = "estimated"

    return SessionCostSummary(
        total_input_tokens=sum(b.input_tokens for b in breakdowns),
        total_output_tokens=sum(b.output_tokens for b in breakdowns),
        total_cache_read_tokens=sum(b.cache_read_tokens for b in breakdowns),
        total_cache_write_tokens=sum(b.cache_write_tokens for b in breakdowns),
        total_api_cost_usd=round(total_api, 6),
        total_credit_cost=round(total_credit, 2),
        total_subscription_equivalent_usd=round(total_sub, 6),
        cost_provenance="provider_reported" if agg_confidence == "reported" else "mixed",
        cost_confidence=agg_confidence,
        tokenizer_version=TOKENIZER_VERSION,
        price_snapshot_version=_PRICE_SNAPSHOT_VERSION,
        per_model=tuple(breakdowns),
    )


def _get_message_model_name(message: object) -> str | None:
    harmonized = getattr(message, "harmonized", None)
    if harmonized is not None:
        model = getattr(harmonized, "model", None)
        if model is not None:
            return str(model)
    model_name = getattr(message, "model_name", None)
    if model_name:
        return str(model_name)
    return None


def _get_message_token_counts(message: object) -> object | None:
    harmonized = getattr(message, "harmonized", None)
    if harmonized is not None:
        result: object | None = getattr(harmonized, "tokens", None)
        return result
    input_tokens = int(getattr(message, "input_tokens", 0) or 0)
    output_tokens = int(getattr(message, "output_tokens", 0) or 0)
    cache_read_tokens = int(getattr(message, "cache_read_tokens", 0) or 0)
    cache_write_tokens = int(getattr(message, "cache_write_tokens", 0) or 0)
    if input_tokens or output_tokens or cache_read_tokens or cache_write_tokens:
        from polylogue.archive.semantic.pricing import CostUsagePayload

        return CostUsagePayload(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            total_tokens=input_tokens + output_tokens + cache_read_tokens + cache_write_tokens,
        )
    return None


def _add_provider_reported_tokens(
    breakdown: SessionCostBreakdown, tokens: object, model_name: str | None
) -> SessionCostBreakdown:
    return SessionCostBreakdown(
        normalized_model=breakdown.normalized_model,
        provider_model_name=model_name or breakdown.provider_model_name,
        input_tokens=breakdown.input_tokens + int(getattr(tokens, "input_tokens", 0) or 0),
        output_tokens=breakdown.output_tokens + int(getattr(tokens, "output_tokens", 0) or 0),
        cache_read_tokens=breakdown.cache_read_tokens + int(getattr(tokens, "cache_read_tokens", 0) or 0),
        cache_write_tokens=breakdown.cache_write_tokens + int(getattr(tokens, "cache_write_tokens", 0) or 0),
        total_tokens=(
            breakdown.total_tokens
            + int(getattr(tokens, "input_tokens", 0) or 0)
            + int(getattr(tokens, "output_tokens", 0) or 0)
            + int(getattr(tokens, "cache_read_tokens", 0) or 0)
            + int(getattr(tokens, "cache_write_tokens", 0) or 0)
        ),
        confidence="reported",
        provenance="provider_reported",
    )


__all__ = ["compute_session_cost"]
