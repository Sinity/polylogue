"""Cost computation for session profiles from session data."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING

from polylogue.archive.semantic.cost_records import ModelUsageTotals, SessionCostBreakdown, SessionCostSummary
from polylogue.archive.semantic.pricing import (
    CATALOG_EFFECTIVE_DATE,
    CATALOG_PROVENANCE,
    _normalize_model,
    estimate_cost,
    estimate_session_cost,
    pricing_catalog_source,
)
from polylogue.archive.semantic.subscription_pricing import compute_credit_cost, credits_to_usd, get_credit_rate
from polylogue.archive.semantic.tokenizer import TOKENIZER_VERSION, estimate_tokens_from_words_split
from polylogue.core.enums import Role

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
    model_usage: Sequence[ModelUsageTotals] | None = None,
    subscription_tier: str | None = None,
) -> SessionCostSummary:
    """Compute per-model cost breakdown and aggregate cost summary.

    ``model_usage`` is the canonical per-model tally from
    ``session_model_usage`` (polylogue-r7p6). When supplied and non-empty it
    is the sole source of per-model token counts -- provider-neutrally, for
    every origin, not just Codex -- because that table is already the single
    substrate the archive's own cost/usage rollups are built from. Per-message
    fields on ``session.messages`` are read only as a fallback for callers
    that build an ad hoc profile without a materialized ``session_model_usage``
    to hand (e.g. a query-time estimate for a session with no persisted
    profile row yet); Codex rarely populates those per-message fields at all
    (its real usage arrives as periodic cumulative ``token_count`` events, not
    per-message ``usage`` blocks), which is what made the per-message fallback
    ~1000x too small for Codex sessions when it was the only source.

    ``subscription_tier`` selects the plan whose ``monthly_fee_usd /
    credit_pool`` ratio prices the subscription-equivalent figure
    (:func:`polylogue.archive.semantic.subscription_pricing.credits_to_usd`).
    ``None`` keeps the conservative ``pro`` default rather than guessing --
    callers that know the archive owner's actual plan (e.g. one that has
    read the ``subscription_tier`` :mod:`user_settings` row, polylogue-at44)
    should pass it explicitly.
    """
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
    per_model: dict[str, SessionCostBreakdown] = (
        _per_model_from_model_usage(model_usage) if model_usage else _per_model_from_messages(session)
    )
    if model_usage and not any(breakdown.total_tokens for breakdown in per_model.values()):
        # session_model_usage carries model identity but no real usage
        # counters for every row (e.g. chatgpt-export/claude-ai-export
        # exports, which don't carry provider token counters -- see
        # docs/cost-model.md's estimate-only disposition for those origins).
        # Fall back to the text-length heuristic over messages instead of
        # reporting a hollow zero-token "reported" breakdown (polylogue-9kjtc).
        message_based = _per_model_from_messages(session)
        if message_based:
            per_model = message_based

    breakdowns: list[SessionCostBreakdown] = []
    total_api = 0.0
    total_credit = 0.0
    total_sub = 0.0
    agg_confidence = "reported"
    has_estimates = False
    has_reported = False
    has_unknown = False

    for _key, breakdown in sorted(per_model.items()):
        norm = breakdown.normalized_model
        api_cost = 0.0
        credit_cost = 0.0
        catalog_priced = norm is not None and pricing_catalog_source(norm) is not None

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
            sub_equivalent = round(credits_to_usd(credit_cost, tier=subscription_tier or "pro"), 6)

        # A model with real, non-zero reported/estimated tokens but no catalog
        # price entry (e.g. a newly-released model the pricing catalog hasn't
        # caught up with) must not surface as a confidently-priced $0.00 --
        # estimate_cost() silently returns 0.0 for an unpriced model, which is
        # indistinguishable from a genuinely free one unless the confidence is
        # downgraded here (polylogue-iuyr).
        effective_confidence = breakdown.confidence
        if norm and not catalog_priced and breakdown.confidence in ("reported", "estimated"):
            effective_confidence = "unknown"

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
            confidence=effective_confidence,
            provenance=breakdown.provenance,
        )
        breakdowns.append(updated)
        total_api += api_cost
        total_credit += credit_cost
        total_sub += sub_equivalent
        if updated.confidence == "estimated":
            has_estimates = True
        elif updated.confidence == "reported":
            has_reported = True
        elif updated.confidence == "unknown":
            has_unknown = True

    if not breakdowns:
        agg_confidence = "unknown"
    elif has_unknown and (has_reported or has_estimates):
        # A mixed session -- e.g. one model with genuine provider-reported or
        # estimated tokens alongside another model whose only evidence is a
        # zero-token _seed_session_model_usage_rows() skeleton row -- must not
        # read as a clean "reported"/"estimated" aggregate. Some of the
        # session's cost is real evidence and some of it is unaccounted for,
        # which is exactly what "partial" means (polylogue-3b607 P2: without
        # this branch, has_reported=True from the genuine row silently
        # overrode the fact that another per-model breakdown was separately
        # marked unknown).
        agg_confidence = "partial"
    elif has_estimates:
        agg_confidence = "estimated"
    elif not has_reported:
        # Every breakdown is a model-identity-only fact with no real token
        # evidence and no text-length estimate to fall back on -- honest
        # disposition is "unknown", not "reported" (polylogue-9kjtc).
        agg_confidence = "unknown"

    if agg_confidence == "reported":
        cost_provenance = "provider_reported"
    elif agg_confidence == "unknown":
        cost_provenance = "unknown"
    else:
        cost_provenance = "mixed"

    return SessionCostSummary(
        total_input_tokens=sum(b.input_tokens for b in breakdowns),
        total_output_tokens=sum(b.output_tokens for b in breakdowns),
        total_cache_read_tokens=sum(b.cache_read_tokens for b in breakdowns),
        total_cache_write_tokens=sum(b.cache_write_tokens for b in breakdowns),
        total_api_cost_usd=round(total_api, 6),
        total_credit_cost=round(total_credit, 2),
        total_subscription_equivalent_usd=round(total_sub, 6),
        cost_provenance=cost_provenance,
        cost_confidence=agg_confidence,
        tokenizer_version=TOKENIZER_VERSION,
        price_snapshot_version=_PRICE_SNAPSHOT_VERSION,
        per_model=tuple(breakdowns),
    )


def _per_model_from_model_usage(model_usage: Sequence[ModelUsageTotals]) -> dict[str, SessionCostBreakdown]:
    """Build the per-model breakdown seed from canonical ``session_model_usage`` rows.

    A row's existence proves a model-identity fact (an ingested usage event or
    message asserted this model was used), but not that the row carries real
    usage *counts* -- some origins (chatgpt-export, claude-ai-export) write a
    row with a real ``model_name`` and zero token counters because their
    exports don't carry provider token counters at all. Only rows whose
    aggregated tokens are actually nonzero are labelled ``reported``/
    ``provider_reported``; a model-identity-only row with no real token
    evidence is labelled ``unknown``/``unknown`` instead, so it can't be
    mistaken for a provider that genuinely reported and billed zero
    (polylogue-9kjtc).
    """
    per_model: dict[str, SessionCostBreakdown] = {}
    for row in model_usage:
        model_name = row.model_name or None
        norm_model = _normalize_model(model_name) if model_name else None
        key = norm_model or "unknown"
        existing = per_model.get(key)
        base_input = existing.input_tokens if existing else 0
        base_output = existing.output_tokens if existing else 0
        base_cache_read = existing.cache_read_tokens if existing else 0
        base_cache_write = existing.cache_write_tokens if existing else 0
        base_total = existing.total_tokens if existing else 0
        per_model[key] = SessionCostBreakdown(
            normalized_model=norm_model,
            provider_model_name=model_name or (existing.provider_model_name if existing else None),
            input_tokens=base_input + row.input_tokens,
            output_tokens=base_output + row.output_tokens,
            cache_read_tokens=base_cache_read + row.cache_read_tokens,
            cache_write_tokens=base_cache_write + row.cache_write_tokens,
            total_tokens=(
                base_total + row.input_tokens + row.output_tokens + row.cache_read_tokens + row.cache_write_tokens
            ),
            confidence="reported",
            provenance="provider_reported",
        )
    for key, breakdown in per_model.items():
        if breakdown.total_tokens == 0:
            per_model[key] = breakdown.model_copy(update={"confidence": "unknown", "provenance": "unknown"})
    return per_model


def _per_model_from_messages(session: Session) -> dict[str, SessionCostBreakdown]:
    """Fallback: estimate per-model tokens by walking ``session.messages``.

    Only used when no ``session_model_usage`` rows are supplied (e.g. an ad
    hoc profile built without a materialized cost/usage rollup to read). Real
    per-message ``input_tokens``/``output_tokens`` fields are sparse-to-absent
    for Codex, which is why this path historically undercounted Codex sessions
    by ~1000x when it was the only source (polylogue-r7p6).

    Two things this must get right for the estimate-only path (polylogue-3b607,
    fixing bugs introduced by polylogue-9kjtc):

    - Messages with no per-message declared model (routine for ChatGPT/Claude
      web *user* turns -- only the assistant turn carries ``model_slug``) fall
      back to the session's dominant declared model instead of an unpriced
      "unknown" bucket. This reuses the same Counter/``most_common`` dominant-
      model pattern ``pricing.py``'s ``_session_level_estimate`` uses, rather
      than inventing a new heuristic.
    - Estimated word-count tokens are classified by the message's role: user/
      human turns are ``input_tokens``, assistant turns are ``output_tokens``.
      Dumping everything into ``input_tokens`` regardless of role systematically
      understated cost, since output tokens are typically priced higher.
    """
    per_model: dict[str, SessionCostBreakdown] = {}

    model_counts: Counter[str] = Counter()
    for message in session.messages:
        declared_model = _get_message_model_name(message)
        if declared_model:
            model_counts[declared_model] += 1
    dominant_model_name = model_counts.most_common(1)[0][0] if model_counts else None

    for message in session.messages:
        model_name = _get_message_model_name(message) or dominant_model_name
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
            is_assistant_turn = getattr(message, "role", None) == Role.ASSISTANT
            est = estimate_tokens_from_words_split(
                input_words=0 if is_assistant_turn else word_count,
                output_words=word_count if is_assistant_turn else 0,
            )
            per_model[key] = SessionCostBreakdown(
                normalized_model=per_model[key].normalized_model,
                provider_model_name=per_model[key].provider_model_name,
                input_tokens=per_model[key].input_tokens + est.input_tokens,
                output_tokens=per_model[key].output_tokens + est.output_tokens,
                total_tokens=per_model[key].total_tokens + est.total_tokens,
                confidence="estimated",
                provenance="heuristic_estimated",
            )
    return per_model


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
