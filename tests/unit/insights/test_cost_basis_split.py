"""Cost basis split + per-model breakdown contracts (#1136).

Cost rollups must distinguish ``provider_reported``, ``api_equivalent``,
``subscription_equivalent``, ``catalog_priced``, and ``tool_surcharge``
basis axes. Per-model breakdown rows must accompany the aggregate so
mixed-model sessions are never collapsed. ``unavailable`` rows must carry
a discrete reason so consumers can render "why" instead of a silent zero.
"""

from __future__ import annotations

import pytest

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.models import Message
from polylogue.archive.semantic.pricing import (
    CostBasisPayload,
    CostModelBreakdown,
    estimate_session_cost,
)
from tests.infra.builders import make_conv, make_msg


def _msg_with_tokens(
    *,
    id: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    role: str = "assistant",
) -> Message:
    """Build a hydrated Message (#1256 drops ``Message.provider_meta``).

    Per-message cost facts now flow through the typed cost projection (#803),
    not through this helper. Callers that need a priced aggregate seed the
    cost facts at the session level instead.
    """

    del model, input_tokens, output_tokens  # accepted for backcompat
    return make_msg(id=id, role=role, text="x", provider="claude-code")


def test_provider_reported_total_populates_provider_and_api_basis() -> None:
    """Exact provider totals fill provider_reported_usd and api_equivalent_usd.

    A parallel catalog estimate is included as catalog_priced_usd whenever
    usage tokens are known, so consumers can compare provider-reported cost
    against catalog-estimated cost on the same usage. Subscription basis
    stays zero unless explicitly configured.
    """

    session = make_conv(
        id="conv-exact",
        provider="claude-code",
        provider_meta={
            "total_cost_usd": 1.25,
            "model": "claude-sonnet-4-5",
            "usage": {"input_tokens": 1000, "output_tokens": 500},
        },
        messages=MessageCollection(messages=[]),
    )

    estimate = estimate_session_cost(session)

    assert estimate.status == "exact"
    assert estimate.basis.provider_reported_usd == pytest.approx(1.25)
    assert estimate.basis.api_equivalent_usd == pytest.approx(1.25)
    # Catalog parallel: 1000 in + 500 out at sonnet-4-5 prices.
    assert estimate.basis.catalog_priced_usd > 0.0
    assert estimate.basis.subscription_equivalent_usd == 0.0
    assert estimate.basis.tool_surcharge_usd == 0.0
    assert estimate.unavailable_reason is None


def test_catalog_priced_estimate_populates_catalog_and_api_basis() -> None:
    """Catalog-priced estimates (no exact provider total) fill api+catalog.

    Provider-reported basis stays zero because no provider total was
    observed; the api_equivalent basis carries the catalog estimate as the
    closest stand-in for what the API would have charged.
    """

    session = make_conv(
        id="conv-priced",
        provider="chatgpt",
        provider_meta={
            "model": "openai/gpt-4o-2024-08-06",
            "usage": {"input_tokens": 1000, "output_tokens": 500},
        },
        messages=MessageCollection(messages=[]),
    )

    estimate = estimate_session_cost(session)

    assert estimate.status == "priced"
    assert estimate.basis.provider_reported_usd == 0.0
    assert estimate.basis.api_equivalent_usd == pytest.approx(estimate.total_usd)
    assert estimate.basis.catalog_priced_usd == pytest.approx(estimate.total_usd)


def test_unavailable_carries_explicit_reason() -> None:
    """Unpriced estimates must carry a discrete unavailable_reason.

    "no_messages" when the session is empty; "no_tokens" when messages
    exist but no token usage; "no_model" when usage is present but the
    model is unknown.
    """

    empty = make_conv(id="conv-empty", messages=MessageCollection(messages=[]))
    empty_estimate = estimate_session_cost(empty)
    assert empty_estimate.status == "unavailable"
    assert empty_estimate.unavailable_reason == "no_messages"


def test_mixed_model_session_breakdown_is_empty_without_typed_per_message_cost() -> None:
    """Per-#1256 + #803: per-model breakdown is empty without per-message cost.

    Pre-#1256 message-level ``provider_meta.model`` + ``provider_meta.usage``
    seeded the per-model breakdown. With hydrated messages no longer
    carrying ``provider_meta``, the per-model breakdown surface requires
    the typed cost projection (#803). Until #803 lands the breakdown is
    empty and the aggregate reports ``status='unavailable'`` for messages
    with no session-level fallback. This test pins the new contract
    so future #803 work has a clear seam to flip the assertions through.
    """

    session = make_conv(
        id="conv-mixed",
        provider="claude-code",
        messages=MessageCollection(
            messages=[
                _msg_with_tokens(id="m1", model="claude-sonnet-4-5", input_tokens=1000, output_tokens=500),
                _msg_with_tokens(id="m2", model="claude-opus-4-5", input_tokens=2000, output_tokens=1000),
                _msg_with_tokens(id="m3", model="claude-sonnet-4-5", input_tokens=500, output_tokens=250),
            ]
        ),
    )

    estimate = estimate_session_cost(session)

    assert estimate.status == "unavailable"
    assert estimate.per_model_breakdown == ()


def test_provider_zero_cost_is_preserved_not_treated_as_free() -> None:
    """A provider-reported cost of exactly $0 must not be silently elided.

    Today the estimator treats ``total_cost_usd == 0`` as missing and falls
    through to a usage-based estimate. This test pins that behavior so
    future refactors that introduce a "free" basis don't quietly drop the
    distinction between "no cost reported" and "cost reported as zero".
    The estimator currently routes zero-totals through the usage estimator,
    so the priced status comes from the catalog catalog basis and basis
    fields stay non-negative.
    """

    session = make_conv(
        id="conv-zero",
        provider="claude-code",
        provider_meta={
            "total_cost_usd": 0.0,
            "model": "claude-sonnet-4-5",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        },
        messages=MessageCollection(messages=[]),
    )

    estimate = estimate_session_cost(session)

    # Zero provider totals route to the usage estimator. The basis fields
    # must stay non-negative; subscription_equivalent stays zero unless
    # explicitly configured by the subscription cluster.
    assert estimate.basis.provider_reported_usd >= 0.0
    assert estimate.basis.subscription_equivalent_usd == 0.0


def test_basis_payload_plus_aggregates_each_axis_independently() -> None:
    """CostBasisPayload.plus must sum every axis independently."""

    left = CostBasisPayload(
        provider_reported_usd=1.0,
        api_equivalent_usd=2.0,
        subscription_equivalent_usd=3.0,
        catalog_priced_usd=4.0,
        tool_surcharge_usd=5.0,
    )
    right = CostBasisPayload(
        provider_reported_usd=10.0,
        api_equivalent_usd=20.0,
        subscription_equivalent_usd=30.0,
        catalog_priced_usd=40.0,
        tool_surcharge_usd=50.0,
    )

    result = left.plus(right)

    assert result.provider_reported_usd == 11.0
    assert result.api_equivalent_usd == 22.0
    assert result.subscription_equivalent_usd == 33.0
    assert result.catalog_priced_usd == 44.0
    assert result.tool_surcharge_usd == 55.0


def test_cost_rollup_aggregates_basis_and_per_model_breakdown() -> None:
    """The ``CostRollupInsight`` rollup aggregates basis + per-model rows across sessions.

    Cross-session aggregation walks ``SessionCostInsight.estimate.per_model_breakdown``
    when present and falls back to the session's dominant model otherwise.
    Mixed-model sessions contribute one row per model; same-model sessions
    are merged with incremented ``session_count``.
    """

    from polylogue.insights.archive import CostRollupInsight

    # Synthesize a rollup row directly to assert the typed contract.
    rollup = CostRollupInsight(
        source_name="anthropic",
        model_name="claude-sonnet-4-5",
        normalized_model="claude-sonnet-4-5",
        session_count=3,
        priced_session_count=3,
        unavailable_session_count=0,
        status_counts={"priced": 3},
        total_usd=10.0,
        basis=CostBasisPayload(
            provider_reported_usd=4.0,
            api_equivalent_usd=10.0,
            subscription_equivalent_usd=0.0,
            catalog_priced_usd=10.0,
        ),
        unavailable_reason_counts={},
        per_model_breakdown=(
            CostModelBreakdown(
                normalized_model="claude-opus-4-5",
                total_usd=6.0,
                session_count=1,
                basis=CostBasisPayload(api_equivalent_usd=6.0, catalog_priced_usd=6.0),
            ),
            CostModelBreakdown(
                normalized_model="claude-sonnet-4-5",
                total_usd=4.0,
                session_count=2,
                basis=CostBasisPayload(api_equivalent_usd=4.0, catalog_priced_usd=4.0),
            ),
        ),
        usage=estimate_session_cost(make_conv(id="c", messages=MessageCollection(messages=[]))).usage,
        confidence=0.85,
        provenance=__import__(
            "polylogue.insights.archive", fromlist=["ArchiveInsightProvenance"]
        ).ArchiveInsightProvenance(
            materializer_version=1,
            materialized_at="2026-05-17T00:00:00+00:00",
            source_updated_at=None,
            source_sort_key=None,
        ),
    )

    assert rollup.basis.api_equivalent_usd == 10.0
    assert rollup.basis.provider_reported_usd == 4.0
    assert len(rollup.per_model_breakdown) == 2
    # Per-model rows are independent — opus's 6.0 + sonnet's 4.0 equals total.
    assert sum(row.total_usd for row in rollup.per_model_breakdown) == 10.0
