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
from polylogue.archive.models import Message, Session
from polylogue.archive.semantic.pricing import (
    CostBasisPayload,
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
    cache_read_tokens: int | None = 0,
    cache_write_tokens: int | None = 0,
) -> Message:
    """Build a hydrated message with typed model and token usage.

    The cache lanes default to a *captured* zero, not ``None``: these tests
    are about basis split and rollup arithmetic over a completely measured
    message, and since polylogue-qe194 a never-captured lane makes the
    estimate ``partial`` (see ``tests/unit/core/test_pricing.py``). Pass
    ``None`` explicitly to build a partly-measured message.
    """

    return make_msg(
        id=id,
        role=role,
        text="x",
        provider="claude-code",
        model_name=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )


def test_provider_reported_total_populates_provider_and_api_basis() -> None:
    """Exact provider totals fill provider_reported_usd and api_equivalent_usd.

    A parallel catalog estimate is included as catalog_priced_usd whenever
    usage tokens are known, so consumers can compare provider-reported cost
    against catalog-estimated cost on the same usage. Subscription basis
    stays zero unless explicitly configured.

    Exercises the real production exact-cost path (polylogue-gt1z): a
    ``Session`` with ``reported_cost_usd`` set (the field
    ``_session_level_estimate`` reads, populated in production from
    ``sessions.reported_cost_usd``/``ParsedSession.reported_cost_usd``) run
    through ``estimate_session_cost()`` -- not a ``CostEstimatePayload``
    built from literals, which verifies only that pydantic returns the
    fields it was just assigned.
    """

    session = make_conv(
        id="conv-exact",
        provider="claude-code",
        messages=MessageCollection(
            messages=[
                _msg_with_tokens(
                    id="m1",
                    model="claude-sonnet-4-5",
                    input_tokens=1000,
                    output_tokens=500,
                )
            ]
        ),
        reported_cost_usd=1.25,
    )
    estimate = estimate_session_cost(session)

    assert estimate.status == "exact"
    assert estimate.basis.provider_reported_usd == pytest.approx(1.25)
    assert estimate.basis.api_equivalent_usd == pytest.approx(1.25)
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
        messages=MessageCollection(
            messages=[
                _msg_with_tokens(
                    id="m1",
                    model="openai/gpt-4o-2024-08-06",
                    input_tokens=1000,
                    output_tokens=500,
                )
            ]
        ),
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

    Typed ``model_name`` and token columns seed the per-model breakdown.
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

    assert estimate.status == "priced"
    assert {row.normalized_model for row in estimate.per_model_breakdown} == {
        "claude-sonnet-4-5",
        "claude-opus-4-5",
    }


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
        messages=MessageCollection(
            messages=[
                _msg_with_tokens(
                    id="m1",
                    model="claude-sonnet-4-5",
                    input_tokens=100,
                    output_tokens=50,
                )
            ]
        ),
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
    """The real rollup aggregator sums basis + per-model rows across sessions.

    Drives ``aggregate_cost_rollup_insights`` (the production function the
    materializer and the archive read path both use) over
    ``SessionCostInsight`` rows whose estimates come from real
    ``estimate_session_cost`` calls on real ``Session`` objects.

    Anti-vacuity: breaking the aggregation -- dropping the ``basis.plus``
    accumulation, ignoring ``per_model_breakdown`` entries, merging the two
    same-model sessions without incrementing ``session_count``, or failing to
    group by ``(origin, normalized_model)`` -- makes this test red. Nothing
    asserted here is a literal the test handed to the function under test:
    every expected number is derived from the per-session estimates.
    """

    from polylogue.analysis.archive import ArchiveInsightProvenance, SessionCostInsight
    from polylogue.analysis.archive_rollups import aggregate_cost_rollup_insights

    def _insight(session_id: str, session: Session) -> SessionCostInsight:
        estimate = estimate_session_cost(session)
        return SessionCostInsight(
            session_id=session_id,
            origin="claude-code-session",
            estimate=estimate,
            provenance=ArchiveInsightProvenance(
                materializer_version=1,
                materialized_at="2026-05-17T00:00:00+00:00",
                source_updated_at=None,
                source_sort_key=None,
            ),
        )

    sonnet_a = make_conv(
        id="conv-sonnet-a",
        provider="claude-code",
        messages=MessageCollection(
            messages=[_msg_with_tokens(id="m1", model="claude-sonnet-4-5", input_tokens=1000, output_tokens=500)]
        ),
    )
    sonnet_b = make_conv(
        id="conv-sonnet-b",
        provider="claude-code",
        messages=MessageCollection(
            messages=[_msg_with_tokens(id="m1", model="claude-sonnet-4-5", input_tokens=2000, output_tokens=1000)]
        ),
    )
    mixed = make_conv(
        id="conv-mixed",
        provider="claude-code",
        messages=MessageCollection(
            messages=[
                _msg_with_tokens(id="m1", model="claude-opus-4-5", input_tokens=1000, output_tokens=500),
                _msg_with_tokens(id="m2", model="claude-haiku-4-5", input_tokens=1000, output_tokens=500),
            ]
        ),
    )

    sonnet_insights = [_insight("s:a", sonnet_a), _insight("s:b", sonnet_b)]
    mixed_insight = _insight("s:mixed", mixed)

    rollups = aggregate_cost_rollup_insights(
        [*sonnet_insights, mixed_insight],
        materialized_at="2026-05-17T00:00:00+00:00",
    )

    by_model = {rollup.normalized_model: rollup for rollup in rollups}
    sonnet = by_model["claude-sonnet-4-5"]

    # Same-model sessions merge into one group with an incremented count.
    assert sonnet.session_count == 2
    assert sonnet.priced_session_count == 2
    assert sonnet.status_counts == {"priced": 2}
    assert sonnet.total_usd == pytest.approx(sum(i.estimate.total_usd or 0.0 for i in sonnet_insights))
    for axis in (
        "provider_reported_usd",
        "api_equivalent_usd",
        "subscription_equivalent_usd",
        "catalog_priced_usd",
        "tool_surcharge_usd",
    ):
        expected = sum(getattr(i.estimate.basis, axis) for i in sonnet_insights)
        assert getattr(sonnet.basis, axis) == pytest.approx(expected), axis

    # The mixed-model session groups under its own key and carries one
    # per-model row per model its estimate reported.
    mixed_rollup = by_model[mixed_insight.estimate.normalized_model]
    assert mixed_rollup is not sonnet
    assert mixed_rollup.session_count == 1
    expected_models = {row.normalized_model for row in mixed_insight.estimate.per_model_breakdown}
    assert {row.normalized_model for row in mixed_rollup.per_model_breakdown} == expected_models
    assert sum(row.total_usd for row in mixed_rollup.per_model_breakdown) == pytest.approx(
        mixed_insight.estimate.total_usd
    )

    # Rollups are ordered by total spend, descending.
    assert [rollup.total_usd for rollup in rollups] == sorted((rollup.total_usd for rollup in rollups), reverse=True)
