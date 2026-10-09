"""Contracts for per-model cost breakdown provenance (polylogue-9kjtc, polylogue-3b607).

``session_model_usage`` rows can carry a real model-identity fact (a
``model_name``) with zero token counters -- this is the normal shape for
origins whose exports don't carry provider token counters at all
(chatgpt-export, claude-ai-export; see docs/cost-model.md's "estimate-only"
disposition table). Before the fix, every such row was unconditionally
labelled ``confidence="reported"``/``provenance="provider_reported"``,
making a hollow zero-token row indistinguishable from a provider that
genuinely reported and billed a zero cost.
"""

from __future__ import annotations

import pytest

from polylogue.archive.semantic.cost_compute import (
    _per_model_from_messages,
    _per_model_from_model_usage,
    compute_session_cost,
)
from polylogue.archive.semantic.cost_records import ModelUsageTotals
from polylogue.archive.semantic.pricing import CostEstimatePayload, CostUsagePayload
from tests.infra.builders import make_conv, make_msg


def test_zero_token_model_usage_row_is_labelled_unknown_not_reported() -> None:
    """A model-identity-only row (real model_name, zero tokens) must not be
    stamped as provider-reported evidence."""

    rows = [ModelUsageTotals(model_name="gpt-4o", input_tokens=0, output_tokens=0, provider_lanes_complete=True)]
    per_model = _per_model_from_model_usage(rows)

    (breakdown,) = per_model.values()
    assert breakdown.confidence == "unknown"
    assert breakdown.provenance == "unknown"


def test_nonzero_model_usage_row_stays_provider_reported() -> None:
    """A row with real token counts is unaffected by the zero-token guard."""

    rows = [ModelUsageTotals(model_name="gpt-4o", input_tokens=100, output_tokens=50, provider_lanes_complete=True)]
    per_model = _per_model_from_model_usage(rows)

    (breakdown,) = per_model.values()
    assert breakdown.confidence == "reported"
    assert breakdown.provenance == "provider_reported"


def test_catalog_priced_session_money_is_not_labelled_provider_reported() -> None:
    """Catalog-priced dollars must not inherit provider token provenance.

    Anti-vacuity: restoring the aggregate mapping from reported token
    confidence to ``provider_reported`` makes this assertion red.
    """

    session = make_conv(id="catalog-priced-money", provider="claude-code", messages=[])
    summary = compute_session_cost(
        session,
        estimate_if_missing=False,
        model_usage=[
            ModelUsageTotals(
                model_name="claude-opus-4-8", input_tokens=1_000, output_tokens=500, provider_lanes_complete=True
            )
        ],
    )

    assert summary.cost_provenance == "catalog_priced"
    assert summary.per_model[0].provenance == "provider_reported"


def test_exact_provider_money_remains_provider_reported() -> None:
    """A provider-stated dollar total keeps its exact money provenance."""

    session = make_conv(id="exact-provider-money", provider="hermes", messages=[])
    summary = compute_session_cost(
        session,
        session_estimate=CostEstimatePayload(
            origin="hermes",
            status="exact",
            total_usd=1.25,
            usage=CostUsagePayload(input_tokens=100, output_tokens=50),
        ),
    )

    assert summary.cost_provenance == "provider_reported"


def test_exact_provider_money_does_not_replace_canonical_model_usage_tokens() -> None:
    """polylogue-t2ugv: reported money and canonical token lanes are orthogonal.

    Anti-vacuity: restoring the exact-cost early return makes the summary
    report the message estimate's zero tokens instead of 100/20.
    """

    session = make_conv(id="exact-provider-with-rollup", provider="chatgpt", messages=[])
    summary = compute_session_cost(
        session,
        session_estimate=CostEstimatePayload(
            origin="chatgpt",
            status="exact",
            total_usd=1.0,
            usage=CostUsagePayload(),
        ),
        model_usage=[
            ModelUsageTotals(model_name="gpt-4o", input_tokens=100, output_tokens=20, provider_lanes_complete=True)
        ],
    )

    assert summary.total_input_tokens == 100
    assert summary.total_output_tokens == 20
    assert summary.total_api_cost_usd == 1.0
    assert sum(item.api_cost_usd for item in summary.per_model) == summary.total_api_cost_usd


def test_routed_names_with_their_own_rates_keep_separate_buckets() -> None:
    """Anti-vacuity: merging routed names by normalized model prices all tokens at the last route's rate."""
    routed = ("amazon.nova-pro-v1:0", "bedrock/us-gov-east-1/amazon.nova-pro-v1:0")
    session = make_conv(id="two-routes", provider="chatgpt", messages=[])
    rows = [
        ModelUsageTotals(model_name=name, input_tokens=1000, output_tokens=100, provider_lanes_complete=True)
        for name in routed
    ]
    forward = compute_session_cost(session, model_usage=rows)
    backward = compute_session_cost(session, model_usage=list(reversed(rows)))
    assert len(forward.per_model) == 2
    assert forward.total_api_cost_usd == backward.total_api_cost_usd


def test_exact_session_total_is_not_attributed_to_an_arbitrary_unpriced_model() -> None:
    """Anti-vacuity: assigning the whole total to the first sorted model fabricates a per-model exact cost."""
    session = make_conv(id="exact-provider-two-unpriced", provider="chatgpt", messages=[])
    summary = compute_session_cost(
        session,
        session_estimate=CostEstimatePayload(origin="chatgpt", status="exact", total_usd=1.0, usage=CostUsagePayload()),
        model_usage=[
            ModelUsageTotals(
                model_name="unpriced-model-a", input_tokens=100, output_tokens=20, provider_lanes_complete=True
            ),
            ModelUsageTotals(
                model_name="unpriced-model-b", input_tokens=50, output_tokens=10, provider_lanes_complete=True
            ),
        ],
    )

    assert summary.total_api_cost_usd == 1.0
    assert [item.api_cost_usd for item in summary.per_model] == [0.0, 0.0]
    assert {item.confidence for item in summary.per_model} == {"unknown"}


def test_exact_total_reconciliation_never_yields_a_negative_share() -> None:
    """Anti-vacuity: rounding each share independently gives the first three models
    $0.000001 each of a $0.000002 total and the last model -$0.000001."""
    session = make_conv(id="exact-provider-tiny-total", provider="chatgpt", messages=[])
    tokens = {"gpt-4o": 255, "gpt-4o-2024-05-13": 128, "gpt-4o-2024-08-06": 255, "gpt-4o-2024-11-20": 200}
    summary = compute_session_cost(
        session,
        session_estimate=CostEstimatePayload(
            origin="chatgpt", status="exact", total_usd=0.000002, usage=CostUsagePayload()
        ),
        model_usage=[
            ModelUsageTotals(model_name=name, input_tokens=n, output_tokens=0, provider_lanes_complete=True)
            for name, n in tokens.items()
        ],
    )

    shares = [item.api_cost_usd for item in summary.per_model]
    assert len(shares) == 4
    assert min(shares) >= 0.0
    assert round(sum(shares) * 1_000_000) == 2


def test_compute_session_cost_falls_back_to_word_count_estimate_for_zero_token_usage() -> None:
    """polylogue-9kjtc AC2: when session_model_usage carries only zero-token
    rows (the chatgpt-export/claude-ai-export shape) but the session's
    messages have real text, compute_session_cost must run the text-length
    heuristic estimate rather than reporting a hollow $0.0 'reported' cost.
    """

    session = make_conv(
        id="chatgpt-zero-token-session",
        provider="chatgpt",
        messages=[
            make_msg(id="m1", role="user", text="a reasonably long user message with several words in it"),
            make_msg(id="m2", role="assistant", text="a reasonably long assistant reply with several words too"),
        ],
    )
    model_usage = [ModelUsageTotals(model_name="gpt-4o", input_tokens=0, output_tokens=0, provider_lanes_complete=True)]

    summary = compute_session_cost(session, estimate_if_missing=False, model_usage=model_usage)

    assert summary.cost_confidence == "estimated"
    assert summary.cost_provenance != "provider_reported"
    assert summary.total_input_tokens > 0
    assert any(b.confidence == "estimated" for b in summary.per_model)


def test_zero_token_fallback_keeps_the_declared_model_identity_and_price() -> None:
    """Messages without a per-message model take the session's one declared model.

    Anti-vacuity (polylogue-jgngs): replacing the canonical map with the
    message-only estimate files every token under ``unknown``, so the
    breakdown loses ``gpt-4o`` and its catalog price.
    """

    session = make_conv(
        id="identity-only-session",
        provider="chatgpt",
        messages=[
            make_msg(id="m1", role="user", text="a reasonably long user message with several words in it"),
            make_msg(id="m2", role="assistant", text="a reasonably long assistant reply with several words too"),
        ],
    )
    model_usage = [ModelUsageTotals(model_name="gpt-4o", input_tokens=0, output_tokens=0, provider_lanes_complete=True)]

    summary = compute_session_cost(session, estimate_if_missing=False, model_usage=model_usage)

    assert [breakdown.normalized_model for breakdown in summary.per_model] == ["gpt-4o"]
    assert summary.per_model[0].provider_model_name == "gpt-4o"
    assert summary.per_model[0].confidence == "estimated"
    assert summary.total_api_cost_usd > 0
    assert summary.cost_provenance == "catalog_priced"


def test_zero_token_fallback_keeps_declared_identities_it_cannot_attribute() -> None:
    """With two declared models and no per-message model, neither is guessed.

    The estimate stays under ``unknown`` and both declared identities remain
    as identity-only rows, so the aggregate is honestly ``partial``.
    Anti-vacuity: replacing the map drops both declared identities.
    """

    session = make_conv(
        id="two-model-session",
        provider="chatgpt",
        messages=[make_msg(id="m1", role="user", text="several words of user text for the estimate")],
    )
    model_usage = [
        ModelUsageTotals(model_name="gpt-4o", input_tokens=0, output_tokens=0, provider_lanes_complete=True),
        ModelUsageTotals(model_name="o3", input_tokens=0, output_tokens=0, provider_lanes_complete=True),
    ]

    summary = compute_session_cost(session, estimate_if_missing=False, model_usage=model_usage)

    models = {breakdown.normalized_model for breakdown in summary.per_model}
    assert {"gpt-4o", "o3"} <= models
    assert None in models
    assert summary.cost_confidence == "partial"


def test_compute_session_cost_is_unknown_when_no_real_evidence_exists() -> None:
    """When session_model_usage carries only zero-token rows AND the
    session's messages carry no text/word-count evidence either, the honest
    disposition is 'unknown', not a default 'reported' with $0.0 cost."""

    session = make_conv(
        id="no-evidence-session",
        provider="chatgpt",
        messages=[],
    )
    model_usage = [ModelUsageTotals(model_name="gpt-4o", input_tokens=0, output_tokens=0, provider_lanes_complete=True)]

    summary = compute_session_cost(session, estimate_if_missing=False, model_usage=model_usage)

    assert summary.cost_confidence == "unknown"
    assert summary.cost_provenance == "unknown"
    assert summary.total_api_cost_usd == 0.0


def test_per_model_from_messages_splits_estimated_tokens_by_role_with_dominant_model_fallback() -> None:
    """polylogue-3b607 P1: a ChatGPT-shaped session where only the assistant
    turn declares ``model_name`` (the user turn does not, which is the normal
    ChatGPT/Claude-web export shape) must:

    (1) attribute the model-less *user* turn's estimated tokens to the
        session's dominant declared model rather than an unpriced "unknown"
        bucket, and
    (2) classify each turn's estimated tokens by role -- user text becomes
        input_tokens, assistant text becomes output_tokens -- instead of
        dumping everything into input_tokens regardless of role.

    Reverting the dominant-model fallback in ``_per_model_from_messages``
    (making model-less messages key on "unknown" again) fails assertion (1);
    reverting the role-based split (using ``estimate_tokens_from_words``
    unconditionally into ``input_tokens`` again) fails assertion (2), since
    the assistant reply's tokens would land in input_tokens and
    output_tokens would stay 0.
    """

    session = make_conv(
        id="chatgpt-role-split-session",
        provider="chatgpt",
        messages=[
            make_msg(
                id="m1",
                role="user",
                text="a reasonably long user message with quite a few words in it here",
            ),
            make_msg(
                id="m2",
                role="assistant",
                text="a reasonably long assistant reply with even more words in it than that",
                model_name="gpt-4o",
            ),
        ],
    )

    per_model = _per_model_from_messages(session)

    # No "unknown" bucket -- the model-less user turn inherited the session's
    # one declared model (gpt-4o) instead of being shunted off unpriced.
    assert set(per_model.keys()) == {"gpt-4o"}
    breakdown = per_model["gpt-4o"]

    # Both roles contributed real estimated tokens, split by role rather than
    # everything landing in input_tokens.
    assert breakdown.input_tokens > 0, "user turn's estimate should be input_tokens"
    assert breakdown.output_tokens > 0, "assistant turn's estimate should be output_tokens"
    assert breakdown.total_tokens == breakdown.input_tokens + breakdown.output_tokens
    assert breakdown.confidence == "estimated"


def test_compute_session_cost_downgrades_to_partial_when_one_model_unknown_alongside_reported() -> None:
    """polylogue-3b607 P2: a two-model session where one model has genuine
    nonzero provider-reported usage and the other is only a zero-token
    ``_seed_session_model_usage_rows`` skeleton row (no telemetry at all) must
    not read as a clean aggregate 'reported' -- has_reported=True from the
    genuine row previously masked the fact that another per-model breakdown
    was separately labelled unknown. The honest aggregate disposition is
    'partial'.

    Reverting the P2 fix (removing the ``has_unknown`` branch in
    ``compute_session_cost``'s aggregate loop) makes this fail: the aggregate
    would fall through to the initial ``agg_confidence = "reported"`` default
    since ``has_estimates`` is False and ``has_reported`` is True.
    """

    model_usage = [
        ModelUsageTotals(model_name="gpt-4o", input_tokens=1000, output_tokens=500, provider_lanes_complete=True),
        ModelUsageTotals(model_name="claude-3-5-sonnet", input_tokens=0, output_tokens=0, provider_lanes_complete=True),
    ]
    session = make_conv(id="mixed-reported-unknown-session", provider="chatgpt", messages=[])

    summary = compute_session_cost(session, estimate_if_missing=False, model_usage=model_usage)

    assert summary.cost_confidence == "partial"
    assert summary.cost_provenance != "provider_reported"
    confidences = {b.confidence for b in summary.per_model}
    assert "reported" in confidences
    assert "unknown" in confidences


def test_catalog_gap_model_with_real_tokens_is_not_a_confident_zero() -> None:
    """polylogue-iuyr: a model with genuine, nonzero reported tokens but no
    pricing-catalog entry must not surface as
    ``confidence='reported'``/``total_api_cost_usd=0.0`` -- that is
    indistinguishable from a provider that genuinely billed zero.
    ``estimate_cost()`` silently returns 0.0 for an unpriced model; the
    confidence must be downgraded to 'unknown' instead of trusting that zero.

    The model here is deliberately synthetic. A real name eventually gains a
    catalog entry and quietly stops exercising the gap this covers.
    """

    model_usage = [
        ModelUsageTotals(
            model_name="not-a-catalogued-model", input_tokens=1000, output_tokens=500, provider_lanes_complete=True
        )
    ]
    session = make_conv(id="catalog-gap-session", provider="claude-code", messages=[])

    summary = compute_session_cost(session, estimate_if_missing=False, model_usage=model_usage)

    assert summary.total_api_cost_usd == 0.0
    assert summary.cost_confidence == "unknown"
    (breakdown,) = summary.per_model
    assert breakdown.confidence == "unknown"
    assert breakdown.total_tokens > 0, "the tokens are real -- this must not be the zero-token 9kjtc case"


def test_catalogued_model_with_real_tokens_stays_reported() -> None:
    """Anti-vacuity twin: a genuinely priced model is unaffected by the
    catalog-gap downgrade."""

    model_usage = [
        ModelUsageTotals(model_name="gpt-4o", input_tokens=1000, output_tokens=500, provider_lanes_complete=True)
    ]
    session = make_conv(id="catalogued-session", provider="chatgpt", messages=[])

    summary = compute_session_cost(session, estimate_if_missing=False, model_usage=model_usage)

    assert summary.cost_confidence == "reported"
    (breakdown,) = summary.per_model
    assert breakdown.confidence == "reported"
    assert breakdown.api_cost_usd > 0.0


def test_compute_session_cost_defaults_to_pro_tier_subscription_equivalent() -> None:
    """No explicit ``subscription_tier`` keeps the conservative ``pro`` default
    (polylogue-at44 AC: "cost compute reads it with a sane default")."""

    model_usage = [
        ModelUsageTotals(
            model_name="claude-sonnet-4-5", input_tokens=1_000_000, output_tokens=0, provider_lanes_complete=True
        )
    ]
    session = make_conv(id="sub-tier-default-session", provider="claude-code", messages=[])

    default_summary = compute_session_cost(session, estimate_if_missing=False, model_usage=model_usage)
    pro_summary = compute_session_cost(
        session, estimate_if_missing=False, model_usage=model_usage, subscription_tier="pro"
    )

    assert default_summary.total_subscription_equivalent_usd > 0
    assert default_summary.total_subscription_equivalent_usd == pro_summary.total_subscription_equivalent_usd


def test_compute_session_cost_honors_explicit_subscription_tier() -> None:
    """A caller that knows the archive owner's real plan (e.g. read from the
    ``subscription_tier`` user setting) gets that plan's cheaper per-credit
    rate reflected in the subscription-equivalent figure, not the hardcoded
    ``pro`` ratio (polylogue-at44)."""

    model_usage = [
        ModelUsageTotals(
            model_name="claude-sonnet-4-5", input_tokens=1_000_000, output_tokens=0, provider_lanes_complete=True
        )
    ]
    session = make_conv(id="sub-tier-explicit-session", provider="claude-code", messages=[])

    pro_summary = compute_session_cost(
        session, estimate_if_missing=False, model_usage=model_usage, subscription_tier="pro"
    )
    max20_summary = compute_session_cost(
        session, estimate_if_missing=False, model_usage=model_usage, subscription_tier="max_20x"
    )

    # Same credit cost either way; only the USD conversion ratio differs.
    assert pro_summary.total_credit_cost == max20_summary.total_credit_cost
    assert max20_summary.total_subscription_equivalent_usd < pro_summary.total_subscription_equivalent_usd
    assert max20_summary.total_subscription_equivalent_usd > 0


def test_message_fallback_reports_partial_when_a_token_lane_was_never_captured() -> None:
    """polylogue-qe194: the message-evidence fallback must not label a
    partly-measured tally "reported".

    ``_get_message_token_counts`` fires when *some* lane is present, then
    coerced every ``None`` sibling to 0 and handed the result to
    ``_add_provider_reported_tokens``, which stamped ``confidence="reported"``
    -- a claim that the provider measured the whole breakdown. Here the
    cache-write lane was never captured, so the summed dollars are a lower
    bound and the disposition is ``partial``.

    Anti-vacuity: the twin below keeps every lane captured (cache lanes at
    zero) and must stay ``reported``; if both report the same confidence, the
    unknown lane has been collapsed back onto a measured zero.
    """

    session = make_conv(
        id="partial-lane-session",
        provider="claude-code",
        messages=[
            make_msg(
                id="m1",
                role="assistant",
                provider="claude-code",
                model_name="claude-sonnet-4-5",
                input_tokens=1000,
                output_tokens=500,
                cache_read_tokens=200,
                cache_write_tokens=None,
            )
        ],
    )

    summary = compute_session_cost(session, estimate_if_missing=False)

    (breakdown,) = summary.per_model
    assert breakdown.confidence == "partial"
    assert summary.cost_confidence == "partial"
    assert summary.total_api_cost_usd > 0.0


def test_message_fallback_stays_reported_when_every_lane_was_captured() -> None:
    """Anti-vacuity twin of the partial-lane case: captured zeros are real
    measurements and keep the ``reported`` disposition (polylogue-qe194)."""

    session = make_conv(
        id="captured-lane-session",
        provider="claude-code",
        messages=[
            make_msg(
                id="m1",
                role="assistant",
                provider="claude-code",
                model_name="claude-sonnet-4-5",
                input_tokens=1000,
                output_tokens=500,
                cache_read_tokens=200,
                cache_write_tokens=0,
            )
        ],
    )

    summary = compute_session_cost(session, estimate_if_missing=False)

    (breakdown,) = summary.per_model
    assert breakdown.confidence == "reported"
    assert summary.cost_confidence == "reported"


def test_runtime_protocol_text_in_an_assistant_envelope_is_not_estimated_output() -> None:
    """Harness-written text (an API-error notice) is not model output.

    Anti-vacuity: drop the RUNTIME_PROTOCOL skip in ``_per_model_from_messages``
    and the error notice's words are estimated as output tokens.
    """
    session = make_conv(
        id="claude-code-api-error-session",
        provider="claude-code",
        messages=[
            make_msg(
                id="m1",
                role="assistant",
                text="API Error: 429 rate limited, please retry after some time has passed",
                material_origin="runtime_protocol",
            ),
        ],
    )

    per_model = _per_model_from_messages(session)

    assert per_model == {}


@pytest.mark.parametrize("role", ["system", "user"])
def test_runtime_protocol_input_is_still_estimated(role: str) -> None:
    """Protocol text sent to the model (a system prompt, a task notification) is input.

    Anti-vacuity: skip every RUNTIME_PROTOCOL message regardless of role and
    the session has no estimated input tokens.
    """
    session = make_conv(
        id=f"runtime-protocol-{role}-session",
        provider="claude-code",
        messages=[
            make_msg(
                id="m1",
                role=role,
                text="You are a helpful assistant working in the operator's repository",
                material_origin="runtime_protocol",
            ),
        ],
    )

    per_model = _per_model_from_messages(session)

    assert sum(breakdown.input_tokens for breakdown in per_model.values()) > 0


def test_unmappable_zero_provider_lanes_do_not_become_text_estimates() -> None:
    session = make_conv(
        id="unmappable-zero",
        provider="codex",
        messages=[make_msg(id="m1", role="assistant", text="a reply long enough to produce estimated tokens")],
    )
    summary = compute_session_cost(
        session,
        estimate_if_missing=False,
        model_usage=[ModelUsageTotals(model_name="gpt-4o", provider_lanes_complete=False)],
    )
    assert summary.total_input_tokens == 0
    assert summary.total_output_tokens == 0
    assert summary.cost_confidence == "unknown"
    assert summary.cost_provenance == "unknown"
    assert summary.per_model[0].confidence == "unknown"


@pytest.mark.parametrize("partial", [False, True])
def test_message_fallback_preserves_cache_and_evidence_under_reorder(partial: bool) -> None:
    """A heuristic addition cannot erase cache usage or become provider-only."""
    from types import SimpleNamespace

    from polylogue.archive.semantic.pricing import catalog_cost_for_tokens
    from polylogue.core.enums import Role

    measured = SimpleNamespace(
        model_name="gpt-4o",
        role=Role.ASSISTANT,
        word_count=20,
        input_tokens=10,
        output_tokens=20,
        cache_read_tokens=100,
        cache_write_tokens=None if partial else 0,
    )
    estimated = SimpleNamespace(
        model_name="gpt-4o",
        role=Role.ASSISTANT,
        word_count=15,
        input_tokens=None,
        output_tokens=None,
        cache_read_tokens=None,
        cache_write_tokens=None,
    )
    forward = compute_session_cost(SimpleNamespace(messages=[measured, estimated]), estimate_if_missing=False)
    backward = compute_session_cost(SimpleNamespace(messages=[estimated, measured]), estimate_if_missing=False)
    assert forward == backward
    assert forward.total_cache_read_tokens == 100
    (breakdown,) = forward.per_model
    assert (
        breakdown.total_tokens == breakdown.input_tokens + breakdown.output_tokens + 100 + breakdown.cache_write_tokens
    )
    assert breakdown.confidence == ("partial" if partial else "estimated")
    assert breakdown.provenance == "mixed"
    expected, _ = catalog_cost_for_tokens("gpt-4o", breakdown.input_tokens, breakdown.output_tokens, 100, 0)
    assert forward.total_api_cost_usd == pytest.approx(round(expected, 6))


def test_measured_zero_usage_does_not_fall_back_to_transcript_estimate() -> None:
    session = make_conv(id="measured-zero", provider="chatgpt", messages=[make_msg(text="synthetic assistant reply")])
    summary = compute_session_cost(
        session,
        estimate_if_missing=False,
        model_usage=[ModelUsageTotals(model_name="gpt-4o", provider_lanes_complete=True, provider_usage_observed=True)],
    )
    assert summary.total_input_tokens == summary.total_output_tokens == 0
    assert summary.total_api_cost_usd == 0
    assert summary.cost_confidence == "reported"
