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

from polylogue.archive.semantic.cost_compute import (
    _per_model_from_messages,
    _per_model_from_model_usage,
    compute_session_cost,
)
from polylogue.archive.semantic.cost_records import ModelUsageTotals
from tests.infra.builders import make_conv, make_msg


def test_zero_token_model_usage_row_is_labelled_unknown_not_reported() -> None:
    """A model-identity-only row (real model_name, zero tokens) must not be
    stamped as provider-reported evidence."""

    rows = [ModelUsageTotals(model_name="gpt-4o", input_tokens=0, output_tokens=0)]
    per_model = _per_model_from_model_usage(rows)

    (breakdown,) = per_model.values()
    assert breakdown.confidence == "unknown"
    assert breakdown.provenance == "unknown"


def test_nonzero_model_usage_row_stays_provider_reported() -> None:
    """A row with real token counts is unaffected by the zero-token guard."""

    rows = [ModelUsageTotals(model_name="gpt-4o", input_tokens=100, output_tokens=50)]
    per_model = _per_model_from_model_usage(rows)

    (breakdown,) = per_model.values()
    assert breakdown.confidence == "reported"
    assert breakdown.provenance == "provider_reported"


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
    model_usage = [ModelUsageTotals(model_name="gpt-4o", input_tokens=0, output_tokens=0)]

    summary = compute_session_cost(session, estimate_if_missing=False, model_usage=model_usage)

    assert summary.cost_confidence == "estimated"
    assert summary.cost_provenance != "provider_reported"
    assert summary.total_input_tokens > 0
    assert any(b.confidence == "estimated" for b in summary.per_model)


def test_compute_session_cost_is_unknown_when_no_real_evidence_exists() -> None:
    """When session_model_usage carries only zero-token rows AND the
    session's messages carry no text/word-count evidence either, the honest
    disposition is 'unknown', not a default 'reported' with $0.0 cost."""

    session = make_conv(
        id="no-evidence-session",
        provider="chatgpt",
        messages=[],
    )
    model_usage = [ModelUsageTotals(model_name="gpt-4o", input_tokens=0, output_tokens=0)]

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
        ModelUsageTotals(model_name="gpt-4o", input_tokens=1000, output_tokens=500),
        ModelUsageTotals(model_name="claude-3-5-sonnet", input_tokens=0, output_tokens=0),
    ]
    session = make_conv(id="mixed-reported-unknown-session", provider="chatgpt", messages=[])

    summary = compute_session_cost(session, estimate_if_missing=False, model_usage=model_usage)

    assert summary.cost_confidence == "partial"
    assert summary.cost_provenance != "provider_reported"
    confidences = {b.confidence for b in summary.per_model}
    assert "reported" in confidences
    assert "unknown" in confidences
