"""Contracts for per-model cost breakdown provenance (polylogue-9kjtc).

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
