"""Per-session usage/cost reconciliation (polylogue-f2qv.6, first slice).

Reproduces the bead's concrete disagreement: one Codex session where
``session_model_usage`` reports an exact rollup of 64,561 uncached input +
723,456 cache read + 7,776 output tokens (795,793 total), the
``session_profiles`` row persists a stale 4,031-token estimate, and the
per-session cost insight (which reads its number off that same profile row)
reports zero/unavailable. ``build_session_usage_reconciliation`` must pick
the exact rollup as the winning token value -- not average the two numbers,
not arbitrarily prefer whichever was passed first -- and must keep the
superseded profile estimate visible as a labeled, lower-authority
contribution rather than silently discarding it.
"""

from __future__ import annotations

import pytest

from polylogue.archive.semantic.cost_records import ModelUsageTotals
from polylogue.archive.semantic.pricing import PRICING, _normalize_model, estimate_cost
from polylogue.core.evidence_value import refine_evidence_value
from polylogue.storage.usage import (
    SESSION_USAGE_RECONCILED_COST_FAMILY,
    SESSION_USAGE_RECONCILED_TOKENS_FAMILY,
    build_session_usage_reconciliation,
)

_OBSERVED_AT = "2026-07-15T00:00:00+00:00"

# The bead's exact reported numbers for the disagreeing live Codex session.
_EXACT_UNCACHED_INPUT = 64_561
_EXACT_CACHE_READ = 723_456
_EXACT_OUTPUT = 7_776
_EXACT_TOTAL = _EXACT_UNCACHED_INPUT + _EXACT_CACHE_READ + _EXACT_OUTPUT

_STALE_PROFILE_ESTIMATE_TOKENS = 4_031

_MODEL = "gpt-5.1-codex"
_NORMALIZED_MODEL = _normalize_model(_MODEL)


def _model_usage_rows() -> tuple[ModelUsageTotals, ...]:
    return (
        ModelUsageTotals(
            model_name=_MODEL,
            input_tokens=_EXACT_UNCACHED_INPUT,
            output_tokens=_EXACT_OUTPUT,
            cache_read_tokens=_EXACT_CACHE_READ,
            cache_write_tokens=0,
        ),
    )


def _normalized_model_name(row: ModelUsageTotals) -> str:
    assert row.model_name is not None
    return _normalize_model(row.model_name)


def test_exact_rollup_wins_over_stale_profile_estimate() -> None:
    """The exact session_model_usage rollup wins, not the stale estimate."""

    reconciliation = build_session_usage_reconciliation(
        "codex-session:disagreeing-example",
        observed_at=_OBSERVED_AT,
        model_usage_rows=_model_usage_rows(),
        profile_total_input_tokens=_STALE_PROFILE_ESTIMATE_TOKENS,
        profile_total_output_tokens=0,
        profile_total_cache_read_tokens=0,
        profile_total_cache_write_tokens=0,
        profile_cost_usd=0.0,
        profile_cost_provenance="heuristic_estimated",
        reconciled_model=_NORMALIZED_MODEL,
    )

    # The three raw sources disagree exactly as the bead describes.
    assert reconciliation.model_usage_tokens_evidence.value == _EXACT_TOTAL
    assert reconciliation.profile_tokens_evidence.value == _STALE_PROFILE_ESTIMATE_TOKENS
    assert reconciliation.model_usage_tokens_evidence.value != reconciliation.profile_tokens_evidence.value

    # The reconciled snapshot is not an average or an arbitrary pick: it is
    # exactly the exact-rollup value, carrying provider-reported authority.
    reconciled = reconciliation.reconciled_tokens_evidence
    assert reconciled.value_state == "known"
    assert reconciled.value == _EXACT_TOTAL
    assert reconciled.value != round((_EXACT_TOTAL + _STALE_PROFILE_ESTIMATE_TOKENS) / 2)
    assert "provider-reported" in reconciled.measurement_authority
    SESSION_USAGE_RECONCILED_TOKENS_FAMILY.require(reconciled)

    # The superseded profile estimate is preserved, not dropped, and is
    # labeled with its own (weaker) authority so a reader can see why it lost.
    superseded = reconciliation.superseded_token_observations()
    assert len(superseded) == 1
    assert superseded[0].value == _STALE_PROFILE_ESTIMATE_TOKENS
    assert superseded[0].value_state == "known"
    assert superseded[0].measurement_authority == ("model-derived",)

    # All source evidence is discoverable on the reconciled value's contributions.
    contribution_values = {observation.value for observation in reconciled.contributions}
    assert contribution_values == {_EXACT_TOTAL, _STALE_PROFILE_ESTIMATE_TOKENS}


def test_reconciled_cost_prefers_fresh_catalog_price_over_stale_zero() -> None:
    """A legacy zero/unavailable cost insight does not survive reconciliation
    when a fresh catalog reprice of the winning tokens is available."""

    reconciliation = build_session_usage_reconciliation(
        "codex-session:disagreeing-example",
        observed_at=_OBSERVED_AT,
        model_usage_rows=_model_usage_rows(),
        profile_total_input_tokens=_STALE_PROFILE_ESTIMATE_TOKENS,
        profile_total_output_tokens=0,
        profile_total_cache_read_tokens=0,
        profile_total_cache_write_tokens=0,
        profile_cost_usd=0.0,
        profile_cost_provenance="heuristic_estimated",
        reconciled_model=_NORMALIZED_MODEL,
    )

    # The legacy cost-insight source is exactly the bug's "zero and unavailable".
    assert reconciliation.legacy_cost_evidence.value_state == "unknown"
    assert reconciliation.legacy_cost_evidence.value is None

    # A fresh catalog reprice of the winning (exact) token evidence is known,
    # and prices EACH category at its own rate rather than collapsing
    # input/output/cache into one combined total priced as pure input --
    # output and cache-read tokens are priced very differently from input,
    # so that shortcut would systematically misprice any real session.
    assert reconciliation.catalog_cost_evidence.value_state == "known"
    assert reconciliation.catalog_cost_evidence.value is not None
    assert reconciliation.catalog_cost_evidence.value > 0.0
    assert _NORMALIZED_MODEL in PRICING
    expected_cost = round(
        estimate_cost(
            input_tokens=_EXACT_UNCACHED_INPUT,
            output_tokens=_EXACT_OUTPUT,
            cache_read_tokens=_EXACT_CACHE_READ,
            cache_write_tokens=0,
            model=_NORMALIZED_MODEL,
        ),
        6,
    )
    mispriced_as_pure_input = round(
        estimate_cost(input_tokens=_EXACT_TOTAL, output_tokens=0, model=_NORMALIZED_MODEL),
        6,
    )
    assert expected_cost != mispriced_as_pure_input, "fixture must exercise a model with non-uniform per-category rates"
    assert reconciliation.catalog_cost_evidence.value == expected_cost

    reconciled_cost = reconciliation.reconciled_cost_evidence
    assert reconciled_cost.value_state == "known"
    assert reconciled_cost.value == reconciliation.catalog_cost_evidence.value
    assert "catalog-derived" in reconciled_cost.measurement_authority
    SESSION_USAGE_RECONCILED_COST_FAMILY.require(reconciled_cost)

    superseded = reconciliation.superseded_cost_observations()
    assert any(observation.value_state == "unknown" for observation in superseded)


@pytest.mark.parametrize(
    ("profile_cost_provenance", "legacy_authority"),
    (
        ("provider_reported", ("provider-reported",)),
        ("catalog_priced", ("model-derived",)),
    ),
)
def test_synthetic_claude_row_preserves_exact_provider_price(
    profile_cost_provenance: str,
    legacy_authority: tuple[str, ...],
) -> None:
    """An exact provider total wins while an estimated legacy total does not."""

    rows = (
        ModelUsageTotals(
            model_name="claude-opus-4-8",
            input_tokens=12_345,
            output_tokens=6_789,
            cache_read_tokens=123_456,
            cache_write_tokens=7_890,
        ),
    )
    reconciliation = build_session_usage_reconciliation(
        "claude-code-session:synthetic-agent-001",
        observed_at="2026-08-31T00:00:00+00:00",
        model_usage_rows=rows,
        profile_total_input_tokens=1_234,
        profile_total_output_tokens=567,
        profile_total_cache_read_tokens=8_901,
        profile_total_cache_write_tokens=234,
        profile_cost_usd=3.21,
        profile_cost_provenance=profile_cost_provenance,
        reconciled_model="claude-opus-4-8",
    )

    expected_catalog = round(
        estimate_cost(
            input_tokens=12_345,
            output_tokens=6_789,
            cache_read_tokens=123_456,
            cache_write_tokens=7_890,
            model="claude-opus-4-8",
        ),
        6,
    )
    assert reconciliation.catalog_cost_evidence.value == expected_catalog
    assert reconciliation.legacy_cost_evidence.value == 3.21
    assert reconciliation.legacy_cost_evidence.measurement_authority == legacy_authority
    reconciled = reconciliation.reconciled_cost_evidence
    assert reconciled.value_state == "known"
    expected_value = 3.21 if profile_cost_provenance == "provider_reported" else expected_catalog
    assert reconciled.value == expected_value
    assert reconciled.conflicts == ()
    expected_superseded = expected_catalog if profile_cost_provenance == "provider_reported" else 3.21
    assert [observation.value for observation in reconciliation.superseded_cost_observations()] == [expected_superseded]
    SESSION_USAGE_RECONCILED_COST_FAMILY.require(reconciled)

    # The winner is determined by the declared authority order, not by the
    # order in which the two observations reach the resolver.
    reversed_inputs = refine_evidence_value(
        reconciliation.legacy_cost_evidence,
        reconciliation.catalog_cost_evidence,
        spec=SESSION_USAGE_RECONCILED_COST_FAMILY,
    )
    assert reversed_inputs.value_state == "known"
    assert reversed_inputs.value == expected_value
    assert reversed_inputs.conflicts == ()


def test_multi_model_reprice_keeps_each_model_rate() -> None:
    """A mixed session is priced as the sum of its per-model projections.

    Anti-vacuity: replacing the implementation with the historical
    ``max(output_tokens)`` model selection makes this red because the two
    catalog models have different rates.
    """

    rows = (
        ModelUsageTotals(
            model_name="gpt-5.1-codex",
            input_tokens=1_000,
            output_tokens=100,
        ),
        ModelUsageTotals(
            model_name="claude-sonnet-4-5",
            input_tokens=2_000,
            output_tokens=1_000,
        ),
    )
    reconciliation = build_session_usage_reconciliation(
        "codex-session:multi-model",
        observed_at=_OBSERVED_AT,
        model_usage_rows=rows,
        profile_cost_usd=0.0,
        profile_cost_provenance="unknown",
        # The old implementation used this selected model for all tokens.
        reconciled_model=_normalize_model("claude-sonnet-4-5"),
    )

    expected = round(
        sum(
            estimate_cost(
                input_tokens=row.input_tokens,
                output_tokens=row.output_tokens,
                cache_read_tokens=row.cache_read_tokens,
                cache_write_tokens=row.cache_write_tokens,
                model=_normalized_model_name(row),
            )
            for row in rows
        ),
        6,
    )
    assert reconciliation.catalog_cost_evidence.value == expected
    assert reconciliation.catalog_cost_evidence.value != round(
        estimate_cost(
            input_tokens=sum(row.input_tokens for row in rows),
            output_tokens=sum(row.output_tokens for row in rows),
            model=_normalize_model("claude-sonnet-4-5"),
        ),
        6,
    )


def test_agreeing_sources_reconcile_without_conflict() -> None:
    """When session_profiles already reflects the exact rollup (post self-heal),
    reconciliation reports a clean agreement, not a spurious conflict."""

    rows = _model_usage_rows()
    reconciliation = build_session_usage_reconciliation(
        "codex-session:agreeing-example",
        observed_at=_OBSERVED_AT,
        model_usage_rows=rows,
        profile_total_input_tokens=_EXACT_UNCACHED_INPUT,
        profile_total_output_tokens=_EXACT_OUTPUT,
        profile_total_cache_read_tokens=_EXACT_CACHE_READ,
        profile_total_cache_write_tokens=0,
        profile_cost_usd=0.05,
        profile_cost_provenance="provider_reported",
        reconciled_model=_NORMALIZED_MODEL,
    )

    reconciled = reconciliation.reconciled_tokens_evidence
    assert reconciled.value == _EXACT_TOTAL
    assert reconciled.conflicts == ()
    assert reconciliation.superseded_token_observations() == ()


def test_no_model_usage_rows_falls_back_to_profile_estimate_labeled_as_weak() -> None:
    """With no session_model_usage rows at all, the only known source is the
    profile estimate -- reconciliation must still surface it, correctly
    labeled with model-derived (not provider-reported) authority."""

    reconciliation = build_session_usage_reconciliation(
        "codex-session:no-rollup-example",
        observed_at=_OBSERVED_AT,
        model_usage_rows=(),
        profile_total_input_tokens=_STALE_PROFILE_ESTIMATE_TOKENS,
        profile_total_output_tokens=0,
        profile_total_cache_read_tokens=0,
        profile_total_cache_write_tokens=0,
        profile_cost_usd=0.0,
        profile_cost_provenance="heuristic_estimated",
        reconciled_model=_NORMALIZED_MODEL,
    )

    assert reconciliation.model_usage_tokens_evidence.value_state == "unknown"
    reconciled = reconciliation.reconciled_tokens_evidence
    assert reconciled.value_state == "known"
    assert reconciled.value == _STALE_PROFILE_ESTIMATE_TOKENS
    # The only known contributor is the profile estimate, so the weakest
    # (and here, sole substantive) authority behind the value is honestly
    # labeled model-derived -- never upgraded to provider-reported just
    # because an *unknown* session_model_usage observation was also present.
    assert reconciled.weakest_measurement_authority == "model-derived"
