"""Contracts for provider/model cost estimation."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.semantic.pricing import (
    _normalize_model,
    canonical_model_family,
    estimate_cost,
    estimate_message_cost,
    estimate_session_cost,
)
from tests.infra.builders import make_conv, make_msg


def test_session_reported_cost_metadata_is_not_read() -> None:
    """Per #803/#1256 the session-level reported-cost metadata bag is no
    longer a cost source.

    Exact provider-reported totals now flow through the typed
    ``session_profiles`` cost projection (see
    ``tests/unit/insights/test_cost_rollup_summary_path.py``);
    ``estimate_session_cost`` on a hydrated ``Session`` prices only from typed
    per-message token/model columns. A session carrying reported-cost metadata
    but no priced messages is therefore ``unavailable`` — the metadata is
    silently ignored rather than producing an ``exact`` total.
    """

    session = make_conv(
        id="conv-exact-cost",
        provider="claude-code",
        provider_meta={"total_cost_usd": 1.25, "model": "claude-sonnet-4-5"},
        messages=MessageCollection(messages=[]),
    )

    estimate = estimate_session_cost(session)

    assert estimate.status == "unavailable"
    assert estimate.total_usd == 0.0
    assert "archive_provider_reported_cost" not in estimate.provenance


def test_token_usage_prices_known_model_with_catalog_provenance() -> None:
    """Per-message token usage prices a known model from the curated catalog.

    Token/model facts are sourced from the typed per-message
    ``model_name``/``input_tokens``/``output_tokens`` columns, not from a
    metadata bag.
    """

    message = make_msg(
        id="m1",
        role="assistant",
        provider="chatgpt",
        model_name="openai/gpt-4o-2024-08-06",
        input_tokens=1000,
        output_tokens=500,
    )

    estimate = estimate_message_cost(message, origin="chatgpt-export")

    assert estimate.status == "priced"
    assert estimate.normalized_model == "gpt-4o"
    assert estimate.total_usd == pytest.approx(0.0075)
    assert estimate.usage.input_tokens == 1000
    assert estimate.usage.output_tokens == 500
    assert estimate.price is not None
    assert estimate.price.source_url.endswith("model_prices_and_context_window.json")


def test_hydrated_messages_report_missing_model_when_no_envelope_cost() -> None:
    """Per #1256, hydrated Message instances no longer carry
    ``provider_meta``; per-message cost facts now flow through the typed
    cost projection (#803). When neither session-level cost nor
    per-message harmonized facts are present, the estimate reports
    ``missing_model`` for each message.
    """

    session = make_conv(
        id="conv-hydrated-no-cost",
        provider="chatgpt",
        messages=MessageCollection(
            messages=[
                make_msg(id="m1", role="assistant", provider="chatgpt"),
                make_msg(id="m2", role="assistant", provider="chatgpt"),
            ]
        ),
    )

    estimate = estimate_session_cost(session)

    assert estimate.status == "unavailable"
    assert estimate.total_usd == 0.0
    # Either missing_model or missing_token_usage is acceptable; hydrated
    # messages contribute no model or usage facts.
    assert estimate.missing_reasons


def test_session_cost_aggregates_typed_message_facts() -> None:
    """Session cost is the sum of typed per-message priced estimates.

    There is no session-level reported-cost short-circuit: the ``costUSD``
    metadata key is ignored, and the total comes from the per-message
    ``model_name``/token columns.
    """

    session = make_conv(
        id="conv-exact-from-envelope",
        provider="chatgpt",
        provider_meta={"costUSD": 0.01, "model": "gpt-4o"},
        messages=MessageCollection(
            messages=[
                make_msg(
                    id="m1",
                    role="assistant",
                    provider="chatgpt",
                    model_name="gpt-4o",
                    input_tokens=1000,
                    output_tokens=500,
                ),
                make_msg(
                    id="m2",
                    role="assistant",
                    provider="chatgpt",
                    model_name="gpt-4o",
                    input_tokens=1000,
                    output_tokens=500,
                ),
            ]
        ),
    )

    estimate = estimate_session_cost(session)

    assert estimate.status == "priced"
    assert estimate.normalized_model == "gpt-4o"
    assert estimate.total_usd == pytest.approx(0.015)


def test_missing_price_is_unavailable_not_zero_precision() -> None:
    """A message with token usage but an unknown model is ``unavailable`` with
    a ``missing_price`` reason — not silently priced at zero."""

    message = make_msg(
        id="m1",
        role="assistant",
        provider="chatgpt",
        model_name="unknown-frontier-model",
        input_tokens=100,
        output_tokens=50,
    )

    estimate = estimate_message_cost(message, origin="chatgpt-export")

    assert estimate.status == "unavailable"
    assert estimate.total_usd == 0.0
    assert estimate.missing_reasons == ("missing_price",)
    assert estimate.unavailable_reason == "no_price"


def test_tokencost_is_not_a_dependency_or_import_anywhere() -> None:
    """polylogue-f2qv.4: LiteLLM is the single pricing source; tokencost must
    stay gone from both the dependency manifest and every source file, not
    just the pricing module. A future accidental re-add (e.g. a helper
    reaching for a familiar package name) would silently reintroduce a
    second, drifting price table.
    """

    repo_root = Path(__file__).resolve().parents[3]
    pyproject_text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert "tokencost" not in pyproject_text.lower()

    offenders: list[str] = []
    for path in (repo_root / "polylogue").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "tokencost" in text.lower():
            offenders.append(str(path.relative_to(repo_root)))
    assert offenders == [], f"tokencost reference(s) found: {offenders}"


def test_no_second_hardcoded_price_table_besides_the_curated_catalog_layer() -> None:
    """polylogue-f2qv.4: exactly one $/token pricing dict feeds every lookup.

    ``PRICING`` is built from ``_load_litellm_catalog()`` (the vendored
    LiteLLM map) overlaid with ``_CURATED_PRICING`` (hand-verified overrides
    for the same catalog, not an independent source) — both merge into the
    single dict every resolution path reads. This pins that composition so a
    future change can't quietly add a second, parallel price map that drifts
    from the vendored catalog.
    """
    from polylogue.archive.semantic import pricing as pricing_module

    expected = {**pricing_module._load_litellm_catalog(), **pricing_module._CURATED_PRICING}
    assert expected == pricing_module.PRICING
    # The curated overrides are a layer on the same catalog, not a second
    # source: every curated key must also resolve (by construction) through
    # the one public resolver used everywhere else in the codebase.
    for key in pricing_module._CURATED_PRICING:
        assert _normalize_model(key) in pricing_module.PRICING


def test_live_archive_shaped_models_resolve_or_are_labelled_unknown() -> None:
    """polylogue-f2qv.4: every model id shape seen in the live archive either
    resolves to a LiteLLM-backed rate via the single resolver, or is labelled
    ``unavailable``/``no_price`` — never silently priced from a second table
    or fabricated as zero-cost-and-priced.
    """
    from polylogue.archive.semantic.pricing import PRICING

    known_shapes = (
        "claude-sonnet-4-5",
        "anthropic/claude-sonnet-4-5-20250929",
        "claude-opus-4-8-20260101",
        "openai/gpt-4o-2024-08-06",
        "gpt-4o-mini",
    )
    for model in known_shapes:
        normalized = _normalize_model(model)
        assert normalized in PRICING, f"{model!r} (normalized {normalized!r}) should resolve"

    message = make_msg(
        id="m-unknown",
        role="assistant",
        provider="chatgpt",
        model_name="totally-unheard-of-model-xyz",
        input_tokens=10,
        output_tokens=5,
    )
    estimate = estimate_message_cost(message, origin="chatgpt-export")
    assert estimate.status == "unavailable"
    assert estimate.unavailable_reason == "no_price"
    assert estimate.total_usd == 0.0


def test_model_normalization_accepts_provider_prefixes_and_version_suffixes() -> None:
    assert _normalize_model("openai/gpt-4o-2024-08-06") == "gpt-4o"
    assert _normalize_model("anthropic/claude-sonnet-4-5-20250929") == "claude-sonnet-4-5"
    assert estimate_cost(1000, 500, "openai/gpt-4o-2024-08-06") == pytest.approx(0.0075)


def test_current_opus_flagships_are_priced_not_zero() -> None:
    """Opus 4.7/4.8 must be priced (regression: they fell through to $0).

    The version-suffix prefix match cannot reach ``claude-opus-4-6`` from a
    ``claude-opus-4-8`` model, so the current flagship needs its own entry.
    """
    for version in ("claude-opus-4-7", "claude-opus-4-8"):
        assert _normalize_model(f"{version}-20260101") == version
        # 1M input + 1M output at Opus rates ($15 + $75) = $90.
        assert estimate_cost(1_000_000, 1_000_000, version) == pytest.approx(90.0)


def test_paid_model_missing_cache_rate_is_flagged_not_silently_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A paid model carrying cache tokens but no catalog cache rate must surface
    a data-quality reason instead of silently pricing that lane at $0 (e.g. a
    model newer than the vendored LiteLLM snapshot)."""
    from polylogue.archive.semantic import pricing as pricing_mod
    from polylogue.archive.semantic.pricing import CostUsagePayload, ModelPricing, _estimate_from_usage

    paid_no_cache = ModelPricing(
        source_name="test",
        input_usd_per_1m=1.0,
        output_usd_per_1m=2.0,
        cache_read_usd_per_1m=0.0,
        cache_write_usd_per_1m=0.0,
    )
    monkeypatch.setitem(pricing_mod.PRICING, "test-paid-nocache", paid_no_cache)

    estimate = _estimate_from_usage(
        origin="unknown-export",
        model_name="test-paid-nocache",
        usage=CostUsagePayload(input_tokens=100, output_tokens=10, cache_read_tokens=1000),
        provenance=("message_token_usage",),
    )
    assert estimate.status == "priced"
    assert "missing_cache_read_price" in estimate.missing_reasons
    # Priced lanes still cost; the unpriced cache lane is $0 but now flagged.
    assert estimate.total_usd > 0


def test_free_model_with_cache_tokens_is_not_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A genuinely free model (all lanes $0, e.g. local-llama) must NOT be
    flagged just because it has cache tokens — the guard is gated on paid."""
    from polylogue.archive.semantic import pricing as pricing_mod
    from polylogue.archive.semantic.pricing import CostUsagePayload, ModelPricing, _estimate_from_usage

    free_model = ModelPricing(source_name="test", input_usd_per_1m=0.0, output_usd_per_1m=0.0)
    monkeypatch.setitem(pricing_mod.PRICING, "test-free-model", free_model)

    estimate = _estimate_from_usage(
        origin="unknown-export",
        model_name="test-free-model",
        usage=CostUsagePayload(input_tokens=100, output_tokens=10, cache_read_tokens=1000),
        provenance=("message_token_usage",),
    )
    assert estimate.status == "priced"
    assert estimate.missing_reasons == ()


def test_paid_model_with_cache_rate_is_not_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A paid model that has a cache rate is not flagged — control case."""
    from polylogue.archive.semantic import pricing as pricing_mod
    from polylogue.archive.semantic.pricing import CostUsagePayload, ModelPricing, _estimate_from_usage

    paid_with_cache = ModelPricing(
        source_name="test",
        input_usd_per_1m=1.0,
        output_usd_per_1m=2.0,
        cache_read_usd_per_1m=0.1,
        cache_write_usd_per_1m=1.25,
    )
    monkeypatch.setitem(pricing_mod.PRICING, "test-paid-cache", paid_with_cache)

    estimate = _estimate_from_usage(
        origin="unknown-export",
        model_name="test-paid-cache",
        usage=CostUsagePayload(input_tokens=100, output_tokens=10, cache_read_tokens=1000, cache_write_tokens=50),
        provenance=("message_token_usage",),
    )
    assert estimate.status == "priced"
    assert estimate.missing_reasons == ()


def test_disjoint_input_cache_lanes_survive_parse_write_and_pricing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Guard the 7.69x-class defect through parse, storage, and pricing."""
    import sqlite3

    from polylogue.archive.semantic import pricing as pricing_mod
    from polylogue.archive.semantic.pricing import CostUsagePayload, ModelPricing, _estimate_from_usage
    from polylogue.sources.parsers.codex import parse
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive

    codex_like = ModelPricing(
        source_name="test",
        input_usd_per_1m=1.0,
        output_usd_per_1m=2.0,
        cache_read_usd_per_1m=0.1,
        cache_write_usd_per_1m=0.0,
    )
    monkeypatch.setitem(pricing_mod.PRICING, "test-codex-like", codex_like)

    message_usage = {"uncached_input_tokens": 100, "cached_input_tokens": 2400, "output_tokens": 50}
    provider_total_usage = {"input_tokens": 2500, "cached_input_tokens": 2400, "output_tokens": 50}
    raw = [
        {
            "type": "message",
            "id": "assistant-1",
            "role": "assistant",
            "model": "test-codex-like",
            "content": [{"type": "output_text", "text": "done"}],
            "usage": message_usage,
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        **provider_total_usage,
                        "reasoning_output_tokens": 30,
                        "total_tokens": 2550,
                    },
                },
            },
        },
    ]

    parsed = parse(raw, "fallback")
    conn = sqlite3.connect(tmp_path / "index.db")
    conn.row_factory = sqlite3.Row
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    session_id = write_parsed_session_to_archive(conn, parsed)

    message_usage = conn.execute(
        """
        SELECT input_tokens, output_tokens, cache_read_tokens, cache_write_tokens
        FROM messages
        WHERE session_id = ? AND native_id = 'assistant-1'
        """,
        (session_id,),
    ).fetchone()
    event_usage = conn.execute(
        """
        SELECT total_input_tokens, total_output_tokens,
               total_cached_input_tokens, total_reasoning_output_tokens
        FROM session_provider_usage_events
        WHERE session_id = ? AND provider_event_type = 'token_count'
        """,
        (session_id,),
    ).fetchone()
    model_usage = conn.execute(
        """
        SELECT input_tokens, output_tokens, cache_read_tokens, cache_write_tokens
        FROM session_model_usage
        WHERE session_id = ? AND model_name = 'test-codex-like'
        """,
        (session_id,),
    ).fetchone()
    conn.close()

    assert dict(message_usage) == {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_tokens": 2400,
        "cache_write_tokens": 0,
    }
    assert dict(event_usage) == {
        "total_input_tokens": 2500,
        "total_output_tokens": 50,
        "total_cached_input_tokens": 2400,
        "total_reasoning_output_tokens": 30,
    }
    completion_output = event_usage["total_output_tokens"] - event_usage["total_reasoning_output_tokens"]
    assert completion_output + event_usage["total_reasoning_output_tokens"] == provider_total_usage["output_tokens"]
    # The model-cost tier keeps priced lanes additive: output is the provider's
    # inclusive total and reasoning stays event-tier evidence, never re-added.
    assert dict(model_usage) == dict(message_usage)

    # Price the values that survived parser -> archive writer, not a second
    # hand-built representation of the corrected usage.
    disjoint = _estimate_from_usage(
        origin="unknown-export",
        model_name="test-codex-like",
        usage=CostUsagePayload(**dict(message_usage)),
        provenance=("message_token_usage",),
    )
    # Pre-fix parser output: input stored inclusive of cache (the bug).
    naive = _estimate_from_usage(
        origin="unknown-export",
        model_name="test-codex-like",
        usage=CostUsagePayload(input_tokens=2500, output_tokens=50, cache_read_tokens=2400),
        provenance=("message_token_usage",),
    )

    assert disjoint.usage.input_tokens + disjoint.usage.cache_read_tokens == provider_total_usage["input_tokens"]
    parsed_message = parsed.messages[0]
    fallback_session = make_conv(
        id="codex-message-fallback",
        provider="codex",
        messages=MessageCollection(
            messages=[
                make_msg(
                    id="assistant-1",
                    role="assistant",
                    provider="codex",
                    model_name=parsed_message.model_name,
                    input_tokens=parsed_message.input_tokens,
                    output_tokens=parsed_message.output_tokens,
                    cache_read_tokens=parsed_message.cache_read_tokens,
                    cache_write_tokens=parsed_message.cache_write_tokens,
                )
            ]
        ),
    )
    fallback_estimate = estimate_session_cost(fallback_session)
    assert fallback_estimate.usage == disjoint.usage
    assert fallback_estimate.total_usd == pytest.approx(disjoint.total_usd)
    # Naive double-bills the entire cached lane at the full input rate on top
    # of the cache-read rate -- on this ~96%-cached ratio the guard requires
    # at least a 5x inflation (real corpus measured 7.69x; the exact multiple
    # depends on the input/cache_read price ratio, not hardcoded here).
    assert naive.total_usd > disjoint.total_usd * 5


def test_canonical_model_family_maps_known_models() -> None:
    """1vpm.1: the enabling primitive for the delegations view's model
    identity. polylogue-4c27: family is derived by pure model-name pattern
    matching (`semantic_model_vendor`), NOT by reusing the pricing catalog's
    own routing-provenance tag -- see the divergence tests below for why
    that distinction is load-bearing."""
    assert canonical_model_family("claude-opus-4-8") == "anthropic"
    assert canonical_model_family("gpt-4o") == "openai"


def test_canonical_model_family_returns_none_for_unknown_model() -> None:
    assert canonical_model_family("some-totally-unknown-model-xyz") is None


def test_canonical_model_family_returns_none_for_empty_or_none() -> None:
    assert canonical_model_family(None) is None
    assert canonical_model_family("") is None


def test_canonical_model_family_does_not_leak_pricing_catalog_routing_tag() -> None:
    """polylogue-4c27 regression: the vendored LiteLLM catalog's bare-name
    entries carry `litellm_provider` routing tags (e.g.
    "vertex_ai-anthropic_models", "bedrock_converse", "openrouter") that are
    catalog/routing provenance, not vendor identity -- multiple provider
    routes to the SAME vendor collide on a bare model name and only one wins
    the catalog dict. `claude-fable-5` is a real vendored catalog entry whose
    bare-key winner is `bedrock_converse`/`vertex_ai-anthropic_models`
    depending on dict insertion order -- neither is "the family". The
    pre-fix `canonical_model_family` returned that raw routing tag; the
    fixed version must return the true semantic vendor regardless of which
    catalog row happens to win."""
    from polylogue.archive.semantic.pricing import PRICING, pricing_catalog_source

    # The catalog's own provenance tag is confirmed messy/non-vendor for this
    # model -- if this assertion ever starts failing because the vendored
    # catalog changed shape, that's fine; the point is canonical_model_family
    # must not track it either way.
    assert PRICING["claude-fable-5"].source_name != "anthropic"
    assert pricing_catalog_source("claude-fable-5") != "anthropic"
    assert canonical_model_family("claude-fable-5") == "anthropic"


def test_semantic_model_vendor_and_pricing_catalog_source_can_diverge() -> None:
    """A marketplace/routed model name and its pricing-catalog provenance
    are genuinely different axes -- the same vendor model routed through a
    marketplace (openrouter) still has vendor=anthropic even though its
    pricing-catalog source is the marketplace/routing layer, not the
    vendor."""
    from polylogue.archive.semantic.pricing import pricing_catalog_source, semantic_model_vendor

    marketplace_routed = "openrouter/anthropic/claude-3.5-sonnet"
    assert semantic_model_vendor(marketplace_routed) == "anthropic"
    assert pricing_catalog_source(marketplace_routed) != "anthropic"


def test_resolve_model_identity_keeps_axes_distinct_across_fixtures() -> None:
    """polylogue-4c27 AC: known Fable, Opus, GPT, Gemini, marketplace, and
    unknown fixtures keep vendor, model line, exact model, pricing source,
    and attribution source distinct fields (not aliases of each other)."""
    from polylogue.archive.semantic.pricing import resolve_model_identity

    fable = resolve_model_identity("claude-fable-5", attribution_source="dispatch_turn")
    assert fable.vendor == "anthropic"
    assert fable.model_line == "fable"
    assert fable.normalized_model == "claude-fable-5"
    assert fable.pricing_source is not None and fable.pricing_source != fable.vendor
    assert fable.attribution_source == "dispatch_turn"

    opus = resolve_model_identity("claude-opus-4-8", attribution_source="child_observed")
    assert opus.vendor == "anthropic"
    assert opus.model_line == "opus"
    assert opus.attribution_source == "child_observed"

    gpt = resolve_model_identity("gpt-4o", attribution_source="requested")
    assert gpt.vendor == "openai"
    assert gpt.model_line == "gpt-4"
    assert gpt.attribution_source == "requested"

    gemini = resolve_model_identity("gemini-2.5-pro", attribution_source="session_dominant_fallback")
    assert gemini.vendor == "google"
    assert gemini.model_line == "gemini-2.5"
    assert gemini.attribution_source == "session_dominant_fallback"

    marketplace = resolve_model_identity("openrouter/anthropic/claude-3.5-sonnet", attribution_source="requested")
    assert marketplace.vendor == "anthropic"
    assert marketplace.pricing_source != marketplace.vendor

    unknown = resolve_model_identity("some-totally-unknown-model-xyz", attribution_source="dispatch_turn")
    assert unknown.vendor is None
    assert unknown.model_line is None
    assert unknown.pricing_source is None
    assert unknown.confidence == "unknown"
    assert unknown.attribution_source == "dispatch_turn"

    unsupported = resolve_model_identity(None, attribution_source="requested")
    assert unsupported.raw_model_name is None
    assert unsupported.vendor is None
    assert unsupported.confidence == "unknown"
    assert unsupported.attribution_source == "requested"

    # None of the fixtures collapse vendor onto attribution_source or onto
    # each other's model line by accident.
    lines = {fixture.model_line for fixture in (fable, opus, gpt, gemini) if fixture.model_line}
    assert len(lines) == 4
