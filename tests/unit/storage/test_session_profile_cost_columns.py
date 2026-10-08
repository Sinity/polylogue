"""Profile rows do not persist usage or pricing mirrors."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.usage import session_usage_costs_for_connection
from tests.infra.index_writer import write_fixture_index_session


def _conn(tmp_path: Path) -> sqlite3.Connection:
    initialize_active_archive_root(tmp_path)
    conn = connect_measured(tmp_path / "index.db")
    conn.row_factory = sqlite3.Row
    return conn


def _session(session_id: str, model: str) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id=session_id,
        title="canonical usage test",
        models_used=[model],
        messages=[
            ParsedMessage(
                provider_message_id="a1",
                role=Role.ASSISTANT,
                text="done",
                model_name=model,
                input_tokens=1_000,
                output_tokens=500,
                cache_read_tokens=200,
                cache_write_tokens=100,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="done")],
            )
        ],
    )


def test_session_profiles_have_no_usage_or_pricing_columns(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    names = {str(row[1]) for row in conn.execute("PRAGMA table_info(session_profiles)")}
    assert not names & {
        "total_input_tokens",
        "total_output_tokens",
        "total_cache_read_tokens",
        "total_cache_write_tokens",
        "total_cost_usd",
        "total_credit_cost",
        "cost_is_estimated",
        "cost_provenance",
        "per_model_cost_json",
        "cost_usd",
        "cost_credits",
        "priced_with",
        "priced_at_ms",
    }
    conn.close()


def test_canonical_usage_projection_preserves_priced_and_unpriced_states(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    write_fixture_index_session(conn, _session("priced", "claude-sonnet-4-5"))
    write_fixture_index_session(conn, _session("unpriced", "totally-unknown-model-xyz"))
    ids = [str(row[0]) for row in conn.execute("SELECT session_id FROM sessions ORDER BY session_id")]
    costs = session_usage_costs_for_connection(conn, ids)
    assert costs["claude-code-session:priced"].total_usd is not None
    assert costs["claude-code-session:unpriced"].total_usd is None
    assert costs["claude-code-session:unpriced"].availability == "unpriced"
    conn.close()


def test_cost_insight_keeps_unpriced_tokens_unknown(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    write_fixture_index_session(conn, _session("unpriced", "totally-unknown-model-xyz"))
    from polylogue.storage.sqlite.archive_tiers.archive import _session_cost_insight_from_archive_row

    row = conn.execute("SELECT * FROM sessions WHERE native_id = 'unpriced'").fetchone()
    insight = _session_cost_insight_from_archive_row(
        conn, row, session_usage_costs_for_connection(conn, [str(row["session_id"])])[str(row["session_id"])]
    )
    assert insight.estimate.total_usd is None
    assert insight.estimate.unavailable_reason == "price_not_materialized"
    conn.close()


def test_session_cost_lookup_pages_at_the_connection_bind_limit(tmp_path: Path) -> None:
    """Both previous IN statements failed when all requested IDs exceeded the limit."""
    conn = _conn(tmp_path)
    try:
        ids = [write_fixture_index_session(conn, _session(f"bind-{i}", "claude-sonnet-4-5")) for i in range(5)]
        previous_limit = conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 2)
        try:
            costs = session_usage_costs_for_connection(conn, [*ids, ids[0]])
        finally:
            conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous_limit)
        assert set(costs) == set(ids)
        assert all(cost.input_tokens == 1_000 and cost.output_tokens == 500 for cost in costs.values())
    finally:
        conn.close()


def test_session_subscription_fallback_normalizes_provider_qualified_models(tmp_path: Path) -> None:
    """The raw qualified name formerly missed the credit catalog and returned zero."""
    from polylogue.archive.semantic.subscription_pricing import compute_credit_cost, credits_to_usd

    conn = _conn(tmp_path)
    try:
        session_id = write_fixture_index_session(conn, _session("qualified-credit", "claude-sonnet-4-5"))
        conn.execute(
            "UPDATE session_model_usage SET model_name = ?, cost_credits = NULL WHERE session_id = ?",
            ("anthropic/claude-sonnet-4-5", session_id),
        )
        cost = session_usage_costs_for_connection(conn, [session_id])[session_id]
        expected = credits_to_usd(compute_credit_cost("claude-sonnet-4-5", 1_000, 500, 200, 100))
        assert expected > 0
        assert cost.subscription_equivalent_usd == pytest.approx(expected)
    finally:
        conn.close()


@pytest.mark.parametrize("configured_tier", [None, "max_20x"])
def test_session_subscription_amount_is_dollars_not_raw_credits(tmp_path: Path, configured_tier: str | None) -> None:
    """The old projection put the raw stored credit count into a USD field."""
    from polylogue.archive.semantic.subscription_pricing import credits_to_usd
    from polylogue.storage.sqlite.archive_tiers.user_settings_write import set_user_setting

    conn = _conn(tmp_path)
    try:
        session_id = write_fixture_index_session(conn, _session("stored-credit", "claude-sonnet-4-5"))
        credits = 1_234_567
        conn.execute("UPDATE session_model_usage SET cost_credits = ? WHERE session_id = ?", (credits, session_id))
        if configured_tier is not None:
            with sqlite3.connect(tmp_path / "user.db") as user_conn:
                set_user_setting(user_conn, "subscription_tier", configured_tier)
        cost = session_usage_costs_for_connection(conn, [session_id])[session_id]
        expected = credits_to_usd(credits, tier=configured_tier or "pro")
        assert expected != credits
        assert cost.subscription_equivalent_usd == pytest.approx(expected)
    finally:
        conn.close()


@pytest.mark.parametrize("stored_tier", [None, "tier-removed-from-catalog"])
def test_session_subscription_amount_is_unknown_without_a_rate_or_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stored_tier: str | None
) -> None:
    """Anti-vacuity: an unrated model or a durable tier setting the current
    catalog no longer declares formerly converted to a reported $0.00
    subscription equivalent instead of an unknown one."""
    from polylogue.storage import usage

    conn = _conn(tmp_path)
    try:
        model = "claude-sonnet-4-5" if stored_tier is not None else "claude-unrated-future-model"
        session_id = write_fixture_index_session(conn, _session("unknown-credit", model))
        conn.execute("UPDATE session_model_usage SET cost_credits = NULL WHERE session_id = ?", (session_id,))
        if stored_tier is not None:
            monkeypatch.setattr(usage, "_resolve_subscription_tier_setting", lambda _root: stored_tier)
        cost = session_usage_costs_for_connection(conn, [session_id])[session_id]
        assert cost.input_tokens == 1_000
        assert cost.subscription_equivalent_usd is None
    finally:
        conn.close()
