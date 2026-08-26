"""Cost query tests — prove cost/model/token data is queryable from typed columns (#803)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


def test_token_columns_exist_on_messages() -> None:
    """Messages table must have input_tokens, output_tokens, model_name columns."""

    from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL

    conn = sqlite3.connect(":memory:")
    conn.executescript(INDEX_DDL)
    cols = {row[1] for row in conn.execute("PRAGMA table_info('messages')").fetchall()}
    assert "input_tokens" in cols, "messages missing input_tokens column"
    assert "output_tokens" in cols, "messages missing output_tokens column"
    assert "model_name" in cols, "messages missing model_name column"
    conn.close()


def test_cost_summary_is_not_persisted_on_session_profiles(tmp_path: Path) -> None:
    """Session profiles must not carry a second cost authority."""

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    db = tmp_path / "test.db"
    initialize_archive_database(db, ArchiveTier.INDEX)
    with sqlite3.connect(str(db)) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info('session_profiles')").fetchall()}
    assert "per_model_cost_json" not in cols


def test_tokenizer_estimate() -> None:
    """Token estimate from text is approximately words * 1.3."""
    from polylogue.archive.semantic.tokenizer import token_estimate_from_text

    est = token_estimate_from_text("hello world this is a test")
    assert est.total_tokens > 0
    assert est.confidence == "estimated"
    assert est.provenance == "heuristic_estimated"


def test_subscription_credit_cost() -> None:
    """Credit cost computation uses per-model rates."""
    from polylogue.archive.semantic.subscription_pricing import compute_credit_cost, get_credit_rate

    cost = compute_credit_cost("claude-sonnet-4-6", input_tokens=1000, output_tokens=500)
    assert cost > 0
    # Verify the model-specific rate is non-trivial
    rate = get_credit_rate("claude-sonnet-4-6")
    assert rate is not None, "expected a credit rate for claude-sonnet-4-6"


def test_subscription_output_credits_are_5x_input() -> None:
    """Claude bills output at 5x input at API rates; credits must mirror that.

    Regression guard: every entry previously set output_credits == input_credits,
    understating output (and therefore total subscription) cost by 5x.
    """
    from polylogue.archive.semantic.subscription_pricing import MODEL_CREDIT_RATES

    assert MODEL_CREDIT_RATES, "expected at least one Claude credit rate"
    for name, rate in MODEL_CREDIT_RATES.items():
        # rates share one divisor, so the 5x ratio holds on the credit ints.
        assert rate.input_divisor == rate.output_divisor, name
        assert rate.output_credits == 5 * rate.input_credits, (
            f"{name}: output_credits {rate.output_credits} must be 5x input "
            f"{rate.input_credits} (API output:input ratio)"
        )
        # cache reads are free on subscription plans; cache writes bill at input rate.
        assert rate.cache_read_credits == 0, name
        assert rate.cache_write_credits == rate.input_credits, name


def test_subscription_credit_cost_output_weight() -> None:
    """Output tokens must cost 5x input tokens for the same count.

    Uses a divisor-aligned token count (15) so math.ceil rounding does not mask
    the exact 5x ratio.
    """
    from polylogue.archive.semantic.subscription_pricing import compute_credit_cost

    input_only = compute_credit_cost("claude-opus-4-6", input_tokens=1500, output_tokens=0)
    output_only = compute_credit_cost("claude-opus-4-6", input_tokens=0, output_tokens=1500)
    assert input_only > 0 and output_only > 0
    assert output_only == 5 * input_only


def test_cost_rollup_unions_normalized_model_session_counts_and_separates_basis_lanes() -> None:
    """polylogue-cfqu0/qvjk5/wdv1x: grouped cohorts must union sessions once.

    Anti-vacuity: summing the raw model/provenance COUNT(DISTINCT) values
    reports two sessions for the one-session variant pair and leaves the
    API-equivalent or USD subscription lane at zero/wrong units.
    """

    from polylogue.archive.semantic.subscription_pricing import credits_to_usd
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY, origin TEXT, sort_key_ms INTEGER,
            updated_at_ms INTEGER, reported_cost_usd REAL
        );
        CREATE TABLE session_model_usage (
            session_id TEXT, model_name TEXT, input_tokens INTEGER,
            output_tokens INTEGER, cache_read_tokens INTEGER,
            cache_write_tokens INTEGER, cost_usd REAL, cost_credits REAL,
            cost_provenance TEXT
        );
        CREATE TABLE session_profiles (session_id TEXT PRIMARY KEY);
        INSERT INTO sessions VALUES ('s1', 'chatgpt-export', 1, 1, NULL);
        INSERT INTO sessions VALUES ('s2', 'chatgpt-export', 2, 2, NULL);
        INSERT INTO session_model_usage VALUES
            ('s1', 'gpt-5-5', 100, 10, 0, 0, 1.0, 100.0, 'priced');
        INSERT INTO session_model_usage VALUES
            ('s2', 'gpt-5-5-pro', 200, 20, 0, 0, 2.0, 200.0, 'estimated');
        """
    )
    archive = ArchiveStore.__new__(ArchiveStore)
    archive._conn = conn

    (rollup,) = archive.list_cost_rollup_insights(origin="chatgpt-export")

    assert rollup.session_count == 2
    assert rollup.per_model_breakdown[0].session_count + rollup.per_model_breakdown[1].session_count == 2
    assert rollup.basis.api_equivalent_usd == pytest.approx(3.0)
    assert rollup.basis.catalog_priced_usd == pytest.approx(3.0)
    assert rollup.basis.subscription_equivalent_usd == pytest.approx(credits_to_usd(300.0))
    conn.close()
