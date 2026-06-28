"""Cost query tests — prove cost/model/token data is queryable from typed columns (#803)."""

from __future__ import annotations

import sqlite3
from pathlib import Path


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


def test_cost_summary_queryable_from_session_profiles(tmp_path: Path) -> None:
    """Session profile must carry per_model_cost_json for analytical queries."""

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    db = tmp_path / "test.db"
    initialize_archive_database(db, ArchiveTier.INDEX)
    with sqlite3.connect(str(db)) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info('session_profiles')").fetchall()}
    assert "per_model_cost_json" in cols, "session_profiles missing per_model_cost_json"


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
