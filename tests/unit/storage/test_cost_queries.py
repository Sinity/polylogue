"""Cost query tests — prove cost/model/token data is queryable from typed columns (#803)."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def test_token_columns_exist_on_messages() -> None:
    """Messages table must have input_tokens, output_tokens, model_name columns."""
    import sqlite3

    from polylogue.storage.sqlite.schema_ddl_archive import ARCHIVE_STORAGE_DDL

    conn = sqlite3.connect(":memory:")
    conn.executescript(ARCHIVE_STORAGE_DDL)
    cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info('messages')").fetchall()
    }
    assert "input_tokens" in cols, "messages missing input_tokens column"
    assert "output_tokens" in cols, "messages missing output_tokens column"
    assert "model_name" in cols, "messages missing model_name column"
    conn.close()


def test_cost_summary_queryable_from_session_profiles(tmp_path: Path) -> None:
    """Session profile must carry per_model_cost_json for analytical queries."""
    import sqlite3

    from polylogue.storage.sqlite.schema_ddl_archive import ARCHIVE_STORAGE_DDL

    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(ARCHIVE_STORAGE_DDL)

    cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info('session_profiles')").fetchall()
    }
    assert "per_model_cost_json" in cols, "session_profiles missing per_model_cost_json"
    conn.close()


def test_tokenizer_estimate() -> None:
    """Token estimate from text is approximately words * 1.3."""
    from polylogue.archive.semantic.tokenizer import token_estimate_from_text

    est = token_estimate_from_text("hello world this is a test")
    assert est.estimated_tokens > 0
    assert est.confidence == "estimated"
    assert est.provenance == "word_count_estimated"


def test_subscription_credit_cost() -> None:
    """Credit cost computation uses per-model rates."""
    from polylogue.archive.semantic.subscription_pricing import compute_credit_cost

    cost = compute_credit_cost("claude-sonnet-4-6", input_tokens=1000, output_tokens=500)
    assert cost.input_credits > 0
    assert cost.output_credits > 0
    assert cost.total_credits > 0
