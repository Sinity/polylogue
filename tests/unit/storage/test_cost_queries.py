"""Cost query tests — prove cost/model/token data is queryable and contracts match (#803, #870)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def test_token_columns_exist_on_messages() -> None:
    """Messages table must have input_tokens, output_tokens, model_name columns."""

    from polylogue.storage.sqlite.schema_ddl_archive import ARCHIVE_STORAGE_DDL

    conn = sqlite3.connect(":memory:")
    conn.executescript(ARCHIVE_STORAGE_DDL)
    cols = {row[1] for row in conn.execute("PRAGMA table_info('messages')").fetchall()}
    assert "input_tokens" in cols, "messages missing input_tokens column"
    assert "output_tokens" in cols, "messages missing output_tokens column"
    assert "model_name" in cols, "messages missing model_name column"
    conn.close()


def test_cost_summary_columns_exist_on_session_profiles(tmp_path: Path) -> None:
    """Session profile must carry cost-related columns."""

    from polylogue.storage.sqlite.schema_ddl import SCHEMA_DDL

    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(SCHEMA_DDL)

    cols = {row[1] for row in conn.execute("PRAGMA table_info('session_profiles')").fetchall()}
    assert "per_model_cost_json" in cols, "session_profiles missing per_model_cost_json"
    assert "total_credit_cost" in cols, "session_profiles missing total_credit_cost"
    assert "cost_provenance" in cols, "session_profiles missing cost_provenance"
    conn.close()


def test_session_profile_record_matches_cost_summary_contract() -> None:
    """SessionProfileRecord cost fields must be a superset of SessionCostSummary storage fields."""

    from polylogue.archive.semantic.cost_records import SessionCostSummary
    from polylogue.storage.insights.session.records import SessionProfileRecord

    summary_model_fields = SessionCostSummary.model_fields
    profile_model_fields = SessionProfileRecord.model_fields

    # Fields mapped from SessionCostSummary into SessionProfileRecord
    mapped_fields = {
        "total_input_tokens",
        "total_output_tokens",
        "total_cache_read_tokens",
        "total_cache_write_tokens",
        "total_credit_cost",
        "cost_provenance",
    }
    for field_name in mapped_fields:
        assert field_name in profile_model_fields, f"SessionProfileRecord missing field: {field_name}"
        assert field_name in summary_model_fields, f"SessionCostSummary missing field: {field_name}"

    # per_model_cost_json stores serialized per_model breakdowns
    assert "per_model_cost_json" in profile_model_fields
    assert "per_model" in summary_model_fields

    # total_api_cost_usd is mapped to total_cost_usd (legacy field name differs)
    assert "total_cost_usd" in profile_model_fields
    assert "total_api_cost_usd" in summary_model_fields


def test_session_cost_summary_per_model_serialization_roundtrip() -> None:
    """Per-model breakdown from SessionCostSummary must roundtrip through JSON
    for storage in session_profiles.per_model_cost_json."""

    from polylogue.archive.semantic.cost_records import SessionCostBreakdown, SessionCostSummary

    breakdown = SessionCostBreakdown(
        normalized_model="claude-sonnet-4-6",
        provider_model_name="claude-sonnet-4-6-20250514",
        input_tokens=1000,
        output_tokens=500,
        total_tokens=1500,
        api_cost_usd=0.0105,
        credit_cost=6000,
        subscription_equivalent_usd=0.00553,
        confidence="reported",
        provenance="provider_reported",
    )

    summary = SessionCostSummary(
        total_input_tokens=1000,
        total_output_tokens=500,
        total_api_cost_usd=0.0105,
        total_credit_cost=6000,
        total_subscription_equivalent_usd=0.00553,
        cost_provenance="provider_reported",
        cost_confidence="reported",
        tokenizer_version="word-count-1.3-v1",
        price_snapshot_version="polylogue-curated-litellm-shaped-seed",
        per_model=(breakdown,),
    )

    serialized = json.dumps([b.model_dump(mode="json") for b in summary.per_model])
    deserialized = json.loads(serialized)
    assert len(deserialized) == 1
    assert deserialized[0]["normalized_model"] == "claude-sonnet-4-6"
    assert deserialized[0]["input_tokens"] == 1000
    assert deserialized[0]["credit_cost"] == 6000


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


def test_usage_outlook_returns_typed_model() -> None:
    """compute_usage_outlook must return a typed UsageOutlookPayload, not a raw dict."""
    from polylogue.archive.semantic.outlook import compute_usage_outlook
    from polylogue.archive.semantic.subscription_models import UsageOutlookPayload

    conversations: list[dict[str, object]] = []
    result = compute_usage_outlook(conversations, plan_name="pro")
    assert isinstance(result, UsageOutlookPayload)
    assert result.plan_name == "pro"
    assert result.plan_confidence.value == "derived"
    assert result.per_model == []
    assert result.anomalies == []
