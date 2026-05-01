from __future__ import annotations

from types import SimpleNamespace

from polylogue.archive.semantic.pricing import CostEstimatePayload
from polylogue.insights.archive import SessionCostInsight
from polylogue.insights.archive_models import ArchiveInsightProvenance
from polylogue.operations.archive import (
    _build_search_snippet,
    _cost_model_matches,
    _cost_status_matches,
    _query_wants_search,
    _row_int,
    _row_str,
    _slice_products,
)


def test_archive_helper_snippets_rows_queries_and_slices() -> None:
    long_text = "prefix " * 20 + "needle" + " suffix" * 40

    assert _build_search_snippet("", "needle") == ""
    assert _build_search_snippet(long_text, "needle").startswith("...")
    assert _build_search_snippet(long_text, "needle").endswith("...")
    assert _row_int({"value": True}, "value") == 1
    assert _row_int({"value": 2}, "value") == 2
    assert _row_int({"value": 2.9}, "value") == 2
    assert _row_int({"value": "3"}, "value") == 3
    assert _row_int({"value": "bad"}, "value") == 0
    assert _row_int({"value": object()}, "value") == 0
    assert _row_str({"value": 7}, "value") == "7"
    assert _row_str({}, "missing", default="fallback") == "fallback"

    assert _query_wants_search(SimpleNamespace(wants_search=True, query=())) is True
    assert _query_wants_search(SimpleNamespace(wants_search=False, query=("needle",))) is False
    assert _query_wants_search(SimpleNamespace(query=("needle",))) is True
    assert _query_wants_search(SimpleNamespace(query=())) is False
    assert _slice_products([1, 2, 3, 4], offset=1, limit=2) == [2, 3]
    assert _slice_products([1, 2, 3], offset=0, limit=None) == [1, 2, 3]


def test_archive_cost_product_filters() -> None:
    product = SessionCostInsight(
        conversation_id="conv-cost",
        provider_name="claude-ai",
        estimate=CostEstimatePayload(
            provider_name="claude-ai",
            conversation_id="conv-cost",
            model_name="claude-sonnet-4-5",
            normalized_model="claude-sonnet-4-5",
            status="priced",
        ),
        provenance=ArchiveInsightProvenance(materializer_version=1, materialized_at="2026-01-01T00:00:00Z"),
    )

    assert _cost_model_matches(product, None) is True
    assert _cost_model_matches(product, "claude-sonnet-4-5") is True
    assert _cost_model_matches(product, "anthropic/claude-sonnet-4-5-20250929") is True
    assert _cost_model_matches(product, "gpt-5") is False
    assert _cost_status_matches(product, None) is True
    assert _cost_status_matches(product, "priced") is True
    assert _cost_status_matches(product, "missing") is False
