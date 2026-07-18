"""Generated-surface contract for parser-gated query discovery."""

from __future__ import annotations

from pathlib import Path

from devtools.render_query_discovery import GENERATED_BEGIN, GENERATED_END, build_block, render_document
from polylogue.archive.query.discovery import (
    QUERY_DISCOVERY_EXAMPLES,
    QUERY_DISCOVERY_GRAMMAR,
    QUERY_DISCOVERY_NEGATIVE_EXAMPLES,
)


def test_search_reference_generated_block_is_in_sync() -> None:
    document = Path("docs/search.md").read_text(encoding="utf-8")

    assert render_document(document) == document
    assert document.count(GENERATED_BEGIN) == 1
    assert document.count(GENERATED_END) == 1


def test_generated_block_projects_counts_semantics_and_shipped_failures() -> None:
    block = build_block()

    assert f"{len(QUERY_DISCOVERY_EXAMPLES)} positive" in block
    assert f"{len(QUERY_DISCOVERY_NEGATIVE_EXAMPLES)} negative" in block
    for semantics in ("exhaustive", "top-k", "sample", "aggregate", "bounded-context", "recursive-page"):
        assert f"`{semantics}`" in block
    for form in QUERY_DISCOVERY_GRAMMAR.values():
        assert form.replace("|", r"\|") in block
    for row in QUERY_DISCOVERY_NEGATIVE_EXAMPLES:
        if row.shipped_at:
            assert row.expression in block
            assert row.corrected_form in block
