"""Hypothesis strategies for source detection and JSON wire-format laws."""

from __future__ import annotations

import json
from typing import Any

from hypothesis import strategies as st


@st.composite
def json_document_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate a small JSON object suitable for streaming round-trip laws."""
    scalar = st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.text(max_size=40),
    )
    value = st.recursive(
        scalar,
        lambda child: st.one_of(
            st.lists(child, max_size=4),
            st.dictionaries(st.text(min_size=1, max_size=12), child, max_size=4),
        ),
        max_leaves=8,
    )
    return draw(
        st.dictionaries(
            st.text(min_size=1, max_size=12),
            value,
            min_size=1,
            max_size=4,
        )
    )


@st.composite
def json_array_bytes_strategy(draw: st.DrawFn) -> tuple[list[dict[str, Any]], bytes]:
    """Generate a root JSON array and its encoded bytes."""
    documents = draw(st.lists(json_document_strategy(), min_size=1, max_size=6))
    return documents, json.dumps(documents).encode("utf-8")


@st.composite
def conversations_wrapper_bytes_strategy(draw: st.DrawFn) -> tuple[list[dict[str, Any]], bytes]:
    """Generate a {"conversations": [...]} document and its encoded bytes."""
    documents = draw(st.lists(json_document_strategy(), min_size=1, max_size=6))
    wrapper = {"conversations": documents}
    return documents, json.dumps(wrapper).encode("utf-8")


@st.composite
def jsonl_bytes_strategy(draw: st.DrawFn) -> tuple[list[dict[str, Any]], bytes]:
    """Generate JSONL bytes with optional blank lines while preserving record order."""
    documents = draw(st.lists(json_document_strategy(), min_size=1, max_size=6))
    prefix_blank_lines = draw(st.integers(min_value=0, max_value=2))
    suffix_blank_lines = draw(st.integers(min_value=0, max_value=2))
    gaps = draw(st.lists(st.integers(min_value=0, max_value=2), min_size=len(documents), max_size=len(documents)))

    lines: list[str] = [""] * prefix_blank_lines
    for document, blank_lines in zip(documents, gaps, strict=True):
        lines.append(json.dumps(document))
        lines.extend([""] * blank_lines)
    lines.extend([""] * suffix_blank_lines)
    return documents, "\n".join(lines).encode("utf-8")
