"""Hypothesis strategies for source detection and JSON wire-format laws."""

from __future__ import annotations

import json
from typing import TypeAlias

from hypothesis import strategies as st

from polylogue.lib.json import JSONDocument, JSONValue

JSONRoundTripScalar: TypeAlias = str | int | bool | None


def _json_value_strategy() -> st.SearchStrategy[JSONValue]:
    """Generate JSON values that preserve shape across the runtime JSON helpers."""
    scalar = st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-(2**63), max_value=2**63 - 1),
        st.text(max_size=40),
    )
    return st.recursive(
        scalar,
        lambda child: st.one_of(
            st.lists(child, max_size=4),
            st.dictionaries(st.text(min_size=1, max_size=12), child, max_size=4),
        ),
        max_leaves=8,
    )


@st.composite
def json_document_strategy(draw: st.DrawFn) -> JSONDocument:
    """Generate a small JSON object suitable for streaming round-trip laws."""
    return draw(
        st.dictionaries(
            st.text(min_size=1, max_size=12),
            _json_value_strategy(),
            min_size=1,
            max_size=4,
        )
    )


@st.composite
def json_array_bytes_strategy(draw: st.DrawFn) -> tuple[list[JSONDocument], bytes]:
    """Generate a root JSON array and its encoded bytes."""
    documents = draw(st.lists(json_document_strategy(), min_size=1, max_size=6))
    return documents, json.dumps(documents).encode("utf-8")


@st.composite
def conversations_wrapper_bytes_strategy(draw: st.DrawFn) -> tuple[list[JSONDocument], bytes]:
    """Generate a {"conversations": [...]} document and its encoded bytes."""
    documents = draw(st.lists(json_document_strategy(), min_size=1, max_size=6))
    wrapper = {"conversations": documents}
    return documents, json.dumps(wrapper).encode("utf-8")


@st.composite
def jsonl_bytes_strategy(draw: st.DrawFn) -> tuple[list[JSONDocument], bytes]:
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
