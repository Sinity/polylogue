"""Property laws for FTS5 search and query escaping.

Supersedes specific parametrized examples in test_fts5.py.
"""
from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from polylogue.storage.search import escape_fts5_query


@given(st.text())
def test_escape_fts5_query_never_crashes(text: str) -> None:
    """escape_fts5_query handles any Unicode input without raising."""
    result = escape_fts5_query(text)
    assert isinstance(result, str)


@given(st.text())
def test_escape_fts5_query_result_is_string(text: str) -> None:
    """escape_fts5_query always returns a string."""
    result = escape_fts5_query(text)
    assert isinstance(result, str)


@given(st.text(min_size=1))
def test_escape_fts5_query_non_empty_input_non_empty_output(text: str) -> None:
    """Non-empty input produces non-empty escaped output."""
    result = escape_fts5_query(text)
    assert len(result) > 0 or text.strip() == ""
