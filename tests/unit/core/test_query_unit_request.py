"""Shared terminal query-unit request normalization."""

from __future__ import annotations

import pytest

from polylogue.archive.query.expression import ExpressionCompileError
from polylogue.archive.query.unit_results import query_unit_request


def test_query_unit_request_compiles_source_and_session_filters() -> None:
    request = query_unit_request(
        expression="messages where text:timeout",
        limit=25,
        offset=3,
        origin="claude-code-session",
        repo="polylogue, sinex",
        has_tool_use=True,
        min_messages="2",
    )

    assert request.expression == "messages where text:timeout"
    assert request.source.unit == "message"
    assert request.limit == 25
    assert request.offset == 3
    assert request.session_filters is not None
    assert request.session_filters["origin"] == "claude-code-session"
    assert request.session_filters["repo_names"] == ("polylogue", "sinex")
    assert request.session_filters["has_tool_use"] is True
    assert request.session_filters["min_messages"] == 2


def test_query_unit_request_rejects_session_expressions() -> None:
    with pytest.raises(ExpressionCompileError, match="requires an explicit"):
        query_unit_request(expression="timeout", limit=10)


def test_query_unit_request_preserves_prebuilt_session_filters() -> None:
    filters = {"origin": "chatgpt-export"}

    request = query_unit_request(
        expression="messages where text:timeout",
        limit=10,
        session_filters=filters,
        origin="claude-code-session",
    )

    assert request.session_filters is filters


def test_query_unit_request_accepts_api_style_tuple_filters() -> None:
    request = query_unit_request(
        expression="messages where text:timeout",
        limit=10,
        origins=("codex-session",),
        tags=("important", "follow-up"),
        repo_names=("polylogue",),
        tool_terms=("bash",),
        action_terms=("file_edit",),
        referenced_paths=("polylogue/archive/query",),
    )

    assert request.session_filters is not None
    assert request.session_filters["origins"] == ("codex-session",)
    assert request.session_filters["tags"] == ("important", "follow-up")
    assert request.session_filters["repo_names"] == ("polylogue",)
    assert request.session_filters["tool_terms"] == ("bash",)
    assert request.session_filters["action_terms"] == ("file_edit",)
    assert request.session_filters["referenced_paths"] == ("polylogue/archive/query",)
