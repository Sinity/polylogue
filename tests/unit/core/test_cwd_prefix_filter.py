"""Tests for the ``cwd_prefix`` filter primitive — CLI/MCP/storage end-to-end."""

from __future__ import annotations

from polylogue.lib.query.spec import ConversationQuerySpec
from polylogue.storage.backends.queries.filter_builder import _build_conversation_filters


def test_cwd_prefix_param_lands_in_spec() -> None:
    spec = ConversationQuerySpec.from_params({"cwd_prefix": "/realm/project/polylogue"})
    assert spec.cwd_prefix == "/realm/project/polylogue"


def test_cwd_prefix_propagates_to_plan() -> None:
    spec = ConversationQuerySpec.from_params({"cwd_prefix": "/x/y"})
    plan = spec.to_plan()
    assert plan.cwd_prefix == "/x/y"


def test_cwd_prefix_propagates_to_record_query() -> None:
    spec = ConversationQuerySpec.from_params({"cwd_prefix": "/x/y"})
    plan = spec.to_plan()
    record_query = plan.record_query
    assert record_query.cwd_prefix == "/x/y"


def test_cwd_prefix_emits_sql_clause() -> None:
    where, params = _build_conversation_filters(cwd_prefix="/realm/project/polylogue")
    assert "json_each" in where
    assert "working_directories" in where
    assert "/realm/project/polylogue%" in params


def test_cwd_prefix_escapes_sql_wildcards() -> None:
    where, params = _build_conversation_filters(cwd_prefix="/path/100%cool_dir")
    assert "ESCAPE" in where
    # The escaped prefix has the wildcards escaped
    assert any(r"100\%cool\_dir%" in str(p) for p in params)


def test_cwd_prefix_absent_emits_no_clause() -> None:
    where, _ = _build_conversation_filters()
    assert "working_directories" not in where


def test_cwd_prefix_combines_with_other_filters() -> None:
    where, params = _build_conversation_filters(
        cwd_prefix="/repo",
        provider="claude-code",
        title_contains="bug",
    )
    assert "provider_name = ?" in where
    assert "title LIKE" in where
    assert "working_directories" in where
    assert "claude-code" in params
    assert any("bug" in str(p) for p in params)


def test_cwd_prefix_in_filter_chain() -> None:
    """Confirm the builder method reaches the plan via spec params alias."""
    spec = ConversationQuerySpec.from_params({"cwd_prefix": "/realm/project/polylogue"})
    plan = spec.to_plan()
    assert plan.cwd_prefix == "/realm/project/polylogue"
    assert plan.record_query.cwd_prefix == "/realm/project/polylogue"


def test_cwd_prefix_in_mcp_request() -> None:
    """MCP tool surface accepts cwd_prefix and threads it to spec."""
    from polylogue.mcp.query_contracts import MCPConversationQueryRequest

    request = MCPConversationQueryRequest(cwd_prefix="/realm/project/polylogue", limit=5)
    spec = request.build_spec(lambda x: int(x) if isinstance(x, int | float | str) else 5)
    assert spec.cwd_prefix == "/realm/project/polylogue"


def test_cwd_prefix_landing_in_sql_pushdown_params() -> None:
    spec = ConversationQuerySpec.from_params({"cwd_prefix": "/x/y"})
    plan = spec.to_plan()
    params = plan.sql_pushdown_params()
    assert params["cwd_prefix"] == "/x/y"


def test_cwd_prefix_describe_includes_label() -> None:
    spec = ConversationQuerySpec.from_params({"cwd_prefix": "/x/y"})
    descriptions = spec.describe()
    assert any("cwd-prefix" in d for d in descriptions)
