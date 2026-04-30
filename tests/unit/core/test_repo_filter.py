"""Tests for the ``--repo`` filter — CLI/MCP/storage end-to-end."""

from __future__ import annotations

from polylogue.lib.query.spec import ConversationQuerySpec
from polylogue.storage.backends.queries.filter_builder import _build_conversation_filters


def test_repo_param_lands_in_spec() -> None:
    spec = ConversationQuerySpec.from_params({"repo": "thoughtspace,sinex"})
    assert spec.repo_names == ("thoughtspace", "sinex")


def test_repo_param_propagates_to_plan() -> None:
    spec = ConversationQuerySpec.from_params({"repo": "thoughtspace"})
    plan = spec.to_plan()
    assert plan.repo_names == ("thoughtspace",)


def test_repo_param_propagates_to_record_query() -> None:
    spec = ConversationQuerySpec.from_params({"repo": "thoughtspace"})
    plan = spec.to_plan()
    record_query = plan.record_query
    assert record_query.repo_names == ("thoughtspace",)


def test_repo_emits_sql_exists_subquery() -> None:
    where, params = _build_conversation_filters(repo_names=["thoughtspace"])
    assert "session_profiles" in where
    assert "json_each" in where
    assert "repo_names_json" in where
    assert "EXISTS" in where
    assert "thoughtspace" in params


def test_repo_multiple_values_use_in_clause() -> None:
    where, params = _build_conversation_filters(repo_names=["thoughtspace", "sinex"])
    assert "IN (?,?)" in where
    assert "thoughtspace" in params
    assert "sinex" in params


def test_repo_absent_emits_no_clause() -> None:
    where, _ = _build_conversation_filters()
    assert "session_profiles" not in where


def test_repo_combines_with_other_filters() -> None:
    where, params = _build_conversation_filters(
        repo_names=["thoughtspace"],
        provider="claude-code",
        title_contains="bug",
    )
    assert "provider_name = ?" in where
    assert "title LIKE" in where
    assert "session_profiles" in where
    assert "claude-code" in params
    assert "thoughtspace" in params
    assert any("bug" in str(p) for p in params)


def test_repo_in_filter_chain() -> None:
    spec = ConversationQuerySpec.from_params({"repo": "thoughtspace"})
    plan = spec.to_plan()
    assert plan.repo_names == ("thoughtspace",)
    assert plan.record_query.repo_names == ("thoughtspace",)


def test_repo_in_mcp_request() -> None:
    from polylogue.mcp.query_contracts import MCPConversationQueryRequest

    request = MCPConversationQueryRequest(repo="thoughtspace", limit=5)
    spec = request.build_spec(lambda x: int(x) if isinstance(x, int | float | str) else 5)
    assert spec.repo_names == ("thoughtspace",)


def test_repo_landing_in_sql_pushdown_params() -> None:
    spec = ConversationQuerySpec.from_params({"repo": "thoughtspace"})
    plan = spec.to_plan()
    params = plan.sql_pushdown_params()
    assert params["repo_names"] == ["thoughtspace"]


def test_repo_describe_includes_label() -> None:
    spec = ConversationQuerySpec.from_params({"repo": "thoughtspace"})
    descriptions = spec.describe()
    assert any("repo: thoughtspace" in d for d in descriptions)


def test_repo_fluent_builder() -> None:
    from polylogue.lib.filter.builder import ConversationFilterBuilderMixin

    filt = ConversationFilterBuilderMixin()
    filt._plan = ConversationQuerySpec().to_plan()
    filt.repo("thoughtspace")
    assert filt._plan.repo_names == ("thoughtspace",)
