"""Query-field descriptor catalog contracts."""

from __future__ import annotations

from datetime import datetime, timezone

from polylogue.lib.query.fields import (
    QUERY_FIELD_DESCRIPTORS,
    active_plan_field_names,
    storage_filters_require_stats_join,
)
from polylogue.lib.query.plan import ConversationQueryPlan
from polylogue.lib.query.spec import ConversationQuerySpec
from polylogue.storage.backends.queries.filter_builder import _needs_stats_join
from polylogue.types import Provider


def test_query_field_catalog_drives_spec_presence_and_descriptions() -> None:
    spec = ConversationQuerySpec(
        query_terms=("sqlite", "locks"),
        referenced_path=("polylogue/storage",),
        providers=(Provider.CODEX,),
        repo_names=("thoughtspace",),
        filter_has_tool_use=True,
        min_messages=3,
        since="2024-01-01",
    )

    assert spec.has_filters() is True
    assert spec.describe() == [
        "search: sqlite locks",
        "referenced-path: polylogue/storage",
        "provider: codex",
        "repo: thoughtspace",
        "has: tool_use (sql)",
        "min_messages: 3",
        "since: 2024-01-01",
    ]


def test_query_field_catalog_drives_plan_presence_descriptions_and_pushdown() -> None:
    since = datetime(2024, 1, 1, tzinfo=timezone.utc)
    plan = ConversationQueryPlan(
        query_terms=("sqlite",),
        referenced_path=("polylogue/storage",),
        tool_terms=("bash",),
        providers=(Provider.CODEX,),
        repo_names=("thoughtspace",),
        title="locks",
        since=since,
        filter_has_tool_use=True,
        min_messages=3,
    )

    assert active_plan_field_names(plan) == (
        "query_terms",
        "referenced_path",
        "tool_terms",
        "providers",
        "repo_names",
        "title",
        "filter_has_tool_use",
        "min_messages",
        "since",
    )
    assert plan.has_filters() is True
    assert plan.describe() == [
        "contains: sqlite",
        "referenced-path: polylogue/storage",
        "tool: bash",
        "provider: codex",
        "repo: thoughtspace",
        "title: locks",
        "has_tool_use",
        "min_messages: 3",
        "since: 2024-01-01T00:00:00+00:00",
    ]

    assert plan.sql_pushdown_params() == {
        "provider": "codex",
        "referenced_path": ["polylogue/storage"],
        "tool_terms": ["bash"],
        "repo_names": ["thoughtspace"],
        "title_contains": "locks",
        "has_tool_use": True,
        "min_messages": 3,
        "since": "2024-01-01T00:00:00+00:00",
    }
    assert all(
        isinstance(value, (str, int, bool))
        or (isinstance(value, list) and all(isinstance(item, str) for item in value))
        for value in plan.sql_pushdown_params().values()
    )

    record_query = plan.record_query
    assert record_query.provider == "codex"
    assert record_query.referenced_path == ("polylogue/storage",)
    assert record_query.tool_terms == ("bash",)
    assert record_query.repo_names == ("thoughtspace",)
    assert record_query.title_contains == "locks"
    assert record_query.has_tool_use is True
    assert record_query.min_messages == 3
    assert record_query.since == "2024-01-01T00:00:00+00:00"


def test_query_field_catalog_marks_storage_stats_join_fields() -> None:
    assert {descriptor.name for descriptor in QUERY_FIELD_DESCRIPTORS if descriptor.requires_stats_join} == {
        "filter_has_tool_use",
        "filter_has_thinking",
        "filter_has_paste",
        "typed_only",
        "min_messages",
        "max_messages",
        "min_words",
    }

    assert storage_filters_require_stats_join({"has_tool_use": True}) is True
    assert storage_filters_require_stats_join({"min_messages": 2}) is True
    assert storage_filters_require_stats_join({"title_contains": "locks"}) is False
    assert _needs_stats_join(has_tool_use=True) is True
    assert _needs_stats_join(min_words=10) is True
    assert _needs_stats_join() is False
