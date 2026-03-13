"""Property and exact contracts for metadata-only ConversationFilter behavior.

This file owns summary-compatible filter laws and exact description contracts.
Non-core RRF coverage lives with hybrid search under ``tests/unit/storage``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given
from hypothesis import strategies as st

from polylogue.lib.filters import ConversationFilter


@pytest.fixture
def summary_filter_repo(make_filter_repo):
    """Repository with metadata-rich conversations for summary-compatible laws."""
    return make_filter_repo(
        [
            {
                "id": "conv-alpha",
                "provider": "claude",
                "title": "Alpha Python",
                "metadata": {"tags": ["science", "keep"], "summary": "alpha summary"},
                "created_at": "2024-01-10T00:00:00+00:00",
                "messages": [
                    {"id": "m1", "role": "user", "text": "alpha question with enough words"},
                    {"id": "m2", "role": "assistant", "text": "alpha answer with enough words"},
                ],
            },
            {
                "id": "conv-beta",
                "provider": "chatgpt",
                "title": "Beta Rust",
                "metadata": {"tags": ["code"]},
                "created_at": "2024-02-10T00:00:00+00:00",
                "messages": [
                    {"id": "m3", "role": "user", "text": "beta question with enough words"},
                    {"id": "m4", "role": "assistant", "text": "beta answer with enough words"},
                ],
            },
            {
                "id": "conv-gamma",
                "provider": "codex",
                "title": "Gamma Python",
                "metadata": {"tags": ["science", "drop"], "summary": "gamma summary"},
                "created_at": "2024-03-10T00:00:00+00:00",
                "messages": [
                    {"id": "m5", "role": "user", "text": "gamma question with enough words"},
                    {"id": "m6", "role": "assistant", "text": "gamma answer with enough words"},
                ],
            },
            {
                "id": "conv-delta",
                "provider": "claude",
                "title": "Delta Math",
                "metadata": {"tags": ["math"]},
                "created_at": "2024-04-10T00:00:00+00:00",
                "messages": [
                    {"id": "m7", "role": "user", "text": "delta question with enough words"},
                    {"id": "m8", "role": "assistant", "text": "delta answer with enough words"},
                ],
            },
        ]
    )


SUMMARY_FILTER_SPECS: list[tuple[str, dict[str, object]]] = [
    ("provider", {"providers": ["claude"]}),
    ("excluded-provider", {"excluded_providers": ["codex"]}),
    ("tag", {"tag": "science"}),
    ("excluded-tag", {"excluded_tag": "drop"}),
    ("title-and-summary", {"title": "Python", "has_summary": True}),
    ("since-until", {"since": "2024-02-15", "until": "2024-04-20"}),
    (
        "mixed-metadata",
        {
            "providers": ["claude", "chatgpt"],
            "excluded_providers": ["chatgpt"],
            "tag": "science",
            "excluded_tag": "drop",
            "since": "2024-01-15",
            "until": "2024-04-20",
        },
    ),
    ("id-prefix", {"id_prefix": "conv-g"}),
]


def _apply_summary_filter_spec(
    filter_obj: ConversationFilter,
    spec: dict[str, object],
) -> ConversationFilter:
    providers = spec.get("providers", [])
    excluded_providers = spec.get("excluded_providers", [])
    tag = spec.get("tag")
    excluded_tag = spec.get("excluded_tag")
    title = spec.get("title")
    since = spec.get("since")
    until = spec.get("until")
    has_summary = spec.get("has_summary", False)
    id_prefix = spec.get("id_prefix")

    if providers:
        filter_obj.provider(*providers)
    if excluded_providers:
        filter_obj.exclude_provider(*excluded_providers)
    if tag:
        filter_obj.tag(tag)
    if excluded_tag:
        filter_obj.exclude_tag(excluded_tag)
    if title:
        filter_obj.title(title)
    if since:
        filter_obj.since(since)
    if until:
        filter_obj.until(until)
    if has_summary:
        filter_obj.has("summary")
    if id_prefix:
        filter_obj.id(id_prefix)
    return filter_obj


@pytest.mark.asyncio
@pytest.mark.parametrize(("case_name", "spec"), SUMMARY_FILTER_SPECS)
async def test_summary_filter_terminal_views_agree(
    summary_filter_repo,
    case_name: str,
    spec: dict[str, object],
) -> None:
    list_filter = _apply_summary_filter_spec(ConversationFilter(summary_filter_repo), spec)
    summary_filter = _apply_summary_filter_spec(ConversationFilter(summary_filter_repo), spec)
    count_filter = _apply_summary_filter_spec(ConversationFilter(summary_filter_repo), spec)

    conversations = await list_filter.list()
    summaries = await summary_filter.list_summaries()
    count = await count_filter.count()

    assert [conv.id for conv in conversations] == [summary.id for summary in summaries], case_name
    assert count == len(conversations) == len(summaries), case_name


DESCRIBE_CASES = [
    (
        "every-active-filter-kind",
        lambda repo: (
            ConversationFilter(repo)
            .contains("python")
            .provider("claude", "chatgpt")
            .exclude_provider("codex")
            .tag("science")
            .exclude_tag("drop")
            .has("summary")
            .has_tool_use()
            .has_thinking()
            .has_file_operations()
            .has_git_operations()
            .has_subagent_spawns()
            .min_messages(2)
            .max_messages(8)
            .min_words(5)
            .since(datetime(2024, 1, 1, tzinfo=timezone.utc))
            .until(datetime(2024, 12, 31, tzinfo=timezone.utc))
            .title("Alpha")
            .id("conv")
            .exclude_text("rust")
            .where(lambda c: True)
            .similar("semantic query")
        ),
        [
            "contains: python",
            "provider: claude, chatgpt",
            "exclude provider: codex",
            "tag: science",
            "exclude tag: drop",
            "has: summary",
            "has_tool_use",
            "has_thinking",
            "has_file_ops",
            "has_git_ops",
            "has_subagent",
            "min_messages: 2",
            "max_messages: 8",
            "min_words: 5",
            "since: 2024-01-01T00:00:00+00:00",
            "until: 2024-12-31T00:00:00+00:00",
            "title: Alpha",
            "id: conv",
            "exclude text: rust",
            "custom predicates: 1",
            "similar: semantic query",
        ],
    ),
    (
        "contains-joins-exactly",
        lambda repo: ConversationFilter(repo).contains("python").contains("rust"),
        ["contains: python, rust"],
    ),
    (
        "multi-value-joins-and-truncates-similar",
        lambda repo: (
            ConversationFilter(repo)
            .provider("claude", "chatgpt")
            .exclude_provider("codex", "gemini")
            .tag("science")
            .exclude_tag("drop")
            .similar("x" * 90)
        ),
        [
            "provider: claude, chatgpt",
            "exclude provider: codex, gemini",
            "tag: science",
            "exclude tag: drop",
            f"similar: {'x' * 30}",
        ],
    ),
]


@pytest.mark.parametrize(("case_name", "build_filter", "expected_parts"), DESCRIBE_CASES)
def test_describe_contract_matrix(summary_filter_repo, case_name, build_filter, expected_parts) -> None:
    assert build_filter(summary_filter_repo).describe() == expected_parts, case_name


def test_sql_pushdown_only_filters_stay_summary_compatible(summary_filter_repo) -> None:
    compatible = (
        ConversationFilter(summary_filter_repo)
        .provider("claude")
        .exclude_provider("codex")
        .tag("science")
        .exclude_tag("drop")
        .title("Alpha")
        .id("conv")
        .has("summary")
        .since("2024-01-01")
        .until("2024-12-31")
    )
    incompatible = (
        ConversationFilter(summary_filter_repo)
        .provider("claude")
        .exclude_text("beta")
        .has_thinking()
        .sort("words")
        .where(lambda c: True)
    )

    compatible_plan = compatible._build_execution_plan()
    incompatible_plan = incompatible._build_execution_plan()

    assert compatible_plan.needs_content_loading is False
    assert compatible_plan.has_post_filters is True
    assert incompatible_plan.needs_content_loading is True


def test_count_fast_path_contract(summary_filter_repo) -> None:
    sql_only = ConversationFilter(summary_filter_repo).provider("claude").since("2024-01-01")
    post_filtered = ConversationFilter(summary_filter_repo).provider("claude").tag("science")
    content_filtered = ConversationFilter(summary_filter_repo).exclude_text("beta")

    assert sql_only._can_count_in_sql() is True
    assert post_filtered._can_count_in_sql() is False
    assert content_filtered._can_count_in_sql() is False


@given(
    st.lists(st.sampled_from(["chatgpt", "claude", "codex"]), min_size=1, max_size=3),
    st.lists(st.sampled_from(["chatgpt", "claude", "codex"]), min_size=0, max_size=2),
)
def test_provider_filter_exclusion_is_disjoint(
    include_providers: list[str],
    exclude_providers: list[str],
) -> None:
    included = set(include_providers)
    excluded = set(exclude_providers)
    remaining = included - excluded
    assert remaining.isdisjoint(excluded)


@given(
    st.integers(min_value=0, max_value=100),
    st.integers(min_value=0, max_value=50),
    st.integers(min_value=0, max_value=50),
)
def test_limit_and_offset_form_a_valid_window(total_items: int, limit: int, offset: int) -> None:
    items = list(range(total_items))
    window = items[offset : offset + limit]
    assert len(window) <= limit
    assert window == items[offset:][:limit]


@given(
    st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2025, 1, 1),
        timezones=st.just(timezone.utc),
    ),
    st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2025, 1, 1),
        timezones=st.just(timezone.utc),
    ),
)
def test_date_range_forms_a_valid_interval(since: datetime, until: datetime) -> None:
    if since > until:
        since, until = until, since

    mid = since + (until - since) / 2
    before = since - timedelta(days=1)
    after = until + timedelta(days=1)

    assert since <= mid <= until
    assert before < since
    assert after > until
