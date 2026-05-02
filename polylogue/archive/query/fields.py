"""Shared query-field descriptors for selection, planning, and storage."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal, Protocol, TypeAlias, cast

from polylogue.storage.query_models import ConversationRecordQuery
from polylogue.types import Provider

PresenceCheck = Callable[[object], bool]
DescriptionRenderer = Callable[[object], str]
StorageValue = Callable[[object], object]
_ReplaceRecordQuery: Callable[..., ConversationRecordQuery] = replace
SqlPushdownValue: TypeAlias = str | int | bool | list[str]
SqlPushdownParams: TypeAlias = dict[str, SqlPushdownValue]
CompletionSource: TypeAlias = Literal[
    "action",
    "action_sequence",
    "conversation_id",
    "cwd_prefix",
    "message_type",
    "provider",
    "repo",
    "retrieval_lane",
    "tag",
    "tool",
]


class _ProviderScopedPlan(Protocol):
    providers: tuple[Provider | str, ...]


def _truthy(value: object) -> bool:
    return bool(value)


def _not_none(value: object) -> bool:
    return value is not None


def _is_true(value: object) -> bool:
    return value is True


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and value > 0


def _not_auto(value: object) -> bool:
    return value != "auto"


def _not_date_sort(value: object) -> bool:
    return value is not None and value != "date"


def _as_tuple(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _join_comma(value: object) -> str:
    return ", ".join(str(item) for item in _as_tuple(value))


def _join_space(value: object) -> str:
    return " ".join(str(item) for item in _as_tuple(value))


def _join_arrow(value: object) -> str:
    return " -> ".join(str(item) for item in _as_tuple(value))


def _provider_values(value: object) -> tuple[str, ...]:
    return tuple(str(Provider.from_string(cast(str | Provider | None, item))) for item in _as_tuple(value))


def _join_providers(value: object) -> str:
    return ", ".join(_provider_values(value))


def _isoformat(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _truncated_text(value: object) -> str:
    return str(value)[:30]


def _label(label: str, formatter: DescriptionRenderer = str) -> DescriptionRenderer:
    def render(value: object) -> str:
        return f"{label}: {formatter(value)}"

    return render


def _literal(text: str) -> DescriptionRenderer:
    def render(_value: object) -> str:
        return text

    return render


def _list_value(value: object) -> object:
    return list(_as_tuple(value))


def _sql_pushdown_value(value: object) -> SqlPushdownValue:
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise TypeError(f"SQL pushdown parameter is not supported: {value!r}")


def _identity(value: object) -> object:
    return value


def _iso_value(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


@dataclass(frozen=True, slots=True)
class QueryFieldDescriptor:
    """Semantic facts about one query field across query layers."""

    name: str
    spec_attr: str | None = None
    plan_attr: str | None = None
    spec_active: PresenceCheck = _truthy
    plan_active: PresenceCheck = _truthy
    spec_description: DescriptionRenderer | None = None
    plan_description: DescriptionRenderer | None = None
    selection_filter: bool = True
    record_attr: str | None = None
    sql_param: str | None = None
    storage_value: StorageValue = _identity
    sql_value: StorageValue | None = None
    requires_stats_join: bool = False
    requires_post_filter: bool = False
    requires_content_loading: bool = False
    blocks_sql_count: bool = False
    blocks_action_event_stats: bool = False
    blocks_simple_message_hit: bool = False
    completion_source: CompletionSource | None = None
    completion_label: str | None = None
    mcp_names: tuple[str, ...] = ()
    api_names: tuple[str, ...] = ()

    def spec_value(self, spec: object) -> object:
        if self.spec_attr is None:
            raise ValueError(f"{self.name} has no spec attribute")
        return cast(object, getattr(spec, self.spec_attr))

    def plan_value(self, plan: object) -> object:
        if self.plan_attr is None:
            raise ValueError(f"{self.name} has no plan attribute")
        return cast(object, getattr(plan, self.plan_attr))

    def is_active_for_spec(self, spec: object) -> bool:
        if self.spec_attr is None:
            return False
        return self.spec_active(self.spec_value(spec))

    def is_active_for_plan(self, plan: object) -> bool:
        if self.plan_attr is None:
            return False
        return self.plan_active(self.plan_value(plan))

    def describe_spec(self, spec: object) -> str | None:
        if self.spec_description is None or not self.is_active_for_spec(spec):
            return None
        return self.spec_description(self.spec_value(spec))

    def describe_plan(self, plan: object) -> str | None:
        if self.plan_description is None or not self.is_active_for_plan(plan):
            return None
        return self.plan_description(self.plan_value(plan))

    def storage_plan_value(self, plan: object) -> object:
        return self.storage_value(self.plan_value(plan))

    def sql_plan_value(self, plan: object) -> object:
        if self.sql_value is not None:
            return self.sql_value(self.plan_value(plan))
        return self.storage_plan_value(plan)


QUERY_FIELD_DESCRIPTORS: tuple[QueryFieldDescriptor, ...] = (
    QueryFieldDescriptor(
        name="query_terms",
        spec_attr="query_terms",
        plan_attr="fts_terms",
        spec_description=_label("search", _join_space),
        plan_description=_label("contains", _join_comma),
        blocks_sql_count=True,
        blocks_action_event_stats=True,
        mcp_names=("query",),
        api_names=("query",),
    ),
    QueryFieldDescriptor(
        name="contains_terms",
        spec_attr="contains_terms",
        spec_description=_label("contains", _join_comma),
    ),
    QueryFieldDescriptor(
        name="exclude_text_terms",
        spec_attr="exclude_text_terms",
        plan_attr="negative_terms",
        spec_description=_label("exclude text", _join_comma),
        plan_description=_label("exclude text", _join_comma),
        requires_post_filter=True,
        requires_content_loading=True,
        blocks_sql_count=True,
        blocks_action_event_stats=True,
        blocks_simple_message_hit=True,
    ),
    QueryFieldDescriptor(
        name="retrieval_lane",
        spec_attr="retrieval_lane",
        plan_attr="retrieval_lane",
        spec_active=_not_auto,
        plan_active=_not_auto,
        spec_description=_label("retrieval"),
        plan_description=_label("retrieval"),
        selection_filter=False,
        completion_source="retrieval_lane",
        completion_label="retrieval lane",
        mcp_names=("retrieval_lane",),
    ),
    QueryFieldDescriptor(
        name="referenced_path",
        spec_attr="referenced_path",
        plan_attr="referenced_path",
        spec_description=_label("referenced-path", _join_comma),
        plan_description=_label("referenced-path", _join_comma),
        record_attr="referenced_path",
        sql_param="referenced_path",
        sql_value=_list_value,
        requires_post_filter=True,
        requires_content_loading=True,
        blocks_sql_count=True,
        blocks_simple_message_hit=True,
        mcp_names=("referenced_path",),
    ),
    QueryFieldDescriptor(
        name="cwd_prefix",
        spec_attr="cwd_prefix",
        plan_attr="cwd_prefix",
        spec_active=_not_none,
        plan_active=_not_none,
        spec_description=_label("cwd-prefix"),
        plan_description=_label("cwd-prefix"),
        record_attr="cwd_prefix",
        sql_param="cwd_prefix",
        blocks_simple_message_hit=True,
        completion_source="cwd_prefix",
        completion_label="working directory",
        mcp_names=("cwd_prefix",),
    ),
    QueryFieldDescriptor(
        name="action_terms",
        spec_attr="action_terms",
        plan_attr="action_terms",
        spec_description=_label("action", _join_comma),
        plan_description=_label("action", _join_comma),
        record_attr="action_terms",
        sql_param="action_terms",
        sql_value=_list_value,
        requires_post_filter=True,
        requires_content_loading=True,
        blocks_sql_count=True,
        blocks_simple_message_hit=True,
        completion_source="action",
        completion_label="action",
        mcp_names=("action",),
    ),
    QueryFieldDescriptor(
        name="excluded_action_terms",
        spec_attr="excluded_action_terms",
        plan_attr="excluded_action_terms",
        spec_description=_label("exclude action", _join_comma),
        plan_description=_label("exclude action", _join_comma),
        record_attr="excluded_action_terms",
        sql_param="excluded_action_terms",
        sql_value=_list_value,
        requires_post_filter=True,
        requires_content_loading=True,
        blocks_sql_count=True,
        blocks_simple_message_hit=True,
        completion_source="action",
        completion_label="action",
        mcp_names=("exclude_action",),
    ),
    QueryFieldDescriptor(
        name="action_sequence",
        spec_attr="action_sequence",
        plan_attr="action_sequence",
        spec_description=_label("action sequence", _join_arrow),
        plan_description=_label("action sequence", _join_arrow),
        sql_param="action_sequence",
        sql_value=_list_value,
        requires_post_filter=True,
        requires_content_loading=True,
        blocks_sql_count=True,
        blocks_action_event_stats=True,
        blocks_simple_message_hit=True,
        completion_source="action_sequence",
        completion_label="action sequence",
        mcp_names=("action_sequence",),
    ),
    QueryFieldDescriptor(
        name="action_text_terms",
        spec_attr="action_text_terms",
        plan_attr="action_text_terms",
        spec_description=_label("action text", _join_comma),
        plan_description=_label("action text", _join_comma),
        sql_param="action_text_terms",
        sql_value=_list_value,
        requires_post_filter=True,
        requires_content_loading=True,
        blocks_sql_count=True,
        blocks_action_event_stats=True,
        blocks_simple_message_hit=True,
        mcp_names=("action_text",),
    ),
    QueryFieldDescriptor(
        name="tool_terms",
        spec_attr="tool_terms",
        plan_attr="tool_terms",
        spec_description=_label("tool", _join_comma),
        plan_description=_label("tool", _join_comma),
        record_attr="tool_terms",
        sql_param="tool_terms",
        sql_value=_list_value,
        requires_post_filter=True,
        requires_content_loading=True,
        blocks_sql_count=True,
        blocks_simple_message_hit=True,
        completion_source="tool",
        completion_label="tool",
        mcp_names=("tool",),
    ),
    QueryFieldDescriptor(
        name="excluded_tool_terms",
        spec_attr="excluded_tool_terms",
        plan_attr="excluded_tool_terms",
        spec_description=_label("exclude tool", _join_comma),
        plan_description=_label("exclude tool", _join_comma),
        record_attr="excluded_tool_terms",
        sql_param="excluded_tool_terms",
        sql_value=_list_value,
        requires_post_filter=True,
        requires_content_loading=True,
        blocks_sql_count=True,
        blocks_simple_message_hit=True,
        completion_source="tool",
        completion_label="tool",
        mcp_names=("exclude_tool",),
    ),
    QueryFieldDescriptor(
        name="providers",
        spec_attr="providers",
        plan_attr="providers",
        spec_description=_label("provider", _join_providers),
        plan_description=_label("provider", _join_providers),
        completion_source="provider",
        completion_label="provider",
        mcp_names=("provider",),
        api_names=("provider", "source"),
    ),
    QueryFieldDescriptor(
        name="excluded_providers",
        spec_attr="excluded_providers",
        plan_attr="excluded_providers",
        spec_description=_label("exclude provider", _join_providers),
        plan_description=_label("exclude provider", _join_providers),
        requires_post_filter=True,
        blocks_sql_count=True,
        blocks_action_event_stats=True,
        blocks_simple_message_hit=True,
        completion_source="provider",
        completion_label="provider",
    ),
    QueryFieldDescriptor(
        name="repo_names",
        spec_attr="repo_names",
        plan_attr="repo_names",
        spec_description=_label("repo", _join_comma),
        plan_description=_label("repo", _join_comma),
        record_attr="repo_names",
        sql_param="repo_names",
        sql_value=_list_value,
        completion_source="repo",
        completion_label="repository",
        mcp_names=("repo",),
    ),
    QueryFieldDescriptor(
        name="tags",
        spec_attr="tags",
        plan_attr="tags",
        spec_description=_label("tag", _join_comma),
        plan_description=_label("tag", _join_comma),
        requires_post_filter=True,
        blocks_sql_count=True,
        blocks_action_event_stats=True,
        blocks_simple_message_hit=True,
        completion_source="tag",
        completion_label="tag",
        mcp_names=("tag",),
    ),
    QueryFieldDescriptor(
        name="excluded_tags",
        spec_attr="excluded_tags",
        plan_attr="excluded_tags",
        spec_description=_label("exclude tag", _join_comma),
        plan_description=_label("exclude tag", _join_comma),
        requires_post_filter=True,
        blocks_sql_count=True,
        blocks_action_event_stats=True,
        blocks_simple_message_hit=True,
        completion_source="tag",
        completion_label="tag",
    ),
    QueryFieldDescriptor(
        name="title",
        spec_attr="title",
        plan_attr="title",
        spec_active=_not_none,
        plan_active=_not_none,
        spec_description=_label("title"),
        plan_description=_label("title"),
        record_attr="title_contains",
        sql_param="title_contains",
        blocks_simple_message_hit=True,
        mcp_names=("title",),
    ),
    QueryFieldDescriptor(
        name="has_types",
        spec_attr="has_types",
        plan_attr="has_types",
        spec_description=_label("has", _join_comma),
        plan_description=_label("has", _join_comma),
        requires_post_filter=True,
        blocks_sql_count=True,
        blocks_action_event_stats=True,
        blocks_simple_message_hit=True,
    ),
    QueryFieldDescriptor(
        name="filter_has_tool_use",
        spec_attr="filter_has_tool_use",
        plan_attr="filter_has_tool_use",
        spec_active=_is_true,
        plan_active=_is_true,
        spec_description=_literal("has: tool_use (sql)"),
        plan_description=_literal("has_tool_use"),
        record_attr="has_tool_use",
        sql_param="has_tool_use",
        requires_stats_join=True,
        blocks_simple_message_hit=True,
        mcp_names=("has_tool_use",),
    ),
    QueryFieldDescriptor(
        name="filter_has_thinking",
        spec_attr="filter_has_thinking",
        plan_attr="filter_has_thinking",
        spec_active=_is_true,
        plan_active=_is_true,
        spec_description=_literal("has: thinking (sql)"),
        plan_description=_literal("has_thinking"),
        record_attr="has_thinking",
        sql_param="has_thinking",
        requires_stats_join=True,
        blocks_simple_message_hit=True,
        mcp_names=("has_thinking",),
    ),
    QueryFieldDescriptor(
        name="filter_has_paste",
        spec_attr="filter_has_paste",
        plan_attr="filter_has_paste",
        spec_active=_is_true,
        plan_active=_is_true,
        spec_description=_literal("has: paste"),
        plan_description=_literal("has_paste"),
        record_attr="has_paste",
        sql_param="has_paste",
        requires_stats_join=True,
        blocks_simple_message_hit=True,
        mcp_names=("has_paste",),
    ),
    QueryFieldDescriptor(
        name="typed_only",
        spec_attr="typed_only",
        plan_attr="typed_only",
        spec_active=_is_true,
        plan_active=_is_true,
        spec_description=_literal("typed-only (no paste)"),
        plan_description=_literal("typed_only"),
        record_attr="typed_only",
        sql_param="typed_only",
        requires_stats_join=True,
        blocks_simple_message_hit=True,
        mcp_names=("typed_only",),
    ),
    QueryFieldDescriptor(
        name="min_messages",
        spec_attr="min_messages",
        plan_attr="min_messages",
        spec_active=_not_none,
        plan_active=_not_none,
        spec_description=_label("min_messages"),
        plan_description=_label("min_messages"),
        record_attr="min_messages",
        sql_param="min_messages",
        requires_stats_join=True,
        blocks_simple_message_hit=True,
        mcp_names=("min_messages",),
    ),
    QueryFieldDescriptor(
        name="max_messages",
        spec_attr="max_messages",
        plan_attr="max_messages",
        spec_active=_not_none,
        plan_active=_not_none,
        spec_description=_label("max_messages"),
        plan_description=_label("max_messages"),
        record_attr="max_messages",
        sql_param="max_messages",
        requires_stats_join=True,
        blocks_simple_message_hit=True,
    ),
    QueryFieldDescriptor(
        name="min_words",
        spec_attr="min_words",
        plan_attr="min_words",
        spec_active=_not_none,
        plan_active=_not_none,
        spec_description=_label("min_words"),
        plan_description=_label("min_words"),
        record_attr="min_words",
        sql_param="min_words",
        requires_stats_join=True,
        blocks_simple_message_hit=True,
        mcp_names=("min_words",),
    ),
    QueryFieldDescriptor(
        name="similar_text",
        spec_attr="similar_text",
        plan_attr="similar_text",
        spec_active=_not_none,
        plan_active=_truthy,
        spec_description=_label("similar"),
        plan_description=_label("similar", _truncated_text),
        requires_content_loading=True,
        blocks_sql_count=True,
        blocks_action_event_stats=True,
        blocks_simple_message_hit=True,
    ),
    QueryFieldDescriptor(
        name="since",
        spec_attr="since",
        plan_attr="since",
        spec_active=_not_none,
        plan_active=_not_none,
        spec_description=_label("since"),
        plan_description=_label("since", _isoformat),
        record_attr="since",
        sql_param="since",
        storage_value=_iso_value,
        mcp_names=("since",),
        api_names=("since",),
    ),
    QueryFieldDescriptor(
        name="until",
        spec_attr="until",
        plan_attr="until",
        spec_active=_not_none,
        plan_active=_not_none,
        spec_description=_label("until"),
        plan_description=_label("until", _isoformat),
        record_attr="until",
        sql_param="until",
        storage_value=_iso_value,
        blocks_simple_message_hit=True,
    ),
    QueryFieldDescriptor(
        name="conversation_id",
        spec_attr="conversation_id",
        plan_attr="conversation_id",
        spec_active=_not_none,
        plan_active=_not_none,
        spec_description=_label("id"),
        plan_description=_label("id"),
        blocks_sql_count=True,
        blocks_simple_message_hit=True,
        completion_source="conversation_id",
        completion_label="conversation",
    ),
    QueryFieldDescriptor(
        name="latest",
        spec_attr="latest",
        spec_active=_is_true,
        selection_filter=True,
    ),
    QueryFieldDescriptor(
        name="sort",
        spec_attr="sort",
        plan_attr="sort",
        spec_active=_not_none,
        plan_active=_not_date_sort,
        spec_description=_label("sort"),
        plan_description=_label("sort"),
        selection_filter=False,
        blocks_simple_message_hit=True,
        mcp_names=("sort",),
    ),
    QueryFieldDescriptor(
        name="reverse",
        spec_attr="reverse",
        plan_attr="reverse",
        spec_active=_is_true,
        plan_active=_is_true,
        spec_description=_literal("reverse"),
        plan_description=_literal("reverse"),
        selection_filter=False,
        blocks_simple_message_hit=True,
    ),
    QueryFieldDescriptor(
        name="limit",
        spec_attr="limit",
        plan_attr="limit",
        spec_active=_not_none,
        plan_active=_not_none,
        spec_description=_label("limit"),
        plan_description=_label("limit"),
        selection_filter=False,
        mcp_names=("limit",),
        api_names=("limit",),
    ),
    QueryFieldDescriptor(
        name="parent_id",
        plan_attr="parent_id",
        plan_active=_not_none,
        plan_description=_label("parent"),
        record_attr="parent_id",
        sql_param="parent_id",
    ),
    QueryFieldDescriptor(
        name="continuation",
        plan_attr="continuation",
        plan_active=_not_none,
        plan_description=lambda value: "continuation" if value is True else "not continuation",
        requires_post_filter=True,
        blocks_sql_count=True,
        blocks_action_event_stats=True,
    ),
    QueryFieldDescriptor(
        name="sidechain",
        plan_attr="sidechain",
        plan_active=_not_none,
        plan_description=lambda value: "sidechain" if value is True else "not sidechain",
        requires_post_filter=True,
        blocks_sql_count=True,
        blocks_action_event_stats=True,
    ),
    QueryFieldDescriptor(
        name="root",
        plan_attr="root",
        plan_active=_not_none,
        plan_description=lambda value: "root" if value is True else "not root",
        requires_post_filter=True,
        blocks_sql_count=True,
        blocks_action_event_stats=True,
    ),
    QueryFieldDescriptor(
        name="has_branches",
        plan_attr="has_branches",
        plan_active=_not_none,
        plan_description=lambda value: "has branches" if value is True else "no branches",
        requires_post_filter=True,
        requires_content_loading=True,
        blocks_sql_count=True,
        blocks_action_event_stats=True,
    ),
    QueryFieldDescriptor(
        name="predicates",
        plan_attr="predicates",
        plan_description=lambda value: f"custom predicates: {len(_as_tuple(value))}",
        requires_post_filter=True,
        requires_content_loading=True,
        blocks_sql_count=True,
        blocks_action_event_stats=True,
    ),
    QueryFieldDescriptor(
        name="sample",
        spec_attr="sample",
        plan_attr="sample",
        plan_active=_not_none,
        spec_active=_not_none,
        spec_description=_label("sample"),
        plan_description=_label("sample"),
        selection_filter=False,
        blocks_simple_message_hit=True,
    ),
    QueryFieldDescriptor(
        name="offset",
        spec_attr="offset",
        plan_attr="offset",
        spec_active=_positive_int,
        plan_active=_positive_int,
        spec_description=_label("offset"),
        plan_description=_label("offset"),
        record_attr="offset",
        storage_value=lambda v: int(cast(int, v)) if v else 0,
        selection_filter=False,
        mcp_names=("offset",),
    ),
    QueryFieldDescriptor(
        name="since_session_id",
        spec_attr="since_session_id",
        plan_attr="since_session_id",
        spec_active=_not_none,
        plan_active=_not_none,
        spec_description=_label("since-session"),
        plan_description=_label("since-session"),
        record_attr="since_session_id",
        storage_value=lambda v: str(v) if v else None,
        selection_filter=True,
        requires_post_filter=True,
        blocks_sql_count=True,
        completion_source="conversation_id",
        completion_label="conversation",
        mcp_names=("since_session",),
    ),
    QueryFieldDescriptor(
        name="message_type",
        spec_attr="message_type",
        plan_attr="message_type",
        spec_active=_not_none,
        plan_active=_not_none,
        spec_description=_label("message-type"),
        plan_description=_label("message-type"),
        record_attr="message_type",
        sql_param="message_type",
        storage_value=lambda v: str(v) if v else None,
        selection_filter=True,
        completion_source="message_type",
        completion_label="message type",
        mcp_names=("message_type",),
    ),
)


def describe_spec_fields(spec: object) -> list[str]:
    return [
        description
        for descriptor in QUERY_FIELD_DESCRIPTORS
        if (description := descriptor.describe_spec(spec)) is not None
    ]


def describe_plan_fields(plan: object) -> list[str]:
    return [
        description
        for descriptor in QUERY_FIELD_DESCRIPTORS
        if (description := descriptor.describe_plan(plan)) is not None
    ]


def query_spec_has_selection_filters(spec: object) -> bool:
    return any(
        descriptor.selection_filter and descriptor.is_active_for_spec(spec) for descriptor in QUERY_FIELD_DESCRIPTORS
    )


def plan_has_selection_filters(plan: object) -> bool:
    return any(
        descriptor.selection_filter and descriptor.is_active_for_plan(plan) for descriptor in QUERY_FIELD_DESCRIPTORS
    )


def active_plan_field_names(plan: object) -> tuple[str, ...]:
    """Return active plan descriptor names for tests and diagnostics."""
    return tuple(descriptor.name for descriptor in QUERY_FIELD_DESCRIPTORS if descriptor.is_active_for_plan(plan))


def plan_has_fields_matching(plan: object, predicate: Callable[[QueryFieldDescriptor], bool]) -> bool:
    return any(predicate(descriptor) and descriptor.is_active_for_plan(plan) for descriptor in QUERY_FIELD_DESCRIPTORS)


def has_message_content_type_filter(plan: object) -> bool:
    for descriptor in QUERY_FIELD_DESCRIPTORS:
        if descriptor.name != "has_types" or not descriptor.is_active_for_plan(plan):
            continue
        return any(str(kind) in {"thinking", "tools", "attachments"} for kind in _as_tuple(descriptor.plan_value(plan)))
    return False


def provider_scope_for_plan(plan: _ProviderScopedPlan) -> tuple[str | None, tuple[str, ...]]:
    values = _provider_values(plan.providers)
    provider = values[0] if len(values) == 1 else None
    provider_group = values if len(values) > 1 else ()
    return provider, provider_group


def conversation_record_query_for_plan(plan: object) -> ConversationRecordQuery:
    provider, providers = provider_scope_for_plan(cast(_ProviderScopedPlan, plan))
    changes: dict[str, object] = {}
    for descriptor in QUERY_FIELD_DESCRIPTORS:
        if descriptor.record_attr is None or not descriptor.is_active_for_plan(plan):
            continue
        changes[descriptor.record_attr] = descriptor.storage_plan_value(plan)
    return _ReplaceRecordQuery(
        ConversationRecordQuery(provider=provider, providers=providers),
        **changes,
    )


def sql_pushdown_params_for_plan(plan: object) -> SqlPushdownParams:
    params: SqlPushdownParams = {}
    provider, providers = provider_scope_for_plan(cast(_ProviderScopedPlan, plan))
    if provider is not None:
        params["provider"] = provider
    elif providers:
        params["providers"] = list(providers)
    for descriptor in QUERY_FIELD_DESCRIPTORS:
        if descriptor.sql_param is None or not descriptor.is_active_for_plan(plan):
            continue
        params[descriptor.sql_param] = _sql_pushdown_value(descriptor.sql_plan_value(plan))
    return params


def storage_filters_require_stats_join(filters: Mapping[str, object]) -> bool:
    for descriptor in QUERY_FIELD_DESCRIPTORS:
        if not descriptor.requires_stats_join or descriptor.record_attr is None:
            continue
        if descriptor.plan_active(filters.get(descriptor.record_attr)):
            return True
    return False


def query_completion_sources() -> tuple[CompletionSource, ...]:
    return tuple(
        sorted(
            {
                descriptor.completion_source
                for descriptor in QUERY_FIELD_DESCRIPTORS
                if descriptor.completion_source is not None
            }
        )
    )


def mcp_query_field_names() -> frozenset[str]:
    names: set[str] = set()
    for descriptor in QUERY_FIELD_DESCRIPTORS:
        names.update(descriptor.mcp_names)
    return frozenset(names)


def api_query_field_names() -> frozenset[str]:
    names: set[str] = set()
    for descriptor in QUERY_FIELD_DESCRIPTORS:
        names.update(descriptor.api_names)
    return frozenset(names)


__all__ = [
    "CompletionSource",
    "QUERY_FIELD_DESCRIPTORS",
    "QueryFieldDescriptor",
    "SqlPushdownParams",
    "SqlPushdownValue",
    "active_plan_field_names",
    "api_query_field_names",
    "conversation_record_query_for_plan",
    "describe_plan_fields",
    "describe_spec_fields",
    "has_message_content_type_filter",
    "mcp_query_field_names",
    "plan_has_fields_matching",
    "plan_has_selection_filters",
    "query_completion_sources",
    "query_spec_has_selection_filters",
    "sql_pushdown_params_for_plan",
    "storage_filters_require_stats_join",
]
