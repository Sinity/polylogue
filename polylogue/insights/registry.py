"""Insight type registry — typed descriptors for durable insight surfaces.

Each insight type is an ``InsightType`` descriptor that defines:
- how to display items in plain text (via ``fields``)
- what the JSON key is for API responses
- what display name to use in CLI output
- how to build a typed query model
- which archive-operations method provides the items
- CLI command metadata (name, help, options)

The rendering is generic: ``render_insight_items()`` handles JSON mode
(all types) and plain-text mode (using field descriptors). Insight
semantics stay in the archive/storage layers; the registry owns only the
transport and presentation contract.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import click

from polylogue.errors import PolylogueError
from polylogue.insights.archive import (
    ArchiveDebtInsightQuery,
    ArchiveInsightModel,
    CostRollupInsightQuery,
    DaySessionSummaryInsightQuery,
    ProviderAnalyticsInsightQuery,
    SessionCostInsightQuery,
    SessionEnrichmentInsightQuery,
    SessionPhaseInsightQuery,
    SessionProfileInsightQuery,
    SessionTagRollupQuery,
    SessionWorkEventInsightQuery,
    WeekSessionSummaryInsightQuery,
    WorkThreadInsightQuery,
)

InsightAccessor: TypeAlias = Callable[[ArchiveInsightModel], str]


@dataclass(frozen=True, slots=True)
class InsightField:
    """Describes one displayable field of an insight item."""

    label: str
    accessor: InsightAccessor
    group: int = 0


@dataclass(frozen=True, slots=True)
class CliOption:
    """Describes one Click option for an insight command."""

    param_name: str
    flags: tuple[str, ...]
    help: str = ""
    type: click.ParamType | type[object] | None = None
    default: object | None = None
    show_default: bool = False
    is_flag: bool = False
    expose_value_as: str | None = None


@dataclass(frozen=True, slots=True)
class InsightType:
    """Descriptor for one kind of derived insight."""

    name: str
    display_name: str
    json_key: str
    fields: tuple[InsightField, ...] = ()
    empty_message: str = "No items matched."
    query_model: type[ArchiveInsightModel] | None = None
    operations_method_name: str = ""
    cli_command_name: str = ""
    cli_help: str = ""
    cli_options: tuple[CliOption, ...] = ()
    mcp_default_limit: int = 50
    export_eligible: bool = True
    reader_panel: str | None = None
    readiness_exempt: bool = False

    @property
    def resolved_cli_command_name(self) -> str:
        return self.cli_command_name or self.name.replace("_", "-")


def _model_payload(item: ArchiveInsightModel) -> dict[str, object]:
    """Convert an insight item to a JSON-serializable payload."""

    return item.model_dump(mode="json")


def insight_items_payload(
    items: Sequence[ArchiveInsightModel],
    insight_type: InsightType,
    *,
    item_key: str | None = None,
) -> dict[str, object]:
    """Return the shared machine payload for an insight list surface."""

    return {
        "count": len(items),
        item_key or insight_type.json_key: [_model_payload(item) for item in items],
    }


def _stringify(value: object | None, default: str = "-") -> str:
    if value is None:
        return default
    if isinstance(value, str) and not value:
        return default
    return str(value)


def render_insight_items(
    items: Sequence[ArchiveInsightModel],
    insight_type: InsightType,
    *,
    json_mode: bool = False,
) -> None:
    """Render insight items using the insight type descriptor."""

    if json_mode:
        from polylogue.cli.shared.machine_errors import emit_success

        emit_success(insight_items_payload(items, insight_type))
        return

    if not items:
        click.echo(insight_type.empty_message)
        return

    click.echo(f"{insight_type.display_name}: {len(items)}\n")

    for item in items:
        groups: dict[int, list[str]] = {}
        for field in insight_type.fields:
            try:
                value = field.accessor(item)
            except (AttributeError, KeyError, TypeError):
                value = "-"
            groups.setdefault(field.group, []).append(f"{field.label}={value}" if field.label else str(value))

        for group_num in sorted(groups):
            prefix = "  " if group_num == 0 else "    "
            click.echo(f"{prefix}{' '.join(groups[group_num])}")


def _attr(name: str, default: str = "-") -> InsightAccessor:
    """Create an accessor that gets an attribute by name."""

    def accessor(item: ArchiveInsightModel) -> str:
        return _stringify(getattr(item, name, None), default)

    return accessor


def _nested(outer: str, inner: str, default: str = "-") -> InsightAccessor:
    """Create an accessor that gets a nested attribute."""

    def accessor(item: ArchiveInsightModel) -> str:
        nested = getattr(item, outer, None)
        if nested is None:
            return default
        return _stringify(getattr(nested, inner, None), default)

    return accessor


def _nested_ms_as_seconds(outer: str, inner: str, default: str = "-") -> InsightAccessor:
    """Create an accessor that renders a nested millisecond field as seconds."""

    def accessor(item: ArchiveInsightModel) -> str:
        nested = getattr(item, outer, None)
        if nested is None:
            return default
        value = getattr(nested, inner, None)
        if isinstance(value, int):
            return str(max(value, 0) // 1000)
        return default

    return accessor


def _id_with_provider(identifier_attr: str) -> InsightAccessor:
    """Accessor rendering an identifier together with the provider name."""

    def accessor(item: ArchiveInsightModel) -> str:
        identifier = _stringify(getattr(item, identifier_attr, None))
        provider = _stringify(getattr(item, "provider_name", None))
        return f"{identifier} [{provider}]"

    return accessor


def _list_preview(name: str, limit: int = 3) -> InsightAccessor:
    """Accessor showing the first N items of a tuple/list attribute."""

    def accessor(item: ArchiveInsightModel) -> str:
        values = getattr(item, name, None)
        if isinstance(values, (list, tuple)):
            preview = ", ".join(str(value) for value in values[:limit])
            return preview or "-"
        return _stringify(values)

    return accessor


def _formatted_float(name: str, *, precision: int = 1, default: str = "-") -> InsightAccessor:
    """Accessor rendering a numeric attribute with fixed precision."""

    def accessor(item: ArchiveInsightModel) -> str:
        value = getattr(item, name, None)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return default
        return f"{float(value):.{precision}f}"

    return accessor


def _count_with_percentage(count_attr: str, percentage_attr: str) -> InsightAccessor:
    """Accessor rendering ``count (pct%)`` pairs from sibling attributes."""

    def accessor(item: ArchiveInsightModel) -> str:
        count = getattr(item, count_attr, None)
        percentage = getattr(item, percentage_attr, None)
        if not isinstance(count, int) or isinstance(count, bool):
            return "-"
        if not isinstance(percentage, (int, float)) or isinstance(percentage, bool):
            return str(count)
        return f"{count} ({float(percentage):.1f}%)"

    return accessor


INSIGHT_REGISTRY: dict[str, InsightType] = {}


def register(insight_type: InsightType) -> InsightType:
    """Register an insight type and return it."""

    INSIGHT_REGISTRY[insight_type.name] = insight_type
    return insight_type


def get_insight_type(name: str) -> InsightType:
    """Look up a registered insight type by name."""

    insight_type = INSIGHT_REGISTRY.get(name)
    if insight_type is None:
        raise KeyError(f"Unknown insight type: {name!r}. Available: {sorted(INSIGHT_REGISTRY)}")
    return insight_type


def list_insight_types() -> list[str]:
    """Return the sorted registered insight type names."""

    return sorted(INSIGHT_REGISTRY)


_SESSION_TIME_OPTIONS = (
    CliOption(
        "first_message_since",
        ("--first-message-since",),
        help="Only sessions whose first message is on/after this timestamp",
    ),
    CliOption(
        "first_message_until",
        ("--first-message-until",),
        help="Only sessions whose first message is on/before this timestamp",
    ),
    CliOption(
        "session_date_since",
        ("--session-date-since",),
        help="Only sessions whose canonical session date is on/after this date",
    ),
    CliOption(
        "session_date_until",
        ("--session-date-until",),
        help="Only sessions whose canonical session date is on/before this date",
    ),
    CliOption(
        "min_wallclock_seconds",
        ("--min-wallclock-seconds",),
        type=int,
        help="Only sessions whose wallclock span is at least this many seconds",
    ),
    CliOption(
        "max_wallclock_seconds",
        ("--max-wallclock-seconds",),
        type=int,
        help="Only sessions whose wallclock span is at most this many seconds",
    ),
)

_QUERY_OPTION = CliOption("query", ("--query",), help="FTS query against insight search text")
_SESSION_TIME_SORT_OPTION = CliOption(
    "sort",
    ("--sort",),
    type=click.Choice(["source", "first-message", "last-message", "wallclock"]),
    default="source",
    show_default=True,
    help="Sort by source recency, first message time, last message time, or wallclock span",
)


register(
    InsightType(
        name="session_profiles",
        display_name="Session Profiles",
        json_key="session_profiles",
        empty_message="No session profiles matched.",
        query_model=SessionProfileInsightQuery,
        operations_method_name="list_session_profile_insights",
        cli_command_name="profiles",
        cli_help="List durable session-profile insights.",
        cli_options=(
            *_SESSION_TIME_OPTIONS,
            _SESSION_TIME_SORT_OPTION,
            CliOption(
                "tier",
                ("--tier",),
                type=click.Choice(["merged", "evidence", "inference"]),
                default="merged",
                show_default=True,
                help="Return merged, evidence-only, or inference-only profile insights",
            ),
            _QUERY_OPTION,
        ),
        fields=(
            InsightField("", _id_with_provider("conversation_id"), group=0),
            InsightField("tier", _attr("semantic_tier"), group=0),
            InsightField("", _attr("title", "(untitled)"), group=0),
            InsightField("session_date", _nested("evidence", "canonical_session_date"), group=1),
            InsightField("first", _nested("evidence", "first_message_at", "-"), group=1),
            InsightField("last", _nested("evidence", "last_message_at", "-"), group=1),
            InsightField("wall_s", _nested_ms_as_seconds("evidence", "wall_duration_ms", "0"), group=1),
            InsightField("ts_cov", _nested("evidence", "timestamp_coverage", "none"), group=1),
            InsightField("messages", _nested("evidence", "message_count", "0"), group=1),
            InsightField("engaged_min", _nested("inference", "engaged_minutes", "0"), group=1),
        ),
    )
)

register(
    InsightType(
        name="session_enrichments",
        display_name="Session Enrichments",
        json_key="session_enrichments",
        empty_message="No session enrichments matched.",
        query_model=SessionEnrichmentInsightQuery,
        operations_method_name="list_session_enrichment_insights",
        cli_command_name="enrichments",
        cli_help="List durable probabilistic session-enrichment insights.",
        cli_options=(
            *_SESSION_TIME_OPTIONS,
            _SESSION_TIME_SORT_OPTION,
            _QUERY_OPTION,
        ),
        fields=(
            InsightField("", _id_with_provider("conversation_id"), group=0),
            InsightField("", _attr("title", "(untitled)"), group=0),
            InsightField("support", _nested("enrichment", "support_level"), group=1),
            InsightField("family", _nested("enrichment_provenance", "enrichment_family"), group=1),
        ),
    )
)

register(
    InsightType(
        name="session_work_events",
        display_name="Work Events",
        json_key="session_work_events",
        empty_message="No work events matched.",
        query_model=SessionWorkEventInsightQuery,
        operations_method_name="list_session_work_event_insights",
        cli_command_name="work-events",
        cli_help="List durable work-event insights.",
        cli_options=(
            CliOption("conversation_id", ("--conversation-id",), help="Only events from one conversation"),
            CliOption("kind", ("--kind",), help="Only this work-event kind"),
            _QUERY_OPTION,
        ),
        fields=(
            InsightField("", _id_with_provider("event_id"), group=0),
            InsightField("kind", _nested("inference", "kind"), group=0),
            InsightField("conv", _attr("conversation_id"), group=0),
            InsightField("start", _nested("evidence", "start_time"), group=1),
            InsightField("end", _nested("evidence", "end_time"), group=1),
            InsightField("duration_ms", _nested("evidence", "duration_ms", "0"), group=1),
        ),
    )
)

register(
    InsightType(
        name="session_phases",
        display_name="Session Phases",
        json_key="session_phases",
        empty_message="No session phases matched.",
        query_model=SessionPhaseInsightQuery,
        operations_method_name="list_session_phase_insights",
        cli_command_name="phases",
        cli_help="List durable session-phase insights.",
        cli_options=(
            CliOption("conversation_id", ("--conversation-id",), help="Only phases from one conversation"),
            CliOption("kind", ("--kind",), help="Only this session phase kind"),
        ),
        fields=(
            InsightField("", _id_with_provider("phase_id"), group=0),
            InsightField("kind", _nested("inference", "kind"), group=0),
            InsightField("conv", _attr("conversation_id"), group=0),
            InsightField("start", _nested("evidence", "start_time"), group=1),
            InsightField("words", _nested("evidence", "word_count", "0"), group=1),
        ),
    )
)

register(
    InsightType(
        name="work_threads",
        display_name="Work Threads",
        json_key="work_threads",
        empty_message="No work threads matched.",
        query_model=WorkThreadInsightQuery,
        operations_method_name="list_work_thread_insights",
        cli_command_name="threads",
        cli_help="List durable work-thread insights.",
        cli_options=(_QUERY_OPTION,),
        fields=(
            InsightField("", _attr("thread_id"), group=0),
            InsightField("repo", _attr("dominant_repo", "-"), group=0),
            InsightField("sessions", _nested("thread", "session_count", "0"), group=0),
            InsightField("messages", _nested("thread", "total_messages", "0"), group=1),
            InsightField("depth", _nested("thread", "depth", "0"), group=1),
        ),
    )
)

register(
    InsightType(
        name="session_tag_rollups",
        display_name="Session Tag Rollups",
        json_key="session_tag_rollups",
        empty_message="No session tag rollups matched.",
        query_model=SessionTagRollupQuery,
        operations_method_name="list_session_tag_rollup_insights",
        cli_command_name="tags",
        cli_help="List durable session-tag rollup insights.",
        cli_options=(CliOption("query", ("--query",), help="Substring match against the tag name"),),
        mcp_default_limit=100,
        fields=(
            InsightField("", _attr("tag"), group=0),
            InsightField("conversations", _attr("conversation_count", "0"), group=0),
            InsightField("explicit", _attr("explicit_count", "0"), group=0),
            InsightField("auto", _attr("auto_count", "0"), group=0),
        ),
    )
)

register(
    InsightType(
        name="day_session_summaries",
        display_name="Day Session Summaries",
        json_key="day_session_summaries",
        empty_message="No day summaries matched.",
        query_model=DaySessionSummaryInsightQuery,
        operations_method_name="list_day_session_summary_insights",
        cli_command_name="day-summaries",
        cli_help="List durable day-level session summary insights.",
        mcp_default_limit=90,
        fields=(
            InsightField("", _attr("date"), group=0),
            InsightField("sessions", _nested("summary", "session_count", "0"), group=0),
            InsightField("messages", _nested("summary", "total_messages", "0"), group=0),
        ),
    )
)

register(
    InsightType(
        name="week_session_summaries",
        display_name="Week Session Summaries",
        json_key="week_session_summaries",
        empty_message="No week summaries matched.",
        query_model=WeekSessionSummaryInsightQuery,
        operations_method_name="list_week_session_summary_insights",
        cli_command_name="week-summaries",
        cli_help="List durable week-level session summary insights.",
        mcp_default_limit=52,
        fields=(
            InsightField("", _attr("iso_week"), group=0),
            InsightField("sessions", _nested("summary", "session_count", "0"), group=0),
            InsightField("messages", _nested("summary", "total_messages", "0"), group=0),
        ),
    )
)

register(
    InsightType(
        name="provider_analytics",
        display_name="Provider Analytics",
        json_key="provider_analytics",
        empty_message="No provider analytics matched.",
        query_model=ProviderAnalyticsInsightQuery,
        operations_method_name="list_provider_analytics_insights",
        cli_command_name="analytics",
        cli_help="List provider-level analytics insights.",
        fields=(
            InsightField("", _attr("provider_name"), group=0),
            InsightField("conversations", _attr("conversation_count", "0"), group=0),
            InsightField("messages", _attr("message_count", "0"), group=0),
            InsightField("avg_messages", _formatted_float("avg_messages_per_conversation"), group=0),
            InsightField("tools", _count_with_percentage("tool_use_count", "tool_use_percentage"), group=1),
            InsightField("thinking", _count_with_percentage("thinking_count", "thinking_percentage"), group=1),
        ),
    )
)

register(
    InsightType(
        name="session_costs",
        display_name="Session Costs",
        json_key="session_costs",
        empty_message="No session cost estimates matched.",
        query_model=SessionCostInsightQuery,
        operations_method_name="list_session_cost_insights",
        cli_command_name="costs",
        cli_help="List session-level cost estimates.",
        readiness_exempt=True,
        cli_options=(
            CliOption("conversation_id", ("--conversation-id",), help="Only one conversation"),
            CliOption("model", ("--model",), help="Only this model or normalized model"),
            CliOption("status", ("--status",), type=click.Choice(["exact", "priced", "partial", "unavailable"])),
        ),
        fields=(
            InsightField("", _id_with_provider("conversation_id"), group=0),
            InsightField("status", _nested("estimate", "status"), group=0),
            InsightField("model", _nested("estimate", "normalized_model"), group=0),
            InsightField("usd", _nested("estimate", "total_usd", "0"), group=1),
            InsightField("confidence", _nested("estimate", "confidence", "0"), group=1),
        ),
    )
)

register(
    InsightType(
        name="cost_rollups",
        display_name="Cost Rollups",
        json_key="cost_rollups",
        empty_message="No cost rollups matched.",
        query_model=CostRollupInsightQuery,
        operations_method_name="list_cost_rollup_insights",
        cli_command_name="cost-rollups",
        cli_help="List provider/model cost rollups.",
        readiness_exempt=True,
        cli_options=(CliOption("model", ("--model",), help="Only this model or normalized model"),),
        fields=(
            InsightField("", _attr("provider_name"), group=0),
            InsightField("model", _attr("normalized_model"), group=0),
            InsightField("sessions", _attr("session_count", "0"), group=0),
            InsightField("priced", _attr("priced_session_count", "0"), group=1),
            InsightField("unavailable", _attr("unavailable_session_count", "0"), group=1),
            InsightField("usd", _attr("total_usd", "0"), group=1),
            InsightField("confidence", _attr("confidence", "0"), group=1),
        ),
    )
)

register(
    InsightType(
        name="archive_debt",
        display_name="Archive Debt",
        json_key="archive_debt",
        empty_message="No archive debt entries matched.",
        query_model=ArchiveDebtInsightQuery,
        operations_method_name="list_archive_debt_insights",
        cli_command_name="debt",
        cli_help="List archive debt and maintenance readiness insights.",
        readiness_exempt=True,
        cli_options=(
            CliOption("category", ("--category",), help="Only this maintenance category"),
            CliOption(
                "only_actionable",
                ("--only-actionable",),
                is_flag=True,
                default=False,
                help="Only debt entries with pending issues",
            ),
        ),
        fields=(
            InsightField("", _attr("debt_name"), group=0),
            InsightField("category", _attr("category"), group=0),
            InsightField("target", _attr("maintenance_target"), group=0),
            InsightField("issues", _attr("issue_count", "0"), group=1),
            InsightField("healthy", _attr("healthy"), group=1),
            InsightField("destructive", _attr("destructive"), group=1),
            InsightField("detail", _attr("detail"), group=2),
        ),
    )
)


class InsightQueryError(PolylogueError):
    """Raised when a registry-backed insight query is invalid."""

    http_status_code = 400


def _build_query(
    insight_type: InsightType,
    **kwargs: object,
) -> ArchiveInsightModel:
    """Build and validate the typed query object for an insight fetch."""

    query_model = insight_type.query_model
    if query_model is None:
        raise InsightQueryError(f"Insight type {insight_type.name} does not declare a query model")
    accepted = set(query_model.model_fields)
    unknown = sorted(set(kwargs) - accepted)
    if unknown:
        unknown_list = ", ".join(unknown)
        accepted_list = ", ".join(sorted(accepted))
        raise InsightQueryError(
            f"Unknown query field(s) for {insight_type.name}: {unknown_list}. Accepted fields: {accepted_list}"
        )
    return query_model(**kwargs)


def fetch_insights(
    insight_type: InsightType,
    operations: object,
    **kwargs: object,
) -> list[ArchiveInsightModel]:
    """Fetch insight items using the registry dispatch metadata."""

    from polylogue.api.sync.bridge import run_coroutine_sync

    query = _build_query(insight_type, **kwargs)
    method = getattr(operations, insight_type.operations_method_name)
    return list(run_coroutine_sync(method(query)))


async def fetch_insights_async(
    insight_type: InsightType,
    operations: object,
    **kwargs: object,
) -> list[ArchiveInsightModel]:
    """Async variant of ``fetch_insights()``."""

    query = _build_query(insight_type, **kwargs)
    method = getattr(operations, insight_type.operations_method_name)
    return list(await method(query))


__all__ = [
    "CliOption",
    "INSIGHT_REGISTRY",
    "InsightField",
    "InsightQueryError",
    "InsightType",
    "fetch_insights",
    "fetch_insights_async",
    "get_insight_type",
    "list_insight_types",
    "insight_items_payload",
    "register",
    "render_insight_items",
]
