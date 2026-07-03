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
from typing import Any, TypeAlias, cast

import click

from polylogue.core.enums import Origin, Provider
from polylogue.core.sources import origin_from_provider
from polylogue.errors import PolylogueError
from polylogue.insights.archive import (
    ArchiveCoverageInsightQuery,
    ArchiveDebtInsightQuery,
    ArchiveInsightModel,
    CostRollupInsightQuery,
    SessionCostInsightQuery,
    SessionPhaseInsightQuery,
    SessionProfileInsightQuery,
    SessionTagRollupQuery,
    SessionWorkEventInsightQuery,
    ThreadInsightQuery,
    UsageTimelineInsightQuery,
)
from polylogue.insights.tool_usage import ToolUsageInsightQuery

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
    type: click.ParamType[Any] | type[object] | None = None
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

    return cast(dict[str, object], project_origin_payload(item.model_dump(mode="json")))


_CANONICAL_ORIGIN_VALUES = frozenset(origin.value for origin in Origin)


def _source_name_origin(source_name: object) -> str:
    value = str(source_name or "")
    if not value:
        return "unknown"
    if value in _CANONICAL_ORIGIN_VALUES:
        return value
    try:
        return origin_from_provider(Provider.from_string(value)).value
    except ValueError:
        return "unknown"


def project_origin_payload(value: object) -> object:
    """Project source-origin fields to public origin vocabulary."""

    if isinstance(value, list):
        return [project_origin_payload(item) for item in value]
    if isinstance(value, tuple):
        return [project_origin_payload(item) for item in value]
    if not isinstance(value, dict):
        return value

    group_by_provider = value.get("group_by") == "provider"
    projected: dict[str, object] = {}
    for key, item in value.items():
        if key in {"source_name", "provider"}:
            projected["origin"] = _source_name_origin(item)
        elif key == "provider_coverage":
            projected["origin_coverage"] = project_origin_payload(item)
        elif key == "provider_breakdown" and isinstance(item, dict):
            projected["origin_breakdown"] = {_source_name_origin(origin): count for origin, count in item.items()}
        elif key == "providers_with_data":
            projected["origins_with_data"] = item
        elif key == "providers_without_data":
            projected["origins_without_data"] = item
        elif key == "group_by" and item == "provider":
            projected[key] = "origin"
        else:
            projected[key] = project_origin_payload(item)
    if group_by_provider and "bucket" in projected:
        projected["bucket"] = _source_name_origin(projected["bucket"])
    return projected


def insight_items_payload(
    items: Sequence[ArchiveInsightModel],
    insight_type: InsightType,
    *,
    item_key: str | None = None,
) -> dict[str, object]:
    """Return the shared machine payload for an insight list surface.

    The envelope follows the same ``{<key>: [...], "total": N}`` shape as
    every other paginated MCP/CLI list surface; the historical
    ``"count"`` field was renamed in #1007.
    """

    return {
        "total": len(items),
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


def _id_with_origin(identifier_attr: str) -> InsightAccessor:
    """Accessor rendering an identifier together with the origin name."""

    def accessor(item: ArchiveInsightModel) -> str:
        identifier = _stringify(getattr(item, identifier_attr, None))
        origin = _source_name_origin(getattr(item, "source_name", None))
        return f"{identifier} [{origin}]"

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
            CliOption("workflow_shape", ("--workflow-shape",), help="Only this workflow-shape label"),
            CliOption("terminal_state", ("--terminal-state",), help="Only this terminal-state label"),
            _QUERY_OPTION,
        ),
        fields=(
            InsightField("", _id_with_origin("session_id"), group=0),
            InsightField("tier", _attr("semantic_tier"), group=0),
            InsightField("", _attr("title", "(untitled)"), group=0),
            InsightField("session_date", _nested("evidence", "canonical_session_date"), group=1),
            InsightField("first", _nested("evidence", "first_message_at", "-"), group=1),
            InsightField("last", _nested("evidence", "last_message_at", "-"), group=1),
            InsightField("wall_s", _nested_ms_as_seconds("evidence", "wall_duration_ms", "0"), group=1),
            InsightField("ts_cov", _nested("evidence", "timestamp_coverage", "none"), group=1),
            InsightField("messages", _nested("evidence", "message_count", "0"), group=1),
            InsightField("engaged_min", _nested("inference", "engaged_minutes", "0"), group=1),
            InsightField("tool_active_min", _nested("inference", "tool_active_minutes", "0"), group=1),
            InsightField("shape", _nested("inference", "workflow_shape", "unknown"), group=1),
            InsightField("state", _nested("inference", "terminal_state", "unknown"), group=1),
            InsightField("think_s", _nested_ms_as_seconds("evidence", "thinking_duration_ms", "0"), group=1),
            InsightField("tool_s", _nested_ms_as_seconds("evidence", "tool_duration_ms", "0"), group=1),
            InsightField("tpm", _nested("evidence", "tool_calls_per_minute", "-"), group=1),
            InsightField("prov", _nested("evidence", "timing_provenance", "-"), group=1),
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
            CliOption("session_id", ("--session-id",), help="Only events from one session"),
            CliOption(
                "session_date_since",
                ("--session-date-since",),
                help="Only events whose canonical session date is on/after this date",
            ),
            CliOption(
                "session_date_until",
                ("--session-date-until",),
                help="Only events whose canonical session date is on/before this date",
            ),
            CliOption(
                "heuristic_label",
                ("--heuristic-label",),
                help="Only this weak heuristic work-event label",
            ),
            _QUERY_OPTION,
        ),
        fields=(
            InsightField("", _id_with_origin("event_id"), group=0),
            InsightField("label", _nested("inference", "heuristic_label"), group=0),
            InsightField("conv", _attr("session_id"), group=0),
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
        cli_options=(CliOption("session_id", ("--session-id",), help="Only phases from one session"),),
        fields=(
            InsightField("", _id_with_origin("phase_id"), group=0),
            InsightField("phase", _attr("phase_index"), group=0),
            InsightField("conv", _attr("session_id"), group=0),
            InsightField("start", _nested("evidence", "start_time"), group=1),
            InsightField("words", _nested("evidence", "word_count", "0"), group=1),
        ),
    )
)

register(
    InsightType(
        name="threads",
        display_name="Work Threads",
        json_key="threads",
        empty_message="No work threads matched.",
        query_model=ThreadInsightQuery,
        operations_method_name="list_thread_insights",
        cli_command_name="threads",
        cli_help="List durable thread insights.",
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
            InsightField("sessions", _attr("session_count", "0"), group=0),
            InsightField("explicit", _attr("explicit_count", "0"), group=0),
            InsightField("auto", _attr("auto_count", "0"), group=0),
        ),
    )
)

register(
    InsightType(
        name="archive_coverage",
        display_name="Archive Coverage",
        json_key="archive_coverage",
        empty_message="No archive coverage buckets matched.",
        query_model=ArchiveCoverageInsightQuery,
        operations_method_name="list_archive_coverage_insights",
        cli_command_name="coverage",
        cli_help="List archive coverage buckets by origin, day, or week.",
        readiness_exempt=True,
        cli_options=(
            CliOption(
                "group_by",
                ("--group-by",),
                type=click.Choice(["origin", "day", "week"]),
                default="origin",
                show_default=True,
                help="Bucket coverage by origin, day, or ISO week",
            ),
            CliOption("provider", ("--origin", "-o"), help="Only this origin"),
            CliOption("since", ("--since",), help="Only buckets at/after this timestamp or date"),
            CliOption("until", ("--until",), help="Only buckets at/before this timestamp or date"),
        ),
        fields=(
            InsightField("", _attr("bucket"), group=0),
            InsightField("group", _attr("group_by"), group=0),
            InsightField("origin", lambda item: _source_name_origin(getattr(item, "source_name", None)), group=0),
            InsightField("sessions", _attr("session_count", "0"), group=1),
            InsightField("messages", _attr("message_count", "0"), group=1),
            InsightField("words", _attr("total_words", "0"), group=1),
            InsightField("provider_user_msgs", _attr("user_message_count", "0"), group=1),
            InsightField("authored_user_msgs", _attr("authored_user_message_count", "0"), group=1),
            InsightField("provider_user_avg_words", _attr("avg_user_words", "0"), group=1),
            InsightField("authored_user_avg_words", _attr("avg_authored_user_words", "0"), group=1),
            InsightField("tool_active_ms", _attr("total_tool_active_duration_ms", "0"), group=1),
        ),
    )
)

register(
    InsightType(
        name="tool_usage",
        display_name="Tool Usage",
        json_key="tool_usage",
        empty_message="No tool usage data available.",
        query_model=ToolUsageInsightQuery,
        operations_method_name="list_tool_usage_insights",
        cli_command_name="tool-usage",
        cli_help="Per-tool, per-origin rollups over canonical actions with coverage map.",
        readiness_exempt=True,
        cli_options=(
            CliOption("tool", ("--tool",), help="Only entries for this normalized tool name"),
            CliOption("mcp_server", ("--mcp-server",), help="Only entries for this MCP server prefix"),
            CliOption(
                "action_kind",
                ("--action-kind",),
                help="Only entries for this action_kind value (e.g. file_read, shell)",
            ),
        ),
        mcp_default_limit=200,
        fields=(
            InsightField("origins_with_data", _attr("providers_with_data", "0"), group=0),
            InsightField("origins_without_data", _attr("providers_without_data", "0"), group=0),
            InsightField("total_calls", _attr("total_call_count", "0"), group=0),
            InsightField("distinct_tools", _attr("total_distinct_tools", "0"), group=0),
            InsightField("coverage_gaps", _attr("has_coverage_gaps"), group=0),
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
            CliOption("session_id", ("--session-id",), help="Only one session"),
            CliOption("model", ("--model",), help="Only this model or normalized model"),
            CliOption("status", ("--status",), type=click.Choice(["exact", "priced", "partial", "unavailable"])),
        ),
        fields=(
            InsightField("", _id_with_origin("session_id"), group=0),
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
        cli_help="List origin/model cost rollups.",
        readiness_exempt=True,
        cli_options=(CliOption("model", ("--model",), help="Only this model or normalized model"),),
        fields=(
            InsightField("", lambda item: _source_name_origin(getattr(item, "source_name", None)), group=0),
            InsightField("model", _attr("normalized_model"), group=0),
            InsightField("sessions", _attr("session_count", "0"), group=0),
            InsightField("priced", _attr("priced_session_count", "0"), group=1),
            InsightField("unavailable", _attr("unavailable_session_count", "0"), group=1),
            InsightField("usd", _attr("total_usd", "0"), group=1),
            InsightField("provider_usd", _nested("basis", "provider_reported_usd", "0"), group=1),
            InsightField("api_usd", _nested("basis", "api_equivalent_usd", "0"), group=1),
            InsightField("sub_usd", _nested("basis", "subscription_equivalent_usd", "0"), group=1),
            InsightField("catalog_usd", _nested("basis", "catalog_priced_usd", "0"), group=1),
            InsightField("confidence", _attr("confidence", "0"), group=1),
        ),
    )
)

register(
    InsightType(
        name="usage_timeline",
        display_name="Usage Timeline",
        json_key="usage_timeline",
        empty_message="No usage timeline rows matched.",
        query_model=UsageTimelineInsightQuery,
        operations_method_name="list_usage_timeline_insights",
        cli_command_name="usage-timeline",
        cli_help="List token, reasoning, cost, and subscription-credit usage by time bucket.",
        readiness_exempt=True,
        cli_options=(
            CliOption("model", ("--model",), help="Only this exact stored model name"),
            CliOption(
                "group_by",
                ("--group-by",),
                type=click.Choice(["month", "month-origin", "month-model", "month-origin-model"]),
                default="month-origin-model",
                show_default=True,
                help="Timeline grouping grain.",
            ),
        ),
        fields=(
            InsightField("", _attr("bucket"), group=0),
            InsightField("origin", lambda item: _source_name_origin(getattr(item, "source_name", None)), group=0),
            InsightField("model", _attr("normalized_model"), group=0),
            InsightField("sessions", _attr("session_count", "0"), group=1),
            InsightField("events", _attr("event_count", "0"), group=1),
            InsightField("tokens", _nested("usage", "total_tokens", "0"), group=1),
            InsightField("cache_read", _nested("usage", "cache_read_tokens", "0"), group=1),
            InsightField("reasoning", _attr("reasoning_output_tokens", "0"), group=1),
            InsightField("stored_usd", _attr("stored_cost_usd", "0"), group=2),
            InsightField("credits", _attr("subscription_credits", "0"), group=2),
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
    "project_origin_payload",
    "register",
    "render_insight_items",
]
