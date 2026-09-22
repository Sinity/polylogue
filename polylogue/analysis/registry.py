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
from typing import Any, Literal, TypeAlias, cast

import click

from polylogue.analysis.archive import (
    ArchiveCoverageInsight,
    ArchiveCoverageInsightQuery,
    ArchiveDebtInsight,
    ArchiveDebtInsightQuery,
    ArchiveInsightModel,
    CostRollupInsight,
    CostRollupInsightQuery,
    SessionCostInsight,
    SessionCostInsightQuery,
    SessionProfileInsight,
    SessionProfileInsightQuery,
    SessionTagRollupInsight,
    SessionTagRollupQuery,
    ThreadInsight,
    ThreadInsightQuery,
    UsageTimelineInsight,
    UsageTimelineInsightQuery,
)
from polylogue.analysis.command_shapes import CommandShapeUsage, CommandShapeUsageQuery
from polylogue.analysis.tool_episodes import ToolEpisodeInsight, ToolEpisodeQuery
from polylogue.analysis.tool_usage import ToolUsageInsight, ToolUsageInsightQuery
from polylogue.core.errors import PolylogueError
from polylogue.core.evidence_families import fact_family_schema

InsightAccessor: TypeAlias = Callable[[ArchiveInsightModel], str]

# Shared evidence declarations are exposed through the existing insight
# registry so discovery and renderers consume one generated schema source.
EVIDENCE_FAMILY_SCHEMAS = fact_family_schema()


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


RetentionDecision: TypeAlias = Literal["keep", "reduce"]


@dataclass(frozen=True, slots=True)
class RetentionVerdict:
    """One recorded adjudication of whether an insight type earns its place.

    polylogue-4p1.3 asked for a per-type keep/reduce/delete verdict with the
    discrimination evidence that decided it. Carrying the verdict on the
    descriptor rather than in prose is what makes it enforceable: ``register``
    refuses a type that arrives without one, so a new insight cannot ship
    unreviewed and the review set can never drift out of step with the
    registry it is supposed to cover.

    ``decision`` is deliberately only ``keep`` or ``reduce``. A ``delete``
    verdict has no descriptor to live on once it is executed; retirements are
    recorded in :data:`RETIRED_INSIGHT_TYPES` instead.
    """

    decision: RetentionDecision
    evidence: str
    """Why this type is distinguishable from a named query over the DSL.

    The load-bearing property (polylogue-4p1.3) is that an insight pairs an
    aggregate with explicit coverage semantics -- it can say "unavailable"
    where a query can only say "zero".
    """
    recorded_in: str
    """The task record that adjudicated this type."""

    def __post_init__(self) -> None:
        if not self.evidence.strip():
            raise ValueError("a retention verdict requires discrimination evidence")
        if not self.recorded_in.strip():
            raise ValueError("a retention verdict must name the record that decided it")


@dataclass(frozen=True, slots=True)
class RetiredInsightType:
    """One insight type whose recorded verdict was executed as a deletion.

    Keeping the retirement is not compatibility: nothing reads a retired type.
    It exists so a later pass cannot re-attempt a completed deletion, and so
    re-registering the name is a hard error rather than a silent resurrection
    of a shape that was already adjudicated away.
    """

    name: str
    evidence: str
    retired_in: str


#: Types deleted by an executed verdict. ``session_phases`` and
#: ``session_work_events`` were removed in dafcfc612; polylogue-4p1.3's
#: verdict list still names them, so re-reading that list without this record
#: re-attempts two completed deletions.
RETIRED_INSIGHT_TYPES: dict[str, RetiredInsightType] = {
    record.name: record
    for record in (
        RetiredInsightType(
            name="session_phases",
            evidence=(
                "82% of materialized rows were a single span, no span carried a label, and the span "
                "timestamps were synthesized from message index rather than observed -- an aggregate "
                "with neither a second value to compare nor a coverage signal to qualify it."
            ),
            retired_in="dafcfc612",
        ),
        RetiredInsightType(
            name="session_work_events",
            evidence=(
                "82% of materialized rows were a single event and the rows duplicated action_pairs, "
                "which the query DSL already reaches directly."
            ),
            retired_in="dafcfc612",
        ),
    )
}


@dataclass(frozen=True, slots=True)
class InsightType:
    """Descriptor for one kind of derived insight."""

    name: str
    display_name: str
    json_key: str
    retention: RetentionVerdict | None = None
    """Recorded keep/reduce adjudication. ``register`` refuses ``None``."""
    fields: tuple[InsightField, ...] = ()
    item_model: type[ArchiveInsightModel] | None = None
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

    return cast(dict[str, object], item.model_dump(mode="json"))


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
        from polylogue.surfaces.machine_envelope import emit_success

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


def _nested2(outer: str, mid: str, inner: str, default: str = "-") -> InsightAccessor:
    """Create an accessor that gets a doubly-nested attribute."""

    def accessor(item: ArchiveInsightModel) -> str:
        outer_value = getattr(item, outer, None)
        if outer_value is None:
            return default
        mid_value = getattr(outer_value, mid, None)
        if mid_value is None:
            return default
        return _stringify(getattr(mid_value, inner, None), default)

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
        origin = _stringify(getattr(item, "origin", None))
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


class InsightRegistrationError(PolylogueError):
    """Raised when an insight type is registered without a recorded verdict."""

    http_status_code = 500


def register(insight_type: InsightType) -> InsightType:
    """Register an insight type and return it.

    Registration is the review boundary: a type with no recorded
    :class:`RetentionVerdict`, or one reusing a retired name, is refused here
    rather than discovered later by whoever reads the registry.
    """

    retired = RETIRED_INSIGHT_TYPES.get(insight_type.name)
    if retired is not None:
        raise InsightRegistrationError(
            f"insight type {insight_type.name!r} was retired in {retired.retired_in}; "
            "record a new verdict under a new name rather than resurrecting it"
        )
    if insight_type.retention is None:
        raise InsightRegistrationError(
            f"insight type {insight_type.name!r} has no recorded retention verdict; "
            "adjudicate it (keep/reduce) with discrimination evidence before registering it"
        )
    INSIGHT_REGISTRY[insight_type.name] = insight_type
    return insight_type


def unverdicted_insight_types() -> tuple[str, ...]:
    """Return registered names carrying no usable retention verdict.

    Derived from :data:`INSIGHT_REGISTRY` so the review set is exactly the
    registered set; nothing restates the count.
    """

    return tuple(
        sorted(
            name
            for name, insight_type in INSIGHT_REGISTRY.items()
            if insight_type.retention is None or not insight_type.retention.evidence.strip()
        )
    )


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
        retention=RetentionVerdict(
            decision="reduce",
            evidence=(
                "Earns its place: the only session read model that separates measured evidence from "
                "probabilistic inference and enrichment, each with its own provenance, support_level and "
                "time_confidence, so a consumer can tell a counted span from an inferred one. Reduce: "
                "polylogue-4p1.3 recorded surplus constant version/family columns and 100%-NULL cost columns "
                "on the materialized table; that reduction is owned by polylogue-f2qv.6 and is not re-derived "
                "here."
            ),
            recorded_in="polylogue-4p1.5",
        ),
        display_name="Session Profiles",
        json_key="session_profiles",
        item_model=SessionProfileInsight,
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
            InsightField("posture", _nested2("enrichment", "objective_posture", "posture", "unknown"), group=1),
            InsightField(
                "posture_authority", _nested2("enrichment", "objective_posture", "authority", "none"), group=1
            ),
            InsightField("think_s", _nested_ms_as_seconds("evidence", "thinking_duration_ms", "0"), group=1),
            InsightField("tool_s", _nested_ms_as_seconds("evidence", "tool_duration_ms", "0"), group=1),
            InsightField("tpm", _nested("evidence", "tool_calls_per_minute", "-"), group=1),
            InsightField("prov", _nested("evidence", "timing_provenance", "-"), group=1),
            InsightField("time", _nested("provenance", "time_confidence", "unknown"), group=1),
        ),
    )
)

register(
    InsightType(
        name="threads",
        retention=RetentionVerdict(
            decision="keep",
            evidence=(
                "Recomposes a root session's lineage tree from parent/child links. Lineage children store "
                "only their divergent tail, so the tree is a recomposition rather than a selection -- the "
                "query DSL can filter sessions but cannot rebuild the replayed prefix. Single-session threads "
                "are correct, not degenerate."
            ),
            recorded_in="polylogue-4p1.5",
        ),
        display_name="Work Threads",
        json_key="threads",
        item_model=ThreadInsight,
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
            InsightField("time", _nested("provenance", "time_confidence", "unknown"), group=1),
        ),
    )
)

register(
    InsightType(
        name="session_tag_rollups",
        retention=RetentionVerdict(
            decision="keep",
            evidence=(
                "Splits explicit_count from auto_count, so an operator-asserted tag is never summed together "
                "with a probabilistic enrichment tag. polylogue-4p1.3 filed this as reduce on the evidence "
                "that explicit_count was constant 0; re-checked at head, build_session_tag_rollups increments "
                "explicit_count from each profile's explicit tag set, so the observed zero was a fact about "
                "the then-live archive's tag population, not a dead column."
            ),
            recorded_in="polylogue-4p1.5",
        ),
        display_name="Session Tag Rollups",
        json_key="session_tag_rollups",
        item_model=SessionTagRollupInsight,
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
        retention=RetentionVerdict(
            decision="keep",
            evidence=(
                "A coverage rollup whose provenance is populated only for day/week grouping, which is itself "
                "the declared signal that a default origin-grouped row carries no materialization stamp. That "
                "'the stamp is absent, and that absence is meaningful' distinction has no expression in the "
                "query DSL."
            ),
            recorded_in="polylogue-4p1.5",
        ),
        display_name="Archive Coverage",
        json_key="archive_coverage",
        item_model=ArchiveCoverageInsight,
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
            CliOption("origin", ("--origin", "-o"), help="Only this origin"),
            CliOption("since", ("--since",), help="Only buckets at/after this timestamp or date"),
            CliOption("until", ("--until",), help="Only buckets at/before this timestamp or date"),
        ),
        fields=(
            InsightField("", _attr("bucket"), group=0),
            InsightField("group", _attr("group_by"), group=0),
            InsightField("origin", _attr("origin"), group=0),
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
        retention=RetentionVerdict(
            decision="keep",
            evidence=(
                "The canonical example of the coverage pairing: has_coverage_gaps and the per-entry "
                "origin_coverage[].data_available separate a genuine zero tool-use count from an origin with "
                "no ingested action evidence at all. A DSL aggregate can only report zero."
            ),
            recorded_in="polylogue-4p1.5",
        ),
        display_name="Tool Usage",
        json_key="tool_usage",
        item_model=ToolUsageInsight,
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
            InsightField("origins_with_data", _attr("origins_with_data", "0"), group=0),
            InsightField("origins_without_data", _attr("origins_without_data", "0"), group=0),
            InsightField("total_calls", _attr("total_call_count", "0"), group=0),
            InsightField("distinct_tools", _attr("total_distinct_tools", "0"), group=0),
            InsightField("coverage_gaps", _attr("has_coverage_gaps"), group=0),
        ),
    )
)

register(
    InsightType(
        name="tool_episodes",
        retention=RetentionVerdict(
            decision="keep",
            evidence=(
                "Not a named query over the actions view. result_state plus caveat state why an outcome is "
                "absent -- 'outcome unknown: no paired structural result' is never collapsed into success -- "
                "and each episode carries a bounded three-message context window either side of the call plus "
                "its follow-up class. actions joins tool_use to tool_result and stops there; neither the "
                "unknown-outcome reason nor the surrounding context is reachable from the DSL."
            ),
            recorded_in="polylogue-4p1.5",
        ),
        display_name="Tool Episodes",
        json_key="tool_episodes",
        item_model=ToolEpisodeInsight,
        query_model=ToolEpisodeQuery,
        operations_method_name="list_tool_episode_insights",
        cli_command_name="tool-episodes",
        cli_help="List call/result episodes with structural outcomes and context.",
        readiness_exempt=True,
        cli_options=(
            CliOption("session_id", ("--session-id",), help="Only episodes from one session"),
            CliOption("tool", ("--tool",), help="Only episodes for this tool"),
            CliOption("result_state", ("--result-state",), help="Only this result state"),
        ),
        mcp_default_limit=100,
        fields=(
            InsightField("", _attr("episode_id")),
            InsightField("tool", _attr("tool_name")),
            InsightField("state", _attr("result_state")),
            InsightField("error", _attr("is_error")),
            InsightField("exit", _attr("exit_code")),
            InsightField("next", _attr("next_action")),
            InsightField("caveat", _attr("caveat"), group=1),
        ),
    )
)

register(
    InsightType(
        name="command_shapes",
        retention=RetentionVerdict(
            decision="reduce",
            evidence=(
                "Earns its place: normalize_command_shapes is a shell-aware normalization (pipeline and "
                "separator splitting, transparent leading env assignments and sh -c wrappers, path-like "
                "positionals dropped so an argument cannot become a new shape) that no GROUP BY over "
                "actions.tool_command can express, and the declared readiness semantics make an empty result "
                "mean 'no execution observed in this window' rather than 'unused'. Reduce, executed with this "
                "verdict: last_used_sort_key was a byte-identical restatement of provenance.source_sort_key "
                "on the same row with no reader anywhere in the repository, and is deleted."
            ),
            recorded_in="polylogue-4p1.5",
        ),
        display_name="Command Shape Usage",
        json_key="command_shapes",
        item_model=CommandShapeUsage,
        empty_message="No executed command shapes matched.",
        query_model=CommandShapeUsageQuery,
        operations_method_name="list_command_shape_usage",
        cli_command_name="command-shapes",
        cli_help="List normalized executed command shapes and last-use times.",
        readiness_exempt=True,
        mcp_default_limit=200,
        cli_options=(
            CliOption("session_id", ("--session-id",), help="Only commands from one session"),
            CliOption("repository", ("--repository", "--repo"), help="Only commands associated with this repository"),
        ),
        fields=(
            InsightField("shape", _attr("command_shape"), group=0),
            InsightField("count", _attr("execution_count", "0"), group=0),
            InsightField("sessions", _attr("session_count", "0"), group=0),
            InsightField("origin", _attr("origin"), group=1),
            InsightField("repo", _attr("repository"), group=1),
            InsightField("last", _attr("last_used_at"), group=1),
        ),
    )
)

register(
    InsightType(
        name="session_costs",
        retention=RetentionVerdict(
            decision="keep",
            evidence=(
                "estimate.status separates exact, priced, partial and unavailable pricing, and "
                "missing_reasons/unavailable_reason name why a row could not be priced. A SUM over stored "
                "costs reports a number for all four cases and cannot say which one it is."
            ),
            recorded_in="polylogue-4p1.5",
        ),
        display_name="Session Costs",
        json_key="session_costs",
        item_model=SessionCostInsight,
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
            # An unavailable estimate carries ``total_usd = None``. Defaulting
            # to ``"0"`` here rendered ``usd=0`` on the plaintext surface while
            # JSON and HTTP emitted ``null`` for the same row, preserving the
            # exact false zero the nullable-cost contract exists to remove.
            # ``-`` is this renderer's own unknown marker, and it reads beside
            # the row's ``status=unavailable``.
            InsightField("usd", _nested("estimate", "total_usd"), group=1),
            InsightField("confidence", _nested("estimate", "confidence"), group=1),
        ),
    )
)

register(
    InsightType(
        name="cost_rollups",
        retention=RetentionVerdict(
            decision="keep",
            evidence=(
                "Carries unavailable_session_count and status_counts beside the totals, so a rollup states "
                "how much of itself is unpriced. A DSL aggregate silently omits the rows it could not price."
            ),
            recorded_in="polylogue-4p1.5",
        ),
        display_name="Cost Rollups",
        json_key="cost_rollups",
        item_model=CostRollupInsight,
        empty_message="No cost rollups matched.",
        query_model=CostRollupInsightQuery,
        operations_method_name="list_cost_rollup_insights",
        cli_command_name="cost-rollups",
        cli_help="List origin/model cost rollups.",
        readiness_exempt=True,
        cli_options=(CliOption("model", ("--model",), help="Only this model or normalized model"),),
        fields=(
            InsightField("", _attr("origin"), group=0),
            InsightField("model", _attr("normalized_model"), group=0),
            InsightField("sessions", _attr("session_count", "0"), group=0),
            InsightField("priced", _attr("priced_session_count", "0"), group=1),
            InsightField("unavailable", _attr("unavailable_session_count", "0"), group=1),
            InsightField("usd", _attr("total_usd", "0"), group=1),
            InsightField("provider_usd", _nested("basis", "provider_reported_usd", "0"), group=1),
            InsightField("api_usd", _nested("basis", "api_equivalent_usd", "0"), group=1),
            InsightField("sub_usd", _nested("basis", "subscription_equivalent_usd", "0"), group=1),
            InsightField("catalog_usd", _nested("basis", "catalog_priced_usd", "0"), group=1),
            InsightField("confidence", _attr("confidence", "uncovered"), group=1),
        ),
    )
)

register(
    InsightType(
        name="usage_timeline",
        retention=RetentionVerdict(
            decision="keep",
            evidence=(
                "cost_provenance_counts reports how much of each bucket is stored versus catalog-estimated, "
                "which matters precisely because subscription_credits is otherwise indistinguishable in the "
                "payload from a stored credit figure."
            ),
            recorded_in="polylogue-4p1.5",
        ),
        display_name="Usage Timeline",
        json_key="usage_timeline",
        item_model=UsageTimelineInsight,
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
            InsightField("origin", _attr("origin"), group=0),
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
        retention=RetentionVerdict(
            decision="keep",
            evidence=(
                "Not an aggregate over archived content at all: each row is a live health check over current "
                "archive tables (FTS sync, orphaned profile rows, materialization staleness). There is no "
                "query-DSL expression of 'this derived tier disagrees with its source'."
            ),
            recorded_in="polylogue-4p1.5",
        ),
        display_name="Archive Debt",
        json_key="archive_debt",
        item_model=ArchiveDebtInsight,
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

    from polylogue.core.async_bridge import run_coroutine_sync

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
    "RETIRED_INSIGHT_TYPES",
    "InsightField",
    "InsightQueryError",
    "InsightRegistrationError",
    "InsightType",
    "RetentionDecision",
    "RetentionVerdict",
    "RetiredInsightType",
    "unverdicted_insight_types",
    "fetch_insights",
    "fetch_insights_async",
    "get_insight_type",
    "list_insight_types",
    "insight_items_payload",
    "register",
    "render_insight_items",
]
