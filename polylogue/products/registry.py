"""Product type registry — typed descriptors for durable product surfaces.

Each product type is a ``ProductType`` descriptor that defines:
- how to display items in plain text (via ``fields``)
- what the JSON key is for API responses
- what display name to use in CLI output
- how to build a typed query model
- which archive-operations method provides the items
- CLI command metadata (name, help, options)

The rendering is generic: ``render_product_items()`` handles JSON mode
(all types) and plain-text mode (using field descriptors). Product
semantics stay in the archive/storage layers; the registry owns only the
transport and presentation contract.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import click

from polylogue.products.archive import (
    ArchiveDebtProductQuery,
    ArchiveProductModel,
    CostRollupProductQuery,
    DaySessionSummaryProductQuery,
    ProviderAnalyticsProductQuery,
    SessionCostProductQuery,
    SessionEnrichmentProductQuery,
    SessionPhaseProductQuery,
    SessionProfileProductQuery,
    SessionTagRollupQuery,
    SessionWorkEventProductQuery,
    WeekSessionSummaryProductQuery,
    WorkThreadProductQuery,
)

ProductAccessor: TypeAlias = Callable[[ArchiveProductModel], str]


@dataclass(frozen=True, slots=True)
class ProductField:
    """Describes one displayable field of a product item."""

    label: str
    accessor: ProductAccessor
    group: int = 0


@dataclass(frozen=True, slots=True)
class CliOption:
    """Describes one Click option for a product command."""

    param_name: str
    flags: tuple[str, ...]
    help: str = ""
    type: click.ParamType | type[object] | None = None
    default: object | None = None
    show_default: bool = False
    is_flag: bool = False
    expose_value_as: str | None = None


@dataclass(frozen=True, slots=True)
class ProductType:
    """Descriptor for one kind of derived product."""

    name: str
    display_name: str
    json_key: str
    fields: tuple[ProductField, ...] = ()
    empty_message: str = "No items matched."
    query_model: type[ArchiveProductModel] | None = None
    operations_method_name: str = ""
    cli_command_name: str = ""
    cli_help: str = ""
    cli_options: tuple[CliOption, ...] = ()
    mcp_default_limit: int = 50

    @property
    def resolved_cli_command_name(self) -> str:
        return self.cli_command_name or self.name.replace("_", "-")


def _model_payload(item: ArchiveProductModel) -> dict[str, object]:
    """Convert a product item to a JSON-serializable payload."""

    return item.model_dump(mode="json")


def product_items_payload(
    items: Sequence[ArchiveProductModel],
    product_type: ProductType,
    *,
    item_key: str | None = None,
) -> dict[str, object]:
    """Return the shared machine payload for a product list surface."""

    return {
        "count": len(items),
        item_key or product_type.json_key: [_model_payload(item) for item in items],
    }


def _stringify(value: object | None, default: str = "-") -> str:
    if value is None:
        return default
    if isinstance(value, str) and not value:
        return default
    return str(value)


def render_product_items(
    items: Sequence[ArchiveProductModel],
    product_type: ProductType,
    *,
    json_mode: bool = False,
) -> None:
    """Render product items using the product type descriptor."""

    if json_mode:
        from polylogue.cli.shared.machine_errors import emit_success

        emit_success(product_items_payload(items, product_type))
        return

    if not items:
        click.echo(product_type.empty_message)
        return

    click.echo(f"{product_type.display_name}: {len(items)}\n")

    for item in items:
        groups: dict[int, list[str]] = {}
        for field in product_type.fields:
            try:
                value = field.accessor(item)
            except (AttributeError, KeyError, TypeError):
                value = "-"
            groups.setdefault(field.group, []).append(f"{field.label}={value}" if field.label else str(value))

        for group_num in sorted(groups):
            prefix = "  " if group_num == 0 else "    "
            click.echo(f"{prefix}{' '.join(groups[group_num])}")


def _attr(name: str, default: str = "-") -> ProductAccessor:
    """Create an accessor that gets an attribute by name."""

    def accessor(item: ArchiveProductModel) -> str:
        return _stringify(getattr(item, name, None), default)

    return accessor


def _nested(outer: str, inner: str, default: str = "-") -> ProductAccessor:
    """Create an accessor that gets a nested attribute."""

    def accessor(item: ArchiveProductModel) -> str:
        nested = getattr(item, outer, None)
        if nested is None:
            return default
        return _stringify(getattr(nested, inner, None), default)

    return accessor


def _nested_ms_as_seconds(outer: str, inner: str, default: str = "-") -> ProductAccessor:
    """Create an accessor that renders a nested millisecond field as seconds."""

    def accessor(item: ArchiveProductModel) -> str:
        nested = getattr(item, outer, None)
        if nested is None:
            return default
        value = getattr(nested, inner, None)
        if isinstance(value, int):
            return str(max(value, 0) // 1000)
        return default

    return accessor


def _id_with_provider(identifier_attr: str) -> ProductAccessor:
    """Accessor rendering an identifier together with the provider name."""

    def accessor(item: ArchiveProductModel) -> str:
        identifier = _stringify(getattr(item, identifier_attr, None))
        provider = _stringify(getattr(item, "provider_name", None))
        return f"{identifier} [{provider}]"

    return accessor


def _list_preview(name: str, limit: int = 3) -> ProductAccessor:
    """Accessor showing the first N items of a tuple/list attribute."""

    def accessor(item: ArchiveProductModel) -> str:
        values = getattr(item, name, None)
        if isinstance(values, (list, tuple)):
            preview = ", ".join(str(value) for value in values[:limit])
            return preview or "-"
        return _stringify(values)

    return accessor


def _formatted_float(name: str, *, precision: int = 1, default: str = "-") -> ProductAccessor:
    """Accessor rendering a numeric attribute with fixed precision."""

    def accessor(item: ArchiveProductModel) -> str:
        value = getattr(item, name, None)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return default
        return f"{float(value):.{precision}f}"

    return accessor


def _count_with_percentage(count_attr: str, percentage_attr: str) -> ProductAccessor:
    """Accessor rendering ``count (pct%)`` pairs from sibling attributes."""

    def accessor(item: ArchiveProductModel) -> str:
        count = getattr(item, count_attr, None)
        percentage = getattr(item, percentage_attr, None)
        if not isinstance(count, int) or isinstance(count, bool):
            return "-"
        if not isinstance(percentage, (int, float)) or isinstance(percentage, bool):
            return str(count)
        return f"{count} ({float(percentage):.1f}%)"

    return accessor


PRODUCT_REGISTRY: dict[str, ProductType] = {}


def register(product_type: ProductType) -> ProductType:
    """Register a product type and return it."""

    PRODUCT_REGISTRY[product_type.name] = product_type
    return product_type


def get_product_type(name: str) -> ProductType:
    """Look up a registered product type by name."""

    product_type = PRODUCT_REGISTRY.get(name)
    if product_type is None:
        raise KeyError(f"Unknown product type: {name!r}. Available: {sorted(PRODUCT_REGISTRY)}")
    return product_type


def list_product_types() -> list[str]:
    """Return the sorted registered product type names."""

    return sorted(PRODUCT_REGISTRY)


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

_QUERY_OPTION = CliOption("query", ("--query",), help="FTS query against product search text")
_SESSION_TIME_SORT_OPTION = CliOption(
    "sort",
    ("--sort",),
    type=click.Choice(["source", "first-message", "last-message", "wallclock"]),
    default="source",
    show_default=True,
    help="Sort by source recency, first message time, last message time, or wallclock span",
)


register(
    ProductType(
        name="session_profiles",
        display_name="Session Profiles",
        json_key="session_profiles",
        empty_message="No session profiles matched.",
        query_model=SessionProfileProductQuery,
        operations_method_name="list_session_profile_products",
        cli_command_name="profiles",
        cli_help="List durable session-profile products.",
        cli_options=(
            *_SESSION_TIME_OPTIONS,
            _SESSION_TIME_SORT_OPTION,
            CliOption(
                "tier",
                ("--tier",),
                type=click.Choice(["merged", "evidence", "inference"]),
                default="merged",
                show_default=True,
                help="Return merged, evidence-only, or inference-only profile products",
            ),
            _QUERY_OPTION,
        ),
        fields=(
            ProductField("", _id_with_provider("conversation_id"), group=0),
            ProductField("tier", _attr("semantic_tier"), group=0),
            ProductField("", _attr("title", "(untitled)"), group=0),
            ProductField("session_date", _nested("evidence", "canonical_session_date"), group=1),
            ProductField("first", _nested("evidence", "first_message_at", "-"), group=1),
            ProductField("last", _nested("evidence", "last_message_at", "-"), group=1),
            ProductField("wall_s", _nested_ms_as_seconds("evidence", "wall_duration_ms", "0"), group=1),
            ProductField("ts_cov", _nested("evidence", "timestamp_coverage", "none"), group=1),
            ProductField("messages", _nested("evidence", "message_count", "0"), group=1),
            ProductField("engaged_min", _nested("inference", "engaged_minutes", "0"), group=1),
        ),
    )
)

register(
    ProductType(
        name="session_enrichments",
        display_name="Session Enrichments",
        json_key="session_enrichments",
        empty_message="No session enrichments matched.",
        query_model=SessionEnrichmentProductQuery,
        operations_method_name="list_session_enrichment_products",
        cli_command_name="enrichments",
        cli_help="List durable probabilistic session-enrichment products.",
        cli_options=(
            *_SESSION_TIME_OPTIONS,
            _SESSION_TIME_SORT_OPTION,
            _QUERY_OPTION,
        ),
        fields=(
            ProductField("", _id_with_provider("conversation_id"), group=0),
            ProductField("", _attr("title", "(untitled)"), group=0),
            ProductField("support", _nested("enrichment", "support_level"), group=1),
            ProductField("family", _nested("enrichment_provenance", "enrichment_family"), group=1),
        ),
    )
)

register(
    ProductType(
        name="session_work_events",
        display_name="Work Events",
        json_key="session_work_events",
        empty_message="No work events matched.",
        query_model=SessionWorkEventProductQuery,
        operations_method_name="list_session_work_event_products",
        cli_command_name="work-events",
        cli_help="List durable work-event products.",
        cli_options=(
            CliOption("conversation_id", ("--conversation-id",), help="Only events from one conversation"),
            CliOption("kind", ("--kind",), help="Only this work-event kind"),
            _QUERY_OPTION,
        ),
        fields=(
            ProductField("", _id_with_provider("event_id"), group=0),
            ProductField("kind", _nested("inference", "kind"), group=0),
            ProductField("conv", _attr("conversation_id"), group=0),
            ProductField("start", _nested("evidence", "start_time"), group=1),
            ProductField("end", _nested("evidence", "end_time"), group=1),
            ProductField("duration_ms", _nested("evidence", "duration_ms", "0"), group=1),
        ),
    )
)

register(
    ProductType(
        name="session_phases",
        display_name="Session Phases",
        json_key="session_phases",
        empty_message="No session phases matched.",
        query_model=SessionPhaseProductQuery,
        operations_method_name="list_session_phase_products",
        cli_command_name="phases",
        cli_help="List durable session-phase products.",
        cli_options=(
            CliOption("conversation_id", ("--conversation-id",), help="Only phases from one conversation"),
            CliOption("kind", ("--kind",), help="Only this session phase kind"),
        ),
        fields=(
            ProductField("", _id_with_provider("phase_id"), group=0),
            ProductField("kind", _nested("inference", "kind"), group=0),
            ProductField("conv", _attr("conversation_id"), group=0),
            ProductField("start", _nested("evidence", "start_time"), group=1),
            ProductField("words", _nested("evidence", "word_count", "0"), group=1),
        ),
    )
)

register(
    ProductType(
        name="work_threads",
        display_name="Work Threads",
        json_key="work_threads",
        empty_message="No work threads matched.",
        query_model=WorkThreadProductQuery,
        operations_method_name="list_work_thread_products",
        cli_command_name="threads",
        cli_help="List durable work-thread products.",
        cli_options=(_QUERY_OPTION,),
        fields=(
            ProductField("", _attr("thread_id"), group=0),
            ProductField("repo", _attr("dominant_repo", "-"), group=0),
            ProductField("sessions", _nested("thread", "session_count", "0"), group=0),
            ProductField("messages", _nested("thread", "total_messages", "0"), group=1),
            ProductField("depth", _nested("thread", "depth", "0"), group=1),
        ),
    )
)

register(
    ProductType(
        name="session_tag_rollups",
        display_name="Session Tag Rollups",
        json_key="session_tag_rollups",
        empty_message="No session tag rollups matched.",
        query_model=SessionTagRollupQuery,
        operations_method_name="list_session_tag_rollup_products",
        cli_command_name="tags",
        cli_help="List durable session-tag rollup products.",
        cli_options=(CliOption("query", ("--query",), help="Substring match against the tag name"),),
        mcp_default_limit=100,
        fields=(
            ProductField("", _attr("tag"), group=0),
            ProductField("conversations", _attr("conversation_count", "0"), group=0),
            ProductField("explicit", _attr("explicit_count", "0"), group=0),
            ProductField("auto", _attr("auto_count", "0"), group=0),
        ),
    )
)

register(
    ProductType(
        name="day_session_summaries",
        display_name="Day Session Summaries",
        json_key="day_session_summaries",
        empty_message="No day summaries matched.",
        query_model=DaySessionSummaryProductQuery,
        operations_method_name="list_day_session_summary_products",
        cli_command_name="day-summaries",
        cli_help="List durable day-level session summary products.",
        mcp_default_limit=90,
        fields=(
            ProductField("", _attr("date"), group=0),
            ProductField("sessions", _nested("summary", "session_count", "0"), group=0),
            ProductField("messages", _nested("summary", "total_messages", "0"), group=0),
        ),
    )
)

register(
    ProductType(
        name="week_session_summaries",
        display_name="Week Session Summaries",
        json_key="week_session_summaries",
        empty_message="No week summaries matched.",
        query_model=WeekSessionSummaryProductQuery,
        operations_method_name="list_week_session_summary_products",
        cli_command_name="week-summaries",
        cli_help="List durable week-level session summary products.",
        mcp_default_limit=52,
        fields=(
            ProductField("", _attr("iso_week"), group=0),
            ProductField("sessions", _nested("summary", "session_count", "0"), group=0),
            ProductField("messages", _nested("summary", "total_messages", "0"), group=0),
        ),
    )
)

register(
    ProductType(
        name="provider_analytics",
        display_name="Provider Analytics",
        json_key="provider_analytics",
        empty_message="No provider analytics matched.",
        query_model=ProviderAnalyticsProductQuery,
        operations_method_name="list_provider_analytics_products",
        cli_command_name="analytics",
        cli_help="List provider-level analytics products.",
        fields=(
            ProductField("", _attr("provider_name"), group=0),
            ProductField("conversations", _attr("conversation_count", "0"), group=0),
            ProductField("messages", _attr("message_count", "0"), group=0),
            ProductField("avg_messages", _formatted_float("avg_messages_per_conversation"), group=0),
            ProductField("tools", _count_with_percentage("tool_use_count", "tool_use_percentage"), group=1),
            ProductField("thinking", _count_with_percentage("thinking_count", "thinking_percentage"), group=1),
        ),
    )
)

register(
    ProductType(
        name="session_costs",
        display_name="Session Costs",
        json_key="session_costs",
        empty_message="No session cost estimates matched.",
        query_model=SessionCostProductQuery,
        operations_method_name="list_session_cost_products",
        cli_command_name="costs",
        cli_help="List session-level cost estimates.",
        cli_options=(
            CliOption("conversation_id", ("--conversation-id",), help="Only one conversation"),
            CliOption("model", ("--model",), help="Only this model or normalized model"),
            CliOption("status", ("--status",), type=click.Choice(["exact", "priced", "partial", "unavailable"])),
        ),
        fields=(
            ProductField("", _id_with_provider("conversation_id"), group=0),
            ProductField("status", _nested("estimate", "status"), group=0),
            ProductField("model", _nested("estimate", "normalized_model"), group=0),
            ProductField("usd", _nested("estimate", "total_usd", "0"), group=1),
            ProductField("confidence", _nested("estimate", "confidence", "0"), group=1),
        ),
    )
)

register(
    ProductType(
        name="cost_rollups",
        display_name="Cost Rollups",
        json_key="cost_rollups",
        empty_message="No cost rollups matched.",
        query_model=CostRollupProductQuery,
        operations_method_name="list_cost_rollup_products",
        cli_command_name="cost-rollups",
        cli_help="List provider/model cost rollups.",
        cli_options=(CliOption("model", ("--model",), help="Only this model or normalized model"),),
        fields=(
            ProductField("", _attr("provider_name"), group=0),
            ProductField("model", _attr("normalized_model"), group=0),
            ProductField("sessions", _attr("session_count", "0"), group=0),
            ProductField("priced", _attr("priced_session_count", "0"), group=1),
            ProductField("unavailable", _attr("unavailable_session_count", "0"), group=1),
            ProductField("usd", _attr("total_usd", "0"), group=1),
            ProductField("confidence", _attr("confidence", "0"), group=1),
        ),
    )
)

register(
    ProductType(
        name="archive_debt",
        display_name="Archive Debt",
        json_key="archive_debt",
        empty_message="No archive debt entries matched.",
        query_model=ArchiveDebtProductQuery,
        operations_method_name="list_archive_debt_products",
        cli_command_name="debt",
        cli_help="List archive debt and maintenance readiness products.",
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
            ProductField("", _attr("debt_name"), group=0),
            ProductField("category", _attr("category"), group=0),
            ProductField("target", _attr("maintenance_target"), group=0),
            ProductField("issues", _attr("issue_count", "0"), group=1),
            ProductField("healthy", _attr("healthy"), group=1),
            ProductField("destructive", _attr("destructive"), group=1),
            ProductField("detail", _attr("detail"), group=2),
        ),
    )
)


class ProductQueryError(ValueError):
    """Raised when a registry-backed product query is invalid."""


def _build_query(
    product_type: ProductType,
    **kwargs: object,
) -> ArchiveProductModel:
    """Build and validate the typed query object for a product fetch."""

    query_model = product_type.query_model
    if query_model is None:
        raise ProductQueryError(f"Product type {product_type.name} does not declare a query model")
    accepted = set(query_model.model_fields)
    unknown = sorted(set(kwargs) - accepted)
    if unknown:
        unknown_list = ", ".join(unknown)
        accepted_list = ", ".join(sorted(accepted))
        raise ProductQueryError(
            f"Unknown query field(s) for {product_type.name}: {unknown_list}. Accepted fields: {accepted_list}"
        )
    return query_model(**kwargs)


def fetch_products(
    product_type: ProductType,
    operations: object,
    **kwargs: object,
) -> list[ArchiveProductModel]:
    """Fetch product items using the registry dispatch metadata."""

    from polylogue.api.sync.bridge import run_coroutine_sync

    query = _build_query(product_type, **kwargs)
    method = getattr(operations, product_type.operations_method_name)
    return list(run_coroutine_sync(method(query)))


async def fetch_products_async(
    product_type: ProductType,
    operations: object,
    **kwargs: object,
) -> list[ArchiveProductModel]:
    """Async variant of ``fetch_products()``."""

    query = _build_query(product_type, **kwargs)
    method = getattr(operations, product_type.operations_method_name)
    return list(await method(query))


__all__ = [
    "CliOption",
    "PRODUCT_REGISTRY",
    "ProductField",
    "ProductQueryError",
    "ProductType",
    "fetch_products",
    "fetch_products_async",
    "get_product_type",
    "list_product_types",
    "product_items_payload",
    "register",
    "render_product_items",
]
