"""Product type registry — data-driven derived product descriptors.

Each product type is a ``ProductType`` descriptor that defines:
- how to display items in plain text (via ``fields``)
- what the JSON key is for API responses
- what display name to use in CLI output
- how to fetch data (query class + operations method name)
- CLI command configuration (name, help, options)

The rendering is generic: ``render_product_items()`` handles JSON mode
(all types) and plain-text mode (using field descriptors). Adding a new
product type requires only a new ``ProductType`` registration — no new
rendering, workflow, command, or MCP tool files.

Product *semantics* (row builders, lifecycle, storage) remain in code;
only the *transport/presentation* layer is generic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import click
from pydantic import BaseModel

from polylogue.archive_products import ArchiveProductModel

# -------------------------------------------------------------------
# Field descriptors
# -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProductField:
    """Describes one displayable field of a product item."""

    label: str
    accessor: Callable[[Any], str]
    """Function that takes an item and returns a display string."""
    group: int = 0
    """Fields with the same group number are displayed on the same line."""


# -------------------------------------------------------------------
# CLI option descriptor
# -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CliOption:
    """Describes one Click option for a product command."""

    param_name: str
    """Python parameter name (snake_case)."""
    flags: tuple[str, ...]
    """Click flags, e.g. ('--provider',)."""
    help: str = ""
    type: type | click.ParamType | None = None
    default: Any = None
    show_default: bool = False
    is_flag: bool = False
    expose_value_as: str | None = None
    """If set, Click will expose the value under this name instead of param_name."""


# -------------------------------------------------------------------
# Product type descriptor
# -------------------------------------------------------------------


@dataclass(frozen=True)
class ProductType:
    """Descriptor for one kind of derived product."""

    name: str
    """Registry key, e.g. 'session_profiles'."""

    display_name: str
    """Human-readable name, e.g. 'Session Profiles'."""

    json_key: str
    """Key used in JSON API responses."""

    fields: list[ProductField] = field(default_factory=list)
    """Fields to display in plain-text mode, grouped by ``group`` number."""

    empty_message: str = "No items matched."
    """Message shown when no items match the query."""

    # --- Dispatch metadata ---

    query_class_path: str = ""
    """Dotted import path to the query class, e.g.
    'polylogue.archive_products.SessionProfileProductQuery'."""

    operations_method: str = ""
    """Name of the async method on ArchiveOperations, e.g.
    'list_session_profile_products'."""

    cli_command_name: str = ""
    """Click command name, e.g. 'profiles'. Defaults to name with _ -> -."""

    cli_help: str = ""
    """Help string for the Click command."""

    cli_options: list[CliOption] = field(default_factory=list)
    """Extra Click options beyond the standard --json, --limit, --offset."""

    mcp_default_limit: int = 50
    """Default limit for MCP tool calls."""

    @property
    def resolved_cli_command_name(self) -> str:
        return self.cli_command_name or self.name.replace("_", "-")


# -------------------------------------------------------------------
# Generic rendering
# -------------------------------------------------------------------


def _model_payload(item: object) -> object:
    """Convert a product item to a JSON-serializable payload."""
    return item.model_dump(mode="json") if hasattr(item, "model_dump") else item


def render_product_items(
    items: list[Any],
    product_type: ProductType,
    *,
    json_mode: bool = False,
) -> None:
    """Render product items to stdout using the product type descriptor.

    JSON mode: emit structured JSON via the machine-error envelope.
    Plain mode: format using field descriptors.
    """
    if json_mode:
        from polylogue.cli.machine_errors import emit_success

        emit_success(
            {
                "count": len(items),
                product_type.json_key: [_model_payload(item) for item in items],
            }
        )
        return

    if not items:
        click.echo(product_type.empty_message)
        return

    click.echo(f"{product_type.display_name}: {len(items)}\n")

    for item in items:
        # Group fields by group number, render each group as one line
        groups: dict[int, list[str]] = {}
        for f in product_type.fields:
            try:
                value = f.accessor(item)
            except (AttributeError, KeyError, TypeError):
                value = "-"
            groups.setdefault(f.group, []).append(f"{f.label}={value}" if f.label else str(value))

        for group_num in sorted(groups):
            prefix = "  " if group_num == 0 else "    "
            click.echo(f"{prefix}{' '.join(groups[group_num])}")


# -------------------------------------------------------------------
# Helper for building accessors
# -------------------------------------------------------------------


def _attr(name: str, default: str = "-") -> Callable[[Any], str]:
    """Create an accessor that gets an attribute by name."""

    def accessor(item: Any) -> str:
        value = getattr(item, name, None)
        if value is None:
            return default
        return str(value)

    return accessor


def _nested(outer: str, inner: str, default: str = "-") -> Callable[[Any], str]:
    """Create an accessor that gets a nested dict value from a model_dump."""

    def accessor(item: Any) -> str:
        obj = getattr(item, outer, None)
        if obj is None:
            return default
        if hasattr(obj, "model_dump"):
            obj = obj.model_dump(mode="json")
        if isinstance(obj, dict):
            value = obj.get(inner)
            return str(value) if value is not None else default
        return default

    return accessor


def _list_preview(name: str, limit: int = 3) -> Callable[[Any], str]:
    """Accessor showing first N items of a list attribute."""

    def accessor(item: Any) -> str:
        values = getattr(item, name, None)
        if not values:
            return "-"
        if isinstance(values, (list, tuple)):
            return ", ".join(str(v) for v in values[:limit]) or "-"
        return str(values)

    return accessor


# -------------------------------------------------------------------
# Registry
# -------------------------------------------------------------------

PRODUCT_REGISTRY: dict[str, ProductType] = {}


def register(pt: ProductType) -> ProductType:
    """Register a product type and return it."""
    PRODUCT_REGISTRY[pt.name] = pt
    return pt


def get_product_type(name: str) -> ProductType:
    """Look up a registered product type by name."""
    pt = PRODUCT_REGISTRY.get(name)
    if pt is None:
        raise KeyError(f"Unknown product type: {name!r}. Available: {sorted(PRODUCT_REGISTRY)}")
    return pt


def list_product_types() -> list[str]:
    """Return sorted list of registered product type names."""
    return sorted(PRODUCT_REGISTRY)


# -------------------------------------------------------------------
# Shared option sets
# -------------------------------------------------------------------

_SESSION_TIME_OPTIONS: list[CliOption] = [
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
]

_QUERY_OPTION = CliOption("query", ("--query",), help="FTS query against product search text")


# -------------------------------------------------------------------
# Product type registrations
# -------------------------------------------------------------------

register(
    ProductType(
        name="session_profiles",
        display_name="Session Profiles",
        json_key="session_profiles",
        empty_message="No session profiles matched.",
        query_class_path="polylogue.archive_products.SessionProfileProductQuery",
        operations_method="list_session_profile_products",
        cli_command_name="profiles",
        cli_help="List durable session-profile products.",
        cli_options=[
            *_SESSION_TIME_OPTIONS,
            CliOption(
                "tier",
                ("--tier",),
                type=click.Choice(["merged", "evidence", "inference"]),
                default="merged",
                show_default=True,
                help="Return merged, evidence-only, or inference-only profile products",
            ),
            _QUERY_OPTION,
        ],
        fields=[
            ProductField("", lambda i: f"{i.conversation_id} [{i.provider_name}]", group=0),
            ProductField("tier", _attr("semantic_tier"), group=0),
            ProductField("", _attr("title", "(untitled)"), group=0),
            ProductField("session_date", _nested("evidence", "canonical_session_date"), group=1),
            ProductField("messages", _nested("evidence", "message_count", "0"), group=1),
            ProductField("engaged_min", _nested("inference", "engaged_minutes", "0"), group=1),
        ],
    )
)

register(
    ProductType(
        name="session_enrichments",
        display_name="Session Enrichments",
        json_key="session_enrichments",
        empty_message="No session enrichments matched.",
        query_class_path="polylogue.archive_products.SessionEnrichmentProductQuery",
        operations_method="list_session_enrichment_products",
        cli_command_name="enrichments",
        cli_help="List durable probabilistic session-enrichment products.",
        cli_options=[
            *_SESSION_TIME_OPTIONS,
            _QUERY_OPTION,
        ],
        fields=[
            ProductField("", lambda i: f"{i.conversation_id} [{i.provider_name}]", group=0),
            ProductField("", _attr("title", "(untitled)"), group=0),
            ProductField("support", _nested("enrichment", "support_level"), group=1),
            ProductField(
                "family",
                lambda i: i.enrichment_provenance.enrichment_family if hasattr(i, "enrichment_provenance") else "-",
                group=1,
            ),
        ],
    )
)

register(
    ProductType(
        name="session_work_events",
        display_name="Work Events",
        json_key="session_work_events",
        empty_message="No work events matched.",
        query_class_path="polylogue.archive_products.SessionWorkEventProductQuery",
        operations_method="list_session_work_event_products",
        cli_command_name="work-events",
        cli_help="List durable work-event products.",
        cli_options=[
            CliOption("conversation_id", ("--conversation-id",), help="Only events from one conversation"),
            CliOption("kind", ("--kind",), help="Only this work-event kind"),
            _QUERY_OPTION,
        ],
        fields=[
            ProductField("", lambda i: f"{i.event_id} [{i.provider_name}]", group=0),
            ProductField("kind", _nested("inference", "kind"), group=0),
            ProductField("conv", _attr("conversation_id"), group=0),
            ProductField("start", _nested("evidence", "start_time"), group=1),
            ProductField("end", _nested("evidence", "end_time"), group=1),
            ProductField("duration_ms", _nested("evidence", "duration_ms", "0"), group=1),
        ],
    )
)

register(
    ProductType(
        name="session_phases",
        display_name="Session Phases",
        json_key="session_phases",
        empty_message="No session phases matched.",
        query_class_path="polylogue.archive_products.SessionPhaseProductQuery",
        operations_method="list_session_phase_products",
        cli_command_name="phases",
        cli_help="List durable session-phase products.",
        cli_options=[
            CliOption("conversation_id", ("--conversation-id",), help="Only phases from one conversation"),
            CliOption("kind", ("--kind",), help="Only this session phase kind"),
        ],
        fields=[
            ProductField("", lambda i: f"{i.phase_id} [{i.provider_name}]", group=0),
            ProductField("kind", _nested("inference", "kind"), group=0),
            ProductField("conv", _attr("conversation_id"), group=0),
            ProductField("start", _nested("evidence", "start_time"), group=1),
            ProductField("words", _nested("evidence", "word_count", "0"), group=1),
        ],
    )
)

register(
    ProductType(
        name="work_threads",
        display_name="Work Threads",
        json_key="work_threads",
        empty_message="No work threads matched.",
        query_class_path="polylogue.archive_products.WorkThreadProductQuery",
        operations_method="list_work_thread_products",
        cli_command_name="threads",
        cli_help="List durable work-thread products.",
        cli_options=[
            _QUERY_OPTION,
        ],
        fields=[
            ProductField("", _attr("thread_id"), group=0),
            ProductField("repo", _attr("dominant_repo", "-"), group=0),
            ProductField(
                "sessions", lambda i: str(i.thread.get("session_count", 0)) if hasattr(i, "thread") else "0", group=0
            ),
            ProductField(
                "messages", lambda i: str(i.thread.get("total_messages", 0)) if hasattr(i, "thread") else "0", group=1
            ),
            ProductField("depth", lambda i: str(i.thread.get("depth", 0)) if hasattr(i, "thread") else "0", group=1),
        ],
    )
)

register(
    ProductType(
        name="session_tag_rollups",
        display_name="Session Tag Rollups",
        json_key="session_tag_rollups",
        empty_message="No session tag rollups matched.",
        query_class_path="polylogue.archive_products.SessionTagRollupQuery",
        operations_method="list_session_tag_rollup_products",
        cli_command_name="tags",
        cli_help="List durable session-tag rollup products.",
        cli_options=[
            CliOption("query", ("--query",), help="Substring match against the tag name"),
        ],
        mcp_default_limit=100,
        fields=[
            ProductField("", _attr("tag"), group=0),
            ProductField("conversations", _attr("conversation_count", "0"), group=0),
            ProductField("explicit", _attr("explicit_count", "0"), group=0),
            ProductField("auto", _attr("auto_count", "0"), group=0),
        ],
    )
)

register(
    ProductType(
        name="day_session_summaries",
        display_name="Day Session Summaries",
        json_key="day_session_summaries",
        empty_message="No day summaries matched.",
        query_class_path="polylogue.archive_products.DaySessionSummaryProductQuery",
        operations_method="list_day_session_summary_products",
        cli_command_name="day-summaries",
        cli_help="List durable day-level session summary products.",
        cli_options=[],
        mcp_default_limit=90,
        fields=[
            ProductField("", _attr("date"), group=0),
            ProductField(
                "sessions", lambda i: str(i.summary.get("session_count", 0)) if hasattr(i, "summary") else "0", group=0
            ),
            ProductField(
                "messages", lambda i: str(i.summary.get("total_messages", 0)) if hasattr(i, "summary") else "0", group=0
            ),
        ],
    )
)

register(
    ProductType(
        name="week_session_summaries",
        display_name="Week Session Summaries",
        json_key="week_session_summaries",
        empty_message="No week summaries matched.",
        query_class_path="polylogue.archive_products.WeekSessionSummaryProductQuery",
        operations_method="list_week_session_summary_products",
        cli_command_name="week-summaries",
        cli_help="List durable week-level session summary products.",
        cli_options=[],
        mcp_default_limit=52,
        fields=[
            ProductField("", _attr("iso_week"), group=0),
            ProductField(
                "sessions", lambda i: str(i.summary.get("session_count", 0)) if hasattr(i, "summary") else "0", group=0
            ),
            ProductField(
                "messages", lambda i: str(i.summary.get("total_messages", 0)) if hasattr(i, "summary") else "0", group=0
            ),
        ],
    )
)

register(
    ProductType(
        name="provider_analytics",
        display_name="Provider Analytics",
        json_key="provider_analytics",
        empty_message="No provider analytics matched.",
        query_class_path="polylogue.archive_products.ProviderAnalyticsProductQuery",
        operations_method="list_provider_analytics_products",
        cli_command_name="analytics",
        cli_help="List provider-level analytics products.",
        cli_options=[],
        fields=[
            ProductField("", _attr("provider_name"), group=0),
            ProductField("conversations", _attr("conversation_count", "0"), group=0),
            ProductField("messages", _attr("message_count", "0"), group=0),
            ProductField(
                "avg_messages",
                lambda i: (
                    f"{i.avg_messages_per_conversation:.1f}" if hasattr(i, "avg_messages_per_conversation") else "-"
                ),
                group=0,
            ),
            ProductField(
                "tools",
                lambda i: f"{i.tool_use_count} ({i.tool_use_percentage:.1f}%)" if hasattr(i, "tool_use_count") else "-",
                group=1,
            ),
            ProductField(
                "thinking",
                lambda i: f"{i.thinking_count} ({i.thinking_percentage:.1f}%)" if hasattr(i, "thinking_count") else "-",
                group=1,
            ),
        ],
    )
)


# -------------------------------------------------------------------
# Generic data fetching
# -------------------------------------------------------------------


def _resolve_query_class(dotted_path: str) -> type[ArchiveProductModel]:
    """Import and return a class from a dotted path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib

    mod = importlib.import_module(module_path)
    resolved = getattr(mod, class_name)
    if not isinstance(resolved, type) or not issubclass(resolved, BaseModel):
        raise TypeError(f"{dotted_path} is not a Pydantic model class")
    return cast(type[ArchiveProductModel], resolved)


class ProductQueryError(ValueError):
    """Raised when a registry-backed product query is invalid."""


def _build_query(
    product_type: ProductType,
    **kwargs: Any,
) -> Any:
    """Build and validate the typed query object for a product fetch."""
    query_cls = _resolve_query_class(product_type.query_class_path)
    accepted = set(query_cls.model_fields)
    unknown = sorted(set(kwargs) - accepted)
    if unknown:
        unknown_list = ", ".join(unknown)
        accepted_list = ", ".join(sorted(accepted))
        raise ProductQueryError(
            f"Unknown query field(s) for {product_type.name}: {unknown_list}. Accepted fields: {accepted_list}"
        )
    return query_cls(**kwargs)


def fetch_products(
    product_type: ProductType,
    operations: object,
    **kwargs: Any,
) -> list[Any]:
    """Fetch product items using the registry dispatch metadata.

    Constructs the query object from ``product_type.query_class_path``
    using **kwargs, then calls the async operations method synchronously.
    """
    from polylogue.sync_bridge import run_coroutine_sync

    query = _build_query(product_type, **kwargs)
    method = getattr(operations, product_type.operations_method)
    return cast(list[Any], run_coroutine_sync(method(query)))


async def fetch_products_async(
    product_type: ProductType,
    operations: object,
    **kwargs: Any,
) -> list[Any]:
    """Async variant of ``fetch_products()``."""
    query = _build_query(product_type, **kwargs)
    method = getattr(operations, product_type.operations_method)
    return cast(list[Any], await method(query))


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
    "register",
    "render_product_items",
]
