"""Product type registry — data-driven derived product descriptors.

Each product type is a ``ProductType`` descriptor that defines:
- how to display items in plain text (via ``fields``)
- what the JSON key is for API responses
- what display name to use in CLI output

The rendering is generic: ``render_product_items()`` handles JSON mode
(all types) and plain-text mode (using field descriptors). Adding a new
product type requires only a new ``ProductType`` registration — no new
rendering, workflow, command, or MCP tool files.

Product *semantics* (row builders, lifecycle, storage) remain in code;
only the *transport/presentation* layer is generic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import click

from polylogue.cli.machine_errors import emit_success


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
        emit_success({
            "count": len(items),
            product_type.json_key: [_model_payload(item) for item in items],
        })
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
# Product type registrations
# -------------------------------------------------------------------

register(ProductType(
    name="session_profiles",
    display_name="Session Profiles",
    json_key="session_profiles",
    empty_message="No session profiles matched.",
    fields=[
        ProductField("", lambda i: f"{i.conversation_id} [{i.provider_name}]", group=0),
        ProductField("tier", _attr("semantic_tier"), group=0),
        ProductField("", _attr("title", "(untitled)"), group=0),
        ProductField("session_date", _nested("evidence", "canonical_session_date"), group=1),
        ProductField("messages", _nested("evidence", "message_count", "0"), group=1),
        ProductField("engaged_min", _nested("inference", "engaged_minutes", "0"), group=1),
    ],
))

register(ProductType(
    name="session_enrichments",
    display_name="Session Enrichments",
    json_key="session_enrichments",
    empty_message="No session enrichments matched.",
    fields=[
        ProductField("", lambda i: f"{i.conversation_id} [{i.provider_name}]", group=0),
        ProductField("", _nested("enrichment", "refined_work_kind"), group=0),
        ProductField("", _attr("title", "(untitled)"), group=0),
        ProductField("support", _nested("enrichment", "support_level"), group=1),
        ProductField("family", lambda i: i.enrichment_provenance.enrichment_family if hasattr(i, "enrichment_provenance") else "-", group=1),
    ],
))

register(ProductType(
    name="session_work_events",
    display_name="Work Events",
    json_key="session_work_events",
    empty_message="No work events matched.",
    fields=[
        ProductField("", lambda i: f"{i.event_id} [{i.provider_name}]", group=0),
        ProductField("kind", _nested("inference", "kind"), group=0),
        ProductField("conv", _attr("conversation_id"), group=0),
        ProductField("start", _nested("evidence", "start_time"), group=1),
        ProductField("end", _nested("evidence", "end_time"), group=1),
        ProductField("duration_ms", _nested("evidence", "duration_ms", "0"), group=1),
    ],
))

register(ProductType(
    name="session_phases",
    display_name="Session Phases",
    json_key="session_phases",
    empty_message="No session phases matched.",
    fields=[
        ProductField("", lambda i: f"{i.phase_id} [{i.provider_name}]", group=0),
        ProductField("kind", _nested("inference", "kind"), group=0),
        ProductField("conv", _attr("conversation_id"), group=0),
        ProductField("start", _nested("evidence", "start_time"), group=1),
        ProductField("words", _nested("evidence", "word_count", "0"), group=1),
    ],
))

register(ProductType(
    name="work_threads",
    display_name="Work Threads",
    json_key="work_threads",
    empty_message="No work threads matched.",
    fields=[
        ProductField("", _attr("thread_id"), group=0),
        ProductField("project", _attr("dominant_project", "-"), group=0),
        ProductField("sessions", lambda i: str(i.thread.get("session_count", 0)) if hasattr(i, "thread") else "0", group=0),
        ProductField("messages", lambda i: str(i.thread.get("total_messages", 0)) if hasattr(i, "thread") else "0", group=1),
        ProductField("depth", lambda i: str(i.thread.get("depth", 0)) if hasattr(i, "thread") else "0", group=1),
    ],
))

register(ProductType(
    name="session_tag_rollups",
    display_name="Session Tag Rollups",
    json_key="session_tag_rollups",
    empty_message="No session tag rollups matched.",
    fields=[
        ProductField("", _attr("tag"), group=0),
        ProductField("conversations", _attr("conversation_count", "0"), group=0),
        ProductField("explicit", _attr("explicit_count", "0"), group=0),
        ProductField("auto", _attr("auto_count", "0"), group=0),
    ],
))

register(ProductType(
    name="day_session_summaries",
    display_name="Day Session Summaries",
    json_key="day_session_summaries",
    empty_message="No day summaries matched.",
    fields=[
        ProductField("", _attr("date"), group=0),
        ProductField("sessions", lambda i: str(i.summary.get("session_count", 0)) if hasattr(i, "summary") else "0", group=0),
        ProductField("messages", lambda i: str(i.summary.get("total_messages", 0)) if hasattr(i, "summary") else "0", group=0),
    ],
))

register(ProductType(
    name="week_session_summaries",
    display_name="Week Session Summaries",
    json_key="week_session_summaries",
    empty_message="No week summaries matched.",
    fields=[
        ProductField("", _attr("iso_week"), group=0),
        ProductField("sessions", lambda i: str(i.summary.get("session_count", 0)) if hasattr(i, "summary") else "0", group=0),
        ProductField("messages", lambda i: str(i.summary.get("total_messages", 0)) if hasattr(i, "summary") else "0", group=0),
    ],
))

register(ProductType(
    name="provider_analytics",
    display_name="Provider Analytics",
    json_key="provider_analytics",
    empty_message="No provider analytics matched.",
    fields=[
        ProductField("", _attr("provider_name"), group=0),
        ProductField("conversations", _attr("conversation_count", "0"), group=0),
        ProductField("messages", _attr("message_count", "0"), group=0),
        ProductField("avg_messages", lambda i: f"{i.avg_messages_per_conversation:.1f}" if hasattr(i, "avg_messages_per_conversation") else "-", group=0),
        ProductField("tools", lambda i: f"{i.tool_use_count} ({i.tool_use_percentage:.1f}%)" if hasattr(i, "tool_use_count") else "-", group=1),
        ProductField("thinking", lambda i: f"{i.thinking_count} ({i.thinking_percentage:.1f}%)" if hasattr(i, "thinking_count") else "-", group=1),
    ],
))


__all__ = [
    "PRODUCT_REGISTRY",
    "ProductField",
    "ProductType",
    "get_product_type",
    "list_product_types",
    "register",
    "render_product_items",
]
