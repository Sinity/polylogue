"""Read, projection, and output parameter descriptors for archive surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

ReadFieldKind: TypeAlias = Literal[
    "aggregation",
    "format",
    "identity",
    "mutation",
    "pagination",
    "projection",
    "selection_display",
]
ReadSurface: TypeAlias = Literal["api", "cli", "mcp", "storage"]


@dataclass(frozen=True, slots=True)
class ReadFieldDescriptor:
    """Semantic ownership for non-filter read/output controls."""

    name: str
    kind: ReadFieldKind
    description: str
    cli_names: tuple[str, ...] = ()
    mcp_names: tuple[str, ...] = ()
    api_names: tuple[str, ...] = ()
    storage_names: tuple[str, ...] = ()

    def names_for_surface(self, surface: ReadSurface) -> tuple[str, ...]:
        return {
            "api": self.api_names,
            "cli": self.cli_names,
            "mcp": self.mcp_names,
            "storage": self.storage_names,
        }[surface]


READ_FIELD_DESCRIPTORS: tuple[ReadFieldDescriptor, ...] = (
    ReadFieldDescriptor(
        name="conversation_id",
        kind="identity",
        description="Conversation/session identifier used by point-read surfaces.",
        cli_names=("conversation_id", "target_terms"),
        mcp_names=("id", "conversation_id"),
        api_names=("conversation_id", "conversation_ids", "session_id"),
        storage_names=("conversation_id",),
    ),
    ReadFieldDescriptor(
        name="message_role",
        kind="projection",
        description="Message role filter for paginated message reads.",
        cli_names=("message_role",),
        mcp_names=("message_role",),
        api_names=("message_role",),
        storage_names=("message_role",),
    ),
    ReadFieldDescriptor(
        name="message_type",
        kind="projection",
        description="Message content-type filter for paginated message reads.",
        cli_names=("message_type",),
        mcp_names=("message_type",),
        api_names=("message_type",),
        storage_names=("message_type",),
    ),
    ReadFieldDescriptor(
        name="limit",
        kind="pagination",
        description="Maximum rows/items returned by read and output surfaces.",
        cli_names=("limit",),
        mcp_names=("limit",),
        api_names=("limit", "related_limit"),
        storage_names=("limit",),
    ),
    ReadFieldDescriptor(
        name="offset",
        kind="pagination",
        description="Zero-based pagination offset for read surfaces.",
        cli_names=("offset",),
        mcp_names=("offset",),
        api_names=("offset",),
        storage_names=("offset",),
    ),
    ReadFieldDescriptor(
        name="output_format",
        kind="format",
        description="Human or machine output encoding for CLI surfaces.",
        cli_names=("output_format",),
    ),
    ReadFieldDescriptor(
        name="fields",
        kind="selection_display",
        description="Selected display/export fields for structured CLI outputs.",
        cli_names=("fields",),
    ),
    ReadFieldDescriptor(
        name="selector_kind",
        kind="selection_display",
        description="Selector target kind for query-backed fuzzy selection.",
        cli_names=("selector_kind",),
    ),
    ReadFieldDescriptor(
        name="select_print_field",
        kind="format",
        description="Field printed after selecting a conversation.",
        cli_names=("print_field",),
    ),
    ReadFieldDescriptor(
        name="stats_by",
        kind="aggregation",
        description="Stats aggregation dimension.",
        cli_names=("stats_by",),
        mcp_names=("group_by",),
        api_names=("group_by",),
    ),
    ReadFieldDescriptor(
        name="print_path",
        kind="format",
        description="Open command mode that prints the render path instead of launching it.",
        cli_names=("print_path",),
    ),
    ReadFieldDescriptor(
        name="delete_preview",
        kind="mutation",
        description="Delete command preview and confirmation controls.",
        cli_names=("dry_run", "force"),
    ),
    ReadFieldDescriptor(
        name="content_projection",
        kind="projection",
        description="Content-kind projection controls shared by CLI, MCP, and API reads.",
        cli_names=(
            "no_code_blocks",
            "no_tool_calls",
            "no_tool_outputs",
            "no_file_reads",
            "prose_only",
        ),
        mcp_names=(
            "no_code_blocks",
            "no_tool_calls",
            "no_tool_outputs",
            "no_file_reads",
            "prose_only",
        ),
        api_names=("content_projection",),
    ),
)


def read_field_descriptor_map() -> dict[str, ReadFieldDescriptor]:
    return {descriptor.name: descriptor for descriptor in READ_FIELD_DESCRIPTORS}


def read_field_names_for_surface(surface: ReadSurface) -> frozenset[str]:
    names: set[str] = set()
    for descriptor in READ_FIELD_DESCRIPTORS:
        names.update(descriptor.names_for_surface(surface))
    return frozenset(names)


__all__ = [
    "READ_FIELD_DESCRIPTORS",
    "ReadFieldDescriptor",
    "ReadFieldKind",
    "ReadSurface",
    "read_field_descriptor_map",
    "read_field_names_for_surface",
]
