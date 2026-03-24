"""Canonical and export-surface semantic proof declarations."""

from __future__ import annotations

from polylogue.rendering.semantic_surface_contract_helpers import (
    declared_loss_contract as _declared_loss,
)
from polylogue.rendering.semantic_surface_contract_helpers import (
    presence_contract as _presence,
)
from polylogue.rendering.semantic_surface_contract_helpers import (
    preserve_contract as _preserve,
)
from polylogue.rendering.semantic_surface_models import SemanticSurfaceSpec

CANONICAL_MARKDOWN_CONTRACTS = (
    _preserve(
        "renderable_messages",
        "canonical markdown must preserve every renderable message section",
        "renderable_messages",
        "message_sections",
    ),
    _preserve(
        "attachment_lines",
        "canonical markdown must preserve attachment presence as attachment lines",
        "attachment_count",
        "attachment_lines",
    ),
    _preserve(
        "timestamp_lines",
        "canonical markdown must preserve timestamps for renderable messages that have them",
        "timestamped_renderable_messages",
        "timestamp_lines",
    ),
    _preserve(
        "role_sections",
        "canonical markdown must preserve renderable message role sections",
        "renderable_role_counts",
        "role_section_counts",
    ),
    _declared_loss(
        "empty_messages",
        "canonical markdown intentionally omits messages with no text and no attachments",
        "empty_messages",
    ),
    _declared_loss(
        "thinking_semantics",
        "canonical markdown preserves display text but not typed thinking markers",
        "thinking_messages",
        "typed_thinking_markers",
    ),
    _declared_loss(
        "tool_semantics",
        "canonical markdown preserves display text but not typed tool markers",
        "tool_messages",
        "typed_tool_markers",
    ),
)

EXPORT_JSON_LIKE_CONTRACTS = (
    _preserve("conversation_id", "surface must preserve the conversation identifier", "conversation_id"),
    _preserve("provider_identity", "surface must preserve provider identity", "provider", "provider"),
    _preserve("title_metadata", "surface must preserve the display title", "title", "title"),
    _preserve("date_metadata", "surface must preserve the display date value when present", "date", "date"),
    _preserve("message_entries", "surface must preserve every message entry", "total_messages", "messages"),
    _preserve("message_ids", "surface must preserve message identifiers", "message_ids", "message_ids"),
    _preserve("role_entries", "surface must preserve message role distribution", "text_role_counts", "role_counts"),
    _preserve(
        "timestamp_values",
        "surface must preserve message timestamps",
        "timestamped_text_messages",
        "timestamped_messages",
    ),
    _declared_loss(
        "attachment_semantics",
        "surface intentionally omits attachment payload semantics",
        "attachment_count",
    ),
    _declared_loss(
        "thinking_semantics",
        "surface preserves display text but not typed thinking markers",
        "thinking_messages",
    ),
    _declared_loss(
        "tool_semantics",
        "surface preserves display text but not typed tool markers",
        "tool_messages",
    ),
    _declared_loss(
        "branch_structure",
        "surface intentionally omits explicit branch topology",
        "branch_messages",
    ),
)

EXPORT_CSV_CONTRACTS = (
    _preserve(
        "conversation_id",
        "export_csv_v1 must preserve the conversation identifier per row",
        "conversation_id",
        "conversation_id",
    ),
    _preserve(
        "text_messages",
        "export_csv_v1 must preserve one row per text-bearing message",
        "text_messages",
        "messages",
    ),
    _preserve(
        "text_message_ids",
        "export_csv_v1 must preserve identifiers for text-bearing messages",
        "text_message_ids",
        "message_ids",
    ),
    _preserve(
        "role_entries",
        "export_csv_v1 must preserve roles for text-bearing messages",
        "text_role_counts",
        "role_counts",
    ),
    _preserve(
        "timestamp_values",
        "export_csv_v1 must preserve timestamps for text-bearing messages",
        "timestamped_text_messages",
        "timestamped_messages",
    ),
    _declared_loss(
        "provider_identity",
        "export_csv_v1 intentionally omits conversation-level provider metadata",
        "provider",
        input_transform="presence_count",
    ),
    _declared_loss(
        "title_metadata",
        "export_csv_v1 intentionally omits conversation-level title metadata",
        "title",
        input_transform="presence_count",
    ),
    _declared_loss(
        "date_metadata",
        "export_csv_v1 intentionally omits conversation-level date metadata",
        "date",
        input_transform="presence_count",
    ),
    _declared_loss(
        "attachment_semantics",
        "export_csv_v1 intentionally omits attachment payload semantics",
        "attachment_count",
    ),
    _declared_loss(
        "thinking_semantics",
        "export_csv_v1 preserves display text but not typed thinking markers",
        "thinking_messages",
    ),
    _declared_loss(
        "tool_semantics",
        "export_csv_v1 preserves display text but not typed tool markers",
        "tool_messages",
    ),
    _declared_loss(
        "branch_structure",
        "export_csv_v1 intentionally omits explicit branch topology",
        "branch_messages",
    ),
)

EXPORT_MARKDOWN_BASE_CONTRACTS = (
    _preserve("title_metadata", "surface must preserve the display title", "title", "title"),
    _preserve("provider_identity", "surface must preserve provider identity at document level", "provider", "provider"),
    _presence(
        "date_metadata",
        "surface must preserve conversation date presence at document level",
        "date",
        "has_date",
        output_transform="bool",
    ),
    _preserve(
        "text_messages",
        "surface must preserve one section per text-bearing message",
        "text_messages",
        "message_sections",
    ),
    _preserve(
        "role_sections",
        "surface must preserve role sections for text-bearing messages",
        "text_role_counts",
        "role_counts",
    ),
    _declared_loss(
        "timestamp_values",
        "surface intentionally omits per-message timestamps",
        "timestamped_text_messages",
        "timestamp_lines",
    ),
    _declared_loss(
        "attachment_semantics",
        "surface intentionally omits attachment payload semantics",
        "attachment_count",
    ),
    _declared_loss(
        "thinking_semantics",
        "surface preserves display text but not typed thinking markers",
        "thinking_messages",
    ),
    _declared_loss(
        "tool_semantics",
        "surface preserves display text but not typed tool markers",
        "tool_messages",
    ),
    _declared_loss(
        "branch_structure",
        "surface intentionally omits explicit branch topology",
        "branch_messages",
        "branch_labels",
    ),
)

EXPORT_MARKDOWN_CONTRACTS = EXPORT_MARKDOWN_BASE_CONTRACTS
EXPORT_OBSIDIAN_CONTRACTS = (
    _preserve(
        "conversation_id",
        "export_obsidian_v1 must preserve conversation identity in frontmatter",
        "conversation_id",
        "conversation_id",
    ),
    *EXPORT_MARKDOWN_BASE_CONTRACTS,
)
EXPORT_ORG_CONTRACTS = EXPORT_MARKDOWN_BASE_CONTRACTS

EXPORT_HTML_CONTRACTS = (
    _preserve("title_metadata", "export_html_v1 must preserve the display title", "title", "title"),
    _preserve(
        "provider_identity",
        "export_html_v1 must preserve provider identity at document level",
        "provider",
        "provider",
    ),
    _presence(
        "date_metadata",
        "export_html_v1 must preserve conversation date presence at document level",
        "date",
        "has_date",
        output_transform="bool",
    ),
    _preserve(
        "text_messages",
        "export_html_v1 must preserve visible message sections for text-bearing messages",
        "text_messages",
        "message_sections",
    ),
    _preserve(
        "role_sections",
        "export_html_v1 must preserve visible role labels for text-bearing messages",
        "text_role_counts",
        "role_counts",
    ),
    _preserve(
        "timestamp_values",
        "export_html_v1 must preserve visible message timestamps",
        "timestamped_text_messages",
        "timestamp_lines",
    ),
    _preserve(
        "branch_structure",
        "export_html_v1 must preserve visible branch groupings for branched messages",
        "branch_messages",
        "branch_labels",
    ),
    _declared_loss(
        "attachment_semantics",
        "export_html_v1 intentionally omits attachment payload semantics",
        "attachment_count",
    ),
    _declared_loss(
        "thinking_semantics",
        "export_html_v1 preserves display text but not typed thinking markers",
        "thinking_messages",
    ),
    _declared_loss(
        "tool_semantics",
        "export_html_v1 preserves display text but not typed tool markers",
        "tool_messages",
    ),
)

CANONICAL_SEMANTIC_SURFACE_SPECS: tuple[SemanticSurfaceSpec, ...] = (
    SemanticSurfaceSpec(
        "canonical_markdown_v1",
        "canonical",
        aliases=("canonical", "canonical_markdown"),
        contracts=CANONICAL_MARKDOWN_CONTRACTS,
    ),
)

EXPORT_SEMANTIC_SURFACE_SPECS: tuple[SemanticSurfaceSpec, ...] = (
    SemanticSurfaceSpec("export_json_v1", "export", aliases=("json",), export_format="json", contracts=EXPORT_JSON_LIKE_CONTRACTS),
    SemanticSurfaceSpec("export_yaml_v1", "export", aliases=("yaml",), export_format="yaml", contracts=EXPORT_JSON_LIKE_CONTRACTS),
    SemanticSurfaceSpec("export_csv_v1", "export", aliases=("csv",), export_format="csv", contracts=EXPORT_CSV_CONTRACTS),
    SemanticSurfaceSpec("export_markdown_v1", "export", aliases=("markdown",), export_format="markdown", contracts=EXPORT_MARKDOWN_CONTRACTS),
    SemanticSurfaceSpec("export_html_v1", "export", aliases=("html",), export_format="html", contracts=EXPORT_HTML_CONTRACTS),
    SemanticSurfaceSpec("export_obsidian_v1", "export", aliases=("obsidian",), export_format="obsidian", contracts=EXPORT_OBSIDIAN_CONTRACTS),
    SemanticSurfaceSpec("export_org_v1", "export", aliases=("org",), export_format="org", contracts=EXPORT_ORG_CONTRACTS),
)

__all__ = [
    "CANONICAL_SEMANTIC_SURFACE_SPECS",
    "EXPORT_SEMANTIC_SURFACE_SPECS",
]
