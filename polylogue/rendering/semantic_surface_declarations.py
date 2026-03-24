"""Declared semantic-proof surface catalog and alias maps."""

from __future__ import annotations

from typing import Any

from polylogue.rendering.semantic_surface_models import SemanticMetricContract, SemanticSurfaceSpec


def _preserve(
    metric: str,
    policy: str,
    input_key: str,
    output_key: str | None = None,
    *,
    input_transform: str = "identity",
    output_transform: str = "identity",
) -> SemanticMetricContract:
    return SemanticMetricContract(
        metric=metric,
        mode="preserve",
        policy=policy,
        input_key=input_key,
        output_key=output_key or input_key,
        input_transform=input_transform,
        output_transform=output_transform,
    )


def _declared_loss(
    metric: str,
    policy: str,
    input_key: str,
    output_key: str | None = None,
    *,
    input_transform: str = "identity",
    output_transform: str = "identity",
    default_output: Any = 0,
) -> SemanticMetricContract:
    return SemanticMetricContract(
        metric=metric,
        mode="declared_loss",
        policy=policy,
        input_key=input_key,
        output_key=output_key,
        input_transform=input_transform,
        output_transform=output_transform,
        default_output=default_output,
    )


def _presence(
    metric: str,
    policy: str,
    input_key: str,
    output_key: str,
    *,
    input_transform: str = "presence_bool",
    output_transform: str = "identity",
) -> SemanticMetricContract:
    return SemanticMetricContract(
        metric=metric,
        mode="presence",
        policy=policy,
        input_key=input_key,
        output_key=output_key,
        input_transform=input_transform,
        output_transform=output_transform,
    )


_CANONICAL_MARKDOWN_CONTRACTS = (
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

_EXPORT_JSON_LIKE_CONTRACTS = (
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

_EXPORT_CSV_CONTRACTS = (
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

_EXPORT_MARKDOWN_BASE_CONTRACTS = (
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

_EXPORT_MARKDOWN_CONTRACTS = _EXPORT_MARKDOWN_BASE_CONTRACTS
_EXPORT_OBSIDIAN_CONTRACTS = (
    _preserve(
        "conversation_id",
        "export_obsidian_v1 must preserve conversation identity in frontmatter",
        "conversation_id",
        "conversation_id",
    ),
    *_EXPORT_MARKDOWN_BASE_CONTRACTS,
)
_EXPORT_ORG_CONTRACTS = _EXPORT_MARKDOWN_BASE_CONTRACTS

_EXPORT_HTML_CONTRACTS = (
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

_QUERY_SUMMARY_JSON_LIKE_CONTRACTS = (
    _preserve("conversation_id", "surface must preserve the conversation identifier", "conversation_id", "conversation_id"),
    _preserve("provider_identity", "surface must preserve provider identity", "provider", "provider"),
    _preserve("title_metadata", "surface must preserve the display title", "title", "title"),
    _preserve("date_metadata", "surface must preserve the summary display date", "date", "date"),
    _preserve("message_count", "surface must preserve summary message counts", "messages", "messages"),
    _preserve("tag_values", "surface must preserve summary tag values", "tags", "tags"),
    _preserve("summary_text", "surface must preserve summary text", "summary", "summary"),
)

_QUERY_SUMMARY_CSV_CONTRACTS = (
    _preserve("conversation_id", "query_summary_csv_v1 must preserve the conversation identifier", "conversation_id", "conversation_id"),
    _preserve("provider_identity", "query_summary_csv_v1 must preserve provider identity", "provider", "provider"),
    _preserve("title_metadata", "query_summary_csv_v1 must preserve the display title", "title", "title"),
    _preserve(
        "date_metadata",
        "query_summary_csv_v1 must preserve the summary display date",
        "date",
        "date",
        input_transform="date_prefix10",
    ),
    _preserve("message_count", "query_summary_csv_v1 must preserve summary message counts", "messages", "messages"),
    _preserve("tag_values", "query_summary_csv_v1 must preserve summary tag values", "tags", "tags"),
    _preserve("summary_text", "query_summary_csv_v1 must preserve summary text", "summary", "summary"),
)

_QUERY_SUMMARY_TEXT_CONTRACTS = (
    _preserve(
        "conversation_id_prefix",
        "query_summary_text_v1 must preserve the visible conversation id prefix",
        "conversation_id",
        "conversation_id_prefix",
        input_transform="id_prefix24",
    ),
    _preserve("provider_identity", "query_summary_text_v1 must preserve provider identity", "provider", "provider"),
    _preserve(
        "date_metadata",
        "query_summary_text_v1 must preserve the visible summary date",
        "date",
        "date",
        input_transform="date_prefix10",
    ),
    _preserve(
        "title_projection",
        "query_summary_text_v1 must preserve the deterministic visible title projection",
        "title",
        "title",
        input_transform="summary_title_projection",
    ),
    _preserve("message_count", "query_summary_text_v1 must preserve summary message counts", "messages", "messages"),
    _declared_loss(
        "tag_values",
        "query_summary_text_v1 intentionally omits tag values",
        "tags",
        input_transform="len",
    ),
    _declared_loss(
        "summary_text",
        "query_summary_text_v1 intentionally omits summary text",
        "summary",
        input_transform="presence_count",
    ),
)

_QUERY_STREAM_PLAINTEXT_CONTRACTS = (
    _preserve(
        "text_messages",
        "query_stream_plaintext_v1 must preserve one visible block per text-bearing message",
        "text_messages",
        "message_sections",
    ),
    _preserve(
        "role_sections",
        "query_stream_plaintext_v1 must preserve visible role labels for streamed messages",
        "text_role_counts",
        "role_counts",
    ),
    _declared_loss(
        "title_metadata",
        "query_stream_plaintext_v1 intentionally omits title metadata",
        "title",
        input_transform="presence_count",
    ),
    _declared_loss(
        "provider_identity",
        "query_stream_plaintext_v1 intentionally omits provider metadata",
        "provider",
        input_transform="presence_count",
    ),
    _declared_loss(
        "date_metadata",
        "query_stream_plaintext_v1 intentionally omits date metadata",
        "date",
        input_transform="presence_count",
    ),
    _declared_loss(
        "timestamp_values",
        "query_stream_plaintext_v1 intentionally omits per-message timestamps",
        "timestamped_text_messages",
    ),
    _declared_loss(
        "attachment_semantics",
        "query_stream_plaintext_v1 intentionally omits attachment semantics",
        "attachment_count",
    ),
    _declared_loss(
        "thinking_semantics",
        "query_stream_plaintext_v1 preserves display text but not typed thinking markers",
        "thinking_messages",
    ),
    _declared_loss(
        "tool_semantics",
        "query_stream_plaintext_v1 preserves display text but not typed tool markers",
        "tool_messages",
    ),
    _declared_loss(
        "branch_structure",
        "query_stream_plaintext_v1 intentionally omits explicit branch topology",
        "branch_messages",
    ),
)

_QUERY_STREAM_MARKDOWN_CONTRACTS = (
    _preserve(
        "title_metadata",
        "query_stream_markdown_v1 must preserve the conversation title",
        "title",
        "title",
    ),
    _preserve(
        "provider_identity",
        "query_stream_markdown_v1 must preserve provider identity in the stream header",
        "provider",
        "provider",
    ),
    _presence(
        "date_metadata",
        "query_stream_markdown_v1 must preserve conversation date presence in the stream header",
        "date",
        "has_date",
        output_transform="bool",
    ),
    _preserve(
        "text_messages",
        "query_stream_markdown_v1 must preserve one visible section per text-bearing message",
        "text_messages",
        "message_sections",
    ),
    _preserve(
        "role_sections",
        "query_stream_markdown_v1 must preserve visible role headings for streamed messages",
        "text_role_counts",
        "role_counts",
    ),
    _preserve(
        "footer_count",
        "query_stream_markdown_v1 must report the number of emitted messages honestly",
        "text_messages",
        "footer_count",
    ),
    _declared_loss(
        "timestamp_values",
        "query_stream_markdown_v1 intentionally omits per-message timestamps",
        "timestamped_text_messages",
    ),
    _declared_loss(
        "attachment_semantics",
        "query_stream_markdown_v1 intentionally omits attachment semantics",
        "attachment_count",
    ),
    _declared_loss(
        "thinking_semantics",
        "query_stream_markdown_v1 preserves display text but not typed thinking markers",
        "thinking_messages",
    ),
    _declared_loss(
        "tool_semantics",
        "query_stream_markdown_v1 preserves display text but not typed tool markers",
        "tool_messages",
    ),
    _declared_loss(
        "branch_structure",
        "query_stream_markdown_v1 intentionally omits explicit branch topology",
        "branch_messages",
    ),
)

_QUERY_STREAM_JSON_LINES_CONTRACTS = (
    _preserve(
        "conversation_id",
        "query_stream_json_lines_v1 must preserve the conversation identifier in the header",
        "conversation_id",
        "conversation_id",
    ),
    _preserve(
        "title_metadata",
        "query_stream_json_lines_v1 must preserve the title in the header",
        "title",
        "title",
    ),
    _preserve(
        "provider_identity",
        "query_stream_json_lines_v1 must preserve provider identity in the header",
        "provider",
        "provider",
    ),
    _preserve(
        "date_metadata",
        "query_stream_json_lines_v1 must preserve the conversation date in the header",
        "date",
        "date",
    ),
    _preserve(
        "text_message_ids",
        "query_stream_json_lines_v1 must preserve identifiers for emitted messages",
        "text_message_ids",
        "message_ids",
    ),
    _preserve(
        "role_sections",
        "query_stream_json_lines_v1 must preserve role distribution for emitted messages",
        "text_role_counts",
        "role_counts",
    ),
    _preserve(
        "timestamp_values",
        "query_stream_json_lines_v1 must preserve timestamps for emitted messages",
        "timestamped_text_messages",
        "timestamped_messages",
    ),
    _preserve(
        "footer_count",
        "query_stream_json_lines_v1 must report the number of emitted messages honestly",
        "text_messages",
        "footer_count",
    ),
    _declared_loss(
        "attachment_semantics",
        "query_stream_json_lines_v1 intentionally omits attachment semantics",
        "attachment_count",
    ),
    _declared_loss(
        "thinking_semantics",
        "query_stream_json_lines_v1 preserves display text but not typed thinking markers",
        "thinking_messages",
    ),
    _declared_loss(
        "tool_semantics",
        "query_stream_json_lines_v1 preserves display text but not typed tool markers",
        "tool_messages",
    ),
    _declared_loss(
        "branch_structure",
        "query_stream_json_lines_v1 intentionally omits explicit branch topology",
        "branch_messages",
    ),
)

_MCP_SUMMARY_CONTRACTS = (
    _preserve("conversation_id", "mcp_summary_json_v1 must preserve the conversation identifier", "conversation_id", "conversation_id"),
    _preserve("provider_identity", "mcp_summary_json_v1 must preserve provider identity", "provider", "provider"),
    _preserve("title_metadata", "mcp_summary_json_v1 must preserve the display title", "title", "title"),
    _preserve("message_count", "mcp_summary_json_v1 must preserve summary message counts", "messages", "messages"),
    _preserve("created_at", "mcp_summary_json_v1 must preserve the created_at timestamp when present", "created_at", "created_at"),
    _preserve("updated_at", "mcp_summary_json_v1 must preserve the updated_at timestamp when present", "updated_at", "updated_at"),
    _declared_loss(
        "tag_values",
        "mcp_summary_json_v1 intentionally omits tags",
        "tags",
        input_transform="len",
    ),
    _declared_loss(
        "summary_text",
        "mcp_summary_json_v1 intentionally omits summary text",
        "summary",
        input_transform="presence_count",
    ),
)

_MCP_DETAIL_CONTRACTS = (
    _preserve("conversation_id", "mcp_detail_json_v1 must preserve the conversation identifier", "conversation_id", "conversation_id"),
    _preserve("provider_identity", "mcp_detail_json_v1 must preserve provider identity", "provider", "provider"),
    _preserve("title_metadata", "mcp_detail_json_v1 must preserve the display title", "title", "title"),
    _preserve("created_at", "mcp_detail_json_v1 must preserve the created_at timestamp when present", "created_at", "created_at"),
    _preserve("updated_at", "mcp_detail_json_v1 must preserve the updated_at timestamp when present", "updated_at", "updated_at"),
    _preserve("message_entries", "mcp_detail_json_v1 must preserve every message entry", "messages", "messages"),
    _preserve("message_ids", "mcp_detail_json_v1 must preserve message identifiers", "message_ids", "message_ids"),
    _preserve("role_entries", "mcp_detail_json_v1 must preserve message role distribution", "role_counts", "role_counts"),
    _preserve("timestamp_values", "mcp_detail_json_v1 must preserve message timestamps", "timestamped_messages", "timestamped_messages"),
    _declared_loss(
        "attachment_semantics",
        "mcp_detail_json_v1 intentionally omits attachment payload semantics",
        "attachment_count",
    ),
    _declared_loss(
        "thinking_semantics",
        "mcp_detail_json_v1 preserves display text but not typed thinking markers",
        "thinking_messages",
    ),
    _declared_loss(
        "tool_semantics",
        "mcp_detail_json_v1 preserves display text but not typed tool markers",
        "tool_messages",
    ),
    _declared_loss(
        "branch_structure",
        "mcp_detail_json_v1 intentionally omits explicit branch topology",
        "branch_messages",
    ),
)

SEMANTIC_SURFACE_SPECS: tuple[SemanticSurfaceSpec, ...] = (
    SemanticSurfaceSpec(
        "canonical_markdown_v1",
        "canonical",
        aliases=("canonical", "canonical_markdown"),
        contracts=_CANONICAL_MARKDOWN_CONTRACTS,
    ),
    SemanticSurfaceSpec("export_json_v1", "export", aliases=("json",), export_format="json", contracts=_EXPORT_JSON_LIKE_CONTRACTS),
    SemanticSurfaceSpec("export_yaml_v1", "export", aliases=("yaml",), export_format="yaml", contracts=_EXPORT_JSON_LIKE_CONTRACTS),
    SemanticSurfaceSpec("export_csv_v1", "export", aliases=("csv",), export_format="csv", contracts=_EXPORT_CSV_CONTRACTS),
    SemanticSurfaceSpec("export_markdown_v1", "export", aliases=("markdown",), export_format="markdown", contracts=_EXPORT_MARKDOWN_CONTRACTS),
    SemanticSurfaceSpec("export_html_v1", "export", aliases=("html",), export_format="html", contracts=_EXPORT_HTML_CONTRACTS),
    SemanticSurfaceSpec("export_obsidian_v1", "export", aliases=("obsidian",), export_format="obsidian", contracts=_EXPORT_OBSIDIAN_CONTRACTS),
    SemanticSurfaceSpec("export_org_v1", "export", aliases=("org",), export_format="org", contracts=_EXPORT_ORG_CONTRACTS),
    SemanticSurfaceSpec("query_summary_json_v1", "query_summary", aliases=("query_summary_json",), contracts=_QUERY_SUMMARY_JSON_LIKE_CONTRACTS),
    SemanticSurfaceSpec("query_summary_yaml_v1", "query_summary", aliases=("query_summary_yaml",), contracts=_QUERY_SUMMARY_JSON_LIKE_CONTRACTS),
    SemanticSurfaceSpec("query_summary_csv_v1", "query_summary", aliases=("query_summary_csv",), contracts=_QUERY_SUMMARY_CSV_CONTRACTS),
    SemanticSurfaceSpec("query_summary_text_v1", "query_summary", aliases=("query_summary_text",), contracts=_QUERY_SUMMARY_TEXT_CONTRACTS),
    SemanticSurfaceSpec(
        "query_stream_plaintext_v1",
        "query_stream",
        aliases=("stream_plaintext",),
        stream_format="plaintext",
        contracts=_QUERY_STREAM_PLAINTEXT_CONTRACTS,
    ),
    SemanticSurfaceSpec(
        "query_stream_markdown_v1",
        "query_stream",
        aliases=("stream_markdown",),
        stream_format="markdown",
        contracts=_QUERY_STREAM_MARKDOWN_CONTRACTS,
    ),
    SemanticSurfaceSpec(
        "query_stream_json_lines_v1",
        "query_stream",
        aliases=("stream_json_lines",),
        stream_format="json-lines",
        contracts=_QUERY_STREAM_JSON_LINES_CONTRACTS,
    ),
    SemanticSurfaceSpec("mcp_summary_json_v1", "mcp", aliases=("mcp_summary",), contracts=_MCP_SUMMARY_CONTRACTS),
    SemanticSurfaceSpec("mcp_detail_json_v1", "mcp", aliases=("mcp_detail",), contracts=_MCP_DETAIL_CONTRACTS),
)

DEFAULT_SEMANTIC_SURFACES: tuple[str, ...] = tuple(spec.name for spec in SEMANTIC_SURFACE_SPECS)

_SPECS_BY_NAME = {spec.name: spec for spec in SEMANTIC_SURFACE_SPECS}

SURFACE_ALIASES: dict[str, tuple[str, ...]] = {
    "all": DEFAULT_SEMANTIC_SURFACES,
    "canonical": ("canonical_markdown_v1",),
    "canonical_markdown": ("canonical_markdown_v1",),
    "canonical_markdown_v1": ("canonical_markdown_v1",),
    "query_summary_all": tuple(
        spec.name for spec in SEMANTIC_SURFACE_SPECS if spec.category == "query_summary"
    ),
    "stream_all": tuple(
        spec.name for spec in SEMANTIC_SURFACE_SPECS if spec.category == "query_stream"
    ),
    "query_all": tuple(
        spec.name for spec in SEMANTIC_SURFACE_SPECS
        if spec.category in {"query_summary", "query_stream"}
    ),
    "mcp_all": tuple(spec.name for spec in SEMANTIC_SURFACE_SPECS if spec.category == "mcp"),
    "read_all": tuple(
        spec.name for spec in SEMANTIC_SURFACE_SPECS
        if spec.category in {"query_summary", "query_stream", "mcp"}
    ),
    "export_all": tuple(spec.name for spec in SEMANTIC_SURFACE_SPECS if spec.category == "export"),
}

for spec in SEMANTIC_SURFACE_SPECS:
    SURFACE_ALIASES[spec.name] = (spec.name,)
    for alias in spec.aliases:
        SURFACE_ALIASES[alias] = (spec.name,)

EXPORT_SURFACE_FORMATS: dict[str, str] = {
    spec.name: spec.export_format
    for spec in SEMANTIC_SURFACE_SPECS
    if spec.export_format is not None
}

STREAM_SURFACE_FORMATS: dict[str, str] = {
    spec.name: spec.stream_format
    for spec in SEMANTIC_SURFACE_SPECS
    if spec.stream_format is not None
}


__all__ = [
    "DEFAULT_SEMANTIC_SURFACES",
    "EXPORT_SURFACE_FORMATS",
    "SEMANTIC_SURFACE_SPECS",
    "STREAM_SURFACE_FORMATS",
    "SURFACE_ALIASES",
    "SemanticMetricContract",
    "SemanticSurfaceSpec",
]
