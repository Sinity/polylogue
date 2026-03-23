"""Query-summary and query-stream semantic proof declarations."""

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

QUERY_SUMMARY_JSON_LIKE_CONTRACTS = (
    _preserve("conversation_id", "surface must preserve the conversation identifier", "conversation_id", "conversation_id"),
    _preserve("provider_identity", "surface must preserve provider identity", "provider", "provider"),
    _preserve("title_metadata", "surface must preserve the display title", "title", "title"),
    _preserve("date_metadata", "surface must preserve the summary display date", "date", "date"),
    _preserve("message_count", "surface must preserve summary message counts", "messages", "messages"),
    _preserve("tag_values", "surface must preserve summary tag values", "tags", "tags"),
    _preserve("summary_text", "surface must preserve summary text", "summary", "summary"),
)

QUERY_SUMMARY_CSV_CONTRACTS = (
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

QUERY_SUMMARY_TEXT_CONTRACTS = (
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

QUERY_STREAM_PLAINTEXT_CONTRACTS = (
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

QUERY_STREAM_MARKDOWN_CONTRACTS = (
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

QUERY_STREAM_JSON_LINES_CONTRACTS = (
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

QUERY_SEMANTIC_SURFACE_SPECS: tuple[SemanticSurfaceSpec, ...] = (
    SemanticSurfaceSpec("query_summary_json_v1", "query_summary", aliases=("query_summary_json",), contracts=QUERY_SUMMARY_JSON_LIKE_CONTRACTS),
    SemanticSurfaceSpec("query_summary_yaml_v1", "query_summary", aliases=("query_summary_yaml",), contracts=QUERY_SUMMARY_JSON_LIKE_CONTRACTS),
    SemanticSurfaceSpec("query_summary_csv_v1", "query_summary", aliases=("query_summary_csv",), contracts=QUERY_SUMMARY_CSV_CONTRACTS),
    SemanticSurfaceSpec("query_summary_text_v1", "query_summary", aliases=("query_summary_text",), contracts=QUERY_SUMMARY_TEXT_CONTRACTS),
    SemanticSurfaceSpec(
        "query_stream_plaintext_v1",
        "query_stream",
        aliases=("stream_plaintext",),
        stream_format="plaintext",
        contracts=QUERY_STREAM_PLAINTEXT_CONTRACTS,
    ),
    SemanticSurfaceSpec(
        "query_stream_markdown_v1",
        "query_stream",
        aliases=("stream_markdown",),
        stream_format="markdown",
        contracts=QUERY_STREAM_MARKDOWN_CONTRACTS,
    ),
    SemanticSurfaceSpec(
        "query_stream_json_lines_v1",
        "query_stream",
        aliases=("stream_json_lines",),
        stream_format="json-lines",
        contracts=QUERY_STREAM_JSON_LINES_CONTRACTS,
    ),
)

__all__ = ["QUERY_SEMANTIC_SURFACE_SPECS"]
