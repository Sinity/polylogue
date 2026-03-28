"""Query-stream semantic proof contracts."""

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


__all__ = [
    "QUERY_STREAM_JSON_LINES_CONTRACTS",
    "QUERY_STREAM_MARKDOWN_CONTRACTS",
    "QUERY_STREAM_PLAINTEXT_CONTRACTS",
]
