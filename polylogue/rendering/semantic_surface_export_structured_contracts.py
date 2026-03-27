"""Structured export-surface semantic proof contracts."""

from __future__ import annotations

from polylogue.rendering.semantic_surface_contract_helpers import (
    declared_loss_contract as _declared_loss,
)
from polylogue.rendering.semantic_surface_contract_helpers import (
    preserve_contract as _preserve,
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


__all__ = [
    "EXPORT_CSV_CONTRACTS",
    "EXPORT_JSON_LIKE_CONTRACTS",
]
