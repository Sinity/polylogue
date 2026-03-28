"""Query-summary semantic proof contracts."""

from __future__ import annotations

from polylogue.rendering.semantic_surface_contract_helpers import (
    declared_loss_contract as _declared_loss,
)
from polylogue.rendering.semantic_surface_contract_helpers import (
    preserve_contract as _preserve,
)

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


__all__ = [
    "QUERY_SUMMARY_CSV_CONTRACTS",
    "QUERY_SUMMARY_JSON_LIKE_CONTRACTS",
    "QUERY_SUMMARY_TEXT_CONTRACTS",
]
