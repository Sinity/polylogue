"""Document-style export-surface semantic proof contracts."""

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


__all__ = [
    "EXPORT_HTML_CONTRACTS",
    "EXPORT_MARKDOWN_BASE_CONTRACTS",
    "EXPORT_MARKDOWN_CONTRACTS",
    "EXPORT_OBSIDIAN_CONTRACTS",
    "EXPORT_ORG_CONTRACTS",
]
