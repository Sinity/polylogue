"""Surface-specific semantic-proof functions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from polylogue.rendering.semantic_proof_facts import (
    _canonical_markdown_output_facts,
    _conversation_input_facts,
    _critical_or_preserved,
    _csv_output_facts,
    _declared_loss_or_preserved,
    _html_output_facts,
    _input_facts,
    _json_like_output_facts,
    _markdown_doc_output_facts,
    _mcp_detail_input_facts,
    _mcp_detail_output_facts,
    _mcp_summary_output_facts,
    _obsidian_output_facts,
    _org_output_facts,
    _presence_check,
    _stream_input_facts,
    _stream_json_lines_output_facts,
    _stream_markdown_output_facts,
    _stream_plaintext_output_facts,
    _summary_csv_output_facts,
    _summary_input_facts,
    _summary_output_facts,
    _summary_text_output_facts,
)
from polylogue.rendering.semantic_proof_models import SemanticConversationProof

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation, ConversationSummary
    from polylogue.storage.store import ConversationRenderProjection


def prove_markdown_projection_semantics(
    projection: ConversationRenderProjection,
    markdown_text: str,
) -> SemanticConversationProof:
    """Compare a repository render projection to canonical markdown output."""
    input_facts = _input_facts(projection)
    output_facts = _canonical_markdown_output_facts(markdown_text)

    checks = [
        _critical_or_preserved(
            metric="renderable_messages",
            policy="canonical markdown must preserve every renderable message section",
            input_value=input_facts["renderable_messages"],
            output_value=output_facts["message_sections"],
        ),
        _critical_or_preserved(
            metric="attachment_lines",
            policy="canonical markdown must preserve attachment presence as attachment lines",
            input_value=input_facts["attachment_count"],
            output_value=output_facts["attachment_lines"],
        ),
        _critical_or_preserved(
            metric="timestamp_lines",
            policy="canonical markdown must preserve timestamps for renderable messages that have them",
            input_value=input_facts["timestamped_renderable_messages"],
            output_value=output_facts["timestamp_lines"],
        ),
        _critical_or_preserved(
            metric="role_sections",
            policy="canonical markdown must preserve renderable message role sections",
            input_value=input_facts["renderable_role_counts"],
            output_value=output_facts["role_section_counts"],
        ),
        _declared_loss_or_preserved(
            metric="empty_messages",
            policy="canonical markdown intentionally omits messages with no text and no attachments",
            input_value=input_facts["empty_messages"],
        ),
        _declared_loss_or_preserved(
            metric="thinking_semantics",
            policy="canonical markdown preserves display text but not typed thinking markers",
            input_value=input_facts["thinking_messages"],
            output_value=output_facts["typed_thinking_markers"],
        ),
        _declared_loss_or_preserved(
            metric="tool_semantics",
            policy="canonical markdown preserves display text but not typed tool markers",
            input_value=input_facts["tool_messages"],
            output_value=output_facts["typed_tool_markers"],
        ),
    ]

    return SemanticConversationProof(
        conversation_id=projection.conversation.conversation_id,
        provider=projection.conversation.provider_name or "unknown",
        surface="canonical_markdown_v1",
        input_facts=input_facts,
        output_facts=output_facts,
        checks=checks,
    )


def _prove_export_json_like_surface(
    *,
    conversation: Conversation,
    rendered_text: str,
    surface: str,
) -> SemanticConversationProof:
    try:
        if surface == "export_yaml_v1":
            import yaml

            parsed = yaml.safe_load(rendered_text)
        else:
            parsed = json.loads(rendered_text)
    except Exception:
        parsed = {}
    payload = parsed if isinstance(parsed, dict) else {}
    input_facts = _conversation_input_facts(conversation)
    output_facts = _json_like_output_facts(payload)
    checks = [
        _critical_or_preserved(
            metric="conversation_id",
            policy=f"{surface} must preserve the conversation identifier",
            input_value=input_facts["conversation_id"],
            output_value=output_facts["conversation_id"],
        ),
        _critical_or_preserved(
            metric="provider_identity",
            policy=f"{surface} must preserve provider identity",
            input_value=input_facts["provider"],
            output_value=output_facts["provider"],
        ),
        _critical_or_preserved(
            metric="title_metadata",
            policy=f"{surface} must preserve the display title",
            input_value=input_facts["title"],
            output_value=output_facts["title"],
        ),
        _critical_or_preserved(
            metric="date_metadata",
            policy=f"{surface} must preserve the display date value when present",
            input_value=input_facts["date"],
            output_value=output_facts["date"],
        ),
        _critical_or_preserved(
            metric="message_entries",
            policy=f"{surface} must preserve every message entry",
            input_value=input_facts["total_messages"],
            output_value=output_facts["messages"],
        ),
        _critical_or_preserved(
            metric="message_ids",
            policy=f"{surface} must preserve message identifiers",
            input_value=input_facts["message_ids"],
            output_value=output_facts["message_ids"],
        ),
        _critical_or_preserved(
            metric="role_entries",
            policy=f"{surface} must preserve message role distribution",
            input_value=input_facts["text_role_counts"],
            output_value=output_facts["role_counts"],
        ),
        _critical_or_preserved(
            metric="timestamp_values",
            policy=f"{surface} must preserve message timestamps",
            input_value=input_facts["timestamped_text_messages"],
            output_value=output_facts["timestamped_messages"],
        ),
        _declared_loss_or_preserved(
            metric="attachment_semantics",
            policy=f"{surface} intentionally omits attachment payload semantics",
            input_value=input_facts["attachment_count"],
        ),
        _declared_loss_or_preserved(
            metric="thinking_semantics",
            policy=f"{surface} preserves display text but not typed thinking markers",
            input_value=input_facts["thinking_messages"],
        ),
        _declared_loss_or_preserved(
            metric="tool_semantics",
            policy=f"{surface} preserves display text but not typed tool markers",
            input_value=input_facts["tool_messages"],
        ),
        _declared_loss_or_preserved(
            metric="branch_structure",
            policy=f"{surface} intentionally omits explicit branch topology",
            input_value=input_facts["branch_messages"],
        ),
    ]
    return SemanticConversationProof(
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        surface=surface,
        input_facts=input_facts,
        output_facts=output_facts,
        checks=checks,
    )


def _prove_export_csv_surface(
    *,
    conversation: Conversation,
    rendered_text: str,
) -> SemanticConversationProof:
    input_facts = _conversation_input_facts(conversation)
    output_facts = _csv_output_facts(rendered_text)
    checks = [
        _critical_or_preserved(
            metric="conversation_id",
            policy="export_csv_v1 must preserve the conversation identifier per row",
            input_value=input_facts["conversation_id"],
            output_value=output_facts["conversation_id"],
        ),
        _critical_or_preserved(
            metric="text_messages",
            policy="export_csv_v1 must preserve one row per text-bearing message",
            input_value=input_facts["text_messages"],
            output_value=output_facts["messages"],
        ),
        _critical_or_preserved(
            metric="text_message_ids",
            policy="export_csv_v1 must preserve identifiers for text-bearing messages",
            input_value=input_facts["text_message_ids"],
            output_value=output_facts["message_ids"],
        ),
        _critical_or_preserved(
            metric="role_entries",
            policy="export_csv_v1 must preserve roles for text-bearing messages",
            input_value=input_facts["text_role_counts"],
            output_value=output_facts["role_counts"],
        ),
        _critical_or_preserved(
            metric="timestamp_values",
            policy="export_csv_v1 must preserve timestamps for text-bearing messages",
            input_value=input_facts["timestamped_text_messages"],
            output_value=output_facts["timestamped_messages"],
        ),
        _declared_loss_or_preserved(
            metric="provider_identity",
            policy="export_csv_v1 intentionally omits conversation-level provider metadata",
            input_value=1 if input_facts["provider"] else 0,
            output_value=0,
        ),
        _declared_loss_or_preserved(
            metric="title_metadata",
            policy="export_csv_v1 intentionally omits conversation-level title metadata",
            input_value=1 if input_facts["title"] else 0,
            output_value=0,
        ),
        _declared_loss_or_preserved(
            metric="date_metadata",
            policy="export_csv_v1 intentionally omits conversation-level date metadata",
            input_value=1 if input_facts["date"] else 0,
            output_value=0,
        ),
        _declared_loss_or_preserved(
            metric="attachment_semantics",
            policy="export_csv_v1 intentionally omits attachment payload semantics",
            input_value=input_facts["attachment_count"],
        ),
        _declared_loss_or_preserved(
            metric="thinking_semantics",
            policy="export_csv_v1 preserves display text but not typed thinking markers",
            input_value=input_facts["thinking_messages"],
        ),
        _declared_loss_or_preserved(
            metric="tool_semantics",
            policy="export_csv_v1 preserves display text but not typed tool markers",
            input_value=input_facts["tool_messages"],
        ),
        _declared_loss_or_preserved(
            metric="branch_structure",
            policy="export_csv_v1 intentionally omits explicit branch topology",
            input_value=input_facts["branch_messages"],
        ),
    ]
    return SemanticConversationProof(
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        surface="export_csv_v1",
        input_facts=input_facts,
        output_facts=output_facts,
        checks=checks,
    )


def _prove_export_markdown_surface(
    *,
    conversation: Conversation,
    rendered_text: str,
) -> SemanticConversationProof:
    input_facts = _conversation_input_facts(conversation)
    output_facts = _markdown_doc_output_facts(rendered_text)
    checks = [
        _critical_or_preserved(
            metric="title_metadata",
            policy="export_markdown_v1 must preserve the display title",
            input_value=input_facts["title"],
            output_value=output_facts["title"],
        ),
        _critical_or_preserved(
            metric="provider_identity",
            policy="export_markdown_v1 must preserve provider identity at document level",
            input_value=input_facts["provider"],
            output_value=output_facts["provider"],
        ),
        _presence_check(
            metric="date_metadata",
            policy="export_markdown_v1 must preserve conversation date presence at document level",
            input_value=input_facts["date"],
            output_present=bool(output_facts["has_date"]),
        ),
        _critical_or_preserved(
            metric="text_messages",
            policy="export_markdown_v1 must preserve one section per text-bearing message",
            input_value=input_facts["text_messages"],
            output_value=output_facts["message_sections"],
        ),
        _critical_or_preserved(
            metric="role_sections",
            policy="export_markdown_v1 must preserve role sections for text-bearing messages",
            input_value=input_facts["text_role_counts"],
            output_value=output_facts["role_counts"],
        ),
        _declared_loss_or_preserved(
            metric="timestamp_values",
            policy="export_markdown_v1 intentionally omits per-message timestamps",
            input_value=input_facts["timestamped_text_messages"],
            output_value=output_facts["timestamp_lines"],
        ),
        _declared_loss_or_preserved(
            metric="attachment_semantics",
            policy="export_markdown_v1 intentionally omits attachment payload semantics",
            input_value=input_facts["attachment_count"],
        ),
        _declared_loss_or_preserved(
            metric="thinking_semantics",
            policy="export_markdown_v1 preserves display text but not typed thinking markers",
            input_value=input_facts["thinking_messages"],
        ),
        _declared_loss_or_preserved(
            metric="tool_semantics",
            policy="export_markdown_v1 preserves display text but not typed tool markers",
            input_value=input_facts["tool_messages"],
        ),
        _declared_loss_or_preserved(
            metric="branch_structure",
            policy="export_markdown_v1 intentionally omits explicit branch topology",
            input_value=input_facts["branch_messages"],
            output_value=output_facts["branch_labels"],
        ),
    ]
    return SemanticConversationProof(
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        surface="export_markdown_v1",
        input_facts=input_facts,
        output_facts=output_facts,
        checks=checks,
    )


def _prove_export_obsidian_surface(
    *,
    conversation: Conversation,
    rendered_text: str,
) -> SemanticConversationProof:
    input_facts = _conversation_input_facts(conversation)
    output_facts = _obsidian_output_facts(rendered_text)
    checks = [
        _critical_or_preserved(
            metric="conversation_id",
            policy="export_obsidian_v1 must preserve conversation identity in frontmatter",
            input_value=input_facts["conversation_id"],
            output_value=output_facts["conversation_id"],
        ),
        _critical_or_preserved(
            metric="title_metadata",
            policy="export_obsidian_v1 must preserve the display title",
            input_value=input_facts["title"],
            output_value=output_facts["title"],
        ),
        _critical_or_preserved(
            metric="provider_identity",
            policy="export_obsidian_v1 must preserve provider identity in frontmatter",
            input_value=input_facts["provider"],
            output_value=output_facts["provider"],
        ),
        _presence_check(
            metric="date_metadata",
            policy="export_obsidian_v1 must preserve conversation date presence in frontmatter",
            input_value=input_facts["date"],
            output_present=bool(output_facts["has_date"]),
        ),
        _critical_or_preserved(
            metric="text_messages",
            policy="export_obsidian_v1 must preserve one section per text-bearing message",
            input_value=input_facts["text_messages"],
            output_value=output_facts["message_sections"],
        ),
        _critical_or_preserved(
            metric="role_sections",
            policy="export_obsidian_v1 must preserve role sections for text-bearing messages",
            input_value=input_facts["text_role_counts"],
            output_value=output_facts["role_counts"],
        ),
        _declared_loss_or_preserved(
            metric="timestamp_values",
            policy="export_obsidian_v1 intentionally omits per-message timestamps",
            input_value=input_facts["timestamped_text_messages"],
            output_value=output_facts["timestamp_lines"],
        ),
        _declared_loss_or_preserved(
            metric="attachment_semantics",
            policy="export_obsidian_v1 intentionally omits attachment payload semantics",
            input_value=input_facts["attachment_count"],
        ),
        _declared_loss_or_preserved(
            metric="thinking_semantics",
            policy="export_obsidian_v1 preserves display text but not typed thinking markers",
            input_value=input_facts["thinking_messages"],
        ),
        _declared_loss_or_preserved(
            metric="tool_semantics",
            policy="export_obsidian_v1 preserves display text but not typed tool markers",
            input_value=input_facts["tool_messages"],
        ),
        _declared_loss_or_preserved(
            metric="branch_structure",
            policy="export_obsidian_v1 intentionally omits explicit branch topology",
            input_value=input_facts["branch_messages"],
            output_value=output_facts["branch_labels"],
        ),
    ]
    return SemanticConversationProof(
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        surface="export_obsidian_v1",
        input_facts=input_facts,
        output_facts=output_facts,
        checks=checks,
    )


def _prove_export_org_surface(
    *,
    conversation: Conversation,
    rendered_text: str,
) -> SemanticConversationProof:
    input_facts = _conversation_input_facts(conversation)
    output_facts = _org_output_facts(rendered_text)
    checks = [
        _critical_or_preserved(
            metric="title_metadata",
            policy="export_org_v1 must preserve the display title",
            input_value=input_facts["title"],
            output_value=output_facts["title"],
        ),
        _critical_or_preserved(
            metric="provider_identity",
            policy="export_org_v1 must preserve provider identity at document level",
            input_value=input_facts["provider"],
            output_value=output_facts["provider"],
        ),
        _presence_check(
            metric="date_metadata",
            policy="export_org_v1 must preserve conversation date presence at document level",
            input_value=input_facts["date"],
            output_present=bool(output_facts["has_date"]),
        ),
        _critical_or_preserved(
            metric="text_messages",
            policy="export_org_v1 must preserve one heading per text-bearing message",
            input_value=input_facts["text_messages"],
            output_value=output_facts["message_sections"],
        ),
        _critical_or_preserved(
            metric="role_sections",
            policy="export_org_v1 must preserve role headings for text-bearing messages",
            input_value=input_facts["text_role_counts"],
            output_value=output_facts["role_counts"],
        ),
        _declared_loss_or_preserved(
            metric="timestamp_values",
            policy="export_org_v1 intentionally omits per-message timestamps",
            input_value=input_facts["timestamped_text_messages"],
            output_value=output_facts["timestamp_lines"],
        ),
        _declared_loss_or_preserved(
            metric="attachment_semantics",
            policy="export_org_v1 intentionally omits attachment payload semantics",
            input_value=input_facts["attachment_count"],
        ),
        _declared_loss_or_preserved(
            metric="thinking_semantics",
            policy="export_org_v1 preserves display text but not typed thinking markers",
            input_value=input_facts["thinking_messages"],
        ),
        _declared_loss_or_preserved(
            metric="tool_semantics",
            policy="export_org_v1 preserves display text but not typed tool markers",
            input_value=input_facts["tool_messages"],
        ),
        _declared_loss_or_preserved(
            metric="branch_structure",
            policy="export_org_v1 intentionally omits explicit branch topology",
            input_value=input_facts["branch_messages"],
            output_value=output_facts["branch_labels"],
        ),
    ]
    return SemanticConversationProof(
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        surface="export_org_v1",
        input_facts=input_facts,
        output_facts=output_facts,
        checks=checks,
    )


def _prove_export_html_surface(
    *,
    conversation: Conversation,
    rendered_text: str,
) -> SemanticConversationProof:
    input_facts = _conversation_input_facts(conversation)
    output_facts = _html_output_facts(rendered_text)
    checks = [
        _critical_or_preserved(
            metric="title_metadata",
            policy="export_html_v1 must preserve the display title",
            input_value=input_facts["title"],
            output_value=output_facts["title"],
        ),
        _critical_or_preserved(
            metric="provider_identity",
            policy="export_html_v1 must preserve provider identity at document level",
            input_value=input_facts["provider"],
            output_value=output_facts["provider"],
        ),
        _presence_check(
            metric="date_metadata",
            policy="export_html_v1 must preserve conversation date presence at document level",
            input_value=input_facts["date"],
            output_present=bool(output_facts["has_date"]),
        ),
        _critical_or_preserved(
            metric="text_messages",
            policy="export_html_v1 must preserve visible message sections for text-bearing messages",
            input_value=input_facts["text_messages"],
            output_value=output_facts["message_sections"],
        ),
        _critical_or_preserved(
            metric="role_sections",
            policy="export_html_v1 must preserve visible role labels for text-bearing messages",
            input_value=input_facts["text_role_counts"],
            output_value=output_facts["role_counts"],
        ),
        _critical_or_preserved(
            metric="timestamp_values",
            policy="export_html_v1 must preserve visible message timestamps",
            input_value=input_facts["timestamped_text_messages"],
            output_value=output_facts["timestamp_lines"],
        ),
        _critical_or_preserved(
            metric="branch_structure",
            policy="export_html_v1 must preserve visible branch groupings for branched messages",
            input_value=input_facts["branch_messages"],
            output_value=output_facts["branch_labels"],
        ),
        _declared_loss_or_preserved(
            metric="attachment_semantics",
            policy="export_html_v1 intentionally omits attachment payload semantics",
            input_value=input_facts["attachment_count"],
        ),
        _declared_loss_or_preserved(
            metric="thinking_semantics",
            policy="export_html_v1 preserves display text but not typed thinking markers",
            input_value=input_facts["thinking_messages"],
        ),
        _declared_loss_or_preserved(
            metric="tool_semantics",
            policy="export_html_v1 preserves display text but not typed tool markers",
            input_value=input_facts["tool_messages"],
        ),
    ]
    return SemanticConversationProof(
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        surface="export_html_v1",
        input_facts=input_facts,
        output_facts=output_facts,
        checks=checks,
    )


def prove_export_surface_semantics(
    conversation: Conversation,
    surface: str,
    rendered_text: str,
) -> SemanticConversationProof:
    """Compare a conversation export surface to the canonical conversation semantics."""
    if surface == "export_json_v1":
        return _prove_export_json_like_surface(
            conversation=conversation,
            rendered_text=rendered_text,
            surface=surface,
        )
    if surface == "export_yaml_v1":
        return _prove_export_json_like_surface(
            conversation=conversation,
            rendered_text=rendered_text,
            surface=surface,
        )
    if surface == "export_csv_v1":
        return _prove_export_csv_surface(conversation=conversation, rendered_text=rendered_text)
    if surface == "export_markdown_v1":
        return _prove_export_markdown_surface(conversation=conversation, rendered_text=rendered_text)
    if surface == "export_obsidian_v1":
        return _prove_export_obsidian_surface(conversation=conversation, rendered_text=rendered_text)
    if surface == "export_org_v1":
        return _prove_export_org_surface(conversation=conversation, rendered_text=rendered_text)
    if surface == "export_html_v1":
        return _prove_export_html_surface(conversation=conversation, rendered_text=rendered_text)
    raise ValueError(f"Unsupported semantic surface: {surface}")


def _prove_query_summary_json_like_surface(
    *,
    summary: ConversationSummary,
    message_count: int,
    rendered_text: str,
    surface: str,
) -> SemanticConversationProof:
    try:
        if surface == "query_summary_yaml_v1":
            import yaml

            parsed = yaml.safe_load(rendered_text)
        else:
            parsed = json.loads(rendered_text)
    except Exception:
        parsed = []

    row = parsed[0] if isinstance(parsed, list) and parsed else {}
    payload = row if isinstance(row, dict) else {}
    input_facts = _summary_input_facts(summary, message_count)
    output_facts = _summary_output_facts(payload)
    checks = [
        _critical_or_preserved(
            metric="conversation_id",
            policy=f"{surface} must preserve the conversation identifier",
            input_value=input_facts["conversation_id"],
            output_value=output_facts["conversation_id"],
        ),
        _critical_or_preserved(
            metric="provider_identity",
            policy=f"{surface} must preserve provider identity",
            input_value=input_facts["provider"],
            output_value=output_facts["provider"],
        ),
        _critical_or_preserved(
            metric="title_metadata",
            policy=f"{surface} must preserve the display title",
            input_value=input_facts["title"],
            output_value=output_facts["title"],
        ),
        _critical_or_preserved(
            metric="date_metadata",
            policy=f"{surface} must preserve the summary display date",
            input_value=input_facts["date"],
            output_value=output_facts["date"],
        ),
        _critical_or_preserved(
            metric="message_count",
            policy=f"{surface} must preserve summary message counts",
            input_value=input_facts["messages"],
            output_value=output_facts["messages"],
        ),
        _critical_or_preserved(
            metric="tag_values",
            policy=f"{surface} must preserve summary tag values",
            input_value=input_facts["tags"],
            output_value=output_facts["tags"],
        ),
        _critical_or_preserved(
            metric="summary_text",
            policy=f"{surface} must preserve summary text",
            input_value=input_facts["summary"],
            output_value=output_facts["summary"],
        ),
    ]
    return SemanticConversationProof(
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        surface=surface,
        input_facts=input_facts,
        output_facts=output_facts,
        checks=checks,
    )


def _prove_query_summary_csv_surface(
    *,
    summary: ConversationSummary,
    message_count: int,
    rendered_text: str,
) -> SemanticConversationProof:
    rows = _summary_csv_output_facts(rendered_text)
    output_facts = rows[0] if rows else {}
    input_facts = _summary_input_facts(summary, message_count)
    checks = [
        _critical_or_preserved(
            metric="conversation_id",
            policy="query_summary_csv_v1 must preserve the conversation identifier",
            input_value=input_facts["conversation_id"],
            output_value=output_facts.get("conversation_id", ""),
        ),
        _critical_or_preserved(
            metric="provider_identity",
            policy="query_summary_csv_v1 must preserve provider identity",
            input_value=input_facts["provider"],
            output_value=output_facts.get("provider", ""),
        ),
        _critical_or_preserved(
            metric="title_metadata",
            policy="query_summary_csv_v1 must preserve the display title",
            input_value=input_facts["title"],
            output_value=output_facts.get("title"),
        ),
        _critical_or_preserved(
            metric="date_metadata",
            policy="query_summary_csv_v1 must preserve the summary display date",
            input_value=input_facts["date"][:10] if input_facts["date"] else None,
            output_value=output_facts.get("date"),
        ),
        _critical_or_preserved(
            metric="message_count",
            policy="query_summary_csv_v1 must preserve summary message counts",
            input_value=input_facts["messages"],
            output_value=output_facts.get("messages", 0),
        ),
        _critical_or_preserved(
            metric="tag_values",
            policy="query_summary_csv_v1 must preserve summary tag values",
            input_value=input_facts["tags"],
            output_value=output_facts.get("tags", []),
        ),
        _critical_or_preserved(
            metric="summary_text",
            policy="query_summary_csv_v1 must preserve summary text",
            input_value=input_facts["summary"],
            output_value=output_facts.get("summary"),
        ),
    ]
    return SemanticConversationProof(
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        surface="query_summary_csv_v1",
        input_facts=input_facts,
        output_facts=output_facts,
        checks=checks,
    )


def _prove_query_summary_text_surface(
    *,
    summary: ConversationSummary,
    message_count: int,
    rendered_text: str,
) -> SemanticConversationProof:
    rows = _summary_text_output_facts(rendered_text)
    output_facts = rows[0] if rows else {}
    input_facts = _summary_input_facts(summary, message_count)
    raw_title = input_facts["title"] or input_facts["conversation_id"][:20]
    expected_title = (raw_title[:47] + "...") if len(raw_title) > 50 else raw_title
    checks = [
        _critical_or_preserved(
            metric="conversation_id_prefix",
            policy="query_summary_text_v1 must preserve the visible conversation id prefix",
            input_value=input_facts["conversation_id"][:24],
            output_value=output_facts.get("conversation_id_prefix", ""),
        ),
        _critical_or_preserved(
            metric="provider_identity",
            policy="query_summary_text_v1 must preserve provider identity",
            input_value=input_facts["provider"],
            output_value=output_facts.get("provider", ""),
        ),
        _critical_or_preserved(
            metric="date_metadata",
            policy="query_summary_text_v1 must preserve the visible summary date",
            input_value=input_facts["date"][:10] if input_facts["date"] else None,
            output_value=output_facts.get("date"),
        ),
        _critical_or_preserved(
            metric="title_projection",
            policy="query_summary_text_v1 must preserve the deterministic visible title projection",
            input_value=expected_title,
            output_value=output_facts.get("title", ""),
        ),
        _critical_or_preserved(
            metric="message_count",
            policy="query_summary_text_v1 must preserve summary message counts",
            input_value=input_facts["messages"],
            output_value=output_facts.get("messages", 0),
        ),
        _declared_loss_or_preserved(
            metric="tag_values",
            policy="query_summary_text_v1 intentionally omits tag values",
            input_value=len(input_facts["tags"]),
        ),
        _declared_loss_or_preserved(
            metric="summary_text",
            policy="query_summary_text_v1 intentionally omits summary text",
            input_value=1 if input_facts["summary"] else 0,
        ),
    ]
    return SemanticConversationProof(
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        surface="query_summary_text_v1",
        input_facts=input_facts,
        output_facts=output_facts,
        checks=checks,
    )


def _prove_query_stream_plaintext_surface(
    *,
    conversation: Conversation,
    rendered_text: str,
) -> SemanticConversationProof:
    input_facts = _stream_input_facts(conversation)
    output_facts = _stream_plaintext_output_facts(rendered_text)
    checks = [
        _critical_or_preserved(
            metric="text_messages",
            policy="query_stream_plaintext_v1 must preserve one visible block per text-bearing message",
            input_value=input_facts["text_messages"],
            output_value=output_facts["message_sections"],
        ),
        _critical_or_preserved(
            metric="role_sections",
            policy="query_stream_plaintext_v1 must preserve visible role labels for streamed messages",
            input_value=input_facts["text_role_counts"],
            output_value=output_facts["role_counts"],
        ),
        _declared_loss_or_preserved(
            metric="title_metadata",
            policy="query_stream_plaintext_v1 intentionally omits title metadata",
            input_value=1 if input_facts["title"] else 0,
        ),
        _declared_loss_or_preserved(
            metric="provider_identity",
            policy="query_stream_plaintext_v1 intentionally omits provider metadata",
            input_value=1 if input_facts["provider"] else 0,
        ),
        _declared_loss_or_preserved(
            metric="date_metadata",
            policy="query_stream_plaintext_v1 intentionally omits date metadata",
            input_value=1 if input_facts["date"] else 0,
        ),
        _declared_loss_or_preserved(
            metric="timestamp_values",
            policy="query_stream_plaintext_v1 intentionally omits per-message timestamps",
            input_value=input_facts["timestamped_text_messages"],
        ),
        _declared_loss_or_preserved(
            metric="attachment_semantics",
            policy="query_stream_plaintext_v1 intentionally omits attachment semantics",
            input_value=input_facts["attachment_count"],
        ),
        _declared_loss_or_preserved(
            metric="thinking_semantics",
            policy="query_stream_plaintext_v1 preserves display text but not typed thinking markers",
            input_value=input_facts["thinking_messages"],
        ),
        _declared_loss_or_preserved(
            metric="tool_semantics",
            policy="query_stream_plaintext_v1 preserves display text but not typed tool markers",
            input_value=input_facts["tool_messages"],
        ),
        _declared_loss_or_preserved(
            metric="branch_structure",
            policy="query_stream_plaintext_v1 intentionally omits explicit branch topology",
            input_value=input_facts["branch_messages"],
        ),
    ]
    return SemanticConversationProof(
        conversation_id=str(conversation.id),
        provider=str(conversation.provider),
        surface="query_stream_plaintext_v1",
        input_facts=input_facts,
        output_facts=output_facts,
        checks=checks,
    )


def _prove_query_stream_markdown_surface(
    *,
    conversation: Conversation,
    rendered_text: str,
) -> SemanticConversationProof:
    input_facts = _stream_input_facts(conversation)
    output_facts = _stream_markdown_output_facts(rendered_text)
    checks = [
        _critical_or_preserved(
            metric="title_metadata",
            policy="query_stream_markdown_v1 must preserve the conversation title",
            input_value=input_facts["title"],
            output_value=output_facts["title"],
        ),
        _critical_or_preserved(
            metric="provider_identity",
            policy="query_stream_markdown_v1 must preserve provider identity in the stream header",
            input_value=input_facts["provider"],
            output_value=output_facts["provider"],
        ),
        _presence_check(
            metric="date_metadata",
            policy="query_stream_markdown_v1 must preserve conversation date presence in the stream header",
            input_value=input_facts["date"],
            output_present=bool(output_facts["has_date"]),
        ),
        _critical_or_preserved(
            metric="text_messages",
            policy="query_stream_markdown_v1 must preserve one visible section per text-bearing message",
            input_value=input_facts["text_messages"],
            output_value=output_facts["message_sections"],
        ),
        _critical_or_preserved(
            metric="role_sections",
            policy="query_stream_markdown_v1 must preserve visible role headings for streamed messages",
            input_value=input_facts["text_role_counts"],
            output_value=output_facts["role_counts"],
        ),
        _critical_or_preserved(
            metric="footer_count",
            policy="query_stream_markdown_v1 must report the number of emitted messages honestly",
            input_value=input_facts["text_messages"],
            output_value=output_facts["footer_count"],
        ),
        _declared_loss_or_preserved(
            metric="timestamp_values",
            policy="query_stream_markdown_v1 intentionally omits per-message timestamps",
            input_value=input_facts["timestamped_text_messages"],
        ),
        _declared_loss_or_preserved(
            metric="attachment_semantics",
            policy="query_stream_markdown_v1 intentionally omits attachment semantics",
            input_value=input_facts["attachment_count"],
        ),
        _declared_loss_or_preserved(
            metric="thinking_semantics",
            policy="query_stream_markdown_v1 preserves display text but not typed thinking markers",
            input_value=input_facts["thinking_messages"],
        ),
        _declared_loss_or_preserved(
            metric="tool_semantics",
            policy="query_stream_markdown_v1 preserves display text but not typed tool markers",
            input_value=input_facts["tool_messages"],
        ),
        _declared_loss_or_preserved(
            metric="branch_structure",
            policy="query_stream_markdown_v1 intentionally omits explicit branch topology",
            input_value=input_facts["branch_messages"],
        ),
    ]
    return SemanticConversationProof(
        conversation_id=str(conversation.id),
        provider=str(conversation.provider),
        surface="query_stream_markdown_v1",
        input_facts=input_facts,
        output_facts=output_facts,
        checks=checks,
    )


def _prove_query_stream_json_lines_surface(
    *,
    conversation: Conversation,
    rendered_text: str,
) -> SemanticConversationProof:
    input_facts = _stream_input_facts(conversation)
    output_facts = _stream_json_lines_output_facts(rendered_text)
    checks = [
        _critical_or_preserved(
            metric="conversation_id",
            policy="query_stream_json_lines_v1 must preserve the conversation identifier in the header",
            input_value=input_facts["conversation_id"],
            output_value=output_facts["conversation_id"],
        ),
        _critical_or_preserved(
            metric="title_metadata",
            policy="query_stream_json_lines_v1 must preserve the title in the header",
            input_value=input_facts["title"],
            output_value=output_facts["title"],
        ),
        _critical_or_preserved(
            metric="provider_identity",
            policy="query_stream_json_lines_v1 must preserve provider identity in the header",
            input_value=input_facts["provider"],
            output_value=output_facts["provider"],
        ),
        _critical_or_preserved(
            metric="date_metadata",
            policy="query_stream_json_lines_v1 must preserve the conversation date in the header",
            input_value=input_facts["date"],
            output_value=output_facts["date"],
        ),
        _critical_or_preserved(
            metric="text_message_ids",
            policy="query_stream_json_lines_v1 must preserve identifiers for emitted messages",
            input_value=input_facts["text_message_ids"],
            output_value=output_facts["message_ids"],
        ),
        _critical_or_preserved(
            metric="role_sections",
            policy="query_stream_json_lines_v1 must preserve role distribution for emitted messages",
            input_value=input_facts["text_role_counts"],
            output_value=output_facts["role_counts"],
        ),
        _critical_or_preserved(
            metric="timestamp_values",
            policy="query_stream_json_lines_v1 must preserve timestamps for emitted messages",
            input_value=input_facts["timestamped_text_messages"],
            output_value=output_facts["timestamped_messages"],
        ),
        _critical_or_preserved(
            metric="footer_count",
            policy="query_stream_json_lines_v1 must report the number of emitted messages honestly",
            input_value=input_facts["text_messages"],
            output_value=output_facts["footer_count"],
        ),
        _declared_loss_or_preserved(
            metric="attachment_semantics",
            policy="query_stream_json_lines_v1 intentionally omits attachment semantics",
            input_value=input_facts["attachment_count"],
        ),
        _declared_loss_or_preserved(
            metric="thinking_semantics",
            policy="query_stream_json_lines_v1 preserves display text but not typed thinking markers",
            input_value=input_facts["thinking_messages"],
        ),
        _declared_loss_or_preserved(
            metric="tool_semantics",
            policy="query_stream_json_lines_v1 preserves display text but not typed tool markers",
            input_value=input_facts["tool_messages"],
        ),
        _declared_loss_or_preserved(
            metric="branch_structure",
            policy="query_stream_json_lines_v1 intentionally omits explicit branch topology",
            input_value=input_facts["branch_messages"],
        ),
    ]
    return SemanticConversationProof(
        conversation_id=str(conversation.id),
        provider=str(conversation.provider),
        surface="query_stream_json_lines_v1",
        input_facts=input_facts,
        output_facts=output_facts,
        checks=checks,
    )


def _prove_mcp_summary_surface(
    *,
    summary: ConversationSummary,
    message_count: int,
    rendered_text: str,
) -> SemanticConversationProof:
    try:
        parsed = json.loads(rendered_text)
    except Exception:
        parsed = []
    row = parsed[0] if isinstance(parsed, list) and parsed else {}
    payload = row if isinstance(row, dict) else {}
    input_facts = {
        "conversation_id": str(summary.id),
        "provider": str(summary.provider),
        "title": summary.display_title,
        "messages": message_count,
        "created_at": summary.created_at.isoformat() if summary.created_at else None,
        "updated_at": summary.updated_at.isoformat() if summary.updated_at else None,
        "tags": list(summary.tags),
        "summary": summary.summary,
    }
    output_facts = _mcp_summary_output_facts(payload)
    checks = [
        _critical_or_preserved(
            metric="conversation_id",
            policy="mcp_summary_json_v1 must preserve the conversation identifier",
            input_value=input_facts["conversation_id"],
            output_value=output_facts["conversation_id"],
        ),
        _critical_or_preserved(
            metric="provider_identity",
            policy="mcp_summary_json_v1 must preserve provider identity",
            input_value=input_facts["provider"],
            output_value=output_facts["provider"],
        ),
        _critical_or_preserved(
            metric="title_metadata",
            policy="mcp_summary_json_v1 must preserve the display title",
            input_value=input_facts["title"],
            output_value=output_facts["title"],
        ),
        _critical_or_preserved(
            metric="message_count",
            policy="mcp_summary_json_v1 must preserve summary message counts",
            input_value=input_facts["messages"],
            output_value=output_facts["messages"],
        ),
        _critical_or_preserved(
            metric="created_at",
            policy="mcp_summary_json_v1 must preserve the created_at timestamp when present",
            input_value=input_facts["created_at"],
            output_value=output_facts["created_at"],
        ),
        _critical_or_preserved(
            metric="updated_at",
            policy="mcp_summary_json_v1 must preserve the updated_at timestamp when present",
            input_value=input_facts["updated_at"],
            output_value=output_facts["updated_at"],
        ),
        _declared_loss_or_preserved(
            metric="tag_values",
            policy="mcp_summary_json_v1 intentionally omits tags",
            input_value=len(input_facts["tags"]),
        ),
        _declared_loss_or_preserved(
            metric="summary_text",
            policy="mcp_summary_json_v1 intentionally omits summary text",
            input_value=1 if input_facts["summary"] else 0,
        ),
    ]
    return SemanticConversationProof(
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        surface="mcp_summary_json_v1",
        input_facts=input_facts,
        output_facts=output_facts,
        checks=checks,
    )


def _prove_mcp_detail_surface(
    *,
    conversation: Conversation,
    rendered_text: str,
) -> SemanticConversationProof:
    try:
        parsed = json.loads(rendered_text)
    except Exception:
        parsed = {}
    payload = parsed if isinstance(parsed, dict) else {}
    input_facts = _mcp_detail_input_facts(conversation)
    output_facts = _mcp_detail_output_facts(payload)
    checks = [
        _critical_or_preserved(
            metric="conversation_id",
            policy="mcp_detail_json_v1 must preserve the conversation identifier",
            input_value=input_facts["conversation_id"],
            output_value=output_facts["conversation_id"],
        ),
        _critical_or_preserved(
            metric="provider_identity",
            policy="mcp_detail_json_v1 must preserve provider identity",
            input_value=input_facts["provider"],
            output_value=output_facts["provider"],
        ),
        _critical_or_preserved(
            metric="title_metadata",
            policy="mcp_detail_json_v1 must preserve the display title",
            input_value=input_facts["title"],
            output_value=output_facts["title"],
        ),
        _critical_or_preserved(
            metric="created_at",
            policy="mcp_detail_json_v1 must preserve the created_at timestamp when present",
            input_value=input_facts["created_at"],
            output_value=output_facts["created_at"],
        ),
        _critical_or_preserved(
            metric="updated_at",
            policy="mcp_detail_json_v1 must preserve the updated_at timestamp when present",
            input_value=input_facts["updated_at"],
            output_value=output_facts["updated_at"],
        ),
        _critical_or_preserved(
            metric="message_entries",
            policy="mcp_detail_json_v1 must preserve every message entry",
            input_value=input_facts["messages"],
            output_value=output_facts["messages"],
        ),
        _critical_or_preserved(
            metric="message_ids",
            policy="mcp_detail_json_v1 must preserve message identifiers",
            input_value=input_facts["message_ids"],
            output_value=output_facts["message_ids"],
        ),
        _critical_or_preserved(
            metric="role_entries",
            policy="mcp_detail_json_v1 must preserve message role distribution",
            input_value=input_facts["role_counts"],
            output_value=output_facts["role_counts"],
        ),
        _critical_or_preserved(
            metric="timestamp_values",
            policy="mcp_detail_json_v1 must preserve message timestamps",
            input_value=input_facts["timestamped_messages"],
            output_value=output_facts["timestamped_messages"],
        ),
        _declared_loss_or_preserved(
            metric="attachment_semantics",
            policy="mcp_detail_json_v1 intentionally omits attachment payload semantics",
            input_value=input_facts["attachment_count"],
        ),
        _declared_loss_or_preserved(
            metric="thinking_semantics",
            policy="mcp_detail_json_v1 preserves display text but not typed thinking markers",
            input_value=input_facts["thinking_messages"],
        ),
        _declared_loss_or_preserved(
            metric="tool_semantics",
            policy="mcp_detail_json_v1 preserves display text but not typed tool markers",
            input_value=input_facts["tool_messages"],
        ),
        _declared_loss_or_preserved(
            metric="branch_structure",
            policy="mcp_detail_json_v1 intentionally omits explicit branch topology",
            input_value=input_facts["branch_messages"],
        ),
    ]
    return SemanticConversationProof(
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        surface="mcp_detail_json_v1",
        input_facts=input_facts,
        output_facts=output_facts,
        checks=checks,
    )


__all__ = [
    "_prove_mcp_detail_surface",
    "_prove_mcp_summary_surface",
    "_prove_query_stream_json_lines_surface",
    "_prove_query_stream_markdown_surface",
    "_prove_query_stream_plaintext_surface",
    "_prove_query_summary_csv_surface",
    "_prove_query_summary_json_like_surface",
    "_prove_query_summary_text_surface",
    "prove_export_surface_semantics",
    "prove_markdown_projection_semantics",
]
