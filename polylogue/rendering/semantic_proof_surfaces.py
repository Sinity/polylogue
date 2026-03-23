"""Surface-specific semantic-proof functions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from polylogue.lib.semantic_facts import (
    build_conversation_semantic_facts,
    build_mcp_detail_semantic_facts,
    build_mcp_summary_semantic_facts,
    build_projection_semantic_facts,
    build_stream_semantic_facts,
    build_summary_semantic_facts,
)
from polylogue.rendering.semantic_proof_facts import (
    _canonical_markdown_output_facts,
    _csv_output_facts,
    _html_output_facts,
    _json_like_output_facts,
    _markdown_doc_output_facts,
    _mcp_detail_output_facts,
    _mcp_summary_output_facts,
    _obsidian_output_facts,
    _org_output_facts,
    _stream_json_lines_output_facts,
    _stream_markdown_output_facts,
    _stream_plaintext_output_facts,
    _summary_csv_output_facts,
    _summary_output_facts,
    _summary_text_output_facts,
)
from polylogue.rendering.semantic_proof_models import SemanticConversationProof
from polylogue.rendering.semantic_surface_registry import evaluate_semantic_contracts

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation, ConversationSummary
    from polylogue.storage.store import ConversationRenderProjection


def _build_surface_proof(
    *,
    surface: str,
    conversation_id: str,
    provider: str,
    input_facts: dict[str, Any],
    output_facts: dict[str, Any],
) -> SemanticConversationProof:
    return SemanticConversationProof(
        conversation_id=conversation_id,
        provider=provider,
        surface=surface,
        input_facts=input_facts,
        output_facts=output_facts,
        checks=evaluate_semantic_contracts(surface, input_facts, output_facts),
    )


def _parse_json_payload(rendered_text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(rendered_text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_json_row(rendered_text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(rendered_text)
    except Exception:
        return {}
    row = parsed[0] if isinstance(parsed, list) and parsed else {}
    return row if isinstance(row, dict) else {}


def _parse_yaml_payload(rendered_text: str) -> dict[str, Any]:
    try:
        import yaml

        parsed = yaml.safe_load(rendered_text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_yaml_row(rendered_text: str) -> dict[str, Any]:
    try:
        import yaml

        parsed = yaml.safe_load(rendered_text)
    except Exception:
        return {}
    row = parsed[0] if isinstance(parsed, list) and parsed else {}
    return row if isinstance(row, dict) else {}


def prove_markdown_projection_semantics(
    projection: ConversationRenderProjection,
    markdown_text: str,
) -> SemanticConversationProof:
    """Compare a repository render projection to canonical markdown output."""
    input_facts = build_projection_semantic_facts(projection).to_proof_input()
    output_facts = _canonical_markdown_output_facts(markdown_text)
    return _build_surface_proof(
        surface="canonical_markdown_v1",
        conversation_id=projection.conversation.conversation_id,
        provider=projection.conversation.provider_name or "unknown",
        input_facts=input_facts,
        output_facts=output_facts,
    )


def _prove_export_json_like_surface(
    *,
    conversation: Conversation,
    rendered_text: str,
    surface: str,
) -> SemanticConversationProof:
    payload = _parse_yaml_payload(rendered_text) if surface == "export_yaml_v1" else _parse_json_payload(rendered_text)
    input_facts = build_conversation_semantic_facts(conversation).to_proof_input()
    output_facts = _json_like_output_facts(payload)
    return _build_surface_proof(
        surface=surface,
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        input_facts=input_facts,
        output_facts=output_facts,
    )


def _prove_export_csv_surface(
    *,
    conversation: Conversation,
    rendered_text: str,
) -> SemanticConversationProof:
    input_facts = build_conversation_semantic_facts(conversation).to_proof_input()
    output_facts = _csv_output_facts(rendered_text)
    return _build_surface_proof(
        surface="export_csv_v1",
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        input_facts=input_facts,
        output_facts=output_facts,
    )


def _prove_export_markdown_surface(
    *,
    conversation: Conversation,
    rendered_text: str,
) -> SemanticConversationProof:
    input_facts = build_conversation_semantic_facts(conversation).to_proof_input()
    output_facts = _markdown_doc_output_facts(rendered_text)
    return _build_surface_proof(
        surface="export_markdown_v1",
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        input_facts=input_facts,
        output_facts=output_facts,
    )


def _prove_export_obsidian_surface(
    *,
    conversation: Conversation,
    rendered_text: str,
) -> SemanticConversationProof:
    input_facts = build_conversation_semantic_facts(conversation).to_proof_input()
    output_facts = _obsidian_output_facts(rendered_text)
    return _build_surface_proof(
        surface="export_obsidian_v1",
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        input_facts=input_facts,
        output_facts=output_facts,
    )


def _prove_export_org_surface(
    *,
    conversation: Conversation,
    rendered_text: str,
) -> SemanticConversationProof:
    input_facts = build_conversation_semantic_facts(conversation).to_proof_input()
    output_facts = _org_output_facts(rendered_text)
    return _build_surface_proof(
        surface="export_org_v1",
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        input_facts=input_facts,
        output_facts=output_facts,
    )


def _prove_export_html_surface(
    *,
    conversation: Conversation,
    rendered_text: str,
) -> SemanticConversationProof:
    input_facts = build_conversation_semantic_facts(conversation).to_proof_input()
    output_facts = _html_output_facts(rendered_text)
    return _build_surface_proof(
        surface="export_html_v1",
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        input_facts=input_facts,
        output_facts=output_facts,
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
    payload = _parse_yaml_row(rendered_text) if surface == "query_summary_yaml_v1" else _parse_json_row(rendered_text)
    input_facts = build_summary_semantic_facts(summary, message_count=message_count).to_proof_input()
    output_facts = _summary_output_facts(payload)
    return _build_surface_proof(
        surface=surface,
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        input_facts=input_facts,
        output_facts=output_facts,
    )


def _prove_query_summary_csv_surface(
    *,
    summary: ConversationSummary,
    message_count: int,
    rendered_text: str,
) -> SemanticConversationProof:
    rows = _summary_csv_output_facts(rendered_text)
    output_facts = rows[0] if rows else {}
    input_facts = build_summary_semantic_facts(summary, message_count=message_count).to_proof_input()
    return _build_surface_proof(
        surface="query_summary_csv_v1",
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        input_facts=input_facts,
        output_facts=output_facts,
    )


def _prove_query_summary_text_surface(
    *,
    summary: ConversationSummary,
    message_count: int,
    rendered_text: str,
) -> SemanticConversationProof:
    rows = _summary_text_output_facts(rendered_text)
    output_facts = rows[0] if rows else {}
    input_facts = build_summary_semantic_facts(summary, message_count=message_count).to_proof_input()
    return _build_surface_proof(
        surface="query_summary_text_v1",
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        input_facts=input_facts,
        output_facts=output_facts,
    )


def _prove_query_stream_plaintext_surface(
    *,
    conversation: Conversation,
    rendered_text: str,
) -> SemanticConversationProof:
    input_facts = build_stream_semantic_facts(conversation).to_proof_input()
    output_facts = _stream_plaintext_output_facts(rendered_text)
    return _build_surface_proof(
        surface="query_stream_plaintext_v1",
        conversation_id=str(conversation.id),
        provider=str(conversation.provider),
        input_facts=input_facts,
        output_facts=output_facts,
    )


def _prove_query_stream_markdown_surface(
    *,
    conversation: Conversation,
    rendered_text: str,
) -> SemanticConversationProof:
    input_facts = build_stream_semantic_facts(conversation).to_proof_input()
    output_facts = _stream_markdown_output_facts(rendered_text)
    return _build_surface_proof(
        surface="query_stream_markdown_v1",
        conversation_id=str(conversation.id),
        provider=str(conversation.provider),
        input_facts=input_facts,
        output_facts=output_facts,
    )


def _prove_query_stream_json_lines_surface(
    *,
    conversation: Conversation,
    rendered_text: str,
) -> SemanticConversationProof:
    input_facts = build_stream_semantic_facts(conversation).to_proof_input()
    output_facts = _stream_json_lines_output_facts(rendered_text)
    return _build_surface_proof(
        surface="query_stream_json_lines_v1",
        conversation_id=str(conversation.id),
        provider=str(conversation.provider),
        input_facts=input_facts,
        output_facts=output_facts,
    )


def _prove_mcp_summary_surface(
    *,
    summary: ConversationSummary,
    message_count: int,
    rendered_text: str,
) -> SemanticConversationProof:
    payload = _parse_json_row(rendered_text)
    input_facts = build_mcp_summary_semantic_facts(summary, message_count=message_count).to_proof_input()
    output_facts = _mcp_summary_output_facts(payload)
    return _build_surface_proof(
        surface="mcp_summary_json_v1",
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        input_facts=input_facts,
        output_facts=output_facts,
    )


def _prove_mcp_detail_surface(
    *,
    conversation: Conversation,
    rendered_text: str,
) -> SemanticConversationProof:
    payload = _parse_json_payload(rendered_text)
    input_facts = build_mcp_detail_semantic_facts(conversation).to_proof_input()
    output_facts = _mcp_detail_output_facts(payload)
    return _build_surface_proof(
        surface="mcp_detail_json_v1",
        conversation_id=input_facts["conversation_id"],
        provider=input_facts["provider"],
        input_facts=input_facts,
        output_facts=output_facts,
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
