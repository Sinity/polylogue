"""Read-surface semantic-proof functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.lib.semantic_facts import (
    build_mcp_detail_semantic_facts,
    build_mcp_summary_semantic_facts,
    build_stream_semantic_facts,
    build_summary_semantic_facts,
)
from polylogue.rendering.semantic_proof_fact_reads import (
    _mcp_detail_output_facts,
    _mcp_summary_output_facts,
    _stream_json_lines_output_facts,
    _stream_markdown_output_facts,
    _stream_plaintext_output_facts,
    _summary_csv_output_facts,
    _summary_output_facts,
    _summary_text_output_facts,
)
from polylogue.rendering.semantic_proof_surface_support import (
    _build_surface_proof,
    _parse_json_payload,
    _parse_json_row,
    _parse_yaml_row,
)

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation, ConversationSummary
    from polylogue.rendering.semantic_proof_models import SemanticConversationProof


def _prove_query_summary_json_like_surface(
    *,
    summary: ConversationSummary,
    message_count: int,
    rendered_text: str,
    surface: str,
) -> SemanticConversationProof:
    payload = (
        _parse_yaml_row(rendered_text)
        if surface == "query_summary_yaml_v1"
        else _parse_json_row(rendered_text)
    )
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
]
