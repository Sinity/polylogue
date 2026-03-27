"""Export-oriented semantic-proof functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.lib.semantic_facts import (
    build_conversation_semantic_facts,
    build_projection_semantic_facts,
)
from polylogue.rendering.semantic_proof_fact_exports import (
    _canonical_markdown_output_facts,
    _csv_output_facts,
    _html_output_facts,
    _json_like_output_facts,
    _markdown_doc_output_facts,
    _obsidian_output_facts,
    _org_output_facts,
)
from polylogue.rendering.semantic_proof_surface_support import (
    _build_surface_proof,
    _parse_json_payload,
    _parse_yaml_payload,
)

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation
    from polylogue.storage.state_views import ConversationRenderProjection
    from polylogue.rendering.semantic_proof_models import SemanticConversationProof


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
    payload = (
        _parse_yaml_payload(rendered_text)
        if surface == "export_yaml_v1"
        else _parse_json_payload(rendered_text)
    )
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
    if surface in {"export_json_v1", "export_yaml_v1"}:
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


__all__ = [
    "prove_export_surface_semantics",
    "prove_markdown_projection_semantics",
]
