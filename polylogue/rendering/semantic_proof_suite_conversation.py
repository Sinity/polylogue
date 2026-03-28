"""Conversation-surface collectors for semantic-proof suites."""

from __future__ import annotations

from polylogue.cli.query_output import render_stream_transcript
from polylogue.mcp.payloads import MCPConversationDetailPayload
from polylogue.rendering.formatting import format_conversation
from polylogue.rendering.semantic_proof_surface_exports import (
    prove_export_surface_semantics,
    prove_markdown_projection_semantics,
)
from polylogue.rendering.semantic_proof_surface_reads import (
    _prove_mcp_detail_surface,
    _prove_query_stream_json_lines_surface,
    _prove_query_stream_markdown_surface,
    _prove_query_stream_plaintext_surface,
)
from polylogue.rendering.semantic_surface_registry import EXPORT_SURFACE_FORMATS, STREAM_SURFACE_FORMATS


async def append_conversation_surface_proofs(
    *,
    repository,
    formatter,
    conversation,
    conversation_id: str,
    surfaces: list[str],
    proofs_by_surface: dict[str, list],
) -> None:
    """Append conversation-backed surface proofs for a single conversation."""
    projection = None
    for surface in surfaces:
        if surface == "canonical_markdown_v1":
            if projection is None:
                projection = await repository.get_render_projection(conversation_id)
            if projection is None:
                continue
            formatted = formatter.format_projection(projection)
            proofs_by_surface[surface].append(
                prove_markdown_projection_semantics(projection, formatted.markdown_text)
            )
            continue

        if conversation is None:
            continue

        if surface in EXPORT_SURFACE_FORMATS:
            rendered_text = format_conversation(
                conversation,
                EXPORT_SURFACE_FORMATS[surface],
                None,
            )
            proofs_by_surface[surface].append(
                prove_export_surface_semantics(conversation, surface, rendered_text)
            )
            continue

        if surface in STREAM_SURFACE_FORMATS:
            rendered_text, _ = render_stream_transcript(
                conversation_id=conversation_id,
                title=conversation.display_title,
                provider=str(conversation.provider),
                display_date=conversation.display_date,
                messages=list(conversation.messages),
                output_format=STREAM_SURFACE_FORMATS[surface],
                dialogue_only=False,
                message_limit=None,
                stats={
                    "total_messages": len(conversation.messages),
                    "dialogue_messages": sum(
                        1 for message in conversation.messages if message.is_dialogue
                    ),
                },
            )
            if surface == "query_stream_plaintext_v1":
                proofs_by_surface[surface].append(
                    _prove_query_stream_plaintext_surface(
                        conversation=conversation,
                        rendered_text=rendered_text,
                    )
                )
            elif surface == "query_stream_markdown_v1":
                proofs_by_surface[surface].append(
                    _prove_query_stream_markdown_surface(
                        conversation=conversation,
                        rendered_text=rendered_text,
                    )
                )
            else:
                proofs_by_surface[surface].append(
                    _prove_query_stream_json_lines_surface(
                        conversation=conversation,
                        rendered_text=rendered_text,
                    )
                )
            continue

        if surface == "mcp_detail_json_v1":
            rendered_text = MCPConversationDetailPayload.from_conversation(conversation).to_json()
            proofs_by_surface[surface].append(
                _prove_mcp_detail_surface(
                    conversation=conversation,
                    rendered_text=rendered_text,
                )
            )


__all__ = ["append_conversation_surface_proofs"]
