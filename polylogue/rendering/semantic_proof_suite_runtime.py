"""Runtime execution helpers for semantic-proof suites."""

from __future__ import annotations

from pathlib import Path

from polylogue.cli.query_output import format_summary_list, render_stream_transcript
from polylogue.mcp.payloads import (
    MCPConversationDetailPayload,
    MCPConversationSummaryListPayload,
    MCPConversationSummaryPayload,
)
from polylogue.rendering.core import ConversationFormatter
from polylogue.rendering.formatting import format_conversation
from polylogue.rendering.semantic_proof_suite_support import build_suite_report
from polylogue.rendering.semantic_proof_surface_exports import (
    prove_export_surface_semantics,
    prove_markdown_projection_semantics,
)
from polylogue.rendering.semantic_proof_surface_reads import (
    _prove_mcp_detail_surface,
    _prove_mcp_summary_surface,
    _prove_query_stream_json_lines_surface,
    _prove_query_stream_markdown_surface,
    _prove_query_stream_plaintext_surface,
    _prove_query_summary_csv_surface,
    _prove_query_summary_json_like_surface,
    _prove_query_summary_text_surface,
)
from polylogue.rendering.semantic_surface_registry import (
    EXPORT_SURFACE_FORMATS,
    STREAM_SURFACE_FORMATS,
)
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository

_SUMMARY_SURFACES = {
    "query_summary_json_v1",
    "query_summary_yaml_v1",
    "query_summary_csv_v1",
    "query_summary_text_v1",
}
_STREAM_SURFACES = {
    "query_stream_plaintext_v1",
    "query_stream_markdown_v1",
    "query_stream_json_lines_v1",
}
_SUMMARY_JSON_SURFACES = {
    "query_summary_json_v1",
    "query_summary_yaml_v1",
}


async def prove_semantic_surface_suite_async(
    *,
    db_path: Path,
    archive_root: Path,
    providers: list[str] | None,
    surfaces: list[str],
    record_limit: int | None,
    record_offset: int,
):
    backend = SQLiteBackend(db_path=db_path)
    repository = ConversationRepository(backend=backend)
    formatter = ConversationFormatter(archive_root=archive_root, db_path=db_path, backend=backend)
    try:
        summaries = await repository.list_summaries(
            limit=record_limit,
            offset=record_offset,
            providers=providers,
        )
        proofs_by_surface = {surface: [] for surface in surfaces}
        export_surfaces = set(EXPORT_SURFACE_FORMATS)
        need_message_counts = any(
            surface in _SUMMARY_SURFACES or surface == "mcp_summary_json_v1"
            for surface in surfaces
        )
        need_full_conversations = any(
            surface in export_surfaces
            or surface in _STREAM_SURFACES
            or surface == "mcp_detail_json_v1"
            for surface in surfaces
        )

        summary_ids = [str(summary.id) for summary in summaries]
        message_counts = (
            await repository.queries.get_message_counts_batch(summary_ids)
            if summary_ids and need_message_counts
            else {}
        )
        conversations_by_id = {}
        if need_full_conversations and summary_ids:
            conversations = await repository.get_many(summary_ids)
            conversations_by_id = {str(conversation.id): conversation for conversation in conversations}

        for summary in summaries:
            await _collect_surface_proofs(
                repository=repository,
                formatter=formatter,
                summary=summary,
                surfaces=surfaces,
                proofs_by_surface=proofs_by_surface,
                message_counts=message_counts,
                conversations_by_id=conversations_by_id,
            )

        return build_suite_report(
            surfaces=surfaces,
            proofs_by_surface=proofs_by_surface,
            record_limit=record_limit,
            record_offset=record_offset,
            provider_filters=list(providers or []),
        )
    finally:
        await backend.close()


async def _collect_surface_proofs(
    *,
    repository: ConversationRepository,
    formatter: ConversationFormatter,
    summary,
    surfaces: list[str],
    proofs_by_surface: dict[str, list],
    message_counts: dict[str, int],
    conversations_by_id: dict[str, object],
) -> None:
    conversation_id = str(summary.id)
    surface_set = set(surfaces)
    conversation = conversations_by_id.get(conversation_id) if conversations_by_id else None
    message_count = message_counts.get(
        conversation_id,
        len(conversation.messages) if conversation is not None else 0,
    )

    if _SUMMARY_JSON_SURFACES & surface_set:
        _append_summary_json_like_proofs(
            proofs_by_surface,
            summary=summary,
            message_count=message_count,
            surface_set=surface_set,
        )

    if "query_summary_csv_v1" in surface_set:
        rendered_text = format_summary_list(
            [summary],
            "csv",
            None,
            message_counts={conversation_id: message_count},
        )
        proofs_by_surface["query_summary_csv_v1"].append(
            _prove_query_summary_csv_surface(
                summary=summary,
                message_count=message_count,
                rendered_text=rendered_text,
            )
        )

    if "query_summary_text_v1" in surface_set:
        rendered_text = format_summary_list(
            [summary],
            "text",
            None,
            message_counts={conversation_id: message_count},
        )
        proofs_by_surface["query_summary_text_v1"].append(
            _prove_query_summary_text_surface(
                summary=summary,
                message_count=message_count,
                rendered_text=rendered_text,
            )
        )

    if "mcp_summary_json_v1" in surface_set:
        rendered_text = MCPConversationSummaryListPayload(
            root=[
                MCPConversationSummaryPayload.from_summary(
                    summary,
                    message_count=message_count,
                )
            ]
        ).to_json()
        proofs_by_surface["mcp_summary_json_v1"].append(
            _prove_mcp_summary_surface(
                summary=summary,
                message_count=message_count,
                rendered_text=rendered_text,
            )
        )

    await _append_conversation_surface_proofs(
        repository=repository,
        formatter=formatter,
        conversation=conversation,
        conversation_id=conversation_id,
        surfaces=surfaces,
        proofs_by_surface=proofs_by_surface,
    )


def _append_summary_json_like_proofs(
    proofs_by_surface: dict[str, list],
    *,
    summary,
    message_count: int,
    surface_set: set[str],
) -> None:
    for surface in sorted(_SUMMARY_JSON_SURFACES & surface_set):
        rendered_text = format_summary_list(
            [summary],
            "json" if surface == "query_summary_json_v1" else "yaml",
            None,
            message_counts={str(summary.id): message_count},
        )
        proofs_by_surface[surface].append(
            _prove_query_summary_json_like_surface(
                summary=summary,
                message_count=message_count,
                rendered_text=rendered_text,
                surface=surface,
            )
        )


async def _append_conversation_surface_proofs(
    *,
    repository: ConversationRepository,
    formatter: ConversationFormatter,
    conversation,
    conversation_id: str,
    surfaces: list[str],
    proofs_by_surface: dict[str, list],
) -> None:
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

        if surface in _STREAM_SURFACES:
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


__all__ = ["prove_semantic_surface_suite_async"]
