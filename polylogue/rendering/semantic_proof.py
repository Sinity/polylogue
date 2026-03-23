"""Semantic preservation proofing for render, export, query, stream, and MCP surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.cli.query_output import format_summary_list, render_stream_transcript
from polylogue.mcp.payloads import (
    MCPConversationDetailPayload,
    MCPConversationSummaryListPayload,
    MCPConversationSummaryPayload,
)
from polylogue.paths import archive_root as default_archive_root
from polylogue.paths import db_path as default_db_path
from polylogue.rendering.core import ConversationFormatter
from polylogue.rendering.formatting import format_conversation
from polylogue.rendering.semantic_proof_models import (
    ProviderSemanticProof,
    SemanticConversationProof,
    SemanticMetricCheck,
    SemanticProofReport,
    SemanticProofSuiteReport,
    _build_provider_reports,
)
from polylogue.rendering.semantic_proof_surfaces import (
    _prove_mcp_detail_surface,
    _prove_mcp_summary_surface,
    _prove_query_stream_json_lines_surface,
    _prove_query_stream_markdown_surface,
    _prove_query_stream_plaintext_surface,
    _prove_query_summary_csv_surface,
    _prove_query_summary_json_like_surface,
    _prove_query_summary_text_surface,
    prove_export_surface_semantics,
    prove_markdown_projection_semantics,
)
from polylogue.rendering.semantic_surface_registry import (
    DEFAULT_SEMANTIC_SURFACES,
    EXPORT_SURFACE_FORMATS,
    STREAM_SURFACE_FORMATS,
    list_semantic_surface_specs,
    resolve_semantic_surfaces,
)
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository
from polylogue.sync_bridge import run_coroutine_sync

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation

def _empty_surface_report(
    surface: str,
    *,
    record_limit: int | None,
    record_offset: int,
    provider_filters: list[str],
) -> SemanticProofReport:
    return SemanticProofReport(
        surface=surface,
        conversations=[],
        provider_reports={},
        record_limit=record_limit,
        record_offset=record_offset,
        provider_filters=provider_filters,
    )


async def _prove_semantic_surface_suite_async(
    *,
    db_path: Path,
    archive_root: Path,
    providers: list[str] | None,
    surfaces: list[str],
    record_limit: int | None,
    record_offset: int,
) -> SemanticProofSuiteReport:
    backend = SQLiteBackend(db_path=db_path)
    repository = ConversationRepository(backend=backend)
    formatter = ConversationFormatter(archive_root=archive_root, db_path=db_path, backend=backend)
    try:
        summaries = await repository.list_summaries(
            limit=record_limit,
            offset=record_offset,
            providers=providers,
        )
        proofs_by_surface: dict[str, list[SemanticConversationProof]] = {surface: [] for surface in surfaces}
        summary_surfaces = {
            "query_summary_json_v1",
            "query_summary_yaml_v1",
            "query_summary_csv_v1",
            "query_summary_text_v1",
        }
        stream_surfaces = {
            "query_stream_plaintext_v1",
            "query_stream_markdown_v1",
            "query_stream_json_lines_v1",
        }
        export_surfaces = set(EXPORT_SURFACE_FORMATS)
        summary_json_surfaces = {
            "query_summary_json_v1",
            "query_summary_yaml_v1",
        }
        need_message_counts = any(
            surface in summary_surfaces or surface == "mcp_summary_json_v1"
            for surface in surfaces
        )
        need_full_conversations = any(
            surface in export_surfaces
            or surface in stream_surfaces
            or surface == "mcp_detail_json_v1"
            for surface in surfaces
        )

        summary_ids = [str(summary.id) for summary in summaries]
        message_counts = (
            await repository.queries.get_message_counts_batch(summary_ids)
            if summary_ids and need_message_counts
            else {}
        )
        conversations_by_id: dict[str, Conversation] = {}
        if need_full_conversations and summary_ids:
            conversations = await repository.get_many(summary_ids)
            conversations_by_id = {str(conversation.id): conversation for conversation in conversations}

        for summary in summaries:
            conversation_id = str(summary.id)
            conversation = conversations_by_id.get(conversation_id) if need_full_conversations else None
            message_count = message_counts.get(
                conversation_id,
                len(conversation.messages) if conversation is not None else 0,
            )
            if summary_json_surfaces & set(surfaces):
                for surface in sorted(summary_json_surfaces & set(surfaces)):
                    rendered_text = format_summary_list(
                        [summary],
                        "json" if surface == "query_summary_json_v1" else "yaml",
                        None,
                        message_counts={conversation_id: message_count},
                    )
                    proofs_by_surface[surface].append(
                        _prove_query_summary_json_like_surface(
                            summary=summary,
                            message_count=message_count,
                            rendered_text=rendered_text,
                            surface=surface,
                        )
                    )

            if "query_summary_csv_v1" in surfaces:
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

            if "query_summary_text_v1" in surfaces:
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

            if "mcp_summary_json_v1" in surfaces:
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

                if surface in export_surfaces:
                    if conversation is None:
                        continue
                    rendered_text = format_conversation(conversation, EXPORT_SURFACE_FORMATS[surface], None)
                    proofs_by_surface[surface].append(
                        prove_export_surface_semantics(conversation, surface, rendered_text)
                    )
                    continue

                if surface in stream_surfaces:
                    if conversation is None:
                        continue
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
                            "dialogue_messages": sum(1 for message in conversation.messages if message.is_dialogue),
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
                    if conversation is None:
                        continue
                    rendered_text = MCPConversationDetailPayload.from_conversation(conversation).to_json()
                    proofs_by_surface[surface].append(
                        _prove_mcp_detail_surface(
                            conversation=conversation,
                            rendered_text=rendered_text,
                        )
                    )

        provider_filters = list(providers or [])
        return SemanticProofSuiteReport(
            surface_reports={
                surface: SemanticProofReport(
                    surface=surface,
                    conversations=proofs_by_surface[surface],
                    provider_reports=_build_provider_reports(proofs_by_surface[surface]),
                    record_limit=record_limit,
                    record_offset=record_offset,
                    provider_filters=provider_filters,
                )
                for surface in surfaces
            },
            record_limit=record_limit,
            record_offset=record_offset,
            provider_filters=provider_filters,
            surface_filters=list(surfaces),
        )
    finally:
        await backend.close()


def prove_semantic_surface_suite(
    *,
    db_path: Path | None = None,
    archive_root: Path | None = None,
    providers: list[str] | None = None,
    surfaces: list[str] | tuple[str, ...] | None = None,
    record_limit: int | None = None,
    record_offset: int = 0,
) -> SemanticProofSuiteReport:
    """Run semantic preservation proof across canonical render and export surfaces."""
    effective_db_path = db_path or default_db_path()
    bounded_limit = max(1, int(record_limit)) if record_limit is not None else None
    bounded_offset = max(0, int(record_offset))
    resolved_surfaces = resolve_semantic_surfaces(surfaces)
    provider_filters = list(providers or [])

    if not effective_db_path.exists():
        return SemanticProofSuiteReport(
            surface_reports={
                surface: _empty_surface_report(
                    surface,
                    record_limit=bounded_limit,
                    record_offset=bounded_offset,
                    provider_filters=provider_filters,
                )
                for surface in resolved_surfaces
            },
            record_limit=bounded_limit,
            record_offset=bounded_offset,
            provider_filters=provider_filters,
            surface_filters=list(resolved_surfaces),
        )

    return run_coroutine_sync(
        _prove_semantic_surface_suite_async(
            db_path=effective_db_path,
            archive_root=archive_root or default_archive_root(),
            providers=providers,
            surfaces=resolved_surfaces,
            record_limit=bounded_limit,
            record_offset=bounded_offset,
        )
    )


def prove_markdown_render_semantics(
    *,
    db_path: Path | None = None,
    archive_root: Path | None = None,
    providers: list[str] | None = None,
    record_limit: int | None = None,
    record_offset: int = 0,
) -> SemanticProofReport:
    """Run semantic preservation proof over canonical markdown rendering."""
    suite = prove_semantic_surface_suite(
        db_path=db_path,
        archive_root=archive_root,
        providers=providers,
        surfaces=["canonical_markdown_v1"],
        record_limit=record_limit,
        record_offset=record_offset,
    )
    return suite.surfaces["canonical_markdown_v1"]


__all__ = [
    "DEFAULT_SEMANTIC_SURFACES",
    "ProviderSemanticProof",
    "SemanticConversationProof",
    "SemanticMetricCheck",
    "SemanticProofReport",
    "SemanticProofSuiteReport",
    "list_semantic_surface_specs",
    "prove_export_surface_semantics",
    "prove_markdown_projection_semantics",
    "prove_markdown_render_semantics",
    "prove_semantic_surface_suite",
    "resolve_semantic_surfaces",
]
