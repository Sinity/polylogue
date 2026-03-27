"""Input loading helpers for semantic-proof suites."""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.rendering.semantic_surface_registry import EXPORT_SURFACE_FORMATS, STREAM_SURFACE_FORMATS

_SUMMARY_SURFACES = {
    "query_summary_json_v1",
    "query_summary_yaml_v1",
    "query_summary_csv_v1",
    "query_summary_text_v1",
}


@dataclass(frozen=True)
class SemanticProofSuiteInputs:
    """Loaded inputs required for semantic-proof suite execution."""

    summaries: list[object]
    message_counts: dict[str, int]
    conversations_by_id: dict[str, object]


async def load_semantic_surface_suite_inputs(
    *,
    repository,
    providers: list[str] | None,
    surfaces: list[str],
    record_limit: int | None,
    record_offset: int,
) -> SemanticProofSuiteInputs:
    """Load only the repository inputs required by the requested surfaces."""
    summaries = await repository.list_summaries(
        limit=record_limit,
        offset=record_offset,
        providers=providers,
    )
    summary_ids = [str(summary.id) for summary in summaries]
    export_surfaces = set(EXPORT_SURFACE_FORMATS)
    need_message_counts = any(
        surface in _SUMMARY_SURFACES or surface == "mcp_summary_json_v1"
        for surface in surfaces
    )
    need_full_conversations = any(
        surface in export_surfaces
        or surface in STREAM_SURFACE_FORMATS
        or surface == "mcp_detail_json_v1"
        for surface in surfaces
    )
    message_counts = (
        await repository.queries.get_message_counts_batch(summary_ids)
        if summary_ids and need_message_counts
        else {}
    )
    conversations_by_id: dict[str, object] = {}
    if need_full_conversations and summary_ids:
        conversations = await repository.get_many(summary_ids)
        conversations_by_id = {str(conversation.id): conversation for conversation in conversations}
    return SemanticProofSuiteInputs(
        summaries=summaries,
        message_counts=message_counts,
        conversations_by_id=conversations_by_id,
    )


__all__ = ["SemanticProofSuiteInputs", "load_semantic_surface_suite_inputs"]
