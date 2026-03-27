"""Runtime execution helpers for semantic-proof suites."""

from __future__ import annotations

from pathlib import Path

from polylogue.rendering.core import ConversationFormatter
from polylogue.rendering.semantic_proof_suite_conversation import append_conversation_surface_proofs
from polylogue.rendering.semantic_proof_suite_loading import load_semantic_surface_suite_inputs
from polylogue.rendering.semantic_proof_suite_summary import append_summary_surface_proofs
from polylogue.rendering.semantic_proof_suite_support import build_suite_report
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository


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
        inputs = await load_semantic_surface_suite_inputs(
            repository=repository,
            providers=providers,
            surfaces=surfaces,
            record_limit=record_limit,
            record_offset=record_offset,
        )
        proofs_by_surface = {surface: [] for surface in surfaces}

        for summary in inputs.summaries:
            await _collect_surface_proofs(
                repository=repository,
                formatter=formatter,
                summary=summary,
                surfaces=surfaces,
                proofs_by_surface=proofs_by_surface,
                message_counts=inputs.message_counts,
                conversations_by_id=inputs.conversations_by_id,
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

    append_summary_surface_proofs(
        proofs_by_surface,
        summary=summary,
        message_count=message_count,
        surface_set=surface_set,
    )
    await append_conversation_surface_proofs(
        repository=repository,
        formatter=formatter,
        conversation=conversation,
        conversation_id=conversation_id,
        surfaces=surfaces,
        proofs_by_surface=proofs_by_surface,
    )


__all__ = ["prove_semantic_surface_suite_async"]
