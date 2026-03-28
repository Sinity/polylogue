"""Async parse preparation facade with explicit transform and enrichment stages."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.logging import get_logger
from polylogue.pipeline.ids import (
    conversation_id as make_conversation_id,
)
from polylogue.pipeline.ids import (
    materialize_attachment_path,
    move_attachment_to_archive,
)
from polylogue.pipeline.prepare_enrichment import _build_single_cache, enrich_bundle_from_db
from polylogue.pipeline.prepare_models import (
    AttachmentMaterializationPlan,
    EnrichedBundle,
    PrepareCache,
    RecordBundle,
    SaveResult,
    TransformResult,
    _timestamp_sort_key,
)
from polylogue.pipeline.prepare_transform import transform_to_records

if TYPE_CHECKING:
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository

logger = get_logger(__name__)


async def save_bundle(bundle: RecordBundle, repository: ConversationRepository) -> SaveResult:
    counts = await repository.save_conversation(
        conversation=bundle.conversation,
        messages=bundle.messages,
        attachments=bundle.attachments,
        content_blocks=bundle.content_blocks,
    )
    return SaveResult(**counts)


async def prepare_records(
    convo,
    source_name: str,
    *,
    archive_root: Path,
    backend: SQLiteBackend | None = None,
    repository: ConversationRepository | None = None,
    raw_id: str | None = None,
    cache: PrepareCache | None = None,
) -> tuple[str, dict[str, int], bool]:
    """Convert a ParsedConversation to storage records and persist them."""
    if repository is None and backend is None:
        raise ValueError("prepare_records requires a repository or backend")
    if repository is None:
        from polylogue.storage.repository import ConversationRepository

        repository = ConversationRepository(backend=backend)
    if backend is None:
        backend = repository.backend

    if not convo.messages:
        cid = make_conversation_id(convo.provider_name, convo.provider_conversation_id)
        logger.debug("Skipping empty conversation (no messages)", conversation_id=cid)
        return (
            cid,
            {
                "conversations": 0,
                "messages": 0,
                "attachments": 0,
                "skipped_conversations": 1,
                "skipped_messages": 0,
                "skipped_attachments": 0,
            },
            False,
        )

    transform = transform_to_records(convo, source_name, archive_root=archive_root)
    if cache is None:
        cache = await _build_single_cache(backend, convo, transform.candidate_cid, transform.candidate_cid)
    enriched = enrich_bundle_from_db(convo, source_name, transform, cache, raw_id=raw_id)

    applied_moves: list[tuple[Path, Path]] = []
    try:
        for source_path, target_path in enriched.materialization_plan.move_before_save:
            materialize_attachment_path(source_path, target_path)
            applied_moves.append((source_path, target_path))

        result = await save_bundle(enriched.bundle, repository=repository)
    except Exception:
        for source_path, target_path in reversed(applied_moves):
            if target_path.exists():
                move_attachment_to_archive(target_path, source_path)
        raise

    for duplicate_source in enriched.materialization_plan.delete_after_save:
        if duplicate_source.exists():
            duplicate_source.unlink()

    return (
        enriched.cid,
        {
            "conversations": result.conversations,
            "messages": result.messages,
            "attachments": result.attachments,
            "skipped_conversations": result.skipped_conversations,
            "skipped_messages": result.skipped_messages,
            "skipped_attachments": result.skipped_attachments,
        },
        enriched.changed,
    )


__all__ = [
    "AttachmentMaterializationPlan",
    "EnrichedBundle",
    "PrepareCache",
    "RecordBundle",
    "SaveResult",
    "TransformResult",
    "_timestamp_sort_key",
    "enrich_bundle_from_db",
    "prepare_records",
    "save_bundle",
    "transform_to_records",
]
