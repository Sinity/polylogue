"""Persistence helpers for acquisition service writes."""

from __future__ import annotations

from polylogue.logging import get_logger
from polylogue.pipeline.stage_models import AcquireResult
from polylogue.storage.artifact_inspection import inspect_raw_artifact
from polylogue.storage.store import RawConversationRecord

logger = get_logger(__name__)


async def persist_raw_record(
    repository: object,
    record: RawConversationRecord,
    *,
    result: AcquireResult,
) -> None:
    """Persist one raw record and update acquisition counters."""
    try:
        observation = inspect_raw_artifact(record)
        inserted = await repository.save_raw_conversation(record)
        await repository.save_artifact_observation(observation)
        if inserted:
            result.acquired += 1
            result.raw_ids.append(record.raw_id)
        else:
            result.skipped += 1
    except Exception as exc:
        logger.error(
            "Failed to store raw conversation",
            source=record.source_name,
            path=record.source_path,
            error=str(exc),
        )
        result.errors += 1


__all__ = ["persist_raw_record"]
