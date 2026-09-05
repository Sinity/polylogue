"""Persistence helpers for acquisition service writes."""

from __future__ import annotations

from polylogue.core.protocols import RawPersistenceStore
from polylogue.logging import get_logger
from polylogue.pipeline.services.acquisition_records import pending_pre_parse_raw_admission_request
from polylogue.pipeline.stage_models import AcquireResult
from polylogue.security.excision_policy import ExcisionPolicySnapshot
from polylogue.storage.artifacts.inspection import inspect_raw_artifact
from polylogue.storage.runtime import RawSessionRecord

logger = get_logger(__name__)


async def persist_raw_record(
    repository: RawPersistenceStore,
    record: RawSessionRecord,
    *,
    result: AcquireResult,
    policy_snapshot: ExcisionPolicySnapshot | None = None,
) -> None:
    """Persist one raw record and update acquisition counters."""
    try:
        admission = await repository.admit_raw(
            pending_pre_parse_raw_admission_request(record, policy_snapshot=policy_snapshot)
        )
        admitted_record = record.model_copy(update={"raw_id": admission.result.raw_id})
        observation = inspect_raw_artifact(admitted_record)
        await repository.save_artifact_observation(observation)
        if admission.inserted:
            result.acquired += 1
            result.raw_ids.append(admission.result.raw_id)
        else:
            result.skipped += 1
    except Exception as exc:
        logger.error(
            "Failed to store raw session",
            source=record.source_name,
            path=record.source_path,
            error=str(exc),
            exc_info=True,
        )
        result.errors += 1


__all__ = ["persist_raw_record"]
