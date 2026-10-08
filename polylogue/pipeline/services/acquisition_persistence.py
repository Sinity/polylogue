"""Persistence helpers for acquisition service writes."""

from __future__ import annotations

from collections.abc import Callable

from polylogue.core.compute import DaemonOperationCancelled
from polylogue.core.protocols import RawPersistenceStore
from polylogue.logging import get_logger
from polylogue.pipeline.services.acquisition_records import pending_pre_parse_raw_admission_request
from polylogue.pipeline.stage_models import AcquireResult
from polylogue.security.excision_policy import ExcisionPolicyError, ExcisionPolicySnapshot
from polylogue.storage.artifacts.inspection import inspect_raw_artifact
from polylogue.storage.cursor_state import CursorFailurePayload
from polylogue.storage.runtime import ArtifactObservationRecord, RawSessionRecord

logger = get_logger(__name__)


async def persist_raw_record(
    repository: RawPersistenceStore,
    record: RawSessionRecord,
    *,
    result: AcquireResult,
    policy_snapshot: ExcisionPolicySnapshot | None = None,
    prepared_observation: ArtifactObservationRecord | None = None,
    preparation_error: Exception | None = None,
    failures: list[CursorFailurePayload] | None = None,
    on_failure: Callable[[Exception], None] | None = None,
) -> str | None:
    """Persist one raw record and update acquisition counters.

    A record that could not be stored is appended to ``failures`` so the
    caller withholds its source's stat cursor: counting the error alone let
    the cursor advance and every later pass skip the unstored file.
    """
    try:
        admission = await repository.admit_raw(
            pending_pre_parse_raw_admission_request(record, policy_snapshot=policy_snapshot)
        )
        admitted_record = record.model_copy(update={"raw_id": admission.result.raw_id})
        if preparation_error is not None:
            raise preparation_error
        observation = (
            inspect_raw_artifact(admitted_record)
            if prepared_observation is None
            else prepared_observation.model_copy(update={"raw_id": admission.result.raw_id})
        )
        await repository.save_artifact_observation(observation)
        if admission.inserted:
            result.acquired += 1
            result.raw_ids.append(admission.result.raw_id)
        else:
            result.skipped += 1
        return admission.result.raw_id
    except DaemonOperationCancelled:
        raise
    except Exception as exc:
        logger.error(
            "Failed to store raw session",
            source=record.source_name,
            path=record.source_path,
            error=str(exc),
            exc_info=True,
        )
        result.errors += 1
        if on_failure is not None:
            on_failure(exc)
        if failures is not None and not isinstance(exc, ExcisionPolicyError):
            # Durably excised content is a permanent refusal: withholding the
            # cursor would retry forbidden bytes on every pass.
            failures.append(CursorFailurePayload(path=record.source_path, error=f"{type(exc).__name__}: {exc}"))
    return None


__all__ = ["persist_raw_record"]
