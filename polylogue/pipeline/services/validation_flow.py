"""Async orchestration for raw validation service flows."""

from __future__ import annotations

import asyncio
import os

from polylogue.logging import get_logger
from polylogue.paths import blob_store_root
from polylogue.pipeline.services.process_pool import process_pool_executor
from polylogue.pipeline.services.validation_runtime import _validate_record_sync, _ValidationOutcome
from polylogue.pipeline.stage_models import ValidatedRawRecord, ValidateResult
from polylogue.protocols import ProgressCallback
from polylogue.storage.store import RawConversationRecord
from polylogue.types import Provider, ValidationMode, ValidationStatus

logger = get_logger(__name__)


def schema_validation_mode(
    *,
    env_var: str,
    default: ValidationMode,
) -> ValidationMode:
    """Return configured schema validation mode."""
    raw = os.environ.get(env_var, str(default))
    try:
        return ValidationMode.from_string(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r, falling back to %s",
            env_var,
            raw,
            default,
        )
        return default


def validation_progress_desc(processed: int, total: int) -> str:
    """Return a stable validation progress description."""
    return f"Validating: {processed:,}/{total:,} raw"


async def validate_raw_ids(
    *,
    repository: object,
    raw_ids: list[str],
    progress_callback: ProgressCallback | None = None,
    persist: bool = True,
    validation_mode: ValidationMode,
    raw_batch_size: int,
) -> ValidateResult:
    """Validate raw records, optionally persisting the resulting status."""
    if not raw_ids:
        return ValidateResult()

    total_raw_ids = len(raw_ids)
    if progress_callback is not None:
        progress_callback(0, desc=validation_progress_desc(0, total_raw_ids))

    if validation_mode is ValidationMode.OFF:
        result = ValidateResult()
        for index, raw_id in enumerate(raw_ids, start=1):
            if persist:
                await repository.mark_raw_validated(
                    raw_id,
                    status=ValidationStatus.SKIPPED,
                    mode=validation_mode,
                )
            result.records.append(
                ValidatedRawRecord(
                    raw_id=raw_id,
                    parseable=True,
                    validation_status=ValidationStatus.SKIPPED,
                    validation_error=None,
                    canonical_provider=Provider.UNKNOWN,
                    payload_provider=None,
                )
            )
            if progress_callback is not None:
                progress_callback(1, desc=validation_progress_desc(index, total_raw_ids))
        return result

    result = ValidateResult()
    for batch_start in range(0, len(raw_ids), raw_batch_size):
        batch_ids = raw_ids[batch_start : batch_start + raw_batch_size]
        raw_records = await repository.get_raw_conversations_batch(batch_ids)
        batch_result = await evaluate_raw_records(
            repository=repository,
            raw_records=raw_records,
            progress_callback=progress_callback,
            persist=persist,
            mode=validation_mode,
            progress_total=total_raw_ids,
            progress_offset=batch_start,
        )
        result.merge(batch_result)

        missing = [raw_id for raw_id in batch_ids if raw_id not in {record.raw_id for record in raw_records}]
        processed = batch_start + len(raw_records)
        for missing_index, raw_id in enumerate(missing, start=1):
            result.errors += 1
            result.records.append(
                ValidatedRawRecord(
                    raw_id=raw_id,
                    parseable=False,
                    validation_status=ValidationStatus.FAILED,
                    validation_error="Missing raw conversation record",
                    canonical_provider=Provider.UNKNOWN,
                    payload_provider=None,
                )
            )
            if progress_callback is not None:
                progress_callback(
                    1,
                    desc=validation_progress_desc(processed + missing_index, total_raw_ids),
                )

    return result


async def evaluate_raw_records(
    *,
    repository: object,
    raw_records: list[RawConversationRecord],
    progress_callback: ProgressCallback | None = None,
    persist: bool = False,
    mode: ValidationMode,
    progress_total: int | None = None,
    progress_offset: int = 0,
) -> ValidateResult:
    """Evaluate raw records using the canonical validation logic."""
    result = ValidateResult()
    if not raw_records:
        return result

    if mode is ValidationMode.OFF:
        for raw_record in raw_records:
            if persist:
                await repository.mark_raw_validated(
                    raw_record.raw_id,
                    status=ValidationStatus.SKIPPED,
                    mode=mode,
                    provider=Provider.from_string(raw_record.provider_name),
                    payload_provider=raw_record.payload_provider,
                )
            result.records.append(
                ValidatedRawRecord(
                    raw_id=raw_record.raw_id,
                    parseable=True,
                    validation_status=ValidationStatus.SKIPPED,
                    validation_error=None,
                    canonical_provider=Provider.from_string(raw_record.provider_name),
                    payload_provider=raw_record.payload_provider,
                )
            )
            if progress_callback is not None:
                total = progress_total or len(raw_records)
                progress_callback(
                    1,
                    desc=validation_progress_desc(progress_offset + len(result.records), total),
                )
        return result

    import time as _time

    total = progress_total or len(raw_records)
    # ProcessPoolExecutor bypasses the GIL: JSON decode (orjson C extension)
    # + Python wrapper code run truly parallel across processes.
    # Measured: Threads(24)=160 MB/s, Process(8)=605 MB/s (3.7x speedup).
    worker_count = min(len(raw_records), os.cpu_count() or 4, 8)
    blob_root_str = str(blob_store_root())
    t_batch = _time.perf_counter()

    def _run_batch():
        try:
            with process_pool_executor(max_workers=worker_count) as executor:
                return list(
                    executor.map(
                        _validate_record_sync,
                        raw_records,
                        [mode] * len(raw_records),
                        [blob_root_str] * len(raw_records),
                        chunksize=max(1, len(raw_records) // worker_count),
                    )
                )
        except (TypeError, _pickle.PicklingError):
            # Fallback for unpicklable records (e.g. MagicMock in tests)
            return [_validate_record_sync(r, mode, blob_root_str) for r in raw_records]

    import pickle as _pickle

    outcomes: list[_ValidationOutcome] = await asyncio.to_thread(_run_batch)
    batch_elapsed = _time.perf_counter() - t_batch
    total_blob_mb = sum(r.blob_size for r in raw_records) / (1024 * 1024)
    if batch_elapsed > 1.0:
        logger.info(
            "validate_batch",
            elapsed_s=round(batch_elapsed, 2),
            records=len(raw_records),
            blob_mb=round(total_blob_mb, 1),
            rate=round(len(raw_records) / batch_elapsed, 1),
            workers=worker_count,
        )

    for index, (raw_record, outcome) in enumerate(zip(raw_records, outcomes, strict=True), start=1):
        result.validated += outcome.counts_delta["validated"]
        result.invalid += outcome.counts_delta["invalid"]
        result.drift += outcome.counts_delta["drift"]
        result.skipped_no_schema += outcome.counts_delta["skipped_no_schema"]
        result.errors += outcome.counts_delta["errors"]
        for prov, cnt in outcome.drift_counts_delta.items():
            result.drift_counts[prov] = result.drift_counts.get(prov, 0) + cnt

        result.records.append(
            ValidatedRawRecord(
                raw_id=raw_record.raw_id,
                parseable=outcome.parseable,
                validation_status=outcome.validation_status,
                validation_error=outcome.validation_error,
                canonical_provider=outcome.canonical_provider,
                payload_provider=outcome.payload_provider,
                drift_count=outcome.drift_count,
            )
        )

        if persist:
            await repository.mark_raw_validated(
                raw_record.raw_id,
                status=outcome.validation_status,
                error=outcome.validation_error,
                drift_count=outcome.drift_count,
                provider=outcome.canonical_provider,
                mode=mode,
                payload_provider=outcome.payload_provider,
            )
            if not outcome.parseable and outcome.validation_error is not None:
                await repository.mark_raw_parsed(
                    raw_record.raw_id,
                    error=outcome.validation_error,
                    payload_provider=outcome.payload_provider,
                )

        if progress_callback is not None:
            progress_callback(
                1,
                desc=validation_progress_desc(progress_offset + index, total),
            )

    return result


__all__ = [
    "evaluate_raw_records",
    "schema_validation_mode",
    "validate_raw_ids",
    "validation_progress_desc",
]
