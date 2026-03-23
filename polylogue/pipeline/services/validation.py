"""Schema validation service for raw conversation payloads.

This service implements a dedicated VALIDATE stage between ACQUIRE and PARSE.
It validates raw payloads against provider schemas and reports drift/invalid data.
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from polylogue.lib.raw_payload import build_raw_payload_envelope
from polylogue.logging import get_logger
from polylogue.pipeline.stage_models import ValidateResult, ValidatedRawRecord
from polylogue.protocols import ProgressCallback
from polylogue.storage.store import RawConversationRecord
from polylogue.types import Provider, ValidationMode, ValidationStatus

if TYPE_CHECKING:
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository

logger = get_logger(__name__)

__all__ = ["ValidationService", "ValidateResult"]


@dataclass
class _ValidationOutcome:
    """Thread-safe result of per-record CPU-bound validation work.

    Produced by _validate_record_sync and consumed by the sequential
    DB-write phase of evaluate_raw_records.  No DB access, no shared
    state mutation — safe to produce from any thread.
    """

    validation_status: ValidationStatus
    validation_error: str | None
    parseable: bool
    canonical_provider: Provider
    payload_provider: Provider | None
    drift_count: int
    counts_delta: dict[str, int] = field(default_factory=dict)
    drift_counts_delta: dict[str, int] = field(default_factory=dict)


def _validate_record_sync(
    raw_record: RawConversationRecord,
    validation_mode: ValidationMode,
) -> _ValidationOutcome:
    """Run CPU-bound validation for a single raw record.

    Pure function — no DB access, no shared state mutation.  Safe to
    call from a ThreadPoolExecutor worker alongside other records.

    The heavy work here (orjson.loads inside build_raw_payload_envelope,
    jsonschema iter_errors) is partially GIL-releasing (orjson is a C
    extension), so a thread pool gives real concurrency for large
    JSONL payloads.
    """
    from polylogue.schemas.validator import SchemaValidator

    counts_delta: dict[str, int] = {
        "validated": 0,
        "invalid": 0,
        "drift": 0,
        "skipped_no_schema": 0,
        "errors": 0,
    }
    drift_counts_delta: dict[str, int] = {}

    stored_payload_provider = getattr(raw_record, "payload_provider", None)
    if not isinstance(stored_payload_provider, str) or not stored_payload_provider.strip():
        stored_payload_provider = None
    canonical_provider = Provider.from_string(stored_payload_provider or raw_record.provider_name)
    payload_provider = stored_payload_provider

    try:
        envelope = build_raw_payload_envelope(
            raw_record.raw_content,
            source_path=raw_record.source_path,
            fallback_provider=raw_record.provider_name,
            payload_provider=stored_payload_provider,
        )
        payload = envelope.payload
        malformed_lines = envelope.malformed_jsonl_lines
        payload_provider = envelope.provider
    except Exception as exc:
        counts_delta["errors"] += 1
        return _ValidationOutcome(
            validation_status=ValidationStatus.FAILED,
            validation_error=f"Unable to decode payload: {exc}",
            parseable=False,
            canonical_provider=canonical_provider,
            payload_provider=payload_provider,
            drift_count=0,
            counts_delta=counts_delta,
        )

    validation_status = ValidationStatus.PASSED
    validation_error: str | None = None
    parseable = True
    drift_count = 0

    if not envelope.artifact.schema_eligible:
        return _ValidationOutcome(
            validation_status=ValidationStatus.SKIPPED,
            validation_error=f"Artifact excluded from conversation schema inference: {envelope.artifact.kind.value}",
            parseable=False,
            canonical_provider=canonical_provider,
            payload_provider=payload_provider,
            drift_count=0,
            counts_delta=counts_delta,
        )

    if malformed_lines:
        malformed_error = f"Malformed JSONL lines: {malformed_lines}"
        if validation_mode is ValidationMode.STRICT:
            counts_delta["invalid"] += 1
            return _ValidationOutcome(
                validation_status=ValidationStatus.FAILED,
                validation_error=malformed_error,
                parseable=False,
                canonical_provider=canonical_provider,
                payload_provider=payload_provider,
                drift_count=0,
                counts_delta=counts_delta,
            )
        else:
            logger.warning(
                "Malformed JSONL lines ignored in advisory mode",
                raw_id=raw_record.raw_id,
                provider=raw_record.provider_name,
                malformed_lines=malformed_lines,
            )

    validator = None
    try:
        validator = SchemaValidator.for_payload(
            envelope.provider,
            envelope.payload,
            source_path=raw_record.source_path,
        )
    except (FileNotFoundError, ImportError):
        counts_delta["skipped_no_schema"] += 1

    collected_errors: list[str] = []
    collected_drift: list[str] = []

    if validator is not None:
        samples = validator.validation_samples(payload)
        if samples:
            invalid_count = 0
            for sample in samples:
                sample_result = validator.validate(sample)
                if not sample_result.is_valid:
                    invalid_count += 1
                    collected_errors.extend(sample_result.errors[:2])
                if sample_result.has_drift:
                    drift_count += 1
                    collected_drift.extend(sample_result.drift_warnings[:3])

            canonical_provider = validator.provider or envelope.provider

            if invalid_count:
                counts_delta["invalid"] += 1
                logger.warning(
                    "Schema validation errors for %s",
                    canonical_provider,
                    raw_id=raw_record.raw_id,
                    samples=len(samples),
                    invalid_samples=invalid_count,
                    errors=collected_errors[:5],
                )
                if validation_mode is ValidationMode.STRICT:
                    first_error = collected_errors[0] if collected_errors else "unknown schema validation error"
                    validation_status = ValidationStatus.FAILED
                    validation_error = f"Schema validation failed for {canonical_provider}: {first_error}"
                    parseable = False
            else:
                counts_delta["validated"] += 1

            if drift_count:
                counts_delta["drift"] += 1
                drift_counts_delta[canonical_provider] = drift_count
                logger.info(
                    "Schema drift detected for %s",
                    canonical_provider,
                    raw_id=raw_record.raw_id,
                    drift=collected_drift[:10],
                )
    elif parseable:
        validation_status = ValidationStatus.SKIPPED

    return _ValidationOutcome(
        validation_status=validation_status,
        validation_error=validation_error,
        parseable=parseable,
        canonical_provider=canonical_provider,
        payload_provider=payload_provider,
        drift_count=drift_count,
        counts_delta=counts_delta,
        drift_counts_delta=drift_counts_delta,
    )


class ValidationService:
    """Validate raw payloads against provider schemas."""

    SCHEMA_VALIDATION_MODE_ENV = "POLYLOGUE_SCHEMA_VALIDATION"
    SCHEMA_VALIDATION_DEFAULT = ValidationMode.STRICT
    SCHEMA_VALIDATION_MODES = frozenset(ValidationMode)

    # Keep batches aligned with parse batching.
    RAW_BATCH_SIZE = 50

    def __init__(self, backend: SQLiteBackend):
        self.backend = backend
        from polylogue.storage.repository import ConversationRepository

        self.repository: ConversationRepository = ConversationRepository(backend=backend)

    def _schema_validation_mode(self) -> ValidationMode:
        """Return configured schema validation mode."""
        raw = os.environ.get(self.SCHEMA_VALIDATION_MODE_ENV, str(self.SCHEMA_VALIDATION_DEFAULT))
        try:
            return ValidationMode.from_string(raw)
        except ValueError:
            pass
        logger.warning(
            "Invalid %s=%r, falling back to %s",
            self.SCHEMA_VALIDATION_MODE_ENV,
            raw,
            self.SCHEMA_VALIDATION_DEFAULT,
        )
        return self.SCHEMA_VALIDATION_DEFAULT

    def _validation_progress_desc(self, processed: int, total: int) -> str:
        """Return a stable validation progress description."""
        return f"Validating: {processed:,}/{total:,} raw"

    async def validate_raw_ids(
        self,
        *,
        raw_ids: list[str],
        progress_callback: ProgressCallback | None = None,
        persist: bool = True,
    ) -> ValidateResult:
        """Validate raw records, optionally persisting the resulting status."""
        if not raw_ids:
            return ValidateResult()

        total_raw_ids = len(raw_ids)
        if progress_callback is not None:
            progress_callback(0, desc=self._validation_progress_desc(0, total_raw_ids))

        validation_mode = self._schema_validation_mode()
        if validation_mode is ValidationMode.OFF:
            result = ValidateResult()
            for index, raw_id in enumerate(raw_ids, start=1):
                if persist:
                    await self.repository.mark_raw_validated(
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
                    progress_callback(1, desc=self._validation_progress_desc(index, total_raw_ids))
            return result

        result = ValidateResult()
        for batch_start in range(0, len(raw_ids), self.RAW_BATCH_SIZE):
            batch_ids = raw_ids[batch_start : batch_start + self.RAW_BATCH_SIZE]
            raw_records = await self.repository.get_raw_conversations_batch(batch_ids)
            batch_result = await self.evaluate_raw_records(
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
                        desc=self._validation_progress_desc(processed + missing_index, total_raw_ids),
                    )

        return result

    async def evaluate_raw_records(
        self,
        *,
        raw_records: list[RawConversationRecord],
        progress_callback: ProgressCallback | None = None,
        persist: bool = False,
        mode: ValidationMode | None = None,
        progress_total: int | None = None,
        progress_offset: int = 0,
    ) -> ValidateResult:
        """Evaluate raw records using the canonical validation logic.

        Phase 1 (concurrent): envelope building + schema validation runs in a
        thread pool.  orjson (C extension) releases the GIL during JSON
        parsing, giving real parallelism for large JSONL payloads.

        Phase 2 (sequential): DB writes and result accumulation remain
        sequential to avoid SQLite contention and preserve ordering.
        """
        result = ValidateResult()
        if not raw_records:
            return result

        validation_mode = mode or self._schema_validation_mode()
        if validation_mode is ValidationMode.OFF:
            for raw_record in raw_records:
                if persist:
                    await self.repository.mark_raw_validated(
                        raw_record.raw_id,
                        status=ValidationStatus.SKIPPED,
                        mode=validation_mode,
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
                        desc=self._validation_progress_desc(progress_offset + len(result.records), total),
                    )
            return result

        total = progress_total or len(raw_records)

        # Phase 1: concurrent CPU-bound work (envelope + validation).
        # Worker count is capped at batch size and CPU count to avoid spawning
        # excessive threads for small batches.
        worker_count = min(len(raw_records), os.cpu_count() or 4)
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            outcomes: list[_ValidationOutcome] = await asyncio.gather(*[
                loop.run_in_executor(executor, _validate_record_sync, raw_record, validation_mode)
                for raw_record in raw_records
            ])

        # Phase 2: sequential DB writes + result accumulation.
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
                await self.repository.mark_raw_validated(
                    raw_record.raw_id,
                    status=outcome.validation_status,
                    error=outcome.validation_error,
                    drift_count=outcome.drift_count,
                    provider=outcome.canonical_provider,
                    mode=validation_mode,
                    payload_provider=outcome.payload_provider,
                )
                if not outcome.parseable and outcome.validation_error is not None:
                    await self.repository.mark_raw_parsed(
                        raw_record.raw_id,
                        error=outcome.validation_error,
                        payload_provider=outcome.payload_provider,
                    )

            if progress_callback is not None:
                progress_callback(
                    1,
                    desc=self._validation_progress_desc(progress_offset + index, total),
                )

        return result
