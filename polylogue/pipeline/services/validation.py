"""Schema validation service for raw conversation payloads.

This service implements a dedicated VALIDATE stage between ACQUIRE and PARSE.
It validates raw payloads against provider schemas and reports drift/invalid data.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from polylogue.logging import get_logger
from polylogue.lib.provider_identity import canonical_runtime_provider
from polylogue.lib.raw_payload import build_raw_payload_envelope
from polylogue.protocols import ProgressCallback
from polylogue.storage.store import RawConversationRecord

if TYPE_CHECKING:
    from polylogue.storage.backends.async_sqlite import SQLiteBackend

logger = get_logger(__name__)

__all__ = ["ValidationService", "ValidateResult"]


class ValidateResult:
    """Result of validating a set of raw records."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {
            "validated": 0,
            "invalid": 0,
            "drift": 0,
            "skipped_no_schema": 0,
            "errors": 0,
        }
        # Raw records that can proceed to parser dispatch.
        self.parseable_raw_ids: list[str] = []
        # Raw records that failed validation checks.
        self.invalid_raw_ids: list[str] = []
        # Provider -> number of payloads with drift warnings.
        self.drift_counts: dict[str, int] = {}

    def merge(self, other: ValidateResult) -> None:
        """Accumulate another validation result into this one."""
        for key, value in other.counts.items():
            self.counts[key] += value
        self.parseable_raw_ids.extend(other.parseable_raw_ids)
        self.invalid_raw_ids.extend(other.invalid_raw_ids)
        for provider, count in other.drift_counts.items():
            self.drift_counts[provider] = self.drift_counts.get(provider, 0) + count


class ValidationService:
    """Validate raw payloads against provider schemas."""

    SCHEMA_VALIDATION_MODE_ENV = "POLYLOGUE_SCHEMA_VALIDATION"
    SCHEMA_VALIDATION_DEFAULT = "strict"
    SCHEMA_VALIDATION_MODES = frozenset({"off", "advisory", "strict"})

    # Keep batches aligned with parse batching.
    RAW_BATCH_SIZE = 50

    def __init__(self, backend: SQLiteBackend):
        self.backend = backend

    def _schema_validation_mode(self) -> str:
        """Return configured schema validation mode."""
        raw = os.environ.get(self.SCHEMA_VALIDATION_MODE_ENV, self.SCHEMA_VALIDATION_DEFAULT)
        mode = raw.strip().lower()
        if mode in self.SCHEMA_VALIDATION_MODES:
            return mode
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
        if validation_mode == "off":
            result = ValidateResult()
            for index, raw_id in enumerate(raw_ids, start=1):
                if persist:
                    await self.backend.mark_raw_validated(
                        raw_id,
                        status="skipped",
                        mode=validation_mode,
                    )
                result.parseable_raw_ids.append(raw_id)
                if progress_callback is not None:
                    progress_callback(1, desc=self._validation_progress_desc(index, total_raw_ids))
            return result

        result = ValidateResult()
        for batch_start in range(0, len(raw_ids), self.RAW_BATCH_SIZE):
            batch_ids = raw_ids[batch_start : batch_start + self.RAW_BATCH_SIZE]
            raw_records = await self.backend.get_raw_conversations_batch(batch_ids)
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
                result.counts["errors"] += 1
                result.invalid_raw_ids.append(raw_id)
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
        mode: str | None = None,
        progress_total: int | None = None,
        progress_offset: int = 0,
    ) -> ValidateResult:
        """Evaluate raw records using the canonical validation logic."""
        from polylogue.schemas.validator import SchemaValidator

        result = ValidateResult()
        if not raw_records:
            return result

        validation_mode = mode or self._schema_validation_mode()
        if validation_mode == "off":
            for raw_record in raw_records:
                if persist:
                    await self.backend.mark_raw_validated(
                        raw_record.raw_id,
                        status="skipped",
                        mode=validation_mode,
                    )
                result.parseable_raw_ids.append(raw_record.raw_id)
                if progress_callback is not None:
                    total = progress_total or len(raw_records)
                    progress_callback(
                        1,
                        desc=self._validation_progress_desc(progress_offset + len(result.parseable_raw_ids), total),
                    )
            return result

        total = progress_total or len(raw_records)
        for index, raw_record in enumerate(raw_records, start=1):
            raw_id = raw_record.raw_id
            validation_status = "passed"
            validation_error: str | None = None
            parseable = True
            stored_payload_provider = getattr(raw_record, "payload_provider", None)
            if not isinstance(stored_payload_provider, str) or not stored_payload_provider.strip():
                stored_payload_provider = None
            canonical_provider = canonical_runtime_provider(
                stored_payload_provider or raw_record.provider_name,
                preserve_unknown=True,
                default=stored_payload_provider or raw_record.provider_name,
            )
            invalid_count = 0
            drift_count = 0
            collected_errors: list[str] = []
            collected_drift: list[str] = []
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
                result.counts["errors"] += 1
                result.invalid_raw_ids.append(raw_id)
                validation_status = "failed"
                validation_error = f"Unable to decode payload: {exc}"
                parseable = False
                payload = None
                malformed_lines = 0

            validator = None
            if payload is not None:
                if malformed_lines:
                    malformed_error = f"Malformed JSONL lines: {malformed_lines}"
                    if validation_mode == "strict":
                        result.counts["invalid"] += 1
                        validation_status = "failed"
                        validation_error = malformed_error
                        parseable = False
                    else:
                        logger.warning(
                            "Malformed JSONL lines ignored in advisory mode",
                            raw_id=raw_id,
                            provider=raw_record.provider_name,
                            malformed_lines=malformed_lines,
                        )

                try:
                    validator = SchemaValidator.for_provider(envelope.provider)
                except (FileNotFoundError, ImportError):
                    result.counts["skipped_no_schema"] += 1

            if parseable and validator is not None and payload is not None:
                samples = validator.validation_samples(payload)

                if samples:
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
                    result.counts["invalid"] += 1
                    logger.warning(
                        "Schema validation errors for %s",
                        canonical_provider,
                        raw_id=raw_id,
                        samples=len(samples),
                        invalid_samples=invalid_count,
                        errors=collected_errors[:5],
                    )
                    if validation_mode == "strict":
                        first_error = collected_errors[0] if collected_errors else "unknown schema validation error"
                        validation_status = "failed"
                        validation_error = f"Schema validation failed for {canonical_provider}: {first_error}"
                        parseable = False
                else:
                    result.counts["validated"] += 1

                if drift_count:
                    result.counts["drift"] += 1
                    result.drift_counts[canonical_provider] = result.drift_counts.get(canonical_provider, 0) + 1
                    logger.info(
                        "Schema drift detected for %s",
                        canonical_provider,
                        raw_id=raw_id,
                        samples=len(samples),
                        drift_samples=drift_count,
                        drift=collected_drift[:10],
                    )
            elif payload is not None and parseable:
                validation_status = "skipped"

            if persist:
                await self.backend.mark_raw_validated(
                    raw_id,
                    status=validation_status,
                    error=validation_error,
                    drift_count=drift_count,
                    provider=canonical_provider,
                    mode=validation_mode,
                    payload_provider=payload_provider,
                )

            if parseable:
                result.parseable_raw_ids.append(raw_id)
            else:
                result.invalid_raw_ids.append(raw_id)
                if persist and validation_error is not None:
                    await self.backend.mark_raw_parsed(
                        raw_id,
                        error=validation_error,
                        payload_provider=payload_provider,
                    )

            if progress_callback is not None:
                progress_callback(
                    1,
                    desc=self._validation_progress_desc(progress_offset + index, total),
                )

        return result
