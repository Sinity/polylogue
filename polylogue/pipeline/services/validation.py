"""Schema validation service for raw conversation payloads.

This service implements a dedicated VALIDATE stage between ACQUIRE and PARSE.
It validates raw payloads against provider schemas and reports drift/invalid data.
"""

from __future__ import annotations

import os
from typing import Any

from polylogue.lib.log import get_logger
from polylogue.lib.raw_payload import decode_raw_payload, infer_payload_provider

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


class ValidationService:
    """Validate raw payloads against provider schemas."""

    SCHEMA_VALIDATION_MODE_ENV = "POLYLOGUE_SCHEMA_VALIDATION"
    SCHEMA_VALIDATION_DEFAULT = "strict"
    SCHEMA_VALIDATION_MODES = frozenset({"off", "advisory", "strict"})
    SCHEMA_VALIDATION_MAX_SAMPLES_ENV = "POLYLOGUE_SCHEMA_VALIDATION_MAX_SAMPLES"
    SCHEMA_VALIDATION_MAX_SAMPLES_DEFAULT = 16

    # Keep batches aligned with parse batching.
    RAW_BATCH_SIZE = 200

    def __init__(self, backend: Any):
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

    def _schema_validation_max_samples(self, payload: Any) -> int:
        """Return sample count for record-style payload validation."""
        raw = os.environ.get(self.SCHEMA_VALIDATION_MAX_SAMPLES_ENV)
        if raw is None:
            return self.SCHEMA_VALIDATION_MAX_SAMPLES_DEFAULT

        value = raw.strip().lower()
        if value in {"all", "0"}:
            if isinstance(payload, list):
                return max(1, sum(1 for item in payload if isinstance(item, dict)))
            return 1

        try:
            parsed = int(value)
            if parsed > 0:
                return parsed
        except ValueError:
            pass

        logger.warning(
            "Invalid %s=%r, falling back to %d",
            self.SCHEMA_VALIDATION_MAX_SAMPLES_ENV,
            raw,
            self.SCHEMA_VALIDATION_MAX_SAMPLES_DEFAULT,
        )
        return self.SCHEMA_VALIDATION_MAX_SAMPLES_DEFAULT

    async def validate_raw_ids(
        self,
        *,
        raw_ids: list[str],
        progress_callback: Any | None = None,
    ) -> ValidateResult:
        """Validate raw records and return parseable/invalid record IDs."""
        from polylogue.schemas.validator import SchemaValidator

        result = ValidateResult()
        if not raw_ids:
            return result

        mode = self._schema_validation_mode()
        if mode == "off":
            for raw_id in raw_ids:
                await self.backend.mark_raw_validated(
                    raw_id,
                    status="skipped",
                    mode=mode,
                )
                result.parseable_raw_ids.append(raw_id)
            return result

        if progress_callback is not None:
            progress_callback(0, desc=f"Validating ({len(raw_ids):,} raw)")

        for batch_start in range(0, len(raw_ids), self.RAW_BATCH_SIZE):
            batch_ids = raw_ids[batch_start : batch_start + self.RAW_BATCH_SIZE]
            raw_records = await self.backend.get_raw_conversations_batch(batch_ids)
            records_by_id = {record.raw_id: record for record in raw_records}

            for raw_id in batch_ids:
                raw_record = records_by_id.get(raw_id)
                if raw_record is None:
                    result.counts["errors"] += 1
                    result.invalid_raw_ids.append(raw_id)
                    if progress_callback is not None:
                        progress_callback(1)
                    continue

                validation_status = "passed"
                validation_error: str | None = None
                parseable = True
                canonical_provider = raw_record.provider_name
                invalid_count = 0
                drift_count = 0
                collected_errors: list[str] = []
                collected_drift: list[str] = []

                try:
                    decoded = decode_raw_payload(raw_record.raw_content)
                    payload = decoded.payload
                    malformed_lines = decoded.malformed_jsonl_lines
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
                        if mode == "strict":
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

                    provider = infer_payload_provider(
                        payload,
                        source_path=raw_record.source_path,
                        fallback_provider=raw_record.provider_name,
                    )

                    try:
                        validator = SchemaValidator.for_provider(provider)
                    except (FileNotFoundError, ImportError):
                        result.counts["skipped_no_schema"] += 1

                if parseable and validator is not None and payload is not None:
                    max_samples = self._schema_validation_max_samples(payload)
                    samples = validator.validation_samples(payload, max_samples=max_samples)

                    if samples:
                        for sample in samples:
                            sample_result = validator.validate(sample)
                            if not sample_result.is_valid:
                                invalid_count += 1
                                collected_errors.extend(sample_result.errors[:2])
                            if sample_result.has_drift:
                                drift_count += 1
                                collected_drift.extend(sample_result.drift_warnings[:3])

                    canonical_provider = validator.provider or provider

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
                        if mode == "strict":
                            first_error = collected_errors[0] if collected_errors else "unknown schema validation error"
                            validation_status = "failed"
                            validation_error = (
                                f"Schema validation failed for {canonical_provider}: {first_error}"
                            )
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
                    # No schema available for this provider. Keep parseable and
                    # record skipped validation state.
                    validation_status = "skipped"

                await self.backend.mark_raw_validated(
                    raw_id,
                    status=validation_status,
                    error=validation_error,
                    drift_count=drift_count,
                    provider=canonical_provider,
                    mode=mode,
                )

                if parseable:
                    result.parseable_raw_ids.append(raw_id)
                else:
                    result.invalid_raw_ids.append(raw_id)
                    if validation_error is not None:
                        # Surface strict/decode failures in parse_error for existing
                        # operator tooling that inspects parse tracking.
                        await self.backend.mark_raw_parsed(raw_id, error=validation_error)

                if progress_callback is not None:
                    progress_callback(1)

        return result
