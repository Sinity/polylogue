"""CPU-bound raw validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from polylogue.archive.raw_payload import RawPayloadEnvelope, build_raw_payload_envelope
from polylogue.logging import get_logger
from polylogue.schemas.validator import SchemaValidator
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.runtime import RawConversationRecord
from polylogue.types import Provider, ValidationMode, ValidationStatus

logger = get_logger(__name__)


ValidationCounts = dict[str, int]


def _normalize_payload_provider(raw_record: RawConversationRecord) -> str | None:
    stored_payload_provider = getattr(raw_record, "payload_provider", None)
    if not isinstance(stored_payload_provider, str):
        return None
    candidate = stored_payload_provider.strip()
    if not candidate:
        return None
    return candidate


def _initial_counts() -> ValidationCounts:
    return {
        "validated": 0,
        "invalid": 0,
        "drift": 0,
        "skipped_no_schema": 0,
        "errors": 0,
    }


def _build_validation_envelope(
    raw_source: Path,
    raw_record: RawConversationRecord,
    payload_provider: str | None,
) -> RawPayloadEnvelope:
    return build_raw_payload_envelope(
        raw_source,
        source_path=raw_record.source_path,
        fallback_provider=raw_record.provider_name,
        payload_provider=payload_provider,
    )


def _validator_for_payload(
    envelope: RawPayloadEnvelope,
    *,
    source_path: str | None,
) -> SchemaValidator | None:
    try:
        return SchemaValidator.for_payload(
            envelope.provider,
            envelope.payload,
            source_path=source_path,
        )
    except (FileNotFoundError, ImportError):
        return None


def _format_malformed_jsonl_error(*, malformed_lines: int, malformed_detail: str | None) -> str:
    message = f"Malformed JSONL lines: {malformed_lines}"
    if malformed_detail:
        return f"{message} (first bad {malformed_detail})"
    return message


@dataclass
class _ValidationOutcome:
    """Thread-safe result of per-record CPU-bound validation work."""

    validation_status: ValidationStatus
    validation_error: str | None
    parse_error: str | None
    parseable: bool
    canonical_provider: Provider
    payload_provider: Provider | None
    drift_count: int
    counts_delta: dict[str, int] = field(default_factory=dict)
    drift_counts_delta: dict[str, int] = field(default_factory=dict)


def _validate_record_sync(
    raw_record: RawConversationRecord,
    validation_mode: ValidationMode,
    blob_root_str: str,
) -> _ValidationOutcome:
    """Run CPU-bound validation for a single raw record."""
    import time as _time

    t_start = _time.perf_counter()
    counts_delta: ValidationCounts = _initial_counts()
    drift_counts_delta: dict[str, int] = {}

    stored_payload_provider = _normalize_payload_provider(raw_record)
    canonical_provider = Provider.from_string(stored_payload_provider or raw_record.provider_name)
    payload_provider: Provider | None = (
        Provider.from_string(stored_payload_provider) if stored_payload_provider else None
    )

    blob_store = BlobStore(Path(blob_root_str))
    raw_source = blob_store.blob_path(raw_record.raw_id)

    try:
        envelope = _build_validation_envelope(
            raw_source,
            raw_record=raw_record,
            payload_provider=stored_payload_provider,
        )
        payload = envelope.payload
        malformed_lines = envelope.malformed_jsonl_lines
        malformed_detail = envelope.malformed_jsonl_detail
        payload_provider = envelope.provider
    except Exception as exc:
        counts_delta["errors"] += 1
        return _ValidationOutcome(
            validation_status=ValidationStatus.FAILED,
            validation_error=f"Unable to decode payload: {exc}",
            parse_error=f"Unable to decode payload: {exc}",
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
            parse_error=None,
            parseable=False,
            canonical_provider=canonical_provider,
            payload_provider=payload_provider,
            drift_count=0,
            counts_delta=counts_delta,
        )

    if malformed_lines:
        malformed_error = _format_malformed_jsonl_error(
            malformed_lines=malformed_lines,
            malformed_detail=malformed_detail,
        )
        if validation_mode is ValidationMode.STRICT:
            counts_delta["invalid"] += 1
            return _ValidationOutcome(
                validation_status=ValidationStatus.FAILED,
                validation_error=malformed_error,
                parse_error=malformed_error,
                parseable=False,
                canonical_provider=canonical_provider,
                payload_provider=payload_provider,
                drift_count=0,
                counts_delta=counts_delta,
            )
        logger.warning(
            "Malformed JSONL lines ignored in advisory mode",
            raw_id=raw_record.raw_id,
            provider=raw_record.provider_name,
            malformed_lines=malformed_lines,
        )

    validator = _validator_for_payload(
        envelope,
        source_path=raw_record.source_path,
    )
    if validator is None:
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

    total_elapsed = _time.perf_counter() - t_start
    if total_elapsed > 1.0:
        logger.info(
            "slow_validate",
            raw_id=raw_record.raw_id[:16],
            elapsed_s=round(total_elapsed, 2),
            blob_mb=round(raw_record.blob_size / (1024 * 1024), 1),
            provider=raw_record.provider_name,
            status=str(validation_status),
        )

    return _ValidationOutcome(
        validation_status=validation_status,
        validation_error=validation_error,
        parse_error=None,
        parseable=parseable,
        canonical_provider=canonical_provider,
        payload_provider=payload_provider,
        drift_count=drift_count,
        counts_delta=counts_delta,
        drift_counts_delta=drift_counts_delta,
    )


__all__ = ["_ValidationOutcome", "_validate_record_sync"]
