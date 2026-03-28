"""Schema verification workflow over the raw artifact corpus."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.lib.raw_payload import build_raw_payload_envelope
from polylogue.schemas.validator import SchemaValidator

from .verification_corpus_rows import candidate_provider as resolve_candidate_provider
from .verification_corpus_rows import iter_verification_rows
from .verification_corpus_runtime import apply_quarantine_updates
from .verification_models import ProviderSchemaVerification, SchemaVerificationReport
from .verification_requests import SchemaVerificationRequest
from .verification_support import bounded_window


def verify_raw_corpus(
    *,
    db_path: Path,
    request: SchemaVerificationRequest,
) -> SchemaVerificationReport:
    """Run non-mutating schema verification over ``raw_conversations``."""
    bounded_limit, bounded_offset = bounded_window(request.record_limit, request.record_offset)
    if not db_path.exists():
        return SchemaVerificationReport(
            providers={},
            max_samples=request.max_samples,
            total_records=0,
            record_limit=bounded_limit,
            record_offset=bounded_offset,
        )

    stats_by_provider: dict[str, ProviderSchemaVerification] = {}
    total_records = 0
    provider_filter = set(request.providers or [])

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        quarantine_updates: list[tuple[str, str, str, str | None]] = []
        _ignored_limit, _ignored_offset, rows = iter_verification_rows(
            conn,
            providers=request.providers,
            record_limit=request.record_limit,
            record_offset=request.record_offset,
        )
        for row in rows:
            candidate_provider, stored_payload_provider = resolve_candidate_provider(row)
            raw_provider = str(row["provider_name"])

            try:
                envelope = build_raw_payload_envelope(
                    row["raw_content"],
                    source_path=str(row["source_path"] or ""),
                    fallback_provider=raw_provider,
                    payload_provider=stored_payload_provider,
                    jsonl_dict_only=True,
                )
                payload = envelope.payload
                malformed_lines = envelope.malformed_jsonl_lines
            except Exception as exc:
                if provider_filter and candidate_provider not in provider_filter:
                    continue
                total_records += 1
                provider_stats = stats_by_provider.setdefault(
                    candidate_provider,
                    ProviderSchemaVerification(provider=candidate_provider),
                )
                provider_stats.total_records += 1
                provider_stats.decode_errors += 1
                if request.quarantine_malformed:
                    raw_id = str(row["raw_id"])
                    reason = f"Unable to decode payload: {type(exc).__name__}"
                    quarantine_updates.append((raw_id, reason, candidate_provider, stored_payload_provider))
                    provider_stats.quarantined_records += 1
                if request.progress_callback is not None:
                    request.progress_callback(1)
                continue

            actual_provider = envelope.provider
            if provider_filter and actual_provider not in provider_filter:
                continue
            total_records += 1
            provider_stats = stats_by_provider.setdefault(
                actual_provider,
                ProviderSchemaVerification(provider=actual_provider),
            )
            provider_stats.total_records += 1
            if not envelope.artifact.schema_eligible:
                provider_stats.skipped_no_schema += 1
                if request.progress_callback is not None:
                    request.progress_callback(1)
                continue
            if malformed_lines:
                provider_stats.decode_errors += 1
                if request.quarantine_malformed:
                    raw_id = str(row["raw_id"])
                    reason = f"Malformed JSONL lines: {malformed_lines}"
                    quarantine_updates.append((raw_id, reason, actual_provider, actual_provider))
                    provider_stats.quarantined_records += 1
                if request.progress_callback is not None:
                    request.progress_callback(1)
                continue

            try:
                validator = SchemaValidator.for_payload(
                    actual_provider,
                    payload,
                    source_path=str(row["source_path"] or ""),
                )
            except (FileNotFoundError, ImportError):
                provider_stats.skipped_no_schema += 1
                if request.progress_callback is not None:
                    request.progress_callback(1)
                continue

            samples = validator.validation_samples(
                payload,
                max_samples=request.max_samples,
            )
            if not samples:
                provider_stats.valid_records += 1
                if request.progress_callback is not None:
                    request.progress_callback(1)
                continue

            invalid_found = False
            drift_found = False
            for sample in samples:
                result = validator.validate(sample)
                if not result.is_valid:
                    invalid_found = True
                if result.has_drift:
                    drift_found = True

            if invalid_found:
                provider_stats.invalid_records += 1
            else:
                provider_stats.valid_records += 1
            if drift_found:
                provider_stats.drift_records += 1

            if request.progress_callback is not None:
                request.progress_callback(1)

        if quarantine_updates:
            apply_quarantine_updates(conn, updates=quarantine_updates)
    finally:
        conn.close()

    return SchemaVerificationReport(
        providers=stats_by_provider,
        max_samples=request.max_samples,
        total_records=total_records,
        record_limit=bounded_limit,
        record_offset=bounded_offset,
    )


__all__ = ["SchemaValidator", "verify_raw_corpus"]
