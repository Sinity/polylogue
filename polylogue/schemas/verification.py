"""Schema verification workflow for full raw-corpus checks.

This module provides a non-mutating verification pass over ``raw_conversations``
so operators can run explicit schema gates before schema promotion/releases.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polylogue.lib.raw_payload import build_raw_payload_envelope
from polylogue.schemas.validator import SchemaValidator
from polylogue.storage.backends.connection import default_db_path


@dataclass
class ProviderSchemaVerification:
    """Per-provider schema verification summary."""

    provider: str
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    drift_records: int = 0
    skipped_no_schema: int = 0
    decode_errors: int = 0
    quarantined_records: int = 0

    def to_dict(self) -> dict[str, int | str]:
        return {
            "provider": self.provider,
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "drift_records": self.drift_records,
            "skipped_no_schema": self.skipped_no_schema,
            "decode_errors": self.decode_errors,
            "quarantined_records": self.quarantined_records,
        }


@dataclass
class SchemaVerificationReport:
    """Aggregate report for schema verification over raw corpus."""

    providers: dict[str, ProviderSchemaVerification]
    max_samples: int | None
    total_records: int
    record_limit: int | None = None
    record_offset: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_samples": self.max_samples if self.max_samples is not None else "all",
            "record_limit": self.record_limit if self.record_limit is not None else "all",
            "record_offset": self.record_offset,
            "total_records": self.total_records,
            "providers": {
                provider: stats.to_dict() for provider, stats in sorted(self.providers.items())
            },
        }


def _max_samples_for_payload(payload: Any, configured_max_samples: int | None) -> int | None:
    if configured_max_samples is None:
        # ``None`` means "all representative dict records" and is handled by
        # ``SchemaValidator.validation_samples`` without a pre-scan pass.
        return None
    return configured_max_samples


def verify_raw_corpus(
    *,
    db_path: Path | None = None,
    providers: list[str] | None = None,
    max_samples: int | None = 16,
    record_limit: int | None = None,
    record_offset: int = 0,
    quarantine_malformed: bool = False,
) -> SchemaVerificationReport:
    """Run non-mutating schema verification over ``raw_conversations``.

    Args:
        db_path: Optional SQLite path. Defaults to polylogue default DB.
        providers: Optional filter by DB provider_name values.
        max_samples: Per-record sample count. ``None`` means validate all dict
            records from list payloads (equivalent to ``all``).
        quarantine_malformed: When true, malformed/decode-failed raw rows are
            marked as failed validation and given parse_error context.
    """
    db_path = db_path or default_db_path()
    if not db_path.exists():
        return SchemaVerificationReport(
            providers={},
            max_samples=max_samples,
            total_records=0,
            record_limit=record_limit,
            record_offset=max(0, record_offset),
        )

    stats_by_provider: dict[str, ProviderSchemaVerification] = {}
    total_records = 0
    bounded_offset = max(0, int(record_offset))
    bounded_limit = max(1, int(record_limit)) if record_limit is not None else None

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        params: list[Any] = []
        query = (
            "SELECT raw_id, provider_name, source_path, raw_content "
            "FROM raw_conversations "
        )
        if providers:
            placeholders = ",".join("?" for _ in providers)
            query += f"WHERE provider_name IN ({placeholders}) "
            params.extend(providers)
        query += "ORDER BY acquired_at DESC "
        if bounded_limit is not None:
            query += "LIMIT ? OFFSET ?"
            params.extend([bounded_limit, bounded_offset])

        cursor = conn.execute(query, tuple(params))
        quarantine_updates: list[tuple[str, str, str]] = []
        while True:
            rows = cursor.fetchmany(250)
            if not rows:
                break
            for row in rows:
                total_records += 1
                raw_provider = str(row["provider_name"])
                provider_stats = stats_by_provider.setdefault(
                    raw_provider,
                    ProviderSchemaVerification(provider=raw_provider),
                )
                provider_stats.total_records += 1

                try:
                    envelope = build_raw_payload_envelope(
                        row["raw_content"],
                        source_path=str(row["source_path"] or ""),
                        fallback_provider=raw_provider,
                        jsonl_dict_only=True,
                    )
                    payload = envelope.payload
                    malformed_lines = envelope.malformed_jsonl_lines
                except Exception as exc:
                    provider_stats.decode_errors += 1
                    if quarantine_malformed:
                        raw_id = str(row["raw_id"])
                        reason = f"Unable to decode payload: {type(exc).__name__}"
                        quarantine_updates.append((raw_id, reason, raw_provider))
                        provider_stats.quarantined_records += 1
                    continue
                if malformed_lines:
                    provider_stats.decode_errors += 1
                    if quarantine_malformed:
                        raw_id = str(row["raw_id"])
                        reason = f"Malformed JSONL lines: {malformed_lines}"
                        quarantine_updates.append((raw_id, reason, raw_provider))
                        provider_stats.quarantined_records += 1
                    continue

                try:
                    validator = SchemaValidator.for_provider(envelope.provider)
                except (FileNotFoundError, ImportError):
                    provider_stats.skipped_no_schema += 1
                    continue

                samples = validator.validation_samples(
                    payload,
                    max_samples=_max_samples_for_payload(payload, max_samples),
                )
                if not samples:
                    provider_stats.valid_records += 1
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

        if quarantine_updates:
            validated_at = datetime.now(tz=timezone.utc).isoformat()
            for raw_id, reason, provider in quarantine_updates:
                conn.execute(
                    """
                    UPDATE raw_conversations
                    SET validation_status = 'failed',
                        validation_error = ?,
                        validation_drift_count = 0,
                        validation_provider = ?,
                        validation_mode = 'strict',
                        validated_at = ?
                    WHERE raw_id = ?
                    """,
                    (reason, provider, validated_at, raw_id),
                )
                conn.execute(
                    """
                    UPDATE raw_conversations
                    SET parse_error = COALESCE(parse_error, ?)
                    WHERE raw_id = ? AND parsed_at IS NULL
                    """,
                    (reason, raw_id),
                )
            conn.commit()

    finally:
        conn.close()

    return SchemaVerificationReport(
        providers=stats_by_provider,
        max_samples=max_samples,
        total_records=total_records,
        record_limit=bounded_limit,
        record_offset=bounded_offset,
    )
