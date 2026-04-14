"""Schema verification workflow over the raw artifact corpus."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polylogue.lib.provider_identity import CORE_RUNTIME_PROVIDERS
from polylogue.lib.raw_payload import build_raw_payload_envelope
from polylogue.schemas.validator import SchemaValidator
from polylogue.storage.blob_store import get_blob_store

from .verification_models import ProviderSchemaVerification, SchemaVerificationReport
from .verification_requests import SchemaVerificationRequest, bounded_window


def _format_malformed_jsonl_error(*, malformed_lines: int, malformed_detail: str | None) -> str:
    message = f"Malformed JSONL lines: {malformed_lines}"
    if malformed_detail:
        return f"{message} (first bad {malformed_detail})"
    return message

# ---------------------------------------------------------------------------
# Row iteration helpers
# ---------------------------------------------------------------------------


def verification_provider_clause(providers: list[str]) -> tuple[str, tuple[Any, ...]]:
    provider_placeholders = ",".join("?" for _ in providers)
    runtime_placeholders = ",".join("?" for _ in CORE_RUNTIME_PROVIDERS)
    clause = (
        f"payload_provider IN ({provider_placeholders}) "
        f"OR (payload_provider IS NULL AND provider_name IN ({provider_placeholders})) "
        f"OR (payload_provider IS NULL AND provider_name NOT IN ({runtime_placeholders}))"
    )
    params: tuple[Any, ...] = (*providers, *providers, *CORE_RUNTIME_PROVIDERS)
    return clause, params


def iter_verification_rows(
    conn: sqlite3.Connection,
    *,
    providers: list[str] | None,
    record_limit: int | None,
    record_offset: int,
) -> tuple[int | None, int, Iterator[sqlite3.Row]]:
    bounded_limit, bounded_offset = bounded_window(record_limit, record_offset)
    provider_where = ""
    where_params: tuple[Any, ...] = ()
    if providers:
        provider_where, where_params = verification_provider_clause(providers)

    def rows() -> Iterator[sqlite3.Row]:
        batch_size_limit = 50
        last_rowid = 0

        if bounded_offset > 0:
            offset_query = "SELECT rowid FROM raw_conversations "
            if provider_where:
                offset_query += f"WHERE {provider_where} "
            offset_query += "ORDER BY rowid LIMIT 1 OFFSET ?"
            row = conn.execute(offset_query, (*where_params, bounded_offset - 1)).fetchone()
            if row is None:
                return
            last_rowid = row[0]

        base_query = "SELECT rowid, raw_id, provider_name, payload_provider, source_path FROM raw_conversations "
        records_fetched = 0
        while True:
            if bounded_limit is not None:
                remaining = bounded_limit - records_fetched
                if remaining <= 0:
                    break
                batch_size = min(batch_size_limit, remaining)
            else:
                batch_size = batch_size_limit

            if provider_where:
                query = base_query + f"WHERE rowid > ? AND ({provider_where}) ORDER BY rowid LIMIT ?"
                params: tuple[Any, ...] = (last_rowid, *where_params, batch_size)
            else:
                query = base_query + "WHERE rowid > ? ORDER BY rowid LIMIT ?"
                params = (last_rowid, batch_size)

            batch = conn.execute(query, params).fetchall()
            if not batch:
                break

            last_rowid = batch[-1]["rowid"]
            records_fetched += len(batch)
            for row in batch:
                yield row

    return bounded_limit, bounded_offset, rows()


def resolve_candidate_provider(row: sqlite3.Row) -> tuple[str, str | None]:
    raw_provider = str(row["provider_name"])
    stored_payload_provider = row["payload_provider"]
    return str(stored_payload_provider or raw_provider), stored_payload_provider


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------


def apply_quarantine_updates(
    conn: sqlite3.Connection,
    *,
    updates: list[tuple[str, str, str, str | None]],
) -> None:
    validated_at = datetime.now(tz=timezone.utc).isoformat()
    for raw_id, reason, provider, payload_provider in updates:
        conn.execute(
            """
            UPDATE raw_conversations
            SET validation_status = 'failed',
                validation_error = ?,
                validation_drift_count = 0,
                validation_provider = ?,
                validation_mode = 'strict',
                validated_at = ?,
                payload_provider = COALESCE(?, payload_provider)
            WHERE raw_id = ?
            """,
            (reason, provider, validated_at, payload_provider, raw_id),
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
        blob_store = get_blob_store()
        for row in rows:
            candidate_provider, stored_payload_provider = resolve_candidate_provider(row)
            raw_provider = str(row["provider_name"])

            raw_id = str(row["raw_id"])
            raw_source = blob_store.blob_path(raw_id)

            try:
                envelope = build_raw_payload_envelope(
                    raw_source,
                    source_path=str(row["source_path"] or ""),
                    fallback_provider=raw_provider,
                    payload_provider=stored_payload_provider,
                    jsonl_dict_only=True,
                )
                payload = envelope.payload
                malformed_lines = envelope.malformed_jsonl_lines
                malformed_detail = envelope.malformed_jsonl_detail
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
                    reason = f"Unable to decode payload: {exc}"
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
                    reason = _format_malformed_jsonl_error(
                        malformed_lines=malformed_lines,
                        malformed_detail=malformed_detail,
                    )
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


__all__ = [
    "SchemaValidator",
    "apply_quarantine_updates",
    "iter_verification_rows",
    "resolve_candidate_provider",
    "verification_provider_clause",
    "verify_raw_corpus",
]
