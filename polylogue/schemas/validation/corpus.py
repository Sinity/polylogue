"""Schema verification workflow over the raw artifact corpus."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeAlias

from polylogue.archive.raw_payload import build_raw_payload_envelope
from polylogue.core.common import format_malformed_jsonl_error as _format_malformed_jsonl_error
from polylogue.core.provider_identity import CORE_RUNTIME_PROVIDERS
from polylogue.schemas.validator import SchemaValidator
from polylogue.storage.blob_store import get_blob_store
from polylogue.storage.sqlite.connection_profile import open_connection

from .models import ProviderSchemaVerification, SchemaVerificationReport
from .requests import SchemaVerificationRequest, bounded_window

VerificationRow: TypeAlias = tuple[str, str, str | None, str]
VerificationSqlParam: TypeAlias = str | int
VerificationSqlParams: TypeAlias = tuple[VerificationSqlParam, ...]
VerificationUpdate: TypeAlias = tuple[str, str, str, str | None]


# ---------------------------------------------------------------------------
# Row iteration helpers
# ---------------------------------------------------------------------------


def verification_provider_clause(providers: list[str]) -> tuple[str, tuple[str, ...]]:
    provider_placeholders = ",".join("?" for _ in providers)
    runtime_placeholders = ",".join("?" for _ in CORE_RUNTIME_PROVIDERS)
    clause = (
        f"payload_provider IN ({provider_placeholders}) "
        f"OR (payload_provider IS NULL AND provider_name IN ({provider_placeholders})) "
        f"OR (payload_provider IS NULL AND provider_name NOT IN ({runtime_placeholders}))"
    )
    params: tuple[str, ...] = (*providers, *providers, *CORE_RUNTIME_PROVIDERS)
    return clause, params


def _row_payload_data(row: sqlite3.Row) -> VerificationRow:
    return (
        str(row["raw_id"]),
        str(row["provider_name"]),
        row["payload_provider"],
        str(row["source_path"] or ""),
    )


def iter_verification_rows(
    conn: sqlite3.Connection,
    *,
    providers: list[str] | None,
    record_limit: int | None,
    record_offset: int,
) -> tuple[int | None, int, Iterator[sqlite3.Row]]:
    bounded_limit, bounded_offset = bounded_window(record_limit, record_offset)
    provider_where = ""
    where_params: tuple[str, ...] = ()
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
                query_params: VerificationSqlParams = (last_rowid, *where_params, batch_size)
            else:
                query = base_query + "WHERE rowid > ? ORDER BY rowid LIMIT ?"
                query_params = (last_rowid, batch_size)

            batch = conn.execute(query, query_params).fetchall()
            if not batch:
                break

            last_rowid = batch[-1]["rowid"]
            records_fetched += len(batch)
            for row in batch:
                yield row

    return bounded_limit, bounded_offset, rows()


def resolve_candidate_provider(row: sqlite3.Row) -> tuple[str, str | None]:
    _, raw_provider, stored_payload_provider, _ = _row_payload_data(row)
    return str(stored_payload_provider or raw_provider), stored_payload_provider


def _provider_matches_filter(provider_filter: set[str], provider: str) -> bool:
    return not provider_filter or provider in provider_filter


def _report_progress(
    callback: Callable[[int], object] | None,
    steps: int = 1,
) -> None:
    if callback is not None:
        callback(steps)


def _track_selected_row(
    *,
    provider_filter: set[str],
    provider: str,
    total_records: int,
    stats_by_provider: dict[str, ProviderSchemaVerification],
) -> tuple[int, ProviderSchemaVerification] | None:
    if not _provider_matches_filter(provider_filter, provider):
        return None

    total_records += 1
    provider_stats = _provider_stats(stats_by_provider=stats_by_provider, provider=provider)
    provider_stats.total_records += 1
    return total_records, provider_stats


def _provider_stats(
    *, stats_by_provider: dict[str, ProviderSchemaVerification], provider: str
) -> ProviderSchemaVerification:
    return stats_by_provider.setdefault(provider, ProviderSchemaVerification(provider=provider))


def _record_decode_error(
    *,
    raw_id: str,
    provider: str,
    payload_provider: str | None,
    reason: str,
    stats_by_provider: dict[str, ProviderSchemaVerification],
    quarantine_updates: list[VerificationUpdate],
    quarantine_malformed: bool,
) -> None:
    provider_stats = _provider_stats(stats_by_provider=stats_by_provider, provider=provider)
    provider_stats.decode_errors += 1
    if quarantine_malformed:
        provider_stats.quarantined_records += 1
        quarantine_updates.append((raw_id, reason, provider, payload_provider))


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------


def apply_quarantine_updates(
    conn: sqlite3.Connection,
    *,
    updates: list[VerificationUpdate],
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

    conn = open_connection(db_path)
    conn.row_factory = sqlite3.Row
    try:
        quarantine_updates: list[VerificationUpdate] = []
        _ignored_limit, _ignored_offset, rows = iter_verification_rows(
            conn,
            providers=request.providers,
            record_limit=request.record_limit,
            record_offset=request.record_offset,
        )
        blob_store = get_blob_store()
        for row in rows:
            raw_id, raw_provider, stored_payload_provider, source_path = _row_payload_data(row)
            candidate_provider = str(stored_payload_provider or raw_provider)

            raw_source = blob_store.blob_path(raw_id)

            try:
                envelope = build_raw_payload_envelope(
                    raw_source,
                    source_path=source_path,
                    fallback_provider=raw_provider,
                    payload_provider=stored_payload_provider,
                    jsonl_dict_only=True,
                )
                payload = envelope.payload
                malformed_lines = envelope.malformed_jsonl_lines
                malformed_detail = envelope.malformed_jsonl_detail
            except Exception as exc:
                tracked_row = _track_selected_row(
                    provider_filter=provider_filter,
                    provider=candidate_provider,
                    total_records=total_records,
                    stats_by_provider=stats_by_provider,
                )
                if tracked_row is None:
                    continue
                total_records, provider_stats = tracked_row
                _record_decode_error(
                    raw_id=raw_id,
                    provider=candidate_provider,
                    payload_provider=stored_payload_provider,
                    reason=f"Unable to decode payload: {exc}",
                    stats_by_provider=stats_by_provider,
                    quarantine_updates=quarantine_updates,
                    quarantine_malformed=request.quarantine_malformed,
                )
                _report_progress(request.progress_callback)
                continue

            actual_provider = envelope.provider
            tracked_row = _track_selected_row(
                provider_filter=provider_filter,
                provider=actual_provider,
                total_records=total_records,
                stats_by_provider=stats_by_provider,
            )
            if tracked_row is None:
                continue
            total_records, provider_stats = tracked_row
            if not envelope.artifact.schema_eligible:
                provider_stats.skipped_no_schema += 1
                _report_progress(request.progress_callback)
                continue
            if malformed_lines:
                if request.quarantine_malformed:
                    _record_decode_error(
                        raw_id=raw_id,
                        provider=actual_provider,
                        payload_provider=actual_provider,
                        reason=_format_malformed_jsonl_error(
                            malformed_lines=malformed_lines,
                            malformed_detail=malformed_detail,
                        ),
                        stats_by_provider=stats_by_provider,
                        quarantine_updates=quarantine_updates,
                        quarantine_malformed=request.quarantine_malformed,
                    )
                else:
                    provider_stats.decode_errors += 1
                _report_progress(request.progress_callback)
                continue

            try:
                validator = SchemaValidator.for_payload(
                    actual_provider,
                    payload,
                    source_path=source_path,
                )
            except (FileNotFoundError, ImportError):
                provider_stats.skipped_no_schema += 1
                _report_progress(request.progress_callback)
                continue

            samples = validator.validation_samples(
                payload,
                max_samples=request.max_samples,
            )
            if not samples:
                provider_stats.valid_records += 1
                _report_progress(request.progress_callback)
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

            _report_progress(request.progress_callback)

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
