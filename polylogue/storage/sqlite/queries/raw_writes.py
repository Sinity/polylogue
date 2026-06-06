"""Raw session persistence helpers."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.runtime import RawSessionRecord


async def save_raw_session(
    conn: aiosqlite.Connection,
    record: RawSessionRecord,
    transaction_depth: int,
) -> bool:
    cursor = await conn.execute(
        """
        INSERT OR IGNORE INTO raw_sessions (
            raw_id,
            source_name,
            payload_provider,
            source_name,
            source_path,
            source_index,
            blob_size,
            acquired_at,
            file_mtime,
            parsed_at,
            parse_error,
            validated_at,
            validation_status,
            validation_error,
            validation_drift_count,
            validation_provider,
            validation_mode,
            detection_warnings
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.raw_id,
            record.source_name,
            record.payload_provider,
            record.source_name,
            record.source_path,
            record.source_index,
            record.blob_size,
            record.acquired_at,
            record.file_mtime,
            record.parsed_at,
            record.parse_error,
            record.validated_at,
            record.validation_status,
            record.validation_error,
            record.validation_drift_count,
            record.validation_provider,
            record.validation_mode,
            record.detection_warnings,
        ),
    )
    inserted = bool(cursor.rowcount > 0)

    if not inserted and record.file_mtime is not None:
        await conn.execute(
            "UPDATE raw_sessions SET file_mtime = ?, source_path = ? "
            "WHERE raw_id = ? AND (file_mtime IS NOT ? OR source_path IS NOT ?)",
            (record.file_mtime, record.source_path, record.raw_id, record.file_mtime, record.source_path),
        )

    if transaction_depth == 0:
        await conn.commit()
    return inserted


__all__ = ["save_raw_session"]
