"""Raw session persistence helpers."""

from __future__ import annotations

import hashlib

import aiosqlite

from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.archive_tiers.write import _timestamp_ms


async def save_raw_session(
    conn: aiosqlite.Connection,
    record: RawSessionRecord,
    transaction_depth: int,
) -> bool:
    # payload_provider wins when the payload has been classified; otherwise fall
    # back to the source_name token (#1743 collapses both onto origin).
    if record.payload_provider is not None:
        origin = origin_from_provider(record.payload_provider)
    else:
        origin = origin_from_provider(Provider.from_string(record.source_name or "unknown"))
    blob_hash_hex = record.blob_hash or record.raw_id
    try:
        blob_hash = bytes.fromhex(blob_hash_hex)
    except ValueError:
        blob_hash = blob_hash_hex.encode("utf-8")
    if len(blob_hash) != 32:
        blob_hash = hashlib.sha256(blob_hash).digest()

    acquired_at_ms = _timestamp_ms(record.acquired_at) or 0
    cursor = await conn.execute(
        """
        INSERT OR IGNORE INTO raw_sessions (
            raw_id, origin, native_id, source_path, source_index, blob_hash,
            blob_size, acquired_at_ms, file_mtime_ms, parsed_at_ms, parse_error,
            validated_at_ms, validation_status, validation_error, validation_drift_count,
            validation_mode, detection_warnings_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.raw_id,
            origin.value,
            None,
            record.source_path,
            int(record.source_index or 0),
            blob_hash,
            int(record.blob_size),
            acquired_at_ms,
            _timestamp_ms(record.file_mtime),
            _timestamp_ms(record.parsed_at),
            record.parse_error,
            _timestamp_ms(record.validated_at),
            record.validation_status.value if record.validation_status is not None else None,
            record.validation_error,
            int(record.validation_drift_count or 0),
            record.validation_mode.value if record.validation_mode is not None else None,
            record.detection_warnings or "[]",
        ),
    )
    inserted = bool(cursor.rowcount > 0)

    if not inserted and record.file_mtime is not None:
        file_mtime_ms = _timestamp_ms(record.file_mtime)
        await conn.execute(
            "UPDATE raw_sessions SET file_mtime_ms = ?, source_path = ? "
            "WHERE raw_id = ? AND (file_mtime_ms IS NOT ? OR source_path IS NOT ?)",
            (file_mtime_ms, record.source_path, record.raw_id, file_mtime_ms, record.source_path),
        )

    await conn.execute(
        "DELETE FROM blob_publication_reservations WHERE blob_hash = ?",
        (blob_hash,),
    )
    if transaction_depth == 0:
        await conn.commit()
    return inserted


__all__ = ["save_raw_session"]
