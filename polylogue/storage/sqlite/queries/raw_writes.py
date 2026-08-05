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
    # Only the acquisition path can assert a capture mode.  A hydrated legacy
    # row has ``None`` here even though its compatibility projection supplies
    # a canonical payload provider; writing that projection back must not turn
    # historical unknown provenance into a guess.
    capture_mode = record.capture_mode
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
            raw_id, origin, capture_mode, native_id, source_path, source_index, blob_hash,
            blob_size, acquired_at_ms, file_mtime_ms, parsed_at_ms, parse_error,
            validated_at_ms, validation_status, validation_error, validation_drift_count,
            validation_mode, detection_warnings_json, logical_source_key, revision_kind,
            source_revision, predecessor_source_revision, predecessor_raw_id, baseline_raw_id, append_start_offset,
            append_end_offset, acquisition_generation, revision_authority, revision_authority_evidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.raw_id,
            origin.value,
            capture_mode.value if capture_mode is not None else None,
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
            record.revision.logical_source_key if record.revision else None,
            record.revision.kind.value if record.revision else "unknown",
            record.revision.source_revision if record.revision else None,
            record.revision.predecessor_source_revision if record.revision else None,
            record.revision.predecessor_raw_id if record.revision else None,
            record.revision.baseline_raw_id if record.revision else None,
            record.revision.append_start_offset if record.revision else None,
            record.revision.append_end_offset if record.revision else None,
            record.revision.acquisition_generation if record.revision else None,
            record.revision.authority.value if record.revision else "quarantined",
            # revision_authority_evidence (migration 017) is never computed at
            # initial-write time -- it is only ever populated later by a
            # dedicated, explicitly operator-invoked maintenance actuator
            # (raw_live_source_reconciliation_apply.py /
            # raw_append_chain_backfill_apply.py) re-verifying the raw
            # against still-present live source bytes. This is `INSERT OR
            # IGNORE`, so binding NULL here for a brand-new row is correct
            # and a duplicate-key insert attempt never overwrites an
            # already-recorded verification verdict.
            None,
        ),
    )
    inserted = bool(cursor.rowcount > 0)

    if not inserted and capture_mode is not None:
        await conn.execute(
            "UPDATE raw_sessions SET capture_mode = ? WHERE raw_id = ? AND capture_mode IS NULL",
            (capture_mode.value, record.raw_id),
        )
    if capture_mode is not None:
        # Durable multimap append (polylogue-buns): the UPDATE above only
        # ever remembers the FIRST known capture_mode for this raw_id, so a
        # content-identical GEMINI export and a live DRIVE acquisition that
        # collide on raw_id would otherwise silently lose whichever mode
        # arrived second. This records every distinct mode ever observed,
        # regardless of insert/conflict/order, so no acquisition evidence is
        # dropped even though the cached column above still only holds one.
        await conn.execute(
            """
            INSERT INTO raw_capture_observations (raw_id, capture_mode, first_observed_at_ms)
            VALUES (?, ?, ?)
            ON CONFLICT(raw_id, capture_mode) DO NOTHING
            """,
            (record.raw_id, capture_mode.value, acquired_at_ms),
        )
    if not inserted and record.file_mtime is not None:
        file_mtime_ms = _timestamp_ms(record.file_mtime)
        await conn.execute(
            "UPDATE raw_sessions SET file_mtime_ms = ?, source_path = ? "
            "WHERE raw_id = ? AND (file_mtime_ms IS NOT ? OR source_path IS NOT ?)",
            (file_mtime_ms, record.source_path, record.raw_id, file_mtime_ms, record.source_path),
        )

    # ``raw_sessions`` and ``blob_refs`` are one durable acquisition contract.
    # This async writer predates the typed source-tier writer and used to stop
    # after inserting the raw row, leaving the payload invisible to blob GC and
    # source-to-index reindex closure checks. Read the retained row back so a
    # duplicate save cannot manufacture reference metadata from a stale caller
    # record, then write the exact persisted identity.
    cursor = await conn.execute(
        "SELECT source_path, blob_hash, blob_size, acquired_at_ms FROM raw_sessions WHERE raw_id = ?",
        (record.raw_id,),
    )
    persisted = await cursor.fetchone()
    if persisted is None:
        raise RuntimeError(f"raw session disappeared before its blob reference was written: {record.raw_id}")
    await conn.execute(
        """
        INSERT OR REPLACE INTO blob_refs (
            blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms
        ) VALUES (?, ?, 'raw_payload', ?, ?, ?)
        """,
        (persisted[1], record.raw_id, persisted[0], persisted[2], persisted[3]),
    )

    if record.blob_publication_receipt_id is not None:
        await conn.execute(
            "DELETE FROM blob_publication_reservations WHERE publication_id = ? AND blob_hash = ?",
            (record.blob_publication_receipt_id, blob_hash),
        )
    if transaction_depth == 0:
        await conn.commit()
    return inserted


__all__ = ["save_raw_session"]
