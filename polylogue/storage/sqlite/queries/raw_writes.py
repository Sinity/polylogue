"""Raw session persistence helpers."""

from __future__ import annotations

import hashlib

import aiosqlite

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.archive_tiers.write import _timestamp_ms


async def save_raw_session(
    conn: aiosqlite.Connection,
    record: RawSessionRecord,
    transaction_depth: int,
) -> bool:
    acquisition_provider = Provider.from_string(record.source_name or "unknown")
    origin = origin_from_provider(acquisition_provider)
    detected_provider = record.payload_provider
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

    # This compatibility/API route is still a production acquisition writer,
    # so it must enter the same typed raw-admission state machine as the sync
    # daemon routes. Older callers do not provide a revision envelope because
    # they persist bytes before parsing; preserve that contract by assigning
    # the explicit pending arm rather than inserting a nullable/unknown row.
    # A supplied envelope is retained for replay/append callers whose byte
    # relation was already adjudicated by their owning acquisition planner.
    revision = record.revision or RawRevisionEnvelope(
        logical_source_key=(
            f"pending:{origin.value}:{record.source_path}:{int(record.source_index or 0)}:{record.raw_id}"
        ),
        kind=RawRevisionKind.FULL,
        source_revision=blob_hash.hex(),
        acquisition_generation=0,
        authority=RawRevisionAuthority.QUARANTINED,
    )

    acquired_at_ms = _timestamp_ms(record.acquired_at) or 0
    file_mtime_ms = _timestamp_ms(record.file_mtime)

    # Validate the retained raw identity before any duplicate-side effects.
    # This async compatibility writer is often called inside a caller-owned
    # transaction; mutating observations or metadata before discovering a
    # conflict would leave that transaction poisoned with partial evidence.
    cursor = await conn.execute(
        """
        SELECT origin, source_path, source_index, blob_hash, blob_size,
               logical_source_key, revision_kind, source_revision,
               predecessor_source_revision, predecessor_raw_id, baseline_raw_id,
               append_start_offset, append_end_offset, acquisition_generation,
               revision_authority
        FROM raw_sessions WHERE raw_id = ?
        """,
        (record.raw_id,),
    )
    retained = await cursor.fetchone()
    if retained is not None:
        retained_values = tuple(retained)
        unknown_origin = "unknown-export"
        if retained_values[0] != origin.value and unknown_origin not in (retained_values[0], origin.value):
            raise ValueError(f"raw id is already bound to a conflicting identity: {record.raw_id}")
        if retained_values[1:5] != (
            record.source_path,
            int(record.source_index or 0),
            blob_hash,
            int(record.blob_size),
        ):
            raise ValueError(f"raw id is already bound to a conflicting identity: {record.raw_id}")
        # A re-save without an envelope is a legacy retry; retained evidence
        # remains authoritative. Only an explicitly supplied envelope can be
        # checked for an identity conflict.
        expected_revision = record.revision
        if expected_revision is not None:
            expected_revision_values = (
                expected_revision.logical_source_key,
                expected_revision.kind.value,
                expected_revision.source_revision,
                expected_revision.predecessor_source_revision,
                expected_revision.predecessor_raw_id,
                expected_revision.baseline_raw_id,
                expected_revision.append_start_offset,
                expected_revision.append_end_offset,
                expected_revision.acquisition_generation,
                expected_revision.authority.value,
            )
            if retained_values[5:] != expected_revision_values:
                raise ValueError(f"raw id is already bound to a conflicting identity: {record.raw_id}")

    cursor = await conn.execute(
        """
        INSERT OR IGNORE INTO raw_sessions (
            raw_id, origin, detected_provider, capture_mode, native_id, source_path, source_index, blob_hash,
            blob_size, acquired_at_ms, file_mtime_ms, parsed_at_ms, parse_error,
            validated_at_ms, validation_status, validation_error, validation_drift_count,
            validation_mode, detection_warnings_json, logical_source_key, revision_kind,
            source_revision, predecessor_source_revision, predecessor_raw_id, baseline_raw_id, append_start_offset,
            append_end_offset, acquisition_generation, revision_authority, revision_authority_evidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.raw_id,
            origin.value,
            detected_provider.value if detected_provider is not None else None,
            capture_mode.value if capture_mode is not None else None,
            None,
            record.source_path,
            int(record.source_index or 0),
            blob_hash,
            int(record.blob_size),
            acquired_at_ms,
            file_mtime_ms,
            _timestamp_ms(record.parsed_at),
            record.parse_error,
            _timestamp_ms(record.validated_at),
            record.validation_status.value if record.validation_status is not None else None,
            record.validation_error,
            int(record.validation_drift_count or 0),
            record.validation_mode.value if record.validation_mode is not None else None,
            record.detection_warnings or "[]",
            revision.logical_source_key,
            revision.kind.value,
            revision.source_revision,
            revision.predecessor_source_revision,
            revision.predecessor_raw_id,
            revision.baseline_raw_id,
            revision.append_start_offset,
            revision.append_end_offset,
            revision.acquisition_generation,
            revision.authority.value,
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
    if not inserted and file_mtime_ms is not None:
        # A later observation may fill missing acquisition evidence, but never
        # replace an established mtime. Source path is identity and was
        # validated above, so it is deliberately not rewritten here.
        await conn.execute(
            "UPDATE raw_sessions SET file_mtime_ms = ? WHERE raw_id = ? AND file_mtime_ms IS NULL",
            (file_mtime_ms, record.raw_id),
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
