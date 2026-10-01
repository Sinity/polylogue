"""Async executor for the canonical raw-admission plan."""

from __future__ import annotations

import aiosqlite

from polylogue.archive.revision_authority import RawRevisionAuthority
from polylogue.core.enums import Origin, Provider
from polylogue.storage.sqlite.archive_tiers.common import require_vocabulary
from polylogue.storage.sqlite.archive_tiers.raw_admission import (
    RawAdmissionExecution,
    RawAdmissionPlan,
    RawAdmissionResult,
)
from polylogue.storage.sqlite.archive_tiers.source_write import ContentExcisedError, require_profile_identity_key


def _revision_values(plan: RawAdmissionPlan) -> tuple[object, ...]:
    revision = plan.revision
    return (
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
    )


async def execute_raw_admission_plan_async(
    conn: aiosqlite.Connection,
    plan: RawAdmissionPlan,
    transaction_depth: int,
) -> RawAdmissionExecution:
    """Apply the canonical pending plan through the async SQLite adapter.

    The plan owns identity, timestamp and pending-revision semantics. This
    adapter retains only SQLite work, mirroring the source-tier writer's
    conflict-before-side-effects, capture observation, blob reference, and
    publication-reservation behavior.
    """
    request = plan.request
    origin_value = require_vocabulary(request.origin, Origin, field="origin")
    capture_mode_value = (
        require_vocabulary(request.capture_mode, Provider, field="capture_mode")
        if request.capture_mode is not None
        else None
    )
    if request.policy_snapshot is not None and not request.policy_snapshot.allows(request.blob_hash):
        # Raise the excision error the ingest orchestrators catch: the base
        # policy error would abort the whole batch instead of skipping one file.
        raise ContentExcisedError(blob_hash=request.blob_hash, source_path=request.source_path)
    cursor = await conn.execute("PRAGMA database_list")
    schemas = {str(row[1]) for row in await cursor.fetchall()}
    excision_schema = "source_tier" if "source_tier" in schemas else "main"
    cursor = await conn.execute(
        f"SELECT 1 FROM {excision_schema}.sqlite_master WHERE type = 'table' AND name = 'excised_content'"
    )
    if await cursor.fetchone() is not None:
        cursor = await conn.execute(
            f"SELECT 1 FROM {excision_schema}.excised_content "
            "WHERE removed_hash = ? AND hash_kind = 'blob_hash' LIMIT 1",
            (request.blob_hash,),
        )
        if await cursor.fetchone() is not None:
            raise ContentExcisedError(blob_hash=request.blob_hash, source_path=request.source_path)

    cursor = await conn.execute(
        """
        SELECT origin, native_id, source_path, source_index, blob_hash, blob_size,
               logical_source_key, revision_kind, source_revision,
               predecessor_source_revision, predecessor_raw_id, baseline_raw_id,
               append_start_offset, append_end_offset, acquisition_generation,
               revision_authority, canonical_source_path
        FROM raw_sessions WHERE raw_id = ?
        """,
        (plan.raw_id,),
    )
    retained = await cursor.fetchone()
    profile_key = request.captured_profile_key
    if profile_key is not None:
        profile_key = require_profile_identity_key(profile_key)
        profile_cursor = await conn.execute(
            "SELECT profile_key FROM raw_profile_identity_receipts WHERE raw_id = ?", (plan.raw_id,)
        )
        profile_receipt = await profile_cursor.fetchone()
        if profile_receipt is not None and profile_receipt[0] != profile_key:
            raise ValueError(f"raw id is already bound to a different profile identity: {plan.raw_id}")
        if retained is not None and profile_receipt is None:
            raise ValueError(f"retained raw is missing its original profile identity receipt: {plan.raw_id}")
    inserted = retained is None
    if retained is not None:
        retained_values = tuple(retained)
        if (
            retained_values[1:6]
            != (
                request.native_id,
                request.source_path,
                request.source_index,
                request.blob_hash,
                request.blob_size,
            )
            or retained_values[-1] != request.canonical_source_path
        ):
            raise ValueError(f"raw id is already bound to different acquisition evidence: {plan.raw_id}")
        incoming_origin = origin_value
        stored_origin = retained_values[0]
        if stored_origin != incoming_origin:
            if stored_origin == "unknown-export" and incoming_origin != "unknown-export":
                await conn.execute(
                    "UPDATE raw_sessions SET origin = ? WHERE raw_id = ? AND origin = 'unknown-export'",
                    (incoming_origin, plan.raw_id),
                )
            elif incoming_origin != "unknown-export":
                raise ValueError(
                    f"raw id is already bound to a conflicting origin: {plan.raw_id} "
                    f"(stored={stored_origin!r}, incoming={incoming_origin!r})"
                )
        # A pre-parse reservation is deliberately replaced by the parser's
        # typed revision envelope.  Re-acquiring the same observation after
        # that bind is idempotent; only an unrelated non-pending conflict is
        # fatal.
        retained_revision = retained_values[6:-1]
        pending_revision = _revision_values(plan)
        if retained_revision != pending_revision and plan.revision.authority is not RawRevisionAuthority.QUARANTINED:
            raise ValueError(f"raw id is already bound to a different revision envelope: {plan.raw_id}")
    else:
        origin = origin_value
        cursor = await conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, capture_mode, native_id, source_path, canonical_source_path, source_index, blob_hash,
                blob_size, acquired_at_ms, file_mtime_ms, logical_source_key, revision_kind,
                source_revision, predecessor_source_revision, predecessor_raw_id, baseline_raw_id,
                append_start_offset, append_end_offset, acquisition_generation, revision_authority
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(raw_id) DO NOTHING
            """,
            (
                plan.raw_id,
                origin,
                capture_mode_value,
                request.native_id,
                request.source_path,
                request.canonical_source_path,
                request.source_index,
                request.blob_hash,
                request.blob_size,
                request.acquired_at_ms,
                request.file_mtime_ms,
                *_revision_values(plan),
            ),
        )
        inserted = bool(cursor.rowcount > 0)
        if not inserted:
            # Another writer won between the read and insert. Re-enter through
            # the exact same validation path before mutable side effects.
            return await execute_raw_admission_plan_async(conn, plan, transaction_depth)

    if profile_key is not None:
        await conn.execute(
            "INSERT INTO raw_profile_identity_receipts(raw_id, profile_key) VALUES (?, ?) ON CONFLICT(raw_id) DO NOTHING",
            (plan.raw_id, profile_key),
        )
    if request.capture_mode is not None:
        capture_mode = capture_mode_value
        await conn.execute(
            "UPDATE raw_sessions SET capture_mode = ? WHERE raw_id = ? AND capture_mode IS NULL",
            (capture_mode, plan.raw_id),
        )
        await conn.execute(
            """
            INSERT INTO raw_capture_observations (raw_id, capture_mode, first_observed_at_ms)
            VALUES (?, ?, ?)
            ON CONFLICT(raw_id, capture_mode) DO NOTHING
            """,
            (plan.raw_id, capture_mode, request.acquired_at_ms),
        )
    if request.file_mtime_ms is not None:
        await conn.execute(
            "UPDATE raw_sessions SET file_mtime_ms = ? WHERE raw_id = ? AND file_mtime_ms IS NULL",
            (request.file_mtime_ms, plan.raw_id),
        )
    await conn.execute(
        """
        INSERT OR REPLACE INTO blob_refs (
            blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms
        ) VALUES (?, ?, 'raw_payload', ?, ?, ?)
        """,
        (request.blob_hash, plan.raw_id, request.source_path, request.blob_size, request.acquired_at_ms),
    )
    if request.blob_publication_receipt_id is not None:
        await conn.execute(
            "DELETE FROM blob_publication_reservations WHERE publication_id = ? AND blob_hash = ?",
            (request.blob_publication_receipt_id, request.blob_hash),
        )
    if transaction_depth == 0:
        await conn.commit()
    return RawAdmissionExecution(
        result=RawAdmissionResult(arm=plan.arm, raw_id=plan.raw_id),
        inserted=inserted,
    )


__all__ = ["execute_raw_admission_plan_async"]
