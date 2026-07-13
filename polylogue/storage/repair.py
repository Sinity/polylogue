"""Consolidated archive repair: orphan detection, FTS repair, session insights, WAL."""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from polylogue.archive.raw_materialization import parsed_non_session_artifact_reason
from polylogue.archive.revision_authority import (
    BYTE_AUTHORITY_CENSUS_DETAIL,
    RawRevisionAuthority,
    RawRevisionKind,
)
from polylogue.config import Config
from polylogue.core.enums import Origin, Provider
from polylogue.core.json import JSONDocument, json_document
from polylogue.core.sources import origin_from_provider, origin_provider_fiber, provider_from_origin
from polylogue.logging import get_logger
from polylogue.maintenance.models import DerivedModelStatus, MaintenanceCategory
from polylogue.maintenance.offline_guard import offline_maintenance_block_reason
from polylogue.maintenance.targets import (
    CLEANUP_TARGETS,
    SAFE_REPAIR_TARGETS,
    MaintenanceTargetSpec,
    build_maintenance_target_catalog,
)
from polylogue.paths import archive_file_set_root_for_paths
from polylogue.pipeline.ids import session_content_hash
from polylogue.pipeline.ids import session_id as make_session_id
from polylogue.protocols import ProgressCallback
from polylogue.sources.dispatch import is_stream_record_provider
from polylogue.storage.blob_repair import count_orphaned_blobs_sync, repair_orphaned_blobs_data
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.insights.session.repair_assessment import (
    assess_session_insight_repairs,
)
from polylogue.storage.insights.session.runtime import SESSION_INSIGHT_MATERIALIZATION_TYPES
from polylogue.storage.message_type_backfill import (
    BackfillResult,
    count_messages_by_type_sync,
    count_unclassified_message_type_sync,
)

logger = get_logger(__name__)
_MAINTENANCE_TARGET_CATALOG = build_maintenance_target_catalog()
_PROBE_ONLY_EXACT_MESSAGE_ROW_LIMIT = 100_000
RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES = 1024 * 1024 * 1024
RAW_MATERIALIZATION_RESOURCE_BLOCK_REASON = "non-stream-safe raw payload exceeds the bounded replay limit"
_TRANSIENT_LOCK_PARSE_ERROR = "OperationalError: database is locked"
_QUARANTINED_ACCEPTED_RAW_REPAIR_DETAIL = "repair:accepted_quarantined_raw_exact_byte_and_semantic_proof"
_QUARANTINED_ACCEPTED_RAW_REPAIR_LIMIT = 100
_QUARANTINED_ACCEPTED_RAW_REPAIR_BLOB_LIMIT_BYTES = 256 * 1024 * 1024
_QUARANTINED_ACCEPTED_RAW_REPAIR_TOTAL_BLOB_LIMIT_BYTES = 512 * 1024 * 1024
_QUARANTINED_ACCEPTED_RAW_REPAIR_RECEIPT_SCHEMA = "polylogue.quarantined-accepted-raw-repair.v1"


@dataclass(frozen=True, slots=True)
class QuarantinedAcceptedRawArtifactWitness:
    artifact_id: str
    origin: str
    source_path: str
    source_index: int
    artifact_kind: str
    support_status: str
    classification_reason: str
    parse_as_session: int
    schema_eligible: int
    malformed_jsonl_lines: int
    decode_error: str | None
    cohort_id: str | None
    link_group_key: str | None
    sidecar_agent_type: str | None
    first_observed_at_ms: int
    last_observed_at_ms: int


@dataclass(frozen=True, slots=True)
class QuarantinedAcceptedRawApplicationWitness:
    decision_id: str
    raw_id: str
    session_id: str
    logical_source_key: str
    source_revision: str
    acquisition_generation: int
    decision: str
    accepted_raw_id: str
    accepted_source_revision: str
    accepted_content_hash: str
    baseline_raw_id: str | None
    predecessor_raw_id: str | None
    append_end_offset: int | None
    detail: str
    decided_at_ms: int


@dataclass(frozen=True, slots=True)
class QuarantinedAcceptedRawRepairItem:
    raw_id: str
    status: str
    reason: str
    logical_source_key: str | None = None
    session_id: str | None = None
    origin: str | None = None
    capture_mode: str | None = None
    source_path: str | None = None
    source_index: int | None = None
    blob_hash: str | None = None
    blob_size: int | None = None
    blob_ref_hash: str | None = None
    blob_ref_source_path: str | None = None
    blob_ref_size: int | None = None
    artifact_witnesses: tuple[QuarantinedAcceptedRawArtifactWitness, ...] = ()
    accepted_source_revision: str | None = None
    accepted_content_hash: str | None = None
    accepted_frontier_kind: str | None = None
    accepted_frontier: int | None = None
    head_decided_at_ms: int | None = None
    acquisition_generation: int | None = None
    application_decision_id: str | None = None
    application_witness: QuarantinedAcceptedRawApplicationWitness | None = None
    authority_context_digest: str | None = None
    parallel_session_head_count: int = 0
    quarantined_sibling_raw_count: int = 0
    membership_row_count: int = 0
    proof_digest: str | None = None
    repaired: bool = False


@dataclass(frozen=True, slots=True)
class QuarantinedAcceptedRawRepairReport:
    mode: str
    requested_count: int
    eligible_count: int
    repaired_count: int
    already_repaired_count: int
    ineligible_count: int
    proof_digest: str
    receipt_path: str | None
    items: tuple[QuarantinedAcceptedRawRepairItem, ...]


def _quarantined_raw_item(raw_id: str, reason: str) -> QuarantinedAcceptedRawRepairItem:
    return QuarantinedAcceptedRawRepairItem(raw_id=raw_id, status="ineligible", reason=reason)


def _bytes_value(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, memoryview):
        return bytes(value)
    raise ValueError("expected SQLite BLOB value")


def _authority_rows_digest(*row_sets: list[sqlite3.Row]) -> str:
    def value_payload(value: object) -> object:
        if isinstance(value, (bytes, memoryview)):
            return {"blob_hex": bytes(value).hex()}
        return value

    payload = [[{key: value_payload(value) for key, value in dict(row).items()} for row in rows] for rows in row_sets]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _attach_repair_index(conn: sqlite3.Connection, index_db: Path) -> None:
    conn.row_factory = sqlite3.Row
    conn.execute("ATTACH DATABASE ? AS index_tier", (f"file:{index_db}?mode=ro",))


def _validate_quarantined_raw_repair_blob_budget(conn: sqlite3.Connection, raw_ids: list[str]) -> None:
    """Reject a bounded repair set before any retained blob is loaded into memory."""
    rows = conn.execute(
        """
        SELECT raw_id, blob_size FROM raw_sessions
        WHERE raw_id IN (SELECT value FROM json_each(?))
        """,
        (json.dumps(raw_ids),),
    ).fetchall()
    oversized = [
        str(row["raw_id"]) for row in rows if int(row["blob_size"]) > _QUARANTINED_ACCEPTED_RAW_REPAIR_BLOB_LIMIT_BYTES
    ]
    if oversized:
        raise RuntimeError(
            "quarantined accepted raw repair exceeds the per-target retained-blob limit: " + ", ".join(oversized)
        )
    total = sum(int(row["blob_size"]) for row in rows)
    if total > _QUARANTINED_ACCEPTED_RAW_REPAIR_TOTAL_BLOB_LIMIT_BYTES:
        raise RuntimeError("quarantined accepted raw repair exceeds the aggregate retained-blob limit")


def _proof_digest(item: QuarantinedAcceptedRawRepairItem) -> str:
    payload = {
        key: value
        for key, value in dataclasses.asdict(item).items()
        if key not in {"proof_digest", "reason", "repaired", "status"}
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _inspect_quarantined_accepted_raw(
    archive_root: Path,
    raw_id: str,
    *,
    conn: sqlite3.Connection,
) -> QuarantinedAcceptedRawRepairItem:
    """Prove one accepted head against source main + attached read-only index."""
    try:
        raw = conn.execute(
            """
            SELECT raw_id, origin, capture_mode, native_id, source_path, source_index, blob_hash, blob_size,
                   file_mtime_ms, logical_source_key, revision_kind, source_revision,
                   predecessor_source_revision, predecessor_raw_id, baseline_raw_id,
                   append_start_offset, append_end_offset, acquisition_generation,
                   revision_authority
            FROM raw_sessions WHERE raw_id = ?
            """,
            (raw_id,),
        ).fetchone()
        if raw is None:
            return _quarantined_raw_item(raw_id, "raw row is missing")
        heads = conn.execute(
            """
            SELECT logical_source_key, session_id, accepted_raw_id,
                   accepted_source_revision, accepted_content_hash,
                   accepted_frontier_kind, accepted_frontier,
                   acquisition_generation, append_end_offset, decided_at_ms
            FROM index_tier.raw_revision_heads WHERE accepted_raw_id = ?
            """,
            (raw_id,),
        ).fetchall()
        if len(heads) != 1:
            return _quarantined_raw_item(raw_id, f"expected one accepted head, found {len(heads)}")
        head = heads[0]
        logical_source_key = str(head["logical_source_key"])
        session_id = str(head["session_id"])
        session_rows = conn.execute(
            "SELECT session_id, raw_id, content_hash FROM index_tier.sessions WHERE raw_id = ?",
            (raw_id,),
        ).fetchall()
        if len(session_rows) != 1 or str(session_rows[0]["session_id"]) != session_id:
            return _quarantined_raw_item(raw_id, "accepted head is not the raw's unique indexed session")
        session_row = session_rows[0]
        applications = conn.execute(
            """
            SELECT decision_id, raw_id, session_id, logical_source_key, source_revision,
                   acquisition_generation, decision, accepted_raw_id,
                   accepted_source_revision, accepted_content_hash, append_end_offset,
                   baseline_raw_id, predecessor_raw_id, detail, decided_at_ms
            FROM index_tier.raw_revision_applications
            WHERE raw_id = ? OR accepted_raw_id = ? OR logical_source_key = ?
            """,
            (raw_id, raw_id, logical_source_key),
        ).fetchall()
        if len(applications) != 1:
            return _quarantined_raw_item(raw_id, "competing raw-revision application authority exists")
        receipt = applications[0]
        competing_head_count = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM index_tier.raw_revision_heads
                WHERE logical_source_key = ?
                  AND NOT (accepted_raw_id = ? AND session_id = ?)
                """,
                (logical_source_key, raw_id, session_id),
            ).fetchone()[0]
        )
        parallel_session_heads = conn.execute(
            """
            SELECT logical_source_key, session_id, accepted_raw_id, accepted_source_revision,
                   accepted_content_hash, accepted_frontier_kind, accepted_frontier, decided_at_ms
            FROM index_tier.raw_revision_heads
            WHERE session_id = ? AND logical_source_key != ?
            ORDER BY logical_source_key
            """,
            (session_id, logical_source_key),
        ).fetchall()
        competing_revision_rows = conn.execute(
            """
            SELECT raw_id, logical_source_key, revision_kind, source_revision,
                   baseline_raw_id, acquisition_generation, revision_authority
            FROM raw_sessions
            WHERE raw_id != ? AND logical_source_key = ?
            ORDER BY raw_id
            """,
            (raw_id, logical_source_key),
        ).fetchall()
        membership_rows = conn.execute(
            """
            SELECT raw_id, logical_source_key, provider_session_id, source_revision,
                   normalized_content_hash, message_count, predecessor_raw_id,
                   acquisition_generation, revision_authority, decision, decided_at_ms
            FROM raw_session_memberships
            WHERE raw_id = ? OR logical_source_key = ?
            ORDER BY logical_source_key, raw_id
            """,
            (raw_id, logical_source_key),
        ).fetchall()
        census_rows = conn.execute(
            """
            SELECT raw_id, parser_fingerprint, status, member_count, censused_at_ms, detail
            FROM raw_membership_census WHERE raw_id = ?
            """,
            (raw_id,),
        ).fetchall()
        blob_ref_rows = conn.execute(
            """
            SELECT blob_hash, source_path, size_bytes FROM blob_refs
            WHERE ref_id = ? AND ref_type = 'raw_payload'
            """,
            (raw_id,),
        ).fetchall()
        artifact_rows = conn.execute(
            """
            SELECT artifact_id, origin, source_path, source_index, artifact_kind,
                   support_status, classification_reason, parse_as_session,
                   schema_eligible, malformed_jsonl_lines, decode_error, cohort_id,
                   link_group_key, sidecar_agent_type, first_observed_at_ms,
                   last_observed_at_ms
            FROM raw_artifacts WHERE raw_id = ? ORDER BY artifact_id
            """,
            (raw_id,),
        ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("quarantined raw repair authority read failed", raw_id=raw_id, error=str(exc))
        return _quarantined_raw_item(raw_id, f"authority tiers are unreadable: {exc}")

    try:
        accepted_hash = _bytes_value(head["accepted_content_hash"])
        stored_hash = _bytes_value(session_row["content_hash"])
        receipt_hash = _bytes_value(receipt["accepted_content_hash"])
        blob_hash_bytes = _bytes_value(raw["blob_hash"])
    except ValueError as exc:
        return _quarantined_raw_item(raw_id, str(exc))
    accepted_revision = str(head["accepted_source_revision"])
    generation = int(head["acquisition_generation"])
    expected_envelope = (
        logical_source_key,
        RawRevisionKind.FULL.value,
        accepted_revision,
        None,
        None,
        raw_id,
        None,
        None,
        generation,
        RawRevisionAuthority.BYTE_PROVEN.value,
    )
    actual_envelope = (
        raw["logical_source_key"],
        str(raw["revision_kind"]),
        raw["source_revision"],
        raw["predecessor_source_revision"],
        raw["predecessor_raw_id"],
        raw["baseline_raw_id"],
        raw["append_start_offset"],
        raw["append_end_offset"],
        raw["acquisition_generation"],
        str(raw["revision_authority"]),
    )
    quarantined_envelope = (
        logical_source_key,
        RawRevisionKind.FULL.value,
        accepted_revision,
        None,
        None,
        None,
        None,
        None,
        generation,
        RawRevisionAuthority.QUARANTINED.value,
    )
    untyped_envelope = (
        None,
        RawRevisionKind.UNKNOWN.value,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        RawRevisionAuthority.QUARANTINED.value,
    )
    if actual_envelope not in {untyped_envelope, quarantined_envelope, expected_envelope}:
        return _quarantined_raw_item(raw_id, "source raw has an incompatible typed authority envelope")
    if str(head["accepted_frontier_kind"]) != "byte" or head["append_end_offset"] is not None:
        return _quarantined_raw_item(raw_id, "accepted head is not a full byte frontier")
    if int(raw["source_index"]) < 0 or int(raw["blob_size"]) != int(head["accepted_frontier"]):
        return _quarantined_raw_item(raw_id, "raw shape or byte length differs from the accepted full frontier")
    if accepted_hash != stored_hash or accepted_hash != receipt_hash:
        return _quarantined_raw_item(raw_id, "head, session, and application content hashes disagree")
    expected_receipt = (
        raw_id,
        session_id,
        logical_source_key,
        accepted_revision,
        generation,
        "selected_baseline",
        raw_id,
        accepted_revision,
        accepted_hash,
        raw_id,
        None,
        None,
        int(head["decided_at_ms"]),
    )
    actual_receipt = (
        str(receipt["raw_id"]),
        str(receipt["session_id"]),
        str(receipt["logical_source_key"]),
        str(receipt["source_revision"]),
        int(receipt["acquisition_generation"]),
        str(receipt["decision"]),
        str(receipt["accepted_raw_id"]),
        str(receipt["accepted_source_revision"]),
        receipt_hash,
        receipt["baseline_raw_id"],
        receipt["predecessor_raw_id"],
        receipt["append_end_offset"],
        int(receipt["decided_at_ms"]),
    )
    if actual_receipt != expected_receipt:
        return _quarantined_raw_item(raw_id, "immutable baseline receipt does not exactly match the accepted head")
    if competing_head_count or any(str(row["revision_authority"]) != "quarantined" for row in competing_revision_rows):
        return _quarantined_raw_item(raw_id, "competing accepted or byte-proven source authority exists")
    target_memberships = [row for row in membership_rows if str(row["raw_id"]) == raw_id]
    if len(target_memberships) != 1 or len(census_rows) != 1:
        return _quarantined_raw_item(raw_id, "expected one target membership and one membership census")
    membership = target_memberships[0]
    census = census_rows[0]
    if (
        str(census["status"]) != "complete"
        or int(census["member_count"]) != 1
        or any(row["decision"] == "applied" and str(row["raw_id"]) != raw_id for row in membership_rows)
    ):
        return _quarantined_raw_item(raw_id, "membership authority is failed, ambiguous, or competitively applied")
    if len(blob_ref_rows) != 1:
        return _quarantined_raw_item(raw_id, "expected exactly one raw-payload blob reference")
    blob_ref = blob_ref_rows[0]
    if (
        _bytes_value(blob_ref["blob_hash"]) != blob_hash_bytes
        or str(blob_ref["source_path"] or "") != str(raw["source_path"] or "")
        or int(blob_ref["size_bytes"]) != int(raw["blob_size"])
        or any(
            str(artifact["origin"]) != str(raw["origin"])
            or str(artifact["source_path"]) != str(raw["source_path"])
            or int(artifact["source_index"]) != int(raw["source_index"])
            for artifact in artifact_rows
        )
    ):
        return _quarantined_raw_item(raw_id, "raw row, blob reference, and artifact identity disagree")

    blob_hash = blob_hash_bytes.hex()
    if int(raw["blob_size"]) > _QUARANTINED_ACCEPTED_RAW_REPAIR_BLOB_LIMIT_BYTES:
        return _quarantined_raw_item(raw_id, "retained raw exceeds the per-target repair byte limit")
    artifact_witnesses = tuple(
        QuarantinedAcceptedRawArtifactWitness(
            artifact_id=str(artifact["artifact_id"]),
            origin=str(artifact["origin"]),
            source_path=str(artifact["source_path"]),
            source_index=int(artifact["source_index"]),
            artifact_kind=str(artifact["artifact_kind"]),
            support_status=str(artifact["support_status"]),
            classification_reason=str(artifact["classification_reason"]),
            parse_as_session=int(artifact["parse_as_session"]),
            schema_eligible=int(artifact["schema_eligible"]),
            malformed_jsonl_lines=int(artifact["malformed_jsonl_lines"]),
            decode_error=str(artifact["decode_error"]) if artifact["decode_error"] is not None else None,
            cohort_id=str(artifact["cohort_id"]) if artifact["cohort_id"] is not None else None,
            link_group_key=str(artifact["link_group_key"]) if artifact["link_group_key"] is not None else None,
            sidecar_agent_type=(
                str(artifact["sidecar_agent_type"]) if artifact["sidecar_agent_type"] is not None else None
            ),
            first_observed_at_ms=int(artifact["first_observed_at_ms"]),
            last_observed_at_ms=int(artifact["last_observed_at_ms"]),
        )
        for artifact in artifact_rows
    )
    blob_store = BlobStore(archive_root / "blob")
    if not blob_store.exists(blob_hash) or not blob_store.verify(blob_hash):
        return _quarantined_raw_item(raw_id, "retained raw blob is missing or fails its content hash")
    payload = blob_store.read_all(blob_hash)
    if len(payload) != int(raw["blob_size"]) or hashlib.sha256(payload).hexdigest() != accepted_revision:
        return _quarantined_raw_item(raw_id, "retained bytes do not prove the accepted source revision")
    try:
        origin = Origin.from_string(str(raw["origin"]))
        capture_mode = str(raw["capture_mode"]) if raw["capture_mode"] is not None else None
        fiber = origin_provider_fiber(origin)
        if len(fiber) > 1 and capture_mode is None:
            return _quarantined_raw_item(raw_id, "non-injective origin lacks durable capture-mode authority")
        capture_provider = Provider.from_string(capture_mode) if capture_mode is not None else None
        if capture_provider is not None and capture_provider not in fiber:
            return _quarantined_raw_item(raw_id, "capture mode falls outside the raw origin provider fiber")
        provider = provider_from_origin(origin, family_hint=capture_provider)
        from polylogue.pipeline.services.ingest_worker import _normalized_session
        from polylogue.sources.revision_backfill import _parse_one

        sessions = _parse_one(provider, payload, str(raw["source_path"]))
        fallback_timestamp = (
            datetime.fromtimestamp(int(raw["file_mtime_ms"]) / 1000, UTC).isoformat()
            if raw["file_mtime_ms"] is not None
            else None
        )
        sessions = [_normalized_session(session, fallback_timestamp=fallback_timestamp) for session in sessions]
    except Exception as exc:
        logger.warning("quarantined raw repair normalization failed", raw_id=raw_id, error=str(exc))
        return _quarantined_raw_item(raw_id, f"retained raw did not normalize cleanly: {type(exc).__name__}: {exc}")
    if len(sessions) != 1:
        return _quarantined_raw_item(raw_id, f"retained raw normalized to {len(sessions)} sessions instead of one")
    parsed = sessions[0]
    if (
        origin_from_provider(parsed.source_name) != origin
        or str(make_session_id(parsed.source_name, parsed.provider_session_id)) != str(head["session_id"])
        or f"{provider.value}:{parsed.provider_session_id}" != logical_source_key
        or bytes.fromhex(session_content_hash(parsed)) != accepted_hash
    ):
        return _quarantined_raw_item(raw_id, "normalized parser identity or content differs from the accepted session")
    parsed_logical_source_key = f"{parsed.source_name.value}:{parsed.provider_session_id}"
    if (
        str(membership["logical_source_key"]) != parsed_logical_source_key
        or str(membership["provider_session_id"]) != parsed.provider_session_id
        or _bytes_value(membership["normalized_content_hash"]) != accepted_hash
        or str(membership["source_revision"]) != accepted_hash.hex()
        or int(membership["message_count"]) != len(parsed.messages)
        or membership["predecessor_raw_id"] is not None
        or int(membership["acquisition_generation"]) != generation
        or str(membership["revision_authority"]) != "quarantined"
        or membership["decision"] == "applied"
    ):
        return _quarantined_raw_item(raw_id, "membership evidence does not match the normalized accepted session")

    status = "already_repaired" if actual_envelope == expected_envelope else "eligible"
    item = QuarantinedAcceptedRawRepairItem(
        raw_id=raw_id,
        status=status,
        reason=(
            "source envelope already matches the proven accepted head"
            if status == "already_repaired"
            else _QUARANTINED_ACCEPTED_RAW_REPAIR_DETAIL
        ),
        logical_source_key=logical_source_key,
        session_id=session_id,
        origin=str(raw["origin"]),
        capture_mode=capture_mode,
        source_path=str(raw["source_path"]),
        source_index=int(raw["source_index"]),
        blob_hash=blob_hash,
        blob_size=int(raw["blob_size"]),
        blob_ref_hash=_bytes_value(blob_ref["blob_hash"]).hex(),
        blob_ref_source_path=str(blob_ref["source_path"] or ""),
        blob_ref_size=int(blob_ref["size_bytes"]),
        artifact_witnesses=artifact_witnesses,
        accepted_source_revision=accepted_revision,
        accepted_content_hash=accepted_hash.hex(),
        accepted_frontier_kind=str(head["accepted_frontier_kind"]),
        accepted_frontier=int(head["accepted_frontier"]),
        head_decided_at_ms=int(head["decided_at_ms"]),
        acquisition_generation=generation,
        application_decision_id=str(receipt["decision_id"]),
        application_witness=QuarantinedAcceptedRawApplicationWitness(
            decision_id=str(receipt["decision_id"]),
            raw_id=str(receipt["raw_id"]),
            session_id=str(receipt["session_id"]),
            logical_source_key=str(receipt["logical_source_key"]),
            source_revision=str(receipt["source_revision"]),
            acquisition_generation=int(receipt["acquisition_generation"]),
            decision=str(receipt["decision"]),
            accepted_raw_id=str(receipt["accepted_raw_id"]),
            accepted_source_revision=str(receipt["accepted_source_revision"]),
            accepted_content_hash=receipt_hash.hex(),
            baseline_raw_id=str(receipt["baseline_raw_id"]) if receipt["baseline_raw_id"] is not None else None,
            predecessor_raw_id=(
                str(receipt["predecessor_raw_id"]) if receipt["predecessor_raw_id"] is not None else None
            ),
            append_end_offset=(int(receipt["append_end_offset"]) if receipt["append_end_offset"] is not None else None),
            detail=str(receipt["detail"]),
            decided_at_ms=int(receipt["decided_at_ms"]),
        ),
        authority_context_digest=_authority_rows_digest(
            applications,
            parallel_session_heads,
            competing_revision_rows,
            membership_rows,
            census_rows,
        ),
        parallel_session_head_count=len(parallel_session_heads),
        quarantined_sibling_raw_count=len(competing_revision_rows),
        membership_row_count=len(membership_rows),
    )
    return dataclasses.replace(item, proof_digest=_proof_digest(item))


def _repair_receipt_targets(items: list[QuarantinedAcceptedRawRepairItem]) -> list[dict[str, object]]:
    targets = [
        {key: value for key, value in dataclasses.asdict(item).items() if key not in {"reason", "repaired", "status"}}
        for item in items
    ]
    return cast(list[dict[str, object]], json.loads(json.dumps(targets, sort_keys=True)))


def _repair_proof_digest(items: list[QuarantinedAcceptedRawRepairItem]) -> str:
    proof_digests = [item.proof_digest for item in items]
    return hashlib.sha256(json.dumps(proof_digests, separators=(",", ":")).encode()).hexdigest()


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(slots=True)
class _LockedQuarantinedRawRepairReceipt:
    path: Path
    descriptor: int
    target_hash: str
    terminal: bool
    repair_intent_raw_ids: tuple[str, ...]
    torn_terminals: tuple[bytes, ...] = ()
    receipt_terminated: bool = True

    def close(self) -> None:
        fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        os.close(self.descriptor)


def _receipt_write(descriptor: int, payload: bytes) -> int:
    return os.write(descriptor, payload)


def _write_receipt_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = _receipt_write(descriptor, payload[offset:])
        if written <= 0:
            raise RuntimeError("operator repair receipt write made no progress")
        offset += written


def _receipt_records(descriptor: int) -> tuple[list[dict[str, object] | bytes], bool]:
    size = os.fstat(descriptor).st_size
    if size > 16 * 1024 * 1024:
        raise RuntimeError("existing repair receipt exceeds the bounded parser limit")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            raise RuntimeError("existing repair receipt changed during its locked read")
        chunks.append(chunk)
        remaining -= len(chunk)
    payload = b"".join(chunks)
    terminated = payload.endswith(b"\n")
    lines = payload.split(b"\n")
    if terminated:
        lines.pop()
    records: list[dict[str, object] | bytes] = []
    for index, line in enumerate(lines):
        if index == len(lines) - 1 and not terminated:
            records.append(line)
            continue
        try:
            parsed: object = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            records.append(line)
            continue
        records.append(cast(dict[str, object], parsed) if isinstance(parsed, dict) else line)
    return records, terminated


def _validate_repair_receipt_records(
    parsed_receipt: tuple[list[dict[str, object] | bytes], bool],
    *,
    targets: list[dict[str, object]],
    target_hash: str,
) -> tuple[bool, tuple[str, ...], tuple[bytes, ...], bool]:
    records, terminated = parsed_receipt
    if not records:
        raise RuntimeError("existing repair receipt is empty")
    planned = records[0]
    if not isinstance(planned, dict):
        raise RuntimeError("existing repair receipt does not start with valid planned JSON")
    planned_keys = {"schema", "state", "target_hash", "targets", "repair_intent_raw_ids", "planned_at_ms"}
    if set(planned) != planned_keys or planned.get("schema") != _QUARANTINED_ACCEPTED_RAW_REPAIR_RECEIPT_SCHEMA:
        raise RuntimeError("existing repair receipt has an invalid planned record schema")
    if planned.get("state") != "planned":
        raise RuntimeError("existing repair receipt must start with a planned record")
    if planned.get("target_hash") != target_hash or planned.get("targets") != targets:
        raise RuntimeError("existing repair receipt targets do not match the proven repair set")
    planned_at_ms = planned.get("planned_at_ms")
    if not isinstance(planned_at_ms, int) or planned_at_ms < 0:
        raise RuntimeError("existing repair receipt planned timestamp is invalid")
    raw_ids = tuple(str(target["raw_id"]) for target in targets)
    intent = planned.get("repair_intent_raw_ids")
    if not isinstance(intent, list) or any(not isinstance(raw_id, str) for raw_id in intent):
        raise RuntimeError("existing repair receipt repair intent is invalid")
    intent_ids = tuple(cast(list[str], intent))
    if len(set(intent_ids)) != len(intent_ids) or any(raw_id not in raw_ids for raw_id in intent_ids):
        raise RuntimeError("existing repair receipt repair intent does not match its targets")
    if len(records) == 1:
        if not terminated:
            raise RuntimeError("existing repair receipt has a torn planned record")
        return False, intent_ids, (), terminated
    tail = records[1:]
    applied = tail[-1] if isinstance(tail[-1], dict) else None
    torn_terminals = (
        tuple(record for record in tail[:-1] if isinstance(record, bytes))
        if applied
        else tuple(record for record in tail if isinstance(record, bytes))
    )
    expected_tail_length = len(torn_terminals) + (1 if applied is not None else 0)
    if len(tail) != expected_tail_length or any(not fragment for fragment in torn_terminals):
        raise RuntimeError("existing repair receipt has an invalid state transition")
    if applied is None:
        return False, intent_ids, torn_terminals, terminated
    if not terminated:
        raise RuntimeError("existing repair receipt has an unterminated applied record")
    recovered = bool(torn_terminals)
    applied_keys = {
        "schema",
        "state",
        "target_hash",
        "applied_at_ms",
        "repaired_raw_ids",
        "proven_raw_ids",
    }
    if recovered:
        applied_keys |= {"torn_terminals"}
    if set(applied) != applied_keys or applied.get("schema") != _QUARANTINED_ACCEPTED_RAW_REPAIR_RECEIPT_SCHEMA:
        raise RuntimeError("existing repair receipt has an invalid applied record schema")
    if applied.get("state") != "applied" or applied.get("target_hash") != target_hash:
        raise RuntimeError("existing repair receipt has an invalid applied target transition")
    applied_at_ms = applied.get("applied_at_ms")
    if not isinstance(applied_at_ms, int) or applied_at_ms < 0:
        raise RuntimeError("existing repair receipt applied timestamp is invalid")
    repaired_ids = applied.get("repaired_raw_ids")
    if (
        applied.get("proven_raw_ids") != list(raw_ids)
        or not isinstance(repaired_ids, list)
        or any(not isinstance(raw_id, str) for raw_id in repaired_ids)
        or len(set(cast(list[str], repaired_ids))) != len(repaired_ids)
        or any(raw_id not in intent_ids for raw_id in cast(list[str], repaired_ids))
    ):
        raise RuntimeError("existing repair receipt applied ids do not match the planned targets")
    expected_torn_witnesses = [
        {"bytes": len(fragment), "sha256": hashlib.sha256(fragment).hexdigest()} for fragment in torn_terminals
    ]
    if recovered and applied.get("torn_terminals") != expected_torn_witnesses:
        raise RuntimeError("existing repair receipt recovery does not match its preserved torn terminal")
    return True, intent_ids, torn_terminals, terminated


def _lock_quarantined_raw_repair_receipt(
    path: Path,
    items: list[QuarantinedAcceptedRawRepairItem],
) -> _LockedQuarantinedRawRepairReceipt:
    """Lock one stable receipt inode and create or validate its planned record."""
    if path.is_symlink():
        raise RuntimeError("repair receipt path must not be a symbolic link")
    targets = _repair_receipt_targets(items)
    target_hash = hashlib.sha256(json.dumps(targets, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    repair_intent_raw_ids = tuple(item.raw_id for item in items if item.status == "eligible")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise RuntimeError("operator repair receipt is already locked by another apply") from exc
    try:
        opened = os.fstat(descriptor)
        named = path.stat(follow_symlinks=False)
        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
            raise RuntimeError("operator repair receipt path changed while it was being locked")
        if opened.st_size:
            terminal, existing_intent, torn_terminals, terminated = _validate_repair_receipt_records(
                _receipt_records(descriptor), targets=targets, target_hash=target_hash
            )
            return _LockedQuarantinedRawRepairReceipt(
                path,
                descriptor,
                target_hash,
                terminal,
                existing_intent,
                torn_terminals,
                terminated,
            )
        planned = {
            "schema": _QUARANTINED_ACCEPTED_RAW_REPAIR_RECEIPT_SCHEMA,
            "state": "planned",
            "target_hash": target_hash,
            "targets": targets,
            "repair_intent_raw_ids": list(repair_intent_raw_ids),
            "planned_at_ms": int(time.time() * 1000),
        }
        encoded = (json.dumps(planned, sort_keys=True, separators=(",", ":")) + "\n").encode()
        _write_receipt_all(descriptor, encoded)
        os.fsync(descriptor)
        _fsync_parent(path)
        return _LockedQuarantinedRawRepairReceipt(path, descriptor, target_hash, False, repair_intent_raw_ids)
    except Exception:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        raise


def _finish_quarantined_raw_repair_receipt(
    receipt: _LockedQuarantinedRawRepairReceipt,
    *,
    items: list[QuarantinedAcceptedRawRepairItem],
) -> None:
    opened = os.fstat(receipt.descriptor)
    named = receipt.path.stat(follow_symlinks=False)
    if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
        raise RuntimeError("operator repair receipt path changed before terminal append")
    os.lseek(receipt.descriptor, 0, os.SEEK_END)
    preserved_torn_terminals = list(receipt.torn_terminals)
    if preserved_torn_terminals and not receipt.receipt_terminated:
        # Make even a complete-JSON prefix permanently distinguishable from a
        # terminal record after the newline is appended. The exact sealed bytes
        # remain in the append-only receipt and are bound into the terminal.
        _write_receipt_all(receipt.descriptor, b"\xff\n")
        preserved_torn_terminals[-1] += b"\xff"
    terminal = {
        "schema": _QUARANTINED_ACCEPTED_RAW_REPAIR_RECEIPT_SCHEMA,
        "state": "applied",
        "target_hash": receipt.target_hash,
        "applied_at_ms": int(time.time() * 1000),
        "repaired_raw_ids": [item.raw_id for item in items if item.repaired],
        "proven_raw_ids": [item.raw_id for item in items],
    }
    if preserved_torn_terminals:
        terminal["torn_terminals"] = [
            {"bytes": len(fragment), "sha256": hashlib.sha256(fragment).hexdigest()}
            for fragment in preserved_torn_terminals
        ]
    _write_receipt_all(
        receipt.descriptor, (json.dumps(terminal, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    os.fsync(receipt.descriptor)
    _fsync_parent(receipt.path)


def _cas_refine_quarantined_accepted_raw(
    source_conn: sqlite3.Connection,
    item: QuarantinedAcceptedRawRepairItem,
) -> None:
    assert item.logical_source_key is not None
    assert item.accepted_source_revision is not None
    assert item.acquisition_generation is not None
    cursor = source_conn.execute(
        """
        UPDATE raw_sessions
        SET logical_source_key = ?, revision_kind = 'full', source_revision = ?,
            baseline_raw_id = raw_id, acquisition_generation = ?,
            revision_authority = 'byte_proven'
        WHERE raw_id = ? AND revision_authority = 'quarantined'
          AND predecessor_source_revision IS NULL AND predecessor_raw_id IS NULL
          AND append_start_offset IS NULL AND append_end_offset IS NULL
          AND (
            (logical_source_key IS NULL AND revision_kind = 'unknown'
             AND source_revision IS NULL AND baseline_raw_id IS NULL
             AND acquisition_generation IS NULL)
            OR
            (logical_source_key = ? AND revision_kind = 'full'
             AND source_revision = ? AND baseline_raw_id IS NULL
             AND acquisition_generation = ?)
          )
        """,
        (
            item.logical_source_key,
            item.accepted_source_revision,
            item.acquisition_generation,
            item.raw_id,
            item.logical_source_key,
            item.accepted_source_revision,
            item.acquisition_generation,
        ),
    )
    if cursor.rowcount != 1:
        raise RuntimeError(f"source authority CAS failed for {item.raw_id}")


def repair_quarantined_accepted_raws(
    config: Config,
    raw_ids: list[str],
    *,
    apply: bool = False,
    receipt_path: Path | None = None,
    proof_digest: str | None = None,
) -> QuarantinedAcceptedRawRepairReport:
    """Refine accepted untyped or typed-quarantined full raws after exact proof."""
    if len(set(raw_ids)) != len(raw_ids):
        raise ValueError("duplicate raw ids are not allowed")
    if not raw_ids or len(raw_ids) > _QUARANTINED_ACCEPTED_RAW_REPAIR_LIMIT:
        raise ValueError(f"raw-id list must contain 1..{_QUARANTINED_ACCEPTED_RAW_REPAIR_LIMIT} entries")
    if any(re.fullmatch(r"[0-9a-f]{64}", raw_id) is None for raw_id in raw_ids):
        raise ValueError("raw ids must be lowercase SHA-256 identifiers")
    block_reason = offline_maintenance_block_reason(config, active=apply, dry_run=not apply)
    if block_reason is not None:
        raise RuntimeError(block_reason)
    archive_root = _raw_materialization_archive_root(config)
    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    if not source_db.exists() or not index_db.exists():
        raise RuntimeError("source or index tier is missing")
    with closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)) as dry_conn:
        _attach_repair_index(dry_conn, index_db)
        _validate_quarantined_raw_repair_blob_budget(dry_conn, raw_ids)
        items = [_inspect_quarantined_accepted_raw(archive_root, raw_id, conn=dry_conn) for raw_id in raw_ids]
    aggregate_proof = _repair_proof_digest(items)
    if apply and any(item.status == "ineligible" for item in items):
        raise RuntimeError("quarantined accepted raw repair refused because one or more targets are ineligible")
    if apply and receipt_path is None:
        raise ValueError("apply requires an explicit operator repair receipt path")
    if apply and proof_digest != aggregate_proof:
        raise RuntimeError("apply proof digest does not match the exact dry-run target list")
    if apply:
        from polylogue.storage.index_generation import RebuildLease

        assert receipt_path is not None
        with RebuildLease(archive_root):
            receipt = _lock_quarantined_raw_repair_receipt(receipt_path, items)
            try:
                with closing(sqlite3.connect(f"file:{source_db}?mode=rw", uri=True)) as source_conn:
                    source_conn.execute("PRAGMA foreign_keys = ON")
                    _attach_repair_index(source_conn, index_db)
                    source_conn.execute("BEGIN IMMEDIATE")
                    try:
                        _validate_quarantined_raw_repair_blob_budget(source_conn, raw_ids)
                        locked_items = [
                            _inspect_quarantined_accepted_raw(archive_root, raw_id, conn=source_conn)
                            for raw_id in raw_ids
                        ]
                        if _repair_proof_digest(locked_items) != proof_digest:
                            raise RuntimeError("authority proof changed after acquiring the repair transaction")
                        if any(item.status == "ineligible" for item in locked_items):
                            raise RuntimeError("a repair target became ineligible after acquiring the transaction")
                        if receipt.terminal and any(item.status != "already_repaired" for item in locked_items):
                            raise RuntimeError("terminal operator receipt disagrees with durable source authority")
                        if receipt.torn_terminals and any(item.status != "already_repaired" for item in locked_items):
                            raise RuntimeError("torn terminal receipt has no matching committed source refinement")
                        for item in locked_items:
                            if item.status == "eligible":
                                _cas_refine_quarantined_accepted_raw(source_conn, item)
                        after_items = [
                            _inspect_quarantined_accepted_raw(archive_root, raw_id, conn=source_conn)
                            for raw_id in raw_ids
                        ]
                        if any(item.status != "already_repaired" for item in after_items):
                            raise RuntimeError("source envelope refinement did not reach the proven terminal state")
                        source_conn.commit()
                    except Exception:
                        source_conn.rollback()
                        raise
                items = [
                    dataclasses.replace(after, repaired=before.status == "eligible")
                    for before, after in zip(locked_items, after_items, strict=True)
                ]
                if not receipt.terminal:
                    _finish_quarantined_raw_repair_receipt(receipt, items=items)
            finally:
                receipt.close()
    return QuarantinedAcceptedRawRepairReport(
        mode="apply" if apply else "dry-run",
        requested_count=len(items),
        eligible_count=sum(item.status == "eligible" for item in items),
        repaired_count=sum(item.repaired for item in items),
        already_repaired_count=sum(item.status == "already_repaired" for item in items),
        ineligible_count=sum(item.status == "ineligible" for item in items),
        proof_digest=aggregate_proof,
        receipt_path=str(receipt_path) if receipt_path is not None else None,
        items=tuple(items),
    )


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(max(value, 0))
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(amount)} B"
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{int(amount)} B"


@dataclass(frozen=True)
class RawMaterializationCandidates:
    raw_ids: list[str]
    missing_blobs: int
    already_parsed: int
    raw_blob_bytes: dict[str, int] = field(default_factory=dict)
    raw_origins: dict[str, str] = field(default_factory=dict)
    raw_source_paths: dict[str, str] = field(default_factory=dict)
    missing_blob_source_available: int = 0
    missing_blob_source_missing: int = 0
    adoption_deferred: int = 0
    authority_quarantined: int = 0
    byte_authority_fragments: int = 0
    byte_authority_quarantined: int = 0
    byte_authority_pending: int = 0
    expanded_raw_ids: tuple[str, ...] = ()
    expanded_blob_bytes: dict[str, int] = field(default_factory=dict)
    authority_components: tuple[tuple[str, ...], ...] = ()

    @property
    def total_blob_bytes(self) -> int:
        return sum(self.raw_blob_bytes.get(raw_id, 0) for raw_id in self.raw_ids)

    @property
    def max_blob_bytes(self) -> int:
        return max((self.raw_blob_bytes.get(raw_id, 0) for raw_id in self.raw_ids), default=0)


def _raw_materialization_origin_from_provider(provider: str | None) -> str | None:
    if provider is None:
        return None
    return origin_from_provider(Provider.from_string(provider)).value


def _raw_materialization_source_available(source_path: str) -> bool:
    if not source_path:
        return False
    path = Path(source_path)
    if path.exists():
        return True
    if ":" in source_path:
        outer, _inner = source_path.split(":", 1)
        return Path(outer).exists()
    return False


def _raw_materialization_archive_root(config: Config) -> Path:
    return archive_file_set_root_for_paths(archive_root_path=config.archive_root, db_anchor=config.db_path)


def _raw_materialization_candidate_ids(
    config: Config,
    *,
    raw_artifact_id: str | None = None,
    provider: str | None = None,
    source_family: str | None = None,
    source_root: Path | None = None,
) -> RawMaterializationCandidates:
    """Return replayable raw ids plus missing-blob debt count.

    Raw evidence is the durable source of truth, but a raw row whose
    content-addressed blob is absent cannot be reparsed without outside
    evidence. Keep those rows as debt instead of mutating or deleting them.
    Broad repair queues raw rows that are not materialized in the attached
    index tier, including rows whose raw evidence was parsed before an index
    reset or interrupted replay. Parse failures, intentionally skipped rows,
    missing blobs, and source-path/native-id aliases remain excluded or counted
    as debt instead of being blindly retried.
    """
    archive_root = _raw_materialization_archive_root(config)
    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    if not source_db.exists() or not index_db.exists():
        return RawMaterializationCandidates([], 0, 0)
    blob_store = BlobStore(archive_root / "blob")
    raw_ids: list[str] = []
    raw_blob_bytes: dict[str, int] = {}
    raw_origins: dict[str, str] = {}
    raw_source_paths: dict[str, str] = {}
    missing_blobs = 0
    missing_blob_source_available = 0
    missing_blob_source_missing = 0
    already_parsed = 0
    expanded_raw_ids: tuple[str, ...] = ()
    expanded_blob_bytes: dict[str, int] = {}
    authority_components: tuple[tuple[str, ...], ...] = ()
    with closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("ATTACH DATABASE ? AS index_tier", (str(index_db),))
        params: list[object] = []
        raw_filter = ""
        if raw_artifact_id is not None:
            raw_filter = "AND r.raw_id = ?"
            params.append(raw_artifact_id)
        origin_filter = ""
        provider_origin = _raw_materialization_origin_from_provider(provider)
        if provider_origin is not None:
            origin_filter += " AND r.origin = ?"
            params.append(provider_origin)
        if source_family is not None:
            origin_filter += " AND r.origin = ?"
            params.append(source_family)
        source_root_filter = ""
        if source_root is not None:
            normalized_root = str(source_root).rstrip("/")
            source_root_filter = " AND (r.source_path = ? OR r.source_path LIKE ?)"
            params.extend((normalized_root, f"{normalized_root}/%"))
        rows = conn.execute(
            f"""
            SELECT r.raw_id, r.origin, r.native_id, r.source_path, r.blob_hash, r.blob_size, r.parsed_at_ms,
                   r.parse_error,
                   EXISTS (
                       SELECT 1
                       FROM index_tier.raw_revision_applications AS a
                       WHERE a.raw_id = r.raw_id
                         AND a.decision = 'deferred'
                         AND a.detail = 'ordinary_replay:incomparable_existing_index_state'
                   ) AS adoption_deferred,
                   EXISTS (
                       SELECT 1
                       FROM index_tier.raw_revision_applications AS a
                       WHERE a.raw_id = r.raw_id
                         AND a.decision IN (
                           'selected_baseline', 'applied_append', 'superseded', 'ambiguous'
                         )
                   ) AS application_terminal,
                   EXISTS (
                       SELECT 1
                       FROM raw_membership_census AS c
                       WHERE c.raw_id = r.raw_id
                         AND c.status = 'complete'
                         AND c.member_count > 0
                         AND c.member_count = (
                           SELECT COUNT(*) FROM raw_session_memberships AS counted
                           WHERE counted.raw_id = c.raw_id
                         )
                         AND NOT EXISTS (
                           SELECT 1 FROM raw_session_memberships AS m
                           WHERE m.raw_id = c.raw_id
                             AND (m.decision IS NULL OR m.decision IN ('ambiguous', 'deferred'))
                         )
                   ) AS membership_authority_complete
                   , EXISTS (
                       SELECT 1
                       FROM raw_membership_census AS c
                       JOIN raw_session_memberships AS m ON m.raw_id = c.raw_id
                       WHERE c.raw_id = r.raw_id
                         AND c.status = 'complete'
                         AND m.decision = 'ambiguous'
                   ) AS membership_authority_quarantined
                   , (r.source_index = -1 AND r.revision_authority = 'byte_proven')
                     AS byte_authority_fragment
                   , (r.source_index = -1
                      AND r.revision_authority != 'byte_proven'
                      AND EXISTS (
                        SELECT 1
                        FROM raw_membership_census AS c
                        WHERE c.raw_id = r.raw_id
                          AND c.status = 'failed'
                          AND c.detail = ?
                      )) AS byte_authority_quarantined
                   , (r.source_index = -1 AND NOT EXISTS (
                       SELECT 1
                       FROM raw_membership_census AS c
                       WHERE c.raw_id = r.raw_id
                         AND c.status = 'failed'
                         AND c.detail = ?
                   )) AS byte_authority_pending
            FROM raw_sessions AS r
            LEFT JOIN index_tier.sessions AS s_by_raw ON s_by_raw.raw_id = r.raw_id
            LEFT JOIN index_tier.sessions AS s_by_native
              ON r.native_id IS NOT NULL
             AND s_by_native.origin = r.origin
             AND s_by_native.native_id = r.native_id
            LEFT JOIN raw_sessions AS existing_native_raw
              ON existing_native_raw.raw_id = s_by_native.raw_id
            WHERE s_by_raw.raw_id IS NULL
              AND (
                s_by_native.native_id IS NULL
                OR existing_native_raw.raw_id IS NULL
              )
              AND (
                r.parse_error IS NULL
                OR r.parse_error = 'OperationalError: database is locked'
                OR (
                  r.parse_error LIKE 'decode:%No such file or directory:%'
                )
              )
              AND NOT (
                COALESCE(r.validation_status, '') = 'skipped'
                AND r.parsed_at_ms IS NOT NULL
                AND r.parse_error IS NULL
              )
              {raw_filter}
              {origin_filter}
              {source_root_filter}
            ORDER BY r.acquired_at_ms DESC, r.raw_id ASC
            """,
            [BYTE_AUTHORITY_CENSUS_DETAIL, BYTE_AUTHORITY_CENSUS_DETAIL, *params],
        ).fetchall()
        adoption_deferred = 0
        authority_quarantined = 0
        byte_authority_fragments = 0
        byte_authority_quarantined = 0
        byte_authority_pending = 0
        for row in rows:
            if bool(row["adoption_deferred"]):
                adoption_deferred += 1
                continue
            if bool(row["application_terminal"]):
                continue
            if bool(row["membership_authority_complete"]):
                continue
            if bool(row["membership_authority_quarantined"]):
                authority_quarantined += 1
                continue
            if bool(row["byte_authority_fragment"]):
                byte_authority_fragments += 1
                continue
            if bool(row["byte_authority_quarantined"]):
                byte_authority_quarantined += 1
                continue
            if bool(row["byte_authority_pending"]):
                byte_authority_pending += 1
                continue
            if row["parse_error"] and not _raw_materialization_retryable_missing_blob_error(row["parse_error"]):
                continue
            if _raw_materialized_by_source_path_native(conn, row):
                continue
            if _raw_materialization_parsed_non_session_artifact(archive_root, row):
                continue
            blob_hash = row["blob_hash"].hex() if isinstance(row["blob_hash"], bytes) else str(row["blob_hash"])
            if blob_store.exists(blob_hash):
                raw_id = str(row["raw_id"])
                raw_ids.append(raw_id)
                raw_origins[raw_id] = str(row["origin"] or "")
                raw_source_paths[raw_id] = str(row["source_path"] or "")
                blob_size = row["blob_size"]
                if isinstance(blob_size, int):
                    raw_blob_bytes[raw_id] = blob_size
                if row["parsed_at_ms"] is not None:
                    already_parsed += 1
            else:
                missing_blobs += 1
                if _raw_materialization_source_available(str(row["source_path"] or "")):
                    missing_blob_source_available += 1
                else:
                    missing_blob_source_missing += 1
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        authority_components = ArchiveStore.raw_membership_selection_components_sync(conn, raw_ids)
        expanded_raw_ids = tuple(sorted({raw_id for component in authority_components for raw_id in component}))
        if expanded_raw_ids:
            placeholders = ",".join("?" for _ in expanded_raw_ids)
            expanded_blob_bytes = {
                str(row[0]): int(row[1] or 0)
                for row in conn.execute(
                    f"SELECT raw_id, blob_size FROM raw_sessions WHERE raw_id IN ({placeholders})",
                    expanded_raw_ids,
                )
            }
    return RawMaterializationCandidates(
        raw_ids=raw_ids,
        missing_blobs=missing_blobs,
        already_parsed=already_parsed,
        raw_blob_bytes=raw_blob_bytes,
        raw_origins=raw_origins,
        raw_source_paths=raw_source_paths,
        missing_blob_source_available=missing_blob_source_available,
        missing_blob_source_missing=missing_blob_source_missing,
        adoption_deferred=adoption_deferred,
        authority_quarantined=authority_quarantined,
        byte_authority_fragments=byte_authority_fragments,
        byte_authority_quarantined=byte_authority_quarantined,
        byte_authority_pending=byte_authority_pending,
        expanded_raw_ids=expanded_raw_ids,
        expanded_blob_bytes=expanded_blob_bytes,
        authority_components=authority_components,
    )


def _raw_materialization_stream_safe(candidates: RawMaterializationCandidates, raw_id: str) -> bool:
    origin = Origin.from_string(candidates.raw_origins.get(raw_id))
    provider = provider_from_origin(origin)
    return is_stream_record_provider(candidates.raw_source_paths.get(raw_id), provider)


def _raw_materialization_retryable_missing_blob_error(parse_error: object) -> bool:
    if not isinstance(parse_error, str):
        return False
    return parse_error == _TRANSIENT_LOCK_PARSE_ERROR or (
        parse_error.startswith("decode:") and "No such file or directory" in parse_error
    )


def _raw_materialization_total_bytes(candidates: RawMaterializationCandidates, raw_ids: list[str]) -> int:
    return sum(candidates.raw_blob_bytes.get(raw_id, 0) for raw_id in raw_ids)


def _raw_materialization_max_bytes(candidates: RawMaterializationCandidates, raw_ids: list[str]) -> int:
    return max((candidates.raw_blob_bytes.get(raw_id, 0) for raw_id in raw_ids), default=0)


def _raw_materialization_bucket_summary(
    candidates: RawMaterializationCandidates,
    *,
    values: dict[str, str],
    key_name: str,
    limit: int,
) -> list[dict[str, object]]:
    buckets: dict[str, dict[str, int | str]] = {}
    for raw_id in candidates.raw_ids:
        key = values.get(raw_id) or "unknown"
        size = candidates.raw_blob_bytes.get(raw_id, 0)
        bucket = buckets.setdefault(key, {key_name: key, "raw_count": 0, "total_blob_bytes": 0, "max_blob_bytes": 0})
        bucket["raw_count"] = int(bucket["raw_count"]) + 1
        bucket["total_blob_bytes"] = int(bucket["total_blob_bytes"]) + size
        bucket["max_blob_bytes"] = max(int(bucket["max_blob_bytes"]), size)
    return [
        dict(bucket)
        for bucket in sorted(
            buckets.values(),
            key=lambda item: (-int(item["total_blob_bytes"]), -int(item["raw_count"]), str(item[key_name])),
        )[:limit]
    ]


def _raw_materialization_missing_blob_detail(candidates: RawMaterializationCandidates, *, final: bool) -> str:
    verb = "remain blocked by" if final else "blocked by"
    detail = f"{candidates.missing_blobs:,} raw rows {verb} missing blobs"
    parts: list[str] = []
    if candidates.missing_blob_source_available:
        parts.append(f"{candidates.missing_blob_source_available:,} with source paths still present")
    if candidates.missing_blob_source_missing:
        parts.append(f"{candidates.missing_blob_source_missing:,} with source paths missing")
    if parts:
        detail += f" ({'; '.join(parts)})"
    return detail


def raw_materialization_replay_backlog(config: Config, *, limit: int = 10) -> dict[str, object]:
    """Return a read-only weighted backlog for raw source-to-index replay.

    The report uses the same candidate selector as ``repair_raw_materialization``
    so diagnostics and actual replay agree about which raw rows are actionable.
    It does not parse raw blobs or mutate the archive.
    """

    archive_root = _raw_materialization_archive_root(config)
    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    if not source_db.exists() or not index_db.exists():
        return {
            "available": False,
            "reason": "source_or_index_tier_missing",
            "execution_blocked": False,
            "execution_block_reason": None,
            "blocked_candidate_count": 0,
            "durable_authority_debt_count": 0,
            "authority_quarantined_count": 0,
            "byte_authority_fragment_count": 0,
            "byte_authority_quarantined_count": 0,
            "byte_authority_pending_count": 0,
            "candidate_count": 0,
            "top_raw_rows": [],
            "origin_summary": [],
            "source_path_summary": [],
        }
    candidates = _raw_materialization_candidate_ids(config)
    raw_ids_by_size = sorted(
        candidates.raw_ids,
        key=lambda raw_id: (-candidates.raw_blob_bytes.get(raw_id, 0), raw_id),
    )
    top_raw_rows = [
        {
            "raw_id": raw_id,
            "origin": candidates.raw_origins.get(raw_id) or "unknown",
            "source_path": candidates.raw_source_paths.get(raw_id) or "",
            "blob_size": candidates.raw_blob_bytes.get(raw_id, 0),
            "oversized": candidates.raw_blob_bytes.get(raw_id, 0) > RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES,
            "stream_safe": _raw_materialization_stream_safe(candidates, raw_id),
        }
        for raw_id in raw_ids_by_size[:limit]
    ]
    expanded_blob_bytes = candidates.expanded_blob_bytes
    expanded_total_blob_bytes = sum(expanded_blob_bytes.values())
    oversized_raw_ids = [
        raw_id for raw_id, size in expanded_blob_bytes.items() if size > RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES
    ]
    oversized_stream_safe = [
        raw_id for raw_id in oversized_raw_ids if _raw_materialization_stream_safe(candidates, raw_id)
    ]
    # Retained-raw authority replay currently loads blob bytes before handing
    # them to the stream parser, so stream-capable format is diagnostic only.
    blocked_components = [
        component
        for component in candidates.authority_components
        if sum(expanded_blob_bytes.get(raw_id, 0) for raw_id in component)
        > RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES
    ]
    blocked_component_raw_ids = {raw_id for component in blocked_components for raw_id in component}
    aggregate_resource_blocked = bool(blocked_components)
    resource_blocked_count = len(blocked_component_raw_ids)
    blocked_candidate_count = resource_blocked_count + candidates.adoption_deferred + candidates.byte_authority_pending
    durable_authority_debt_count = (
        candidates.authority_quarantined
        + candidates.byte_authority_fragments
        + candidates.byte_authority_quarantined
        + candidates.byte_authority_pending
    )
    return {
        "available": True,
        "execution_blocked": blocked_candidate_count > 0,
        "execution_block_reason": (
            RAW_MATERIALIZATION_RESOURCE_BLOCK_REASON
            if resource_blocked_count
            else "revision adoption deferred behind incomparable existing index state"
            if candidates.adoption_deferred
            else "append fragments remain pending byte-authority adjudication"
            if candidates.byte_authority_pending
            else None
        ),
        "blocked_candidate_count": blocked_candidate_count,
        "durable_authority_debt_count": durable_authority_debt_count,
        "authority_quarantined_count": candidates.authority_quarantined,
        "byte_authority_fragment_count": candidates.byte_authority_fragments,
        "byte_authority_quarantined_count": candidates.byte_authority_quarantined,
        "byte_authority_pending_count": candidates.byte_authority_pending,
        "adoption_deferred_count": candidates.adoption_deferred,
        "candidate_count": len(candidates.raw_ids),
        "missing_blob_count": candidates.missing_blobs,
        "missing_blob_source_available_count": candidates.missing_blob_source_available,
        "missing_blob_source_missing_count": candidates.missing_blob_source_missing,
        "already_parsed_count": candidates.already_parsed,
        "total_blob_bytes": candidates.total_blob_bytes,
        "max_blob_bytes": candidates.max_blob_bytes,
        "execute_blob_limit_bytes": RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES,
        "expanded_candidate_count": len(candidates.expanded_raw_ids),
        "expanded_total_blob_bytes": expanded_total_blob_bytes,
        "expanded_aggregate_blocked": aggregate_resource_blocked,
        "authority_component_count": len(candidates.authority_components),
        "blocked_authority_component_count": len(blocked_components),
        "executable_authority_component_count": len(candidates.authority_components) - len(blocked_components),
        "oversized_count": len(oversized_raw_ids),
        "oversized_stream_safe_count": len(oversized_stream_safe),
        "top_raw_rows": top_raw_rows,
        "origin_summary": _raw_materialization_bucket_summary(
            candidates,
            values=candidates.raw_origins,
            key_name="origin",
            limit=limit,
        ),
        "source_path_summary": _raw_materialization_bucket_summary(
            candidates,
            values=candidates.raw_source_paths,
            key_name="source_path",
            limit=limit,
        ),
    }


def _raw_materialized_by_source_path_native(conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
    origin = str(row["origin"] or "")
    if not origin:
        return False
    for native_id in _source_path_native_id_candidates(str(row["source_path"] or "")):
        existing = conn.execute(
            """
            SELECT 1
            FROM index_tier.sessions AS s
            JOIN raw_sessions AS existing_raw ON existing_raw.raw_id = s.raw_id
            WHERE s.origin = ?
              AND s.native_id = ?
            LIMIT 1
            """,
            (origin, native_id),
        ).fetchone()
        if existing is not None:
            return True
    return False


def _raw_materialization_parsed_non_session_artifact(archive_root: Path, row: sqlite3.Row) -> bool:
    keys = set(row.keys())
    parse_error = row["parse_error"] if "parse_error" in keys else None
    parsed_at_ms = row["parsed_at_ms"] if "parsed_at_ms" in keys else None
    blob_hash = row["blob_hash"] if "blob_hash" in keys else None
    if parse_error or parsed_at_ms is None:
        return False
    return (
        parsed_non_session_artifact_reason(
            archive_root=archive_root,
            origin=str(row["origin"] or ""),
            source_path=str(row["source_path"] or ""),
            blob_hash=blob_hash,
        )
        is not None
    )


def _source_path_native_id_candidates(source_path: str) -> tuple[str, ...]:
    if not source_path:
        return ()
    name = Path(source_path).name
    candidates: list[str] = []
    current = name
    for _ in range(4):
        stem = Path(current).stem
        if stem == current:
            break
        current = stem
        if current and current not in candidates:
            candidates.append(current)
        unsplit = re.sub(r"_\d+$", "", current)
        if unsplit and unsplit != current and unsplit not in candidates:
            candidates.append(unsplit)
    return tuple(candidates)


def _open_archive_index_connection() -> sqlite3.Connection:
    from polylogue.paths import active_index_db_path

    conn = sqlite3.connect(active_index_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _resolve_convergence_debt(
    *,
    ops_db: Path,
    stage: str,
    target_type: str,
    target_id: str,
) -> None:
    """Best-effort resolution for ops-tier convergence debt.

    Maintenance targets are explicit convergence actuators. When one proves a
    target ready, stale daemon debt for the same target must stop appearing as
    actionable work.
    """
    if not ops_db.exists():
        return
    try:
        with sqlite3.connect(ops_db) as conn:
            table_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='convergence_debt'"
            ).fetchone()
            if not table_exists:
                return
            conn.execute(
                """
                DELETE FROM convergence_debt
                WHERE stage = ? AND target_type = ? AND target_id = ?
                """,
                (stage, target_type, target_id),
            )
            conn.commit()
    except sqlite3.Error as exc:
        logger.warning(
            "convergence_debt_resolve_failed",
            stage=stage,
            target_type=target_type,
            target_id=target_id,
            error=str(exc),
        )


def _resolve_session_insight_convergence_debt(
    *,
    ops_db: Path,
    session_ids: tuple[str, ...] | None,
) -> None:
    """Clear proven session-insight convergence debt after maintenance repair."""
    if not ops_db.exists():
        return
    try:
        with sqlite3.connect(ops_db) as conn:
            table_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='convergence_debt'"
            ).fetchone()
            if not table_exists:
                return
            if session_ids is None:
                conn.execute(
                    """
                    DELETE FROM convergence_debt
                    WHERE stage = 'insights'
                      AND target_type = 'session_id'
                    """
                )
            else:
                for session_id in session_ids:
                    conn.execute(
                        """
                        DELETE FROM convergence_debt
                        WHERE stage = 'insights'
                          AND target_type = 'session_id'
                          AND target_id = ?
                        """,
                        (session_id,),
                    )
            conn.commit()
    except sqlite3.Error as exc:
        logger.warning(
            "session_insight_convergence_debt_resolve_failed",
            session_ids=session_ids,
            error=str(exc),
        )


def _session_insight_materializer_version() -> int:
    from polylogue.storage.runtime import SESSION_INSIGHT_MATERIALIZER_VERSION

    return SESSION_INSIGHT_MATERIALIZER_VERSION


def _session_insight_requires_archive_wide_rebuild(status: object) -> bool:
    return any(
        int(getattr(status, attr, 0) or 0) > 0
        for attr in (
            "orphan_profile_row_count",
            "orphan_latency_profile_row_count",
            "orphan_work_event_inference_count",
            "orphan_phase_inference_count",
            "stale_day_summary_count",
        )
    )


def _session_insight_aggregate_debt_count(status: object) -> int:
    return sum(
        int(getattr(status, attr, 0) or 0)
        for attr in (
            "missing_thread_materialization_count",
            "stale_thread_count",
            "orphan_thread_count",
            "stale_tag_rollup_count",
            "stale_day_summary_count",
        )
    )


def _targeted_session_insight_rebuild_ids(
    conn: sqlite3.Connection | None,
    status: object,
) -> tuple[str, ...] | None:
    if conn is None or _session_insight_requires_archive_wide_rebuild(status):
        return None

    materialization_selects = "\nUNION\n".join(
        """
        SELECT s.session_id
        FROM sessions AS s
        WHERE NOT EXISTS (
            SELECT 1
            FROM insight_materialization AS m
            WHERE m.insight_type = ?
              AND m.session_id = s.session_id
              AND m.materializer_version = ?
              AND ABS(COALESCE(m.source_sort_key_ms, 0) - COALESCE(s.sort_key_ms, 0)) = 0
        )
        """
        for _insight_type in SESSION_INSIGHT_MATERIALIZATION_TYPES
        if _insight_type != "thread"
    )
    materializer_version = _session_insight_materializer_version()
    rows = conn.execute(
        f"""
        SELECT DISTINCT session_id
        FROM (
            SELECT s.session_id
            FROM sessions AS s
            WHERE NOT EXISTS (
                SELECT 1 FROM session_profiles AS p WHERE p.session_id = s.session_id
            )
            UNION
            SELECT s.session_id
            FROM sessions AS s
            JOIN session_profiles AS p ON p.session_id = s.session_id
            WHERE p.materializer_version != ?
               OR ABS(COALESCE(p.source_sort_key, 0.0) - COALESCE(CAST(s.sort_key_ms AS REAL)/1000.0, 0.0)) > 0.000001
            UNION
            SELECT p.session_id
            FROM session_profiles AS p
            JOIN sessions AS s ON s.session_id = p.session_id
            WHERE NOT EXISTS (
                SELECT 1 FROM session_latency_profiles AS lp WHERE lp.session_id = p.session_id
            )
            UNION
            SELECT lp.session_id
            FROM session_latency_profiles AS lp
            JOIN sessions AS s ON s.session_id = lp.session_id
            WHERE lp.materializer_version != ?
               OR ABS(COALESCE(lp.source_sort_key, 0.0) - COALESCE(CAST(s.sort_key_ms AS REAL)/1000.0, 0.0)) > 0.000001
            UNION
            SELECT p.session_id
            FROM session_profiles AS p
            WHERE p.work_event_count != (
                SELECT COUNT(*) FROM session_work_events AS e WHERE e.session_id = p.session_id
            )
            UNION
            SELECT p.session_id
            FROM session_profiles AS p
            WHERE p.phase_count != (
                SELECT COUNT(*) FROM session_phases AS ph WHERE ph.session_id = p.session_id
            )
            UNION
            {materialization_selects}
        )
        ORDER BY session_id
        """,
        (
            materializer_version,
            materializer_version,
            *(
                value
                for insight_type in SESSION_INSIGHT_MATERIALIZATION_TYPES
                if insight_type != "thread"
                for value in (insight_type, materializer_version)
            ),
        ),
    ).fetchall()
    return tuple(str(row["session_id"] if isinstance(row, sqlite3.Row) else row[0]) for row in rows)


def _archive_index_present(config: Config) -> bool:
    index_db = config.archive_root / "index.db"
    if not index_db.exists():
        return False
    try:
        with sqlite3.connect(f"file:{index_db}?mode=ro", uri=True) as conn:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
    except sqlite3.Error:
        return False
    return version > 0


def offline_maintenance_blockers(
    config: Config,
    *,
    repair: bool,
    cleanup: bool,
    dry_run: bool,
    targets: tuple[str, ...] = (),
) -> list[RepairResult]:
    detail = offline_maintenance_block_reason(config, active=repair or cleanup, dry_run=dry_run)
    if detail is None:
        return []
    selected_targets = targets or tuple(SAFE_REPAIR_TARGETS if repair else ()) + tuple(
        CLEANUP_TARGETS if cleanup else ()
    )
    return [
        _repair_result(target_name, repaired_count=0, success=False, detail=detail) for target_name in selected_targets
    ]


@dataclass
class RepairResult:
    name: str
    category: MaintenanceCategory
    destructive: bool
    repaired_count: int
    success: bool
    detail: str = ""
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "name": self.name,
                "category": self.category.value,
                "destructive": self.destructive,
                "repaired_count": self.repaired_count,
                "success": self.success,
                "detail": self.detail,
                "metrics": dict(self.metrics),
            }
        )


# ---------------------------------------------------------------------------
# Orphan count queries (formerly archive_debt_counts)
# ---------------------------------------------------------------------------


def count_orphaned_messages_sync(conn: sqlite3.Connection) -> int:
    """Count messages whose parent session row is missing.

    keys each message to ``sessions`` via
    ``messages.session_id REFERENCES sessions(session_id) ON DELETE
    CASCADE``. The cascade makes a message without its session
    structurally impossible — deleting a session deletes its messages in
    the same statement. This query therefore reports the honest native
    invariant: it joins ``messages`` to ``sessions`` and counts the rows
    with no matching session, which is always 0 on a consistent archive
    archive. It is retained as an integrity probe so a corrupted file
    (FK disabled during a hand edit) is still observable.
    """
    return int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM messages m
            LEFT JOIN sessions s ON s.session_id = m.session_id
            WHERE s.session_id IS NULL
            """
        ).fetchone()[0]
    )


def has_orphaned_messages_sync(conn: sqlite3.Connection) -> bool:
    return bool(
        conn.execute(
            """
            SELECT 1
            FROM messages m
            LEFT JOIN sessions s ON s.session_id = m.session_id
            WHERE s.session_id IS NULL
            LIMIT 1
            """
        ).fetchone()
    )


def count_empty_sessions_sync(conn: sqlite3.Connection) -> int:
    """Count sessions that carry no messages.

    The native session/message tree replaces the legacy
    session/message tables: an "empty session" is a ``sessions``
    row with no ``messages`` row referencing it.
    """
    return int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.session_id
            WHERE m.session_id IS NULL
            """
        ).fetchone()[0]
    )


def count_orphaned_attachments_sync(conn: sqlite3.Connection) -> int:
    """Count attachment refs without a parent and attachments without refs.

    Native ``attachment_refs`` keys to ``sessions``/``messages`` with
    ``ON DELETE CASCADE`` / ``SET NULL``; ``attachments`` carry a
    materialized ``ref_count``. A ref without a live parent or an
    attachment with no surviving ref is the archive orphan signature.
    """
    orphaned_refs = int(
        conn.execute(
            """
            SELECT COUNT(*) FROM attachment_refs ar
            WHERE (ar.message_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.message_id = ar.message_id))
               OR NOT EXISTS (SELECT 1 FROM sessions s WHERE s.session_id = ar.session_id)
            """
        ).fetchone()[0]
    )
    unreferenced_attachments = int(
        conn.execute(
            """
            SELECT COUNT(*) FROM attachments a
            WHERE NOT EXISTS (SELECT 1 FROM attachment_refs ar WHERE ar.attachment_id = a.attachment_id)
            """
        ).fetchone()[0]
    )
    return orphaned_refs + unreferenced_attachments


def _table_has_more_than(conn: sqlite3.Connection, table_name: str, row_limit: int) -> bool:
    row = conn.execute(f"SELECT 1 FROM {table_name} LIMIT 1 OFFSET ?", (max(0, row_limit),)).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Derived repair count helpers (formerly archive_debt_repairs)
# ---------------------------------------------------------------------------


def session_insight_repair_count(derived_statuses: dict[str, DerivedModelStatus]) -> int:
    keys = [
        "session_profile_rows",
        "session_work_events",
        "session_work_events_fts",
        "session_phases",
        "threads",
        "threads_fts",
        "session_tag_rollups",
    ]
    maybe_statuses = [derived_statuses.get(k) for k in keys]
    if not all(status is not None for status in maybe_statuses):
        return 0
    statuses = [status for status in maybe_statuses if status is not None]
    total = 0
    for s in statuses:
        total += max(0, int(s.pending_documents or 0))
        total += max(0, int(s.pending_rows or 0))
        total += max(0, int(s.stale_rows or 0))
        total += max(0, int(s.orphan_rows or 0))
    return total


# ---------------------------------------------------------------------------
# Archive debt collection (formerly archive_debt.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchiveDebtStatus:
    """Simple debt/orphan status for a single maintenance target."""

    name: str
    category: MaintenanceCategory
    destructive: bool
    issue_count: int
    detail: str
    maintenance_target: str
    skipped: bool = False

    @property
    def healthy(self) -> bool:
        return self.issue_count == 0 and not self.skipped

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "name": self.name,
                "category": self.category.value,
                "destructive": self.destructive,
                "issue_count": self.issue_count,
                "detail": self.detail,
                "maintenance_target": self.maintenance_target,
                "healthy": self.healthy,
                "skipped": self.skipped,
            }
        )


def _maintenance_target_spec(name: str) -> MaintenanceTargetSpec:
    spec = _MAINTENANCE_TARGET_CATALOG.resolve_name(name)
    if spec is None:
        raise KeyError(f"Unknown maintenance target: {name}")
    return spec


def _repair_result(
    target_name: str,
    *,
    repaired_count: int,
    success: bool,
    detail: str,
    metrics: dict[str, float] | None = None,
) -> RepairResult:
    spec = _maintenance_target_spec(target_name)
    return RepairResult(
        name=spec.name,
        category=spec.category,
        destructive=spec.destructive,
        repaired_count=repaired_count,
        success=success,
        detail=detail,
        metrics=dict(metrics or {}),
    )


def _internal_derived_repair_result(
    name: str,
    *,
    repaired_count: int,
    success: bool,
    detail: str,
    metrics: dict[str, float] | None = None,
) -> RepairResult:
    return RepairResult(
        name=name,
        category=MaintenanceCategory.DERIVED_REPAIR,
        destructive=False,
        repaired_count=repaired_count,
        success=success,
        detail=detail,
        metrics=dict(metrics or {}),
    )


def _archive_debt_status(
    target_name: str,
    *,
    issue_count: int,
    detail: str,
    skipped: bool = False,
) -> ArchiveDebtStatus:
    spec = _maintenance_target_spec(target_name)
    return ArchiveDebtStatus(
        name=spec.name,
        category=spec.category,
        destructive=spec.destructive,
        issue_count=issue_count,
        detail=detail,
        maintenance_target=spec.name,
        skipped=skipped,
    )


def collect_archive_debt_statuses_sync(
    conn: sqlite3.Connection,
    *,
    db_path: Path | str | None = None,
    derived_statuses: dict[str, DerivedModelStatus] | None = None,
    include_expensive: bool = True,
    probe_only: bool = False,
    target_names: tuple[str, ...] = (),
) -> dict[str, ArchiveDebtStatus]:
    from polylogue.storage.derived.derived_status import collect_derived_model_statuses_sync

    selected = set(target_names) if target_names else set(_MAINTENANCE_TARGET_CATALOG.names())
    needs_session_insights = "session_insights" in selected
    statuses = (
        derived_statuses or collect_derived_model_statuses_sync(conn, verify_full=include_expensive)
        if needs_session_insights
        else {}
    )

    skip_large_message_scans = (
        probe_only
        and not include_expensive
        and _table_has_more_than(conn, "messages", _PROBE_ONLY_EXACT_MESSAGE_ROW_LIMIT)
    )
    debt_statuses: dict[str, ArchiveDebtStatus] = {}

    if "orphaned_messages" in selected:
        orphaned_messages = 0 if skip_large_message_scans else count_orphaned_messages_sync(conn)
        debt_statuses["orphaned_messages"] = _archive_debt_status(
            "orphaned_messages",
            issue_count=orphaned_messages,
            detail=(
                "Skipped exact orphaned-message scan in probe mode; use --deep for exact count"
                if skip_large_message_scans
                else "No orphaned messages"
                if orphaned_messages == 0
                else (
                    "Orphaned messages present; use --deep for exact count"
                    if probe_only and not include_expensive
                    else f"{orphaned_messages:,} orphaned messages"
                )
            ),
            skipped=skip_large_message_scans,
        )
    if "empty_sessions" in selected:
        empty_sessions = 0 if skip_large_message_scans else count_empty_sessions_sync(conn)
        debt_statuses["empty_sessions"] = _archive_debt_status(
            "empty_sessions",
            issue_count=empty_sessions,
            detail=(
                "Skipped exact empty-session scan in probe mode; use --deep for exact count"
                if skip_large_message_scans
                else "No empty sessions"
                if empty_sessions == 0
                else f"{empty_sessions:,} empty sessions"
            ),
            skipped=skip_large_message_scans,
        )
    if "orphaned_attachments" in selected:
        orphaned_attachments = count_orphaned_attachments_sync(conn)
        debt_statuses["orphaned_attachments"] = _archive_debt_status(
            "orphaned_attachments",
            issue_count=orphaned_attachments,
            detail="No orphaned attachments"
            if orphaned_attachments == 0
            else f"{orphaned_attachments:,} orphaned attachment rows",
        )
    if "session_insights" in selected:
        session_insights = session_insight_repair_count(statuses)
        debt_statuses["session_insights"] = _archive_debt_status(
            "session_insights",
            issue_count=session_insights,
            detail="Session insight read models ready"
            if session_insights == 0
            else f"{session_insights:,} pending/stale/orphaned session-insight rows",
        )
    if "message_type_backfill" in selected:
        unclassified = 0 if skip_large_message_scans else count_unclassified_message_type_sync(conn)
        debt_statuses["message_type_backfill"] = _archive_debt_status(
            "message_type_backfill",
            issue_count=unclassified,
            detail=(
                "Skipped exact message-type backfill scan in probe mode; use --deep for exact count"
                if skip_large_message_scans
                else "No messages need context/protocol classification"
                if unclassified == 0
                else f"{unclassified:,} messages would be classified as context or protocol"
            ),
            skipped=skip_large_message_scans,
        )
    if include_expensive and "orphaned_blobs" in selected:
        orphaned_blobs = count_orphaned_blobs_sync(conn, db_path=db_path)
        debt_statuses["orphaned_blobs"] = _archive_debt_status(
            "orphaned_blobs",
            issue_count=orphaned_blobs,
            detail="No orphaned blobs" if orphaned_blobs == 0 else f"{orphaned_blobs:,} orphaned blob files on disk",
        )
    if include_expensive and "superseded_raw_snapshots" in selected:
        superseded_raw_snapshots = count_superseded_raw_snapshots_sync(conn)
        debt_statuses["superseded_raw_snapshots"] = _archive_debt_status(
            "superseded_raw_snapshots",
            issue_count=superseded_raw_snapshots,
            detail=(
                "No superseded live raw snapshots"
                if superseded_raw_snapshots == 0
                else f"{superseded_raw_snapshots:,} superseded live raw snapshots"
            ),
        )
    return debt_statuses


def preview_counts_from_archive_debt(
    statuses: dict[str, ArchiveDebtStatus],
) -> dict[str, int]:
    preview_targets = set(_MAINTENANCE_TARGET_CATALOG.preview_target_names())
    return {
        status.maintenance_target: status.issue_count
        for status in statuses.values()
        if status.issue_count > 0 or status.maintenance_target in preview_targets
    }


# ---------------------------------------------------------------------------
# Generic SQL repair helper
# ---------------------------------------------------------------------------


def _run_sql_repair(
    target_name: str,
    *,
    count_sql: str,
    action_sql: str | None,
    dry_run: bool,
    conn: sqlite3.Connection,
) -> RepairResult:
    try:
        count = conn.execute(count_sql).fetchone()[0]
        if dry_run:
            return _repair_result(
                target_name,
                repaired_count=count,
                success=True,
                detail=f"Would: {count} rows affected" if count else "Would: No issues found",
            )
        if action_sql:
            result = conn.execute(action_sql)
            conn.commit()
            return _repair_result(
                target_name,
                repaired_count=result.rowcount,
                success=True,
                detail=f"Repaired {result.rowcount} rows" if result.rowcount else "No repairs needed",
            )
        return _repair_result(
            target_name,
            repaired_count=0,
            success=True,
            detail="No action SQL provided",
        )
    except Exception as exc:
        return _repair_result(
            target_name,
            repaired_count=0,
            success=False,
            detail=f"Repair failed: {exc}",
        )


# ---------------------------------------------------------------------------
# Cleanup repairs (orphans, empty sessions, attachments)
# ---------------------------------------------------------------------------


def repair_orphaned_messages(config: Config, dry_run: bool = False) -> RepairResult:
    """Delete messages whose parent session row is missing.

    On the archive ``messages.session_id`` cascades from
    ``sessions``, so a session-less message can only exist after a
    file-level corruption (FK disabled during a hand edit). The repair
    counts such rows via :func:`count_orphaned_messages_sync` and, when
    any are found, deletes the orphan ``messages`` rows directly; the
    ``blocks`` rows beneath them cascade away through
    ``blocks.message_id REFERENCES messages ON DELETE CASCADE``.
    """
    with _open_archive_index_connection() as conn:
        count = count_orphaned_messages_sync(conn)
        if count == 0:
            return _repair_result(
                "orphaned_messages",
                repaired_count=0,
                success=True,
                detail="No orphaned messages found",
            )
        try:
            if dry_run:
                return _repair_result(
                    "orphaned_messages",
                    repaired_count=count,
                    success=True,
                    detail=f"Would: Delete {count} orphaned messages",
                )
            result = conn.execute(
                """
                DELETE FROM messages
                WHERE NOT EXISTS (
                    SELECT 1 FROM sessions s WHERE s.session_id = messages.session_id
                )
                """
            )
            conn.commit()
            return _repair_result(
                "orphaned_messages",
                repaired_count=result.rowcount,
                success=True,
                detail=f"Deleted {result.rowcount} orphaned messages",
            )
        except Exception as exc:
            return _repair_result(
                "orphaned_messages",
                repaired_count=0,
                success=False,
                detail=f"Failed to delete orphaned messages: {exc}",
            )


def preview_orphaned_messages(*, count: int) -> RepairResult:
    return _repair_result(
        "orphaned_messages",
        repaired_count=count,
        success=True,
        detail=f"Would: Delete {count} orphaned messages" if count else "Would: No orphaned messages found",
    )


def repair_empty_sessions(config: Config, dry_run: bool = False) -> RepairResult:
    with _open_archive_index_connection() as conn:
        return _run_sql_repair(
            "empty_sessions",
            count_sql="SELECT COUNT(*) FROM sessions c WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.session_id = c.session_id)",
            action_sql="DELETE FROM sessions WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.session_id = sessions.session_id)",
            dry_run=dry_run,
            conn=conn,
        )


def preview_empty_sessions(*, count: int) -> RepairResult:
    return _repair_result(
        "empty_sessions",
        repaired_count=count,
        success=True,
        detail=f"Would: {count} rows affected" if count else "Would: No issues found",
    )


# ---------------------------------------------------------------------------
# Blob cleanup
# ---------------------------------------------------------------------------


def repair_orphaned_blobs(config: Config, dry_run: bool = False) -> RepairResult:
    outcome = repair_orphaned_blobs_data(config, dry_run=dry_run)
    return _repair_result(
        "orphaned_blobs",
        repaired_count=outcome.repaired_count,
        success=outcome.success,
        detail=outcome.detail,
    )


def count_superseded_raw_snapshots_sync(conn: sqlite3.Connection) -> int:
    from polylogue.storage.raw_retention import superseded_raw_snapshot_candidates

    return len(superseded_raw_snapshot_candidates(conn, limit=10_000))


def repair_superseded_raw_snapshots(config: Config, dry_run: bool = False) -> RepairResult:
    from polylogue.storage.raw_retention import (
        RawRetentionSafetyError,
        active_raw_retention_authority,
        cleanup_superseded_raw_snapshots,
    )
    from polylogue.storage.sqlite.connection_profile import open_connection

    archive_root = _raw_materialization_archive_root(config)
    repair_db_path = archive_root / "source.db"
    if repair_db_path.exists():
        index_db_path = archive_root / "index.db"
        with closing(open_connection(repair_db_path)) as conn, conn:
            conn.row_factory = sqlite3.Row
            try:
                retention_authority = active_raw_retention_authority(
                    conn,
                    index_db_path=index_db_path,
                )
            except RawRetentionSafetyError as exc:
                return _repair_result(
                    "superseded_raw_snapshots",
                    repaired_count=0,
                    success=False,
                    detail=f"Skipped destructive raw cleanup: {exc}",
                )
            result = cleanup_superseded_raw_snapshots(
                conn,
                dry_run=dry_run,
                limit=10_000,
                protected_raw_ids=retention_authority.protected_raw_ids,
                eligible_raw_ids=retention_authority.eligible_raw_ids,
            )
    else:
        with closing(open_connection(config.db_path)) as conn, conn:
            try:
                retention_authority = active_raw_retention_authority(
                    conn,
                    index_db_path=config.db_path,
                )
            except RawRetentionSafetyError as exc:
                return _repair_result(
                    "superseded_raw_snapshots",
                    repaired_count=0,
                    success=False,
                    detail=f"Skipped destructive raw cleanup: {exc}",
                )
            result = cleanup_superseded_raw_snapshots(
                conn,
                dry_run=dry_run,
                limit=10_000,
                protected_raw_ids=retention_authority.protected_raw_ids,
                eligible_raw_ids=retention_authority.eligible_raw_ids,
            )
    if dry_run:
        skipped_detail = (
            f"; skipped {result.skipped_referenced_count:,} active revision raw rows"
            if result.skipped_referenced_count
            else ""
        )
        return _repair_result(
            "superseded_raw_snapshots",
            repaired_count=result.candidate_count,
            success=True,
            detail=(
                f"Would: delete {result.candidate_count:,} superseded raw snapshots "
                f"({result.deleted_raw_bytes:,} referenced bytes)"
                f"{skipped_detail}"
            ),
        )
    skipped_detail = (
        f"; skipped {result.skipped_referenced_count:,} active revision raw rows"
        if result.skipped_referenced_count
        else ""
    )
    return _repair_result(
        "superseded_raw_snapshots",
        repaired_count=result.deleted_raw_count,
        success=not result.errors,
        detail=(
            f"Deleted {result.deleted_raw_count:,} raw rows and {result.deleted_blob_count:,} blob files "
            f"({result.deleted_blob_bytes:,} bytes)"
            f"{skipped_detail}" + (f"; errors: {'; '.join(result.errors[:3])}" if result.errors else "")
        ),
    )


def preview_orphaned_blobs(*, count: int) -> RepairResult:
    return _repair_result(
        "orphaned_blobs",
        repaired_count=count,
        success=True,
        detail=f"Would: delete {count} orphaned blobs" if count else "Would: No orphaned blobs found",
    )


def preview_superseded_raw_snapshots(*, count: int) -> RepairResult:
    return _repair_result(
        "superseded_raw_snapshots",
        repaired_count=count,
        success=True,
        detail=(
            f"Would: delete {count} superseded live raw snapshots"
            if count
            else "Would: No superseded live raw snapshots found"
        ),
    )


def repair_orphaned_attachments(config: Config, dry_run: bool = False) -> RepairResult:
    try:
        with _open_archive_index_connection() as conn:
            if dry_run:
                return preview_orphaned_attachments(count=count_orphaned_attachments_sync(conn))

            ref_result = conn.execute(
                "DELETE FROM attachment_refs WHERE message_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.message_id = attachment_refs.message_id)"
            )
            refs_deleted = ref_result.rowcount

            conv_ref_result = conn.execute(
                "DELETE FROM attachment_refs WHERE NOT EXISTS (SELECT 1 FROM sessions c WHERE c.session_id = attachment_refs.session_id)"
            )
            conv_refs_deleted = conv_ref_result.rowcount

            att_result = conn.execute(
                "DELETE FROM attachments WHERE NOT EXISTS (SELECT 1 FROM attachment_refs ar WHERE ar.attachment_id = attachments.attachment_id)"
            )
            atts_deleted = att_result.rowcount
            conn.commit()

            total = refs_deleted + conv_refs_deleted + atts_deleted
            return _repair_result(
                "orphaned_attachments",
                repaired_count=total,
                success=True,
                detail=f"Cleaned {refs_deleted} orphaned refs, {conv_refs_deleted} conv refs, {atts_deleted} attachments",
            )
    except Exception as exc:
        return _repair_result(
            "orphaned_attachments",
            repaired_count=0,
            success=False,
            detail=f"Failed to clean orphaned attachments: {exc}",
        )


def preview_orphaned_attachments(*, count: int) -> RepairResult:
    return _repair_result(
        "orphaned_attachments",
        repaired_count=count,
        success=True,
        detail=f"Would: Clean {count} orphaned attachment rows" if count else "Would: No orphaned attachments found",
    )


# ---------------------------------------------------------------------------
# Derived repairs (session insights, actions, FTS, WAL)
# ---------------------------------------------------------------------------


def repair_session_insights(
    config: Config,
    dry_run: bool = False,
    *,
    progress_callback: ProgressCallback | None = None,
    progress_total: int | None = None,
    session_ids: tuple[str, ...] | None = None,
    archive_root_override: Path | None = None,
    owned_inactive_generation: tuple[str, str] | None = None,
) -> RepairResult:
    """Repair / rebuild session insights.

    When ``session_ids`` is given, the rebuild is narrowed to that
    set instead of touching the full archive — used by the maintenance
    planner to honor :class:`MaintenanceScopeFilter.session_ids`.
    """
    from polylogue.api.archive import _rebuild_archive_session_insights
    from polylogue.paths import active_index_db_path
    from polylogue.storage.insights.session.rebuild import refresh_session_insight_aggregates_sync
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    try:
        archive_root = archive_root_override or active_index_db_path().parent
        archive_context = (
            ArchiveStore.open_owned_inactive_generation(
                archive_root,
                generation_id=owned_inactive_generation[0],
                owner_id=owned_inactive_generation[1],
            )
            if owned_inactive_generation is not None
            else ArchiveStore.open_existing(archive_root, read_only=False)
        )
        with archive_context as archive:
            status = archive.session_insight_status()
            assessment = assess_session_insight_repairs(status)
            aggregate_debt = _session_insight_aggregate_debt_count(status)
            targeted_session_ids = (
                None
                if session_ids is not None or assessment.row_debt == 0
                else _targeted_session_insight_rebuild_ids(getattr(archive, "_conn", None), status)
            )

            if dry_run:
                if session_ids is not None:
                    pending = min(assessment.row_debt, len(session_ids))
                    detail = (
                        "Would: session insights already ready"
                        if pending == 0
                        else f"Would: rebuild session insights for {pending:,} scoped session(s)"
                    )
                elif targeted_session_ids is not None:
                    pending = len(targeted_session_ids) + aggregate_debt
                    detail = (
                        "Would: session insights already ready"
                        if pending == 0
                        else (
                            "Would: rebuild session insights for "
                            f"{len(targeted_session_ids):,} candidate session(s)"
                            f" and refresh {aggregate_debt:,} aggregate/thread-materialization debt row(s)"
                            f" to repair {assessment.row_debt:,} total debt row(s)"
                        )
                    )
                elif assessment.row_debt == 0:
                    pending = 0
                    detail = "Would: session insights already ready"
                else:
                    pending = status.total_sessions
                    detail = (
                        "Would: rebuild archive-wide session insights "
                        f"for {pending:,} session(s) to repair {assessment.row_debt:,} debt row(s)"
                    )
                return _repair_result(
                    "session_insights",
                    repaired_count=pending,
                    success=True,
                    detail=detail,
                )

            if session_ids is None and assessment.row_debt == 0:
                return _repair_result(
                    "session_insights",
                    repaired_count=0,
                    success=True,
                    detail="Session insights already ready",
                )

            rebuild_session_ids = session_ids if session_ids is not None else targeted_session_ids
            rebuilt = _rebuild_archive_session_insights(
                archive,
                session_ids=rebuild_session_ids,
                progress_callback=progress_callback,
            )
            rebuilt_count = rebuilt.total()
            refreshed = archive.session_insight_status()
            if session_ids is None and _session_insight_aggregate_debt_count(refreshed) > 0:
                aggregate_counts = refresh_session_insight_aggregates_sync(
                    archive._conn,
                    progress_callback=progress_callback,
                )
                rebuilt_count += aggregate_counts.total()
                refreshed = archive.session_insight_status()
            # A narrowed rebuild only attests its own slice; do not
            # demand global readiness for a scope-filtered call.
            success = True if session_ids is not None else assess_session_insight_repairs(refreshed).row_debt == 0
            if success:
                _resolve_session_insight_convergence_debt(
                    ops_db=config.archive_root / "ops.db",
                    session_ids=session_ids,
                )
            return _repair_result(
                "session_insights",
                repaired_count=rebuilt_count,
                success=success,
                detail="Session insights ready" if success else "Session insights still incomplete",
            )
    except Exception as exc:
        return _repair_result(
            "session_insights",
            repaired_count=0,
            success=False,
            detail=f"Failed to repair session insights: {exc}",
        )


def preview_session_insights(*, count: int) -> RepairResult:
    return _repair_result(
        "session_insights",
        repaired_count=count,
        success=True,
        detail="Would: session insights already ready"
        if count == 0
        else f"Would: rebuild session-insight rows/fts for {count:,} pending items",
    )


def repair_raw_materialization(
    config: Config,
    dry_run: bool = False,
    *,
    raw_artifact_id: str | None = None,
    provider: str | None = None,
    source_family: str | None = None,
    source_root: Path | None = None,
    raw_artifact_limit: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> RepairResult:
    """Converge retained raws through typed per-session revision authority."""
    candidates = _raw_materialization_candidate_ids(
        config,
        raw_artifact_id=raw_artifact_id,
        provider=provider,
        source_family=source_family,
        source_root=source_root,
    )
    candidate_raw_ids = candidates.raw_ids
    raw_ids = list(candidate_raw_ids)
    if raw_artifact_limit is not None:
        raw_ids = sorted(raw_ids, key=lambda raw_id: (candidates.raw_blob_bytes.get(raw_id, 0), raw_id))
        raw_ids = raw_ids[:raw_artifact_limit]
    missing_blobs = candidates.missing_blobs
    selected_total_bytes = _raw_materialization_total_bytes(candidates, raw_ids)
    selected_max_bytes = _raw_materialization_max_bytes(candidates, raw_ids)
    metrics = {
        "raw_materialization_candidate_count": float(len(candidate_raw_ids)),
        "raw_materialization_selected_count": float(len(raw_ids)),
        "raw_materialization_missing_blob_count": float(missing_blobs),
        "raw_materialization_missing_blob_source_available_count": float(candidates.missing_blob_source_available),
        "raw_materialization_missing_blob_source_missing_count": float(candidates.missing_blob_source_missing),
        "raw_materialization_already_parsed_count": float(candidates.already_parsed),
        "raw_materialization_total_blob_bytes": float(candidates.total_blob_bytes),
        "raw_materialization_max_blob_bytes": float(candidates.max_blob_bytes),
        "raw_materialization_selected_total_blob_bytes": float(selected_total_bytes),
        "raw_materialization_selected_max_blob_bytes": float(selected_max_bytes),
        "raw_materialization_adoption_deferred_count": float(candidates.adoption_deferred),
        "raw_materialization_authority_quarantined_count": float(candidates.authority_quarantined),
        "raw_materialization_byte_authority_fragment_count": float(candidates.byte_authority_fragments),
        "raw_materialization_byte_authority_quarantined_count": float(candidates.byte_authority_quarantined),
        "raw_materialization_byte_authority_pending_count": float(candidates.byte_authority_pending),
    }
    if raw_artifact_limit is not None:
        metrics["raw_materialization_limit"] = float(raw_artifact_limit)
    oversized_candidate_raw_ids = [
        raw_id
        for raw_id in raw_ids
        if candidates.raw_blob_bytes.get(raw_id, 0) > RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES
    ]
    oversized_stream_safe_raw_ids = [
        raw_id for raw_id in oversized_candidate_raw_ids if _raw_materialization_stream_safe(candidates, raw_id)
    ]
    # The retained-raw reader materializes bytes before stream parsing, so
    # stream-capable format is diagnostic only until that reader is replaced.
    oversized_raw_ids = oversized_candidate_raw_ids
    if oversized_raw_ids:
        metrics["raw_materialization_oversized_count"] = float(len(oversized_raw_ids))
        metrics["raw_materialization_resource_blocked_count"] = float(len(oversized_raw_ids))
        metrics["raw_materialization_execute_blob_limit_bytes"] = float(RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES)
    if oversized_stream_safe_raw_ids:
        metrics["raw_materialization_stream_oversized_count"] = float(len(oversized_stream_safe_raw_ids))
    if not raw_ids:
        detail = "Executable raw replay converged"
        if (
            candidates.authority_quarantined
            or candidates.byte_authority_fragments
            or candidates.byte_authority_quarantined
            or candidates.byte_authority_pending
        ):
            detail += (
                f"; {candidates.authority_quarantined:,} explicit authority quarantine(s), "
                f"{candidates.byte_authority_fragments:,} byte-authority fragment(s) excluded from replay, "
                f"{candidates.byte_authority_quarantined:,} append authority quarantine(s), "
                f"{candidates.byte_authority_pending:,} append fragment(s) pending byte-authority adjudication"
            )
        if candidates.adoption_deferred:
            detail = (
                f"Raw materialization blocked: {candidates.adoption_deferred:,} revision adoption decision(s) "
                "remain deferred behind incomparable existing index state"
            )
        if missing_blobs:
            detail += f"; {_raw_materialization_missing_blob_detail(candidates, final=True)}"
        return _internal_derived_repair_result(
            "raw_materialization",
            repaired_count=0,
            success=(
                missing_blobs == 0
                and candidates.adoption_deferred == 0
                and candidates.byte_authority_pending == 0
                and (
                    raw_artifact_id is None
                    or (
                        candidates.authority_quarantined == 0
                        and candidates.byte_authority_fragments == 0
                        and candidates.byte_authority_quarantined == 0
                    )
                )
            ),
            detail=detail,
            metrics=metrics,
        )
    if dry_run:
        detail = (
            f"Would: classify and replay {len(raw_ids):,} selected raw row(s) through per-session revision authority; "
            f"selected raw payload bytes total={_format_bytes(selected_total_bytes)}, "
            f"largest={_format_bytes(selected_max_bytes)}"
        )
        if candidates.already_parsed:
            detail += f"; {candidates.already_parsed:,} already parsed but not materialized"
        if missing_blobs:
            detail += f"; {_raw_materialization_missing_blob_detail(candidates, final=True)}"
        if oversized_raw_ids:
            detail += (
                f"; {len(oversized_raw_ids):,} raw rows exceed replay size advisory "
                f"{_format_bytes(RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES)}"
            )
        if oversized_stream_safe_raw_ids:
            detail += f"; {len(oversized_stream_safe_raw_ids):,} oversized stream-record raw rows are stream-capable"
        return _internal_derived_repair_result(
            "raw_materialization", repaired_count=0, success=False, detail=detail, metrics=metrics
        )

    from polylogue.sources.revision_backfill import (
        RawRevisionReplayResourceBlockedError,
        backfill_historical_revision_evidence,
    )

    archive_root = _raw_materialization_archive_root(config)
    blocked_component_raw_ids = {
        raw_id
        for component in candidates.authority_components
        if sum(candidates.expanded_blob_bytes.get(member, 0) for member in component)
        > RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES
        for raw_id in component
    }
    if blocked_component_raw_ids:
        metrics["raw_materialization_resource_blocked_count"] = float(len(blocked_component_raw_ids))
    executable_raw_ids = [raw_id for raw_id in raw_ids if raw_id not in blocked_component_raw_ids]
    metrics["raw_materialization_executed_count"] = float(len(executable_raw_ids))
    if executable_raw_ids:
        try:
            replay = backfill_historical_revision_evidence(
                archive_root,
                selected_raw_ids=executable_raw_ids,
                max_payload_bytes=RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES,
            )
        except RawRevisionReplayResourceBlockedError as exc:
            metrics["raw_materialization_resource_blocked_count"] = float(len(exc.raw_ids))
            metrics["raw_materialization_executed_count"] = 0.0
            return _internal_derived_repair_result(
                "raw_materialization",
                repaired_count=0,
                success=False,
                detail=(
                    f"Raw materialization blocked: {len(exc.raw_ids):,} expanded-cohort raw row(s), "
                    f"aggregate {_format_bytes(exc.total_bytes)}, exceed execution limit "
                    f"{_format_bytes(exc.limit_bytes)}"
                ),
                metrics=metrics,
            )
    else:
        from polylogue.sources.revision_backfill import RevisionBackfillResult

        replay = RevisionBackfillResult(scanned=0, classified_full=0, replayed_logical_sources=0, quarantined=0)
    remaining = _raw_materialization_candidate_ids(
        config,
        raw_artifact_id=raw_artifact_id,
        provider=provider,
        source_family=source_family,
        source_root=source_root,
    )
    metrics.update(
        {
            "raw_materialization_scanned_raw_count": float(replay.scanned),
            "raw_materialization_classified_full_count": float(replay.classified_full),
            "raw_materialization_replayed_logical_source_count": float(replay.replayed_logical_sources),
            "raw_materialization_quarantined_count": float(replay.quarantined),
            "raw_materialization_adoption_deferred_count": float(replay.adoption_deferred),
            "raw_materialization_remaining_candidate_count": float(len(remaining.raw_ids)),
            "raw_materialization_remaining_authority_quarantined_count": float(remaining.authority_quarantined),
            "raw_materialization_remaining_byte_authority_fragment_count": float(remaining.byte_authority_fragments),
            "raw_materialization_remaining_byte_authority_quarantined_count": float(
                remaining.byte_authority_quarantined
            ),
            "raw_materialization_remaining_byte_authority_pending_count": float(remaining.byte_authority_pending),
        }
    )
    success = (
        not remaining.raw_ids
        and remaining.missing_blobs == 0
        and replay.adoption_deferred == 0
        and remaining.adoption_deferred == 0
        and remaining.byte_authority_pending == 0
        and (
            raw_artifact_id is None
            or (
                remaining.authority_quarantined == 0
                and remaining.byte_authority_fragments == 0
                and remaining.byte_authority_quarantined == 0
            )
        )
    )
    detail = (
        f"Replayed {replay.replayed_logical_sources:,} logical source(s) through typed revision authority; "
        f"{len(remaining.raw_ids):,} replay candidate(s) remain"
    )
    if replay.quarantined:
        detail += f"; {replay.quarantined:,} ambiguous/legacy revision(s) quarantined"
    if replay.adoption_deferred:
        detail += f"; {replay.adoption_deferred:,} revision(s) deferred behind incomparable existing index state"
    if remaining.adoption_deferred:
        detail += f"; {remaining.adoption_deferred:,} deferred adoption decision(s) remain blocked"
    if (
        remaining.authority_quarantined
        or remaining.byte_authority_fragments
        or remaining.byte_authority_quarantined
        or remaining.byte_authority_pending
    ):
        detail += (
            f"; durable authority debt: {remaining.authority_quarantined:,} quarantine(s), "
            f"{remaining.byte_authority_fragments:,} governed append fragment(s), "
            f"{remaining.byte_authority_quarantined:,} append authority quarantine(s), "
            f"{remaining.byte_authority_pending:,} append fragment(s) pending adjudication"
        )
    if oversized_raw_ids:
        detail += (
            f"; {len(oversized_raw_ids):,} non-stream-safe raw row(s) exceed execution limit "
            f"{_format_bytes(RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES)}"
        )
    elif blocked_component_raw_ids:
        detail += (
            f"; {len(blocked_component_raw_ids):,} raw row(s) belong to authority components whose aggregate "
            f"payload exceeds {_format_bytes(RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES)}"
        )
    if remaining.missing_blobs:
        detail += f"; {_raw_materialization_missing_blob_detail(remaining, final=True)}"
    if progress_callback is not None:
        progress_callback(replay.replayed_logical_sources, detail)
    return _internal_derived_repair_result(
        "raw_materialization",
        repaired_count=replay.replayed_logical_sources,
        success=success,
        detail=detail,
        metrics=metrics,
    )


def _to_repair_result(result: BackfillResult) -> RepairResult:
    """Adapt a ``BackfillResult`` to the shared ``RepairResult`` shape."""
    return RepairResult(
        name=result.name,
        category=result.category,
        destructive=result.destructive,
        repaired_count=result.repaired_count,
        success=result.success,
        detail=result.detail,
    )


def preview_message_type_backfill(*, count: int) -> RepairResult:
    """Preview handler for the #839 message_type backfill.

    Thin shim over ``message_type_backfill.preview_backfill`` so the
    repair orchestrator's preview dispatch keeps working.
    """
    from polylogue.storage.message_type_backfill import preview_backfill

    return _to_repair_result(preview_backfill(count=count))


def repair_message_type_backfill(config: Config, dry_run: bool = False) -> RepairResult:
    """Backfill ``message_type`` for pre-#839 rows.

    Delegates to ``storage.message_type_backfill.run_backfill``; the
    implementation lives there to keep this module under its file-size
    budget (see ``docs/plans/file-size-budgets.yaml``).
    """
    from polylogue.storage.message_type_backfill import run_backfill

    return _to_repair_result(run_backfill(config, dry_run=dry_run))


_PREVIEW_HANDLERS: dict[str, Callable[..., RepairResult]] = {
    "session_insights": preview_session_insights,
    "message_type_backfill": preview_message_type_backfill,
    "orphaned_messages": preview_orphaned_messages,
    "empty_sessions": preview_empty_sessions,
    "orphaned_attachments": preview_orphaned_attachments,
    "orphaned_blobs": preview_orphaned_blobs,
    "superseded_raw_snapshots": preview_superseded_raw_snapshots,
}


_REPAIR_HANDLERS: dict[str, Callable[..., RepairResult]] = {
    "session_insights": repair_session_insights,
    "message_type_backfill": repair_message_type_backfill,
    "orphaned_messages": repair_orphaned_messages,
    "empty_sessions": repair_empty_sessions,
    "orphaned_attachments": repair_orphaned_attachments,
    "orphaned_blobs": repair_orphaned_blobs,
    "superseded_raw_snapshots": repair_superseded_raw_snapshots,
}


# ---------------------------------------------------------------------------
# Orchestration (run_safe_repairs, run_archive_cleanup, run_selected_maintenance)
# ---------------------------------------------------------------------------


def run_safe_repairs(
    config: Config,
    dry_run: bool = False,
    *,
    preview_counts: dict[str, int] | None = None,
    targets: tuple[str, ...] = (),
    session_insight_progress_callback: ProgressCallback | None = None,
    session_insight_progress_total: int | None = None,
) -> list[RepairResult]:
    preview_counts = preview_counts or {}
    selected = set(targets) if targets else set(SAFE_REPAIR_TARGETS)
    results: list[RepairResult] = []
    for target_name in SAFE_REPAIR_TARGETS:
        if target_name not in selected:
            continue
        if dry_run and target_name in preview_counts:
            preview = _PREVIEW_HANDLERS.get(target_name)
            if preview is not None:
                results.append(preview(count=preview_counts[target_name]))
                continue
        repair = _REPAIR_HANDLERS[target_name]
        if target_name == "session_insights":
            results.append(
                repair(
                    config,
                    dry_run=dry_run,
                    progress_callback=session_insight_progress_callback,
                    progress_total=session_insight_progress_total,
                )
            )
            continue
        results.append(repair(config, dry_run=dry_run))
    return results


def run_archive_cleanup(
    config: Config,
    dry_run: bool = False,
    *,
    preview_counts: dict[str, int] | None = None,
    targets: tuple[str, ...] = (),
) -> list[RepairResult]:
    preview_counts = preview_counts or {}
    selected = set(targets) if targets else set(CLEANUP_TARGETS)
    results: list[RepairResult] = []
    for target_name in CLEANUP_TARGETS:
        if target_name not in selected:
            continue
        if dry_run and target_name in preview_counts:
            results.append(_PREVIEW_HANDLERS[target_name](count=preview_counts[target_name]))
            continue
        results.append(_REPAIR_HANDLERS[target_name](config, dry_run=dry_run))
    return results


def run_selected_maintenance(
    config: Config,
    *,
    repair: bool,
    cleanup: bool,
    dry_run: bool = False,
    preview_counts: dict[str, int] | None = None,
    targets: tuple[str, ...] = (),
    session_insight_progress_callback: ProgressCallback | None = None,
    session_insight_progress_total: int | None = None,
) -> list[RepairResult]:
    blockers = offline_maintenance_blockers(
        config,
        repair=repair,
        cleanup=cleanup,
        dry_run=dry_run,
        targets=targets,
    )
    if blockers:
        return blockers
    results: list[RepairResult] = []
    repair_targets = tuple(name for name in targets if name in SAFE_REPAIR_TARGETS)
    cleanup_targets = tuple(name for name in targets if name in CLEANUP_TARGETS)
    if repair:
        results.extend(
            run_safe_repairs(
                config,
                dry_run=dry_run,
                preview_counts=preview_counts,
                targets=repair_targets,
                session_insight_progress_callback=session_insight_progress_callback,
                session_insight_progress_total=session_insight_progress_total,
            )
        )
    if cleanup:
        results.extend(
            run_archive_cleanup(config, dry_run=dry_run, preview_counts=preview_counts, targets=cleanup_targets)
        )
    return results


__all__ = [
    "ArchiveDebtStatus",
    "RepairResult",
    "collect_archive_debt_statuses_sync",
    "count_empty_sessions_sync",
    "count_orphaned_attachments_sync",
    "count_orphaned_blobs_sync",
    "count_superseded_raw_snapshots_sync",
    "count_orphaned_messages_sync",
    "count_messages_by_type_sync",
    "count_unclassified_message_type_sync",
    "preview_counts_from_archive_debt",
    "preview_empty_sessions",
    "preview_orphaned_attachments",
    "preview_orphaned_blobs",
    "preview_superseded_raw_snapshots",
    "preview_orphaned_messages",
    "preview_message_type_backfill",
    "preview_session_insights",
    "raw_materialization_replay_backlog",
    "repair_empty_sessions",
    "repair_message_type_backfill",
    "repair_orphaned_attachments",
    "repair_orphaned_blobs",
    "repair_raw_materialization",
    "repair_superseded_raw_snapshots",
    "repair_orphaned_messages",
    "repair_session_insights",
    "run_archive_cleanup",
    "run_safe_repairs",
    "run_selected_maintenance",
    "session_insight_repair_count",
]
