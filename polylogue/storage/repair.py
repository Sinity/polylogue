"""Consolidated archive repair: orphan detection, FTS repair, session insights, WAL."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
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
from polylogue.archive.revision_replay import ApplicationDecision
from polylogue.config import Config
from polylogue.core.enums import Origin, Provider
from polylogue.core.json import JSONDocument, json_document
from polylogue.core.protocols import ProgressCallback
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
from polylogue.pipeline.ids import session_content_hash, session_revision_projection
from polylogue.pipeline.ids import session_id as make_session_id
from polylogue.sources.dispatch import detect_provider, is_stream_record_provider
from polylogue.storage.blob_repair import count_orphaned_blobs_sync, repair_orphaned_blobs_data
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.insights.session.repair_assessment import (
    assess_session_insight_repairs,
)
from polylogue.storage.insights.session.runtime import (
    SESSION_INSIGHT_MATERIALIZATION_TYPES,
    session_profile_stale_predicate,
)
from polylogue.storage.message_type_backfill import (
    BackfillResult,
    count_messages_by_type_sync,
    count_unclassified_message_type_sync,
)
from polylogue.storage.raw_authority import (
    RawAuthorityCensusReceipt,
    RawReplayPlan,
    RawReplayPlanOutcome,
    RawReplayPlanStatus,
    build_raw_replay_plans,
    finalize_raw_authority_census,
    latest_raw_authority_census_receipt,
    raw_replay_application_receipt,
    raw_replay_plan_deferred_for_envelope,
    raw_replay_plan_last_attempts,
    record_raw_authority_census,
    record_raw_replay_outcome,
    recover_interrupted_raw_authority_censuses,
    reject_invalid_raw_replay_application,
    reject_stale_raw_replay_plan,
    validate_raw_replay_application_receipt,
    validate_raw_replay_plan,
)

logger = get_logger(__name__)
_MAINTENANCE_TARGET_CATALOG = build_maintenance_target_catalog()
_PROBE_ONLY_EXACT_MESSAGE_ROW_LIMIT = 100_000
RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES = 1024 * 1024 * 1024
RAW_MATERIALIZATION_RESOURCE_BLOCK_REASON = "non-stream-safe raw payload exceeds the bounded replay limit"
RAW_MATERIALIZATION_CENSUS_COMPONENT_LIMIT = 25
RAW_MATERIALIZATION_OUTCOME_SAMPLE_LIMIT = 8
_TRANSIENT_LOCK_PARSE_ERROR = "OperationalError: database is locked"
_QUARANTINED_ACCEPTED_RAW_REPAIR_DETAIL = "repair:accepted_quarantined_raw_exact_byte_and_semantic_proof"
_QUARANTINED_ACCEPTED_RAW_REPAIR_LIMIT = 100
_QUARANTINED_ACCEPTED_RAW_REPAIR_BLOB_LIMIT_BYTES = 256 * 1024 * 1024
_QUARANTINED_ACCEPTED_RAW_REPAIR_TOTAL_BLOB_LIMIT_BYTES = 512 * 1024 * 1024
_QUARANTINED_CENSUS_STAGE_FINGERPRINT = "repair-quarantined-accepted-raw-v1"
_QUARANTINED_CENSUS_STAGE_DETAIL = "census-only evidence staged before accepted-head authority refinement"
_BROWSER_ORIGIN_SEMANTIC_HISTORICAL_WITNESS_LIMIT = 8


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
    census_stage_raw_ids: tuple[str, ...] = ()
    parsed_message_count: int | None = None
    proof_digest: str | None = None
    repaired: bool = False


@dataclass(frozen=True, slots=True)
class BrowserCaptureOriginRepairItem:
    raw_id: str
    status: str
    reason: str
    origin: str | None = None
    source_path: str | None = None
    source_index: int | None = None
    blob_hash: str | None = None
    blob_size: int | None = None
    old_logical_source_key: str | None = None
    canonical_provider: str | None = None
    canonical_origin: str | None = None
    canonical_logical_source_key: str | None = None
    session_id: str | None = None
    accepted_content_hash: str | None = None
    parsed_message_count: int | None = None
    accepted_frontier: int | None = None
    repair_strategy: str | None = None
    replacement_raw_id: str | None = None
    replacement_source_revision: str | None = None
    replacement_content_hash: str | None = None
    replacement_frontier_kind: str | None = None
    replacement_frontier: int | None = None
    copy_forward_raw_id: str | None = None
    copy_forward_source_path: str | None = None
    copy_forward_source_complete: bool = False
    semantic_canonical_raw_id: str | None = None
    semantic_historical_raw_ids: tuple[str, ...] = ()
    semantic_head_snapshot: dict[str, object] | None = None
    semantic_witness_digest: str | None = None
    terminal_byte_witness_digest: str | None = None
    legacy_null_native_id: bool = False
    parser_derived_native_id: str | None = None
    byte_proven_null_native_id_rekey: bool = False
    evidence_digest: str | None = None
    proof_digest: str | None = None
    repaired: bool = False


@dataclass(frozen=True, slots=True)
class _SemanticCanonicalWitness:
    raw_id: str
    historical_raw_ids: tuple[str, ...]
    head_snapshot: dict[str, object]
    digest: str


@dataclass(frozen=True, slots=True)
class BrowserCanonicalAuthorityConflictWitness:
    """Read-only evidence packet for one unresolved browser-capture authority conflict.

    Built for polylogue-lkrc.3: unlike :class:`BrowserCaptureOriginRepairItem`,
    which collapses every rejection into a terse ``reason`` string, this packet
    keeps the competing-head evidence that ``_inspect_browser_capture_origin_mismatch``
    computes but discards on the ineligible path, so an operator can adjudicate
    without re-deriving it by hand. Building this packet never mutates state and
    never chooses an authority -- see ``record_browser_canonical_authority_conflict_blockers``
    for the separate, explicit step that persists it as a durable candidate blocker.
    """

    raw_id: str
    status: str
    reason: str
    session_id: str | None = None
    old_logical_source_key: str | None = None
    canonical_logical_source_key: str | None = None
    unknown_raw_content_hash: str | None = None
    unknown_source_revision: str | None = None
    unknown_frontier_kind: str | None = None
    unknown_frontier: int | None = None
    unknown_decided_at_ms: int | None = None
    unknown_raw_message_count: int | None = None
    competing_raw_id: str | None = None
    competing_content_hash: str | None = None
    competing_source_revision: str | None = None
    competing_frontier_kind: str | None = None
    competing_frontier: int | None = None
    competing_decided_at_ms: int | None = None
    competing_decision: str | None = None
    competing_message_count: int | None = None
    divergent_message_index: int | None = None
    divergence_note: str | None = None
    evidence_digest: str | None = None


@dataclass(frozen=True, slots=True)
class BrowserCanonicalAuthorityConflictReport:
    requested_count: int
    conflict_count: int
    resolved_count: int
    items: tuple[BrowserCanonicalAuthorityConflictWitness, ...]


def _quarantined_raw_item(raw_id: str, reason: str) -> QuarantinedAcceptedRawRepairItem:
    return QuarantinedAcceptedRawRepairItem(raw_id=raw_id, status="ineligible", reason=reason)


def _bytes_value(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, memoryview):
        return bytes(value)
    raise ValueError("expected SQLite BLOB value")


def _authority_rows_digest(*row_sets: Sequence[sqlite3.Row | Mapping[str, object]]) -> str:
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


def _raw_sessions_capture_mode_available(conn: sqlite3.Connection, *, schema: str = "main") -> bool:
    if schema not in {"main", "source"}:
        raise ValueError(f"unsupported raw-session schema: {schema}")
    return any(
        str(row[1]) == "capture_mode" for row in conn.execute(f"PRAGMA {schema}.table_info(raw_sessions)").fetchall()
    )


def _browser_origin_source_envelope_is_exact(
    conn: sqlite3.Connection,
    *,
    schema: str,
    raw_id: str,
    origin: str,
    capture_mode: str,
    native_id: str | None,
    source_path: str,
    source_index: int,
    blob_hash: bytes,
    blob_size: int,
    logical_source_key: str | None,
    revision_kind: RawRevisionKind,
    source_revision: str | None,
    baseline_raw_id: str | None,
    acquisition_generation: int | None,
    revision_authority: RawRevisionAuthority,
) -> sqlite3.Row | None:
    """Prove a complete source envelope and its raw-payload blob reference.

    Browser-origin repair only admits full snapshots.  Keep this one proof
    shared across preflight, locked staging, exact canonical, and semantic
    witnesses so lineage/capture metadata cannot drift between those routes.
    """
    if schema not in {"main", "source"}:
        raise ValueError(f"unsupported raw-session schema: {schema}")
    has_capture_mode = _raw_sessions_capture_mode_available(conn, schema=schema)
    capture_projection = "capture_mode" if has_capture_mode else "NULL AS capture_mode"
    row = conn.execute(
        f"""
        SELECT origin, {capture_projection}, native_id, source_path, source_index,
               blob_hash, blob_size, logical_source_key, revision_kind,
               source_revision, predecessor_source_revision, predecessor_raw_id,
               baseline_raw_id, append_start_offset, append_end_offset,
               acquisition_generation, revision_authority
        FROM {schema}.raw_sessions WHERE raw_id = ?
        """,
        (raw_id,),
    ).fetchone()
    expected = (
        origin,
        capture_mode if has_capture_mode else None,
        native_id,
        source_path,
        source_index,
        blob_hash,
        blob_size,
        logical_source_key,
        revision_kind.value,
        source_revision,
        None,
        None,
        baseline_raw_id,
        None,
        None,
        acquisition_generation,
        revision_authority.value,
    )
    if row is None or tuple(row) != expected:
        return None
    blob_refs = conn.execute(
        f"""
        SELECT blob_hash, source_path, size_bytes
        FROM {schema}.blob_refs
        WHERE ref_id = ? AND ref_type = 'raw_payload'
        """,
        (raw_id,),
    ).fetchall()
    if len(blob_refs) != 1 or tuple(blob_refs[0]) != (blob_hash, source_path, blob_size):
        return None
    return cast(sqlite3.Row, row)


def _stageable_quarantined_census_cohort(
    archive_root: Path,
    *,
    conn: sqlite3.Connection,
    raw: sqlite3.Row,
    origin: Origin,
    provider: Provider,
    logical_source_key: str,
    session_id: str,
    accepted_hash: bytes,
) -> tuple[tuple[str, ...], str | None]:
    """Prove a source-v7 same-path cohort is safe for census-only staging.

    This deliberately proves every raw before inserting any membership evidence.
    It never changes a raw envelope or index authority; the caller may refine only
    the requested accepted raw after this witness exists.
    """
    rows = conn.execute(
        """
        SELECT raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size,
               file_mtime_ms, logical_source_key, revision_kind, source_revision,
               predecessor_source_revision, predecessor_raw_id, baseline_raw_id,
               append_start_offset, append_end_offset, acquisition_generation,
               revision_authority
        FROM raw_sessions WHERE source_path = ? ORDER BY raw_id
        """,
        (str(raw["source_path"]),),
    ).fetchall()
    if not rows or str(rows[0]["source_path"]) != str(raw["source_path"]):
        return (), "same-source-path cohort is missing"
    oversized = [
        str(candidate["raw_id"])
        for candidate in rows
        if int(candidate["blob_size"]) > _QUARANTINED_ACCEPTED_RAW_REPAIR_BLOB_LIMIT_BYTES
    ]
    if oversized:
        return (), "same-source-path cohort exceeds the per-raw retained-blob repair limit"
    if sum(int(candidate["blob_size"]) for candidate in rows) > _QUARANTINED_ACCEPTED_RAW_REPAIR_TOTAL_BLOB_LIMIT_BYTES:
        return (), "same-source-path cohort exceeds the aggregate retained-blob repair limit"
    store = BlobStore(archive_root / "blob")
    staged_ids: list[str] = []
    for candidate in rows:
        candidate_id = str(candidate["raw_id"])
        try:
            candidate_blob_hash = _bytes_value(candidate["blob_hash"])
        except ValueError:
            return (), "same-source-path cohort has a malformed blob hash"
        expected_envelope = (
            logical_source_key,
            RawRevisionKind.FULL.value,
            candidate_blob_hash.hex(),
            None,
            None,
            None,
            None,
            None,
            0,
            RawRevisionAuthority.QUARANTINED.value,
        )
        repaired_target_envelope = (
            logical_source_key,
            RawRevisionKind.FULL.value,
            candidate_blob_hash.hex(),
            None,
            None,
            candidate_id,
            None,
            None,
            0,
            RawRevisionAuthority.BYTE_PROVEN.value,
        )
        actual_envelope = (
            candidate["logical_source_key"],
            str(candidate["revision_kind"]),
            candidate["source_revision"],
            candidate["predecessor_source_revision"],
            candidate["predecessor_raw_id"],
            candidate["baseline_raw_id"],
            candidate["append_start_offset"],
            candidate["append_end_offset"],
            candidate["acquisition_generation"],
            str(candidate["revision_authority"]),
        )
        blob_ref = conn.execute(
            """
            SELECT blob_hash, source_path, size_bytes FROM blob_refs
            WHERE ref_id = ? AND ref_type = 'raw_payload'
            """,
            (candidate_id,),
        ).fetchall()
        memberships = conn.execute(
            """
            SELECT logical_source_key, provider_session_id, source_revision,
                   normalized_content_hash, message_count, predecessor_raw_id,
                   acquisition_generation, revision_authority, decision, decided_at_ms
            FROM raw_session_memberships WHERE raw_id = ?
            """,
            (candidate_id,),
        ).fetchall()
        census = conn.execute(
            """
            SELECT parser_fingerprint, status, member_count, detail
            FROM raw_membership_census WHERE raw_id = ?
            """,
            (candidate_id,),
        ).fetchall()
        if (
            str(candidate["origin"]) != origin.value
            or int(candidate["source_index"]) < 0
            or actual_envelope
            not in (
                {expected_envelope, repaired_target_envelope}
                if candidate_id == str(raw["raw_id"])
                else {expected_envelope}
            )
            or len(blob_ref) != 1
            or tuple(blob_ref[0]) != (candidate_blob_hash, str(candidate["source_path"]), int(candidate["blob_size"]))
        ):
            return (), "same-source-path cohort has incompatible durable authority"
        if not store.exists(candidate_blob_hash.hex()) or not store.verify(candidate_blob_hash.hex()):
            return (), "same-source-path cohort has a missing or invalid retained blob"
        payload = store.read_all(candidate_blob_hash.hex())
        if len(payload) != int(candidate["blob_size"]) or hashlib.sha256(payload).digest() != candidate_blob_hash:
            return (), "same-source-path cohort bytes do not match their raw envelope"
        try:
            from polylogue.pipeline.services.ingest_worker import _normalized_session
            from polylogue.sources.revision_backfill import _parse_one

            fallback_timestamp = (
                datetime.fromtimestamp(int(candidate["file_mtime_ms"]) / 1000, UTC).isoformat()
                if candidate["file_mtime_ms"] is not None
                else None
            )
            sessions = [
                _normalized_session(session, fallback_timestamp=fallback_timestamp)
                for session in _parse_one(provider, payload, str(candidate["source_path"]))
            ]
        except Exception as exc:
            logger.warning(
                "quarantined census staging normalization failed",
                raw_id=candidate_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return (), "same-source-path cohort did not normalize cleanly"
        if (
            len(sessions) != 1
            or str(make_session_id(provider, sessions[0].provider_session_id)) != session_id
            or f"{provider.value}:{sessions[0].provider_session_id}" != logical_source_key
            or bytes.fromhex(session_content_hash(sessions[0])) != accepted_hash
        ):
            return (), "same-source-path cohort differs from the accepted session"
        expected_membership = (
            logical_source_key,
            sessions[0].provider_session_id,
            accepted_hash.hex(),
            accepted_hash,
            len(sessions[0].messages),
            None,
            0,
            RawRevisionAuthority.QUARANTINED.value,
            None,
            None,
        )
        expected_census = (_QUARANTINED_CENSUS_STAGE_FINGERPRINT, "complete", 1, _QUARANTINED_CENSUS_STAGE_DETAIL)
        if (memberships or census) and not (
            len(memberships) == 1
            and tuple(memberships[0]) == expected_membership
            and len(census) == 1
            and tuple(census[0]) == expected_census
        ):
            return (), "same-source-path cohort has pre-existing membership authority"
        staged_ids.append(candidate_id)
    return tuple(staged_ids), None


def _stage_quarantined_census_cohort(
    conn: sqlite3.Connection,
    item: QuarantinedAcceptedRawRepairItem,
) -> None:
    """Insert only the pre-proven singleton census/membership evidence."""
    assert item.logical_source_key is not None
    assert item.session_id is not None
    assert item.accepted_content_hash is not None
    assert item.parsed_message_count is not None
    for staged_raw_id in item.census_stage_raw_ids:
        row = conn.execute("SELECT raw_id FROM raw_sessions WHERE raw_id = ?", (staged_raw_id,)).fetchone()
        if row is None:
            raise RuntimeError(f"census staging witness changed for {staged_raw_id}")
        membership_count = int(
            conn.execute("SELECT COUNT(*) FROM raw_session_memberships WHERE raw_id = ?", (staged_raw_id,)).fetchone()[
                0
            ]
        )
        census_count = int(
            conn.execute("SELECT COUNT(*) FROM raw_membership_census WHERE raw_id = ?", (staged_raw_id,)).fetchone()[0]
        )
        if membership_count == census_count == 1:
            continue
        if membership_count or census_count:
            raise RuntimeError(f"census staging evidence became partial for {staged_raw_id}")
        conn.execute(
            """
            INSERT INTO raw_session_memberships (
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count, acquisition_generation,
                revision_authority
            ) VALUES (?, ?, ?, ?, ?, ?, 0, 'quarantined')
            """,
            (
                staged_raw_id,
                item.logical_source_key,
                item.session_id.split(":", 1)[1],
                item.accepted_content_hash,
                bytes.fromhex(item.accepted_content_hash),
                item.parsed_message_count,
            ),
        )
        conn.execute(
            """
            INSERT INTO raw_membership_census (
                raw_id, parser_fingerprint, status, member_count, censused_at_ms, detail
            ) VALUES (?, ?, 'complete', 1, ?, ?)
            """,
            (
                staged_raw_id,
                _QUARANTINED_CENSUS_STAGE_FINGERPRINT,
                int(time.time() * 1000),
                _QUARANTINED_CENSUS_STAGE_DETAIL,
            ),
        )


def _inspect_quarantined_accepted_raw(
    archive_root: Path,
    raw_id: str,
    *,
    conn: sqlite3.Connection,
) -> QuarantinedAcceptedRawRepairItem:
    """Prove one accepted head against source main + attached read-only index."""
    capture_mode_available = _raw_sessions_capture_mode_available(conn)
    capture_mode_projection = "capture_mode" if capture_mode_available else "NULL AS capture_mode"
    try:
        raw = conn.execute(
            f"""
            SELECT raw_id, origin, {capture_mode_projection}, native_id, source_path, source_index, blob_hash, blob_size,
                   file_mtime_ms, logical_source_key, revision_kind, source_revision,
                   predecessor_source_revision, predecessor_raw_id, baseline_raw_id,
                   append_start_offset, append_end_offset, acquisition_generation,
                   revision_authority
            FROM raw_sessions WHERE raw_id = ?
            """,  # nosec B608 - projection is selected from two fixed identifiers above
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
        if not capture_mode_available and len(fiber) != 1:
            return _quarantined_raw_item(raw_id, "source-v7 origin is not injective without capture-mode authority")
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
    target_memberships = [row for row in membership_rows if str(row["raw_id"]) == raw_id]
    census_stage_raw_ids: tuple[str, ...] = ()
    if len(target_memberships) != 1 or len(census_rows) != 1:
        if capture_mode_available or target_memberships or census_rows:
            return _quarantined_raw_item(raw_id, "expected one target membership and one membership census")
        census_stage_raw_ids, stage_reason = _stageable_quarantined_census_cohort(
            archive_root,
            conn=conn,
            raw=raw,
            origin=origin,
            provider=provider,
            logical_source_key=logical_source_key,
            session_id=session_id,
            accepted_hash=accepted_hash,
        )
        if stage_reason is not None or raw_id not in census_stage_raw_ids:
            return _quarantined_raw_item(raw_id, stage_reason or "target is absent from its census staging cohort")
        # The existing source-v7 rows have no membership evidence yet.  The
        # cohort helper above parses every byte before allowing its precise
        # census-only insert; use the parsed target shape for the stable proof.
        membership = None
        census = None
    else:
        membership = target_memberships[0]
        census = census_rows[0]
    if (
        membership is not None
        and census is not None
        and (
            str(census["status"]) != "complete"
            or int(census["member_count"]) != 1
            or any(row["decision"] == "applied" and str(row["raw_id"]) != raw_id for row in membership_rows)
        )
    ):
        return _quarantined_raw_item(raw_id, "membership authority is failed, ambiguous, or competitively applied")
    parsed_logical_source_key = f"{parsed.source_name.value}:{parsed.provider_session_id}"
    if membership is not None and (
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
    if (
        not capture_mode_available
        and membership is not None
        and census is not None
        and str(census["parser_fingerprint"]) == _QUARANTINED_CENSUS_STAGE_FINGERPRINT
        and str(census["detail"]) == _QUARANTINED_CENSUS_STAGE_DETAIL
    ):
        census_stage_raw_ids, stage_reason = _stageable_quarantined_census_cohort(
            archive_root,
            conn=conn,
            raw=raw,
            origin=origin,
            provider=provider,
            logical_source_key=logical_source_key,
            session_id=session_id,
            accepted_hash=accepted_hash,
        )
        if stage_reason is not None or raw_id not in census_stage_raw_ids:
            return _quarantined_raw_item(raw_id, stage_reason or "census staging witness no longer includes target")

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
            [] if census_stage_raw_ids else membership_rows,
            [] if census_stage_raw_ids else census_rows,
        ),
        parallel_session_head_count=len(parallel_session_heads),
        quarantined_sibling_raw_count=len(competing_revision_rows),
        membership_row_count=0 if census_stage_raw_ids else len(membership_rows),
        census_stage_raw_ids=census_stage_raw_ids,
        parsed_message_count=len(parsed.messages),
    )
    return dataclasses.replace(item, proof_digest=_proof_digest(item))


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


def inspect_quarantined_accepted_raws(
    config: Config,
    raw_ids: list[str],
) -> tuple[QuarantinedAcceptedRawRepairItem, ...]:
    """Return exact typed quarantine-refinement proofs without mutation."""
    if len(set(raw_ids)) != len(raw_ids):
        raise ValueError("duplicate raw ids are not allowed")
    if not raw_ids or len(raw_ids) > _QUARANTINED_ACCEPTED_RAW_REPAIR_LIMIT:
        raise ValueError(f"raw-id list must contain 1..{_QUARANTINED_ACCEPTED_RAW_REPAIR_LIMIT} entries")
    if any(re.fullmatch(r"[0-9a-f]{64}", raw_id) is None for raw_id in raw_ids):
        raise ValueError("raw ids must be lowercase SHA-256 identifiers")
    archive_root = _raw_materialization_archive_root(config)
    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    if not source_db.exists() or not index_db.exists():
        raise RuntimeError("source or index tier is missing")
    with closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)) as conn:
        _attach_repair_index(conn, index_db)
        _validate_quarantined_raw_repair_blob_budget(conn, raw_ids)
        return tuple(_inspect_quarantined_accepted_raw(archive_root, raw_id, conn=conn) for raw_id in raw_ids)


def _browser_origin_ineligible(raw_id: str, reason: str) -> BrowserCaptureOriginRepairItem:
    return BrowserCaptureOriginRepairItem(raw_id=raw_id, status="ineligible", reason=reason)


def _browser_origin_item_payload(item: BrowserCaptureOriginRepairItem) -> dict[str, object]:
    """Canonical exact strategy witness for the shared raw-authority plan."""
    # ``copy_forward_source_complete`` is an execution checkpoint: it flips
    # after the source row is staged but before the index CAS completes.  It
    # therefore cannot be part of the immutable strategy identity.  The
    # actual copy/raw footprint and terminal byte witness remain bound.
    excluded = {"status", "reason", "proof_digest", "repaired", "copy_forward_source_complete"}
    payload = {key: value for key, value in dataclasses.asdict(item).items() if key not in excluded}
    return cast(dict[str, object], json.loads(json.dumps(payload, sort_keys=True, separators=(",", ":"))))


def _browser_origin_item_digest(item: BrowserCaptureOriginRepairItem) -> str:
    return hashlib.sha256(
        json.dumps(_browser_origin_item_payload(item), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _browser_origin_copy_forward_detail(item: BrowserCaptureOriginRepairItem) -> str:
    """Encode the proven semantic-head snapshot in the immutable copy receipt."""
    payload: dict[str, object] = {
        "kind": "browser_capture_origin_copy_forward_v2",
        "raw_id": item.raw_id,
        "semantic_head": item.semantic_head_snapshot,
    }
    if item.legacy_null_native_id:
        assert item.parser_derived_native_id is not None
        payload = {
            "kind": "browser_capture_legacy_native_id_copy_forward_v1",
            "raw_id": item.raw_id,
            "legacy_null_native_id": True,
            "parser_derived_native_id": item.parser_derived_native_id,
            "semantic_head": item.semantic_head_snapshot,
        }
    elif item.byte_proven_null_native_id_rekey:
        assert item.parser_derived_native_id is not None
        payload = {
            "kind": "browser_capture_byte_proven_rekey_v1",
            "raw_id": item.raw_id,
            "byte_proven_null_native_id_rekey": True,
            "parser_derived_native_id": item.parser_derived_native_id,
            "semantic_head": item.semantic_head_snapshot,
        }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )


def _browser_origin_semantic_head_from_copy_detail(
    detail: object,
    *,
    raw_id: str,
    parser_derived_native_id: str | None = None,
    byte_proven_null_native_id_rekey: bool = False,
) -> tuple[bool, dict[str, object] | None]:
    """Read a prior semantic-head snapshot only from the immutable index receipt."""
    try:
        payload = json.loads(str(detail))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False, None
    if not isinstance(payload, dict):
        return False, None
    snapshot = payload.get("semantic_head")
    ordinary = payload.get("kind") == "browser_capture_origin_copy_forward_v2" and set(payload) == {
        "kind",
        "raw_id",
        "semantic_head",
    }
    legacy = (
        parser_derived_native_id is not None
        and payload.get("kind") == "browser_capture_legacy_native_id_copy_forward_v1"
        and set(payload) == {"kind", "raw_id", "legacy_null_native_id", "parser_derived_native_id", "semantic_head"}
        and payload.get("legacy_null_native_id") is True
        and payload.get("parser_derived_native_id") == parser_derived_native_id
    )
    byte_proven_rekey = (
        byte_proven_null_native_id_rekey
        and parser_derived_native_id is not None
        and payload.get("kind") == "browser_capture_byte_proven_rekey_v1"
        and set(payload)
        == {"kind", "raw_id", "byte_proven_null_native_id_rekey", "parser_derived_native_id", "semantic_head"}
        and payload.get("byte_proven_null_native_id_rekey") is True
        and payload.get("parser_derived_native_id") == parser_derived_native_id
    )
    if payload.get("raw_id") != raw_id or not (ordinary or legacy or byte_proven_rekey):
        return False, None
    if snapshot is None:
        return True, None
    if not isinstance(snapshot, dict):
        return False, None
    expected_keys = {
        "session_id",
        "accepted_raw_id",
        "accepted_source_revision",
        "accepted_content_hash",
        "accepted_frontier_kind",
        "accepted_frontier",
        "acquisition_generation",
        "decided_at_ms",
    }
    if set(snapshot) != expected_keys:
        return False, None
    try:
        if (
            not isinstance(snapshot["session_id"], str)
            or not isinstance(snapshot["accepted_raw_id"], str)
            or not isinstance(snapshot["accepted_source_revision"], str)
            or not isinstance(snapshot["accepted_content_hash"], str)
            or len(bytes.fromhex(snapshot["accepted_content_hash"])) != 32
            or snapshot["accepted_frontier_kind"] != "semantic"
            or not isinstance(snapshot["accepted_frontier"], int)
            or snapshot["accepted_frontier"] < 0
            or not isinstance(snapshot["acquisition_generation"], int)
            or snapshot["acquisition_generation"] < 0
            or not isinstance(snapshot["decided_at_ms"], int)
            or snapshot["decided_at_ms"] < 0
        ):
            return False, None
    except ValueError:
        return False, None
    return True, cast(dict[str, object], snapshot)


def _browser_origin_copy_raw_id(
    origin: Origin,
    source_path: str,
    source_index: int,
    blob_hash: bytes,
    native_id: str,
) -> str:
    digest = hashlib.sha256()
    for value in (origin.value, source_path, str(source_index)):
        digest.update(value.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
    digest.update(blob_hash)
    digest.update(b"\0")
    digest.update(native_id.encode("utf-8", errors="surrogatepass"))
    return digest.hexdigest()


def _verify_browser_origin_copy_forward_source_stage(
    archive_root: Path,
    conn: sqlite3.Connection,
    item: BrowserCaptureOriginRepairItem,
    *,
    source_schema: str = "main",
) -> None:
    """Reprove source-only witnesses while its write transaction is held."""
    if source_schema not in {"main", "source"}:
        raise ValueError(f"unsupported source schema: {source_schema}")
    assert item.canonical_origin is not None
    assert item.canonical_provider is not None
    assert item.canonical_logical_source_key is not None
    assert item.session_id is not None
    assert item.old_logical_source_key is not None
    assert item.source_path is not None
    assert item.source_index is not None
    assert item.blob_hash is not None
    assert item.blob_size is not None
    assert item.accepted_content_hash is not None
    null_native_id_mode = item.legacy_null_native_id or item.byte_proven_null_native_id_rekey
    native_id = None if null_native_id_mode else item.session_id.split(":", 1)[1]
    source_authority = (
        RawRevisionAuthority.BYTE_PROVEN if item.byte_proven_null_native_id_rekey else RawRevisionAuthority.QUARANTINED
    )
    membership_authority = source_authority.value
    membership_decision: str | None = "applied" if item.byte_proven_null_native_id_rekey else None
    baseline_raw_id = item.raw_id if item.byte_proven_null_native_id_rekey else None
    if (
        _browser_origin_source_envelope_is_exact(
            conn,
            schema=source_schema,
            raw_id=item.raw_id,
            origin=Origin.UNKNOWN_EXPORT.value,
            capture_mode=Provider.UNKNOWN.value,
            native_id=native_id,
            source_path=item.source_path,
            source_index=item.source_index,
            blob_hash=bytes.fromhex(item.blob_hash),
            blob_size=item.blob_size,
            logical_source_key=item.old_logical_source_key,
            revision_kind=RawRevisionKind.FULL,
            source_revision=item.blob_hash,
            baseline_raw_id=baseline_raw_id,
            acquisition_generation=0,
            revision_authority=source_authority,
        )
        is None
    ):
        raise RuntimeError(f"source evidence changed before copy-forward stage for {item.raw_id}")
    store = BlobStore(archive_root / "blob")
    payload = store.read_all(item.blob_hash)
    if len(payload) != item.blob_size or hashlib.sha256(payload).hexdigest() != item.blob_hash:
        raise RuntimeError(f"retained blob changed before copy-forward stage for {item.raw_id}")
    try:
        provider = detect_provider(json.loads(payload))
        if provider is None or provider.value != item.canonical_provider:
            raise ValueError("provider identity changed")
        from polylogue.sources.revision_backfill import _parse_one

        sessions = _parse_one(provider, payload, item.source_path)
    except Exception as exc:
        raise RuntimeError(f"browser parser evidence changed before copy-forward stage for {item.raw_id}") from exc
    if len(sessions) != 1 or str(make_session_id(provider, sessions[0].provider_session_id)) != item.session_id:
        raise RuntimeError(f"normalized session identity changed before copy-forward stage for {item.raw_id}")
    projection = session_revision_projection(sessions[0])
    if projection.session_hash.hex() != item.accepted_content_hash:
        raise RuntimeError(f"normalized session content changed before copy-forward stage for {item.raw_id}")
    membership = conn.execute(
        f"""
        SELECT provider_session_id, source_revision, normalized_content_hash,
               message_count, acquisition_generation, revision_authority, decision
        FROM {source_schema}.raw_session_memberships WHERE raw_id = ? AND logical_source_key = ?
        """,
        (item.raw_id, item.canonical_logical_source_key),
    ).fetchone()
    membership_count = conn.execute(
        f"SELECT COUNT(*) FROM {source_schema}.raw_session_memberships WHERE raw_id = ?", (item.raw_id,)
    ).fetchone()
    census = conn.execute(
        f"SELECT status, member_count FROM {source_schema}.raw_membership_census WHERE raw_id = ?", (item.raw_id,)
    ).fetchone()
    if item.byte_proven_null_native_id_rekey:
        if membership is not None or membership_count is None or int(membership_count[0]) != 0 or census is not None:
            raise RuntimeError(f"membership evidence changed before copy-forward stage for {item.raw_id}")
        return
    if (
        membership is None
        or membership_count is None
        or int(membership_count[0]) != 1
        or tuple(membership)
        != (
            sessions[0].provider_session_id,
            item.accepted_content_hash,
            projection.session_hash,
            len(projection.message_hashes),
            0,
            membership_authority,
            membership_decision,
        )
        or census is None
        or tuple(census) != ("complete", 1)
    ):
        raise RuntimeError(f"membership evidence changed before copy-forward stage for {item.raw_id}")


def _canonical_browser_origin_head_is_exact(
    conn: sqlite3.Connection,
    *,
    archive_root: Path,
    canonical_head: sqlite3.Row,
    canonical_key: str,
    canonical_origin: Origin,
    session_id: str,
    accepted_hash: bytes,
    message_count: int,
) -> bool:
    """Return whether an existing canonical head has full byte authority witnesses."""
    raw_id = str(canonical_head["accepted_raw_id"])
    source_revision = str(canonical_head["accepted_source_revision"])
    frontier = int(canonical_head["accepted_frontier"])
    generation = int(canonical_head["acquisition_generation"])
    raw_locator = conn.execute(
        """
        SELECT source_path, source_index
        FROM source.raw_sessions WHERE raw_id = ?
        """,
        (raw_id,),
    ).fetchone()
    membership = conn.execute(
        """
        SELECT provider_session_id, source_revision, normalized_content_hash, message_count,
               acquisition_generation, revision_authority, decision
        FROM source.raw_session_memberships WHERE raw_id = ? AND logical_source_key = ?
        """,
        (raw_id, canonical_key),
    ).fetchone()
    census = conn.execute(
        "SELECT status, member_count FROM source.raw_membership_census WHERE raw_id = ?", (raw_id,)
    ).fetchone()
    application = conn.execute(
        """
        SELECT session_id, source_revision, acquisition_generation, decision, accepted_raw_id,
               accepted_source_revision, accepted_content_hash, baseline_raw_id
        FROM raw_revision_applications
        WHERE raw_id = ? AND logical_source_key = ? AND decision = 'selected_baseline'
        """,
        (raw_id, canonical_key),
    ).fetchone()
    native_id = session_id.split(":", 1)[1]
    raw = (
        _browser_origin_source_envelope_is_exact(
            conn,
            schema="source",
            raw_id=raw_id,
            origin=canonical_origin.value,
            capture_mode=canonical_key.split(":", 1)[0],
            native_id=native_id,
            source_path=str(raw_locator["source_path"]) if raw_locator is not None else "",
            source_index=0,
            blob_hash=bytes.fromhex(source_revision),
            blob_size=frontier,
            logical_source_key=canonical_key,
            revision_kind=RawRevisionKind.FULL,
            source_revision=source_revision,
            baseline_raw_id=raw_id,
            acquisition_generation=generation,
            revision_authority=RawRevisionAuthority.BYTE_PROVEN,
        )
        if raw_locator is not None and int(raw_locator["source_index"]) == 0
        else None
    )
    if (
        raw is None
        or membership is None
        or census is None
        or application is None
        or str(canonical_head["session_id"]) != session_id
        or _bytes_value(canonical_head["accepted_content_hash"]) != accepted_hash
        or str(canonical_head["accepted_frontier_kind"]) != "byte"
    ):
        return False
    try:
        blob_hash = bytes.fromhex(source_revision)
        store = BlobStore(archive_root / "blob")
        blob_path = store.blob_path(source_revision)
        if not blob_path.is_file() or blob_path.stat().st_size != frontier:
            return False
        payload = store.read_all(source_revision)
        provider = detect_provider(json.loads(payload))
        if provider is None or origin_from_provider(provider) is not canonical_origin:
            return False
        from polylogue.sources.revision_backfill import _parse_one

        sessions = _parse_one(provider, payload, str(raw["source_path"]))
    except Exception as exc:
        logger.warning(
            "semantic canonical browser head normalization failed",
            raw_id=raw_id,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return False
    if (
        hashlib.sha256(payload).digest() != blob_hash
        or len(sessions) != 1
        or str(make_session_id(provider, sessions[0].provider_session_id)) != session_id
        or f"{provider.value}:{sessions[0].provider_session_id}" != canonical_key
        or bytes.fromhex(session_content_hash(sessions[0])) != accepted_hash
    ):
        return False
    return (
        tuple(membership)
        == (
            native_id,
            source_revision,
            accepted_hash,
            message_count,
            generation,
            RawRevisionAuthority.BYTE_PROVEN.value,
            "applied",
        )
        and tuple(census) == ("complete", 1)
        and tuple(application)
        == (
            session_id,
            source_revision,
            generation,
            ApplicationDecision.SELECTED_BASELINE.value,
            raw_id,
            source_revision,
            accepted_hash,
            raw_id,
        )
    )


def _semantic_head_snapshot(head: Mapping[str, object] | sqlite3.Row) -> dict[str, object]:
    """Normalize the semantic-head fields that survive a copy-forward CAS.

    A byte-proven copy replaces ``raw_revision_heads``.  The copy receipt keeps
    this compact, typed snapshot in its immutable application detail so a later
    invocation can prove the original semantic authority rather than trusting a
    value copied out of an operator receipt.
    """
    accepted_content_hash = head["accepted_content_hash"]
    if isinstance(accepted_content_hash, str):
        if len(bytes.fromhex(accepted_content_hash)) != 32:
            raise ValueError("semantic head snapshot has an invalid content hash")
        content_hash = accepted_content_hash.lower()
    else:
        content_hash = _bytes_value(accepted_content_hash).hex()
    return {
        "session_id": str(head["session_id"]),
        "accepted_raw_id": str(head["accepted_raw_id"]),
        "accepted_source_revision": str(head["accepted_source_revision"]),
        "accepted_content_hash": content_hash,
        "accepted_frontier_kind": str(head["accepted_frontier_kind"]),
        "accepted_frontier": cast(int, head["accepted_frontier"]),
        "acquisition_generation": cast(int, head["acquisition_generation"]),
        "decided_at_ms": cast(int, head["decided_at_ms"]),
    }


def _revision_application_decision_id_is_exact(application: sqlite3.Row) -> bool:
    """Reject a receipt whose immutable identity no longer matches its fields."""
    try:
        decision = ApplicationDecision(str(application["decision"]))
    except ValueError:
        return False
    payload = {
        "accepted_raw_id": application["accepted_raw_id"],
        "accepted_source_revision": application["accepted_source_revision"],
        "decision": decision.value,
        "logical_source_key": application["logical_source_key"],
        "raw_id": application["raw_id"],
        "session_id": application["session_id"],
        "source_revision": application["source_revision"],
    }
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return str(application["decision_id"]) == expected


def _canonical_browser_origin_head_is_semantically_equivalent(
    conn: sqlite3.Connection,
    *,
    archive_root: Path,
    canonical_head: Mapping[str, object] | sqlite3.Row,
    canonical_key: str,
    canonical_origin: Origin,
    session_id: str,
    accepted_hash: bytes,
    message_count: int,
    indexed_raw_id: str,
    excluded_application_decision_id: str | None = None,
) -> _SemanticCanonicalWitness | None:
    """Prove a quarantined semantic head is evidence, not a replacement authority.

    The caller deliberately creates a new byte-proven canonical raw instead of
    repointing at this old head.  That keeps semantic and byte evidence distinct.
    """
    head_snapshot = _semantic_head_snapshot(canonical_head)
    raw_id = str(head_snapshot["accepted_raw_id"])
    native_id = session_id.split(":", 1)[1]
    if (
        str(head_snapshot["session_id"]) != session_id
        or str(head_snapshot["accepted_content_hash"]) != accepted_hash.hex()
        or str(head_snapshot["accepted_source_revision"]) != accepted_hash.hex()
        or str(head_snapshot["accepted_frontier_kind"]) != "semantic"
        or cast(int, head_snapshot["accepted_frontier"]) < 0
        or cast(int, head_snapshot["acquisition_generation"]) < 0
        or cast(int, head_snapshot["decided_at_ms"]) < 0
        or str(indexed_raw_id) == raw_id
    ):
        return None
    raw_locator = conn.execute(
        """
        SELECT source_path, source_index, blob_hash, blob_size
        FROM source.raw_sessions WHERE raw_id = ?
        """,
        (raw_id,),
    ).fetchone()
    memberships = conn.execute(
        """
        SELECT provider_session_id, source_revision, normalized_content_hash, message_count,
               predecessor_raw_id, acquisition_generation, revision_authority, decision
        FROM source.raw_session_memberships WHERE raw_id = ? AND logical_source_key = ?
        """,
        (raw_id, canonical_key),
    ).fetchall()
    census = conn.execute(
        "SELECT status, member_count FROM source.raw_membership_census WHERE raw_id = ?", (raw_id,)
    ).fetchall()
    applications = conn.execute(
        """
        SELECT decision_id, raw_id, session_id, logical_source_key, source_revision, acquisition_generation,
               decision, accepted_raw_id, accepted_source_revision, accepted_content_hash,
               baseline_raw_id, predecessor_raw_id, append_end_offset, detail, decided_at_ms
        FROM raw_revision_applications
        WHERE logical_source_key = ? AND (? IS NULL OR decision_id != ?)
        ORDER BY raw_id
        """,
        (canonical_key, excluded_application_decision_id, excluded_application_decision_id),
    ).fetchall()
    if raw_locator is None or len(memberships) != 1 or len(census) != 1 or not applications:
        return None
    blob_ref = conn.execute(
        "SELECT blob_hash, source_path, size_bytes FROM source.blob_refs WHERE ref_id = ? AND ref_type = 'raw_payload'",
        (raw_id,),
    ).fetchone()
    if blob_ref is None:
        return None
    try:
        blob_hash = _bytes_value(raw_locator["blob_hash"])
        raw = _browser_origin_source_envelope_is_exact(
            conn,
            schema="source",
            raw_id=raw_id,
            origin=canonical_origin.value,
            capture_mode=canonical_key.split(":", 1)[0],
            native_id=native_id,
            source_path=str(raw_locator["source_path"]),
            source_index=0,
            blob_hash=blob_hash,
            blob_size=int(raw_locator["blob_size"]),
            logical_source_key=None,
            revision_kind=RawRevisionKind.UNKNOWN,
            source_revision=None,
            baseline_raw_id=None,
            acquisition_generation=None,
            revision_authority=RawRevisionAuthority.QUARANTINED,
        )
        if raw is None or int(raw_locator["source_index"]) != 0:
            return None
        store = BlobStore(archive_root / "blob")
        payload = store.read_all(blob_hash.hex())
        provider = detect_provider(json.loads(payload))
        if provider is None or origin_from_provider(provider) is not canonical_origin:
            return None
        from polylogue.sources.revision_backfill import _parse_one

        sessions = _parse_one(provider, payload, str(raw["source_path"]))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    membership = memberships[0]
    selected_applications = [application for application in applications if str(application["raw_id"]) == raw_id]
    historical_supersessions = [application for application in applications if str(application["raw_id"]) != raw_id]
    if (
        len(selected_applications) != 1
        or len(historical_supersessions) > _BROWSER_ORIGIN_SEMANTIC_HISTORICAL_WITNESS_LIMIT
        or not _revision_application_decision_id_is_exact(selected_applications[0])
    ):
        return None
    application = selected_applications[0]
    historical_raw_rows: list[sqlite3.Row] = []
    historical_blob_refs: list[sqlite3.Row] = []
    historical_membership_rows: list[sqlite3.Row] = []
    historical_census_rows: list[sqlite3.Row] = []
    # A semantic head can legitimately retain prior equivalent raw evidence.
    # It is not replacement authority: every additional application must be an
    # explicit supersession *to this exact head*.  Any independent selected or
    # divergent historical receipt still makes the proof ineligible.
    for historical in historical_supersessions:
        historical_raw_id = str(historical["raw_id"])
        if (
            not _revision_application_decision_id_is_exact(historical)
            or str(historical["session_id"]) != session_id
            or str(historical["logical_source_key"]) != canonical_key
            or str(historical["source_revision"]) != accepted_hash.hex()
            or int(historical["acquisition_generation"]) < 0
            or str(historical["decision"]) != ApplicationDecision.SUPERSEDED.value
            or str(historical["accepted_raw_id"]) != raw_id
            or str(historical["accepted_source_revision"]) != str(head_snapshot["accepted_source_revision"])
            or _bytes_value(historical["accepted_content_hash"]) != accepted_hash
            or int(historical["decided_at_ms"]) < 0
            or any(
                historical[name] is not None for name in ("baseline_raw_id", "predecessor_raw_id", "append_end_offset")
            )
        ):
            return None
        historical_raw = conn.execute(
            """
            SELECT origin, native_id, source_path, source_index, blob_hash, blob_size,
                   logical_source_key, revision_kind, source_revision,
                   predecessor_source_revision, predecessor_raw_id, baseline_raw_id,
                   append_start_offset, append_end_offset, acquisition_generation,
                   revision_authority
            FROM source.raw_sessions WHERE raw_id = ?
            """,
            (historical_raw_id,),
        ).fetchone()
        historical_blob_ref = conn.execute(
            """
            SELECT blob_hash, source_path, size_bytes FROM source.blob_refs
            WHERE ref_id = ? AND ref_type = 'raw_payload'
            """,
            (historical_raw_id,),
        ).fetchone()
        historical_memberships = conn.execute(
            """
            SELECT provider_session_id, source_revision, normalized_content_hash, message_count,
                   predecessor_raw_id, acquisition_generation, revision_authority, decision
            FROM source.raw_session_memberships WHERE raw_id = ? AND logical_source_key = ?
            """,
            (historical_raw_id, canonical_key),
        ).fetchall()
        historical_census = conn.execute(
            "SELECT status, member_count FROM source.raw_membership_census WHERE raw_id = ?",
            (historical_raw_id,),
        ).fetchall()
        if historical_raw is None or historical_blob_ref is None:
            return None
        historical_blob_hash = _bytes_value(historical_raw["blob_hash"])
        historical_blob_size = int(historical_raw["blob_size"])
        if (
            historical_blob_size > _QUARANTINED_ACCEPTED_RAW_REPAIR_BLOB_LIMIT_BYTES
            or int(historical_raw["source_index"]) != 0
            or _browser_origin_source_envelope_is_exact(
                conn,
                schema="source",
                raw_id=historical_raw_id,
                origin=canonical_origin.value,
                capture_mode=canonical_key.split(":", 1)[0],
                native_id=native_id,
                source_path=str(historical_raw["source_path"]),
                source_index=0,
                blob_hash=historical_blob_hash,
                blob_size=historical_blob_size,
                logical_source_key=None,
                revision_kind=RawRevisionKind.UNKNOWN,
                source_revision=None,
                baseline_raw_id=None,
                acquisition_generation=None,
                revision_authority=RawRevisionAuthority.QUARANTINED,
            )
            is None
            or len(historical_memberships) != 1
            or tuple(historical_memberships[0])
            != (
                native_id,
                accepted_hash.hex(),
                accepted_hash,
                message_count,
                None,
                0,
                RawRevisionAuthority.QUARANTINED.value,
                None,
            )
            or len(historical_census) != 1
            or tuple(historical_census[0]) != ("complete", 1)
        ):
            return None
        try:
            historical_payload = store.read_all(historical_blob_hash.hex())
            historical_provider = detect_provider(json.loads(historical_payload))
            if historical_provider is None or origin_from_provider(historical_provider) is not canonical_origin:
                return None
            historical_sessions = _parse_one(
                historical_provider, historical_payload, str(historical_raw["source_path"])
            )
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        if (
            len(historical_payload) != historical_blob_size
            or hashlib.sha256(historical_payload).digest() != historical_blob_hash
            or len(historical_sessions) != 1
            or str(make_session_id(historical_provider, historical_sessions[0].provider_session_id)) != session_id
            or f"{historical_provider.value}:{historical_sessions[0].provider_session_id}" != canonical_key
            or bytes.fromhex(session_content_hash(historical_sessions[0])) != accepted_hash
        ):
            return None
        historical_raw_rows.append(historical_raw)
        historical_blob_refs.append(historical_blob_ref)
        historical_membership_rows.extend(historical_memberships)
        historical_census_rows.extend(historical_census)
    if not (
        len(payload) == int(raw["blob_size"])
        and hashlib.sha256(payload).digest() == blob_hash
        and len(sessions) == 1
        and str(make_session_id(provider, sessions[0].provider_session_id)) == session_id
        and f"{provider.value}:{sessions[0].provider_session_id}" == canonical_key
        and bytes.fromhex(session_content_hash(sessions[0])) == accepted_hash
        and str(membership["provider_session_id"]) == native_id
        and str(membership["source_revision"]) == accepted_hash.hex()
        and _bytes_value(membership["normalized_content_hash"]) == accepted_hash
        and int(membership["message_count"]) == message_count
        and membership["predecessor_raw_id"] is None
        and int(membership["acquisition_generation"]) == 0
        and str(membership["revision_authority"]) == RawRevisionAuthority.QUARANTINED.value
        and membership["decision"] is None
        and tuple(census[0]) == ("complete", 1)
        and str(application["raw_id"]) == raw_id
        and str(application["session_id"]) == session_id
        and str(application["logical_source_key"]) == canonical_key
        and str(application["source_revision"]) == str(head_snapshot["accepted_source_revision"])
        and int(application["acquisition_generation"]) == cast(int, head_snapshot["acquisition_generation"])
        and str(application["decision"]) == ApplicationDecision.SELECTED_BASELINE.value
        and str(application["accepted_raw_id"]) == raw_id
        and str(application["accepted_source_revision"]) == str(head_snapshot["accepted_source_revision"])
        and _bytes_value(application["accepted_content_hash"]) == accepted_hash
        and all(application[name] is None for name in ("baseline_raw_id", "predecessor_raw_id", "append_end_offset"))
        and int(application["decided_at_ms"]) >= 0
        and int(application["decided_at_ms"]) == cast(int, head_snapshot["decided_at_ms"])
    ):
        return None
    return _SemanticCanonicalWitness(
        raw_id=raw_id,
        historical_raw_ids=tuple(str(row["raw_id"]) for row in historical_supersessions),
        head_snapshot=head_snapshot,
        digest=_authority_rows_digest(
            [head_snapshot],
            [raw],
            [blob_ref],
            memberships,
            census,
            [application],
            historical_supersessions,
            historical_raw_rows,
            historical_blob_refs,
            historical_membership_rows,
            historical_census_rows,
        ),
    )


def _inspect_browser_capture_origin_mismatch(
    archive_root: Path,
    raw_id: str,
    *,
    conn: sqlite3.Connection,
    allow_legacy_null_native_id: bool = False,
    allow_byte_proven_null_native_id_rekey: bool = False,
) -> BrowserCaptureOriginRepairItem:
    if allow_legacy_null_native_id and allow_byte_proven_null_native_id_rekey:
        raise ValueError("browser repair modes are mutually exclusive")
    conn.row_factory = sqlite3.Row
    raw = conn.execute(
        """
        SELECT raw_id, origin, native_id, source_path, source_index, blob_hash,
               blob_size, logical_source_key, revision_kind, source_revision,
               predecessor_source_revision, predecessor_raw_id, baseline_raw_id,
               append_start_offset, append_end_offset, acquisition_generation,
               revision_authority
        FROM source.raw_sessions WHERE raw_id = ?
        """,
        (raw_id,),
    ).fetchone()
    if raw is None:
        return _browser_origin_ineligible(raw_id, "source raw is missing")
    if str(raw["origin"]) != Origin.UNKNOWN_EXPORT.value:
        return _browser_origin_ineligible(raw_id, "source raw is not typed under unknown-export")
    source_path = str(raw["source_path"])
    if "browser-capture" not in source_path:
        return _browser_origin_ineligible(raw_id, "source raw is not a browser-capture artifact")
    expected_source_authority = (
        RawRevisionAuthority.BYTE_PROVEN if allow_byte_proven_null_native_id_rekey else RawRevisionAuthority.QUARANTINED
    )
    if (
        str(raw["revision_kind"]) != RawRevisionKind.FULL.value
        or str(raw["revision_authority"]) != expected_source_authority.value
        or raw["source_revision"] is None
        or raw["acquisition_generation"] is None
        or int(raw["acquisition_generation"]) != 0
        or any(
            raw[name] is not None
            for name in (
                "predecessor_source_revision",
                "predecessor_raw_id",
                "append_start_offset",
                "append_end_offset",
            )
        )
        or (
            (raw["baseline_raw_id"] != raw_id)
            if allow_byte_proven_null_native_id_rekey
            else raw["baseline_raw_id"] is not None
        )
    ):
        return _browser_origin_ineligible(raw_id, "source raw is not the exact quarantined full mismatch shape")
    blob_hash = _bytes_value(raw["blob_hash"])
    blob_size = int(raw["blob_size"])
    if len(blob_hash) != 32 or blob_size < 1 or blob_size > _QUARANTINED_ACCEPTED_RAW_REPAIR_BLOB_LIMIT_BYTES:
        return _browser_origin_ineligible(raw_id, "retained browser capture exceeds the bounded proof shape")
    blob_hash_hex = blob_hash.hex()
    if str(raw["source_revision"]) != blob_hash_hex:
        return _browser_origin_ineligible(raw_id, "source revision does not equal the retained blob digest")
    blob_ref = conn.execute(
        """
        SELECT blob_hash, source_path, size_bytes FROM source.blob_refs
        WHERE ref_id = ? AND ref_type = 'raw_payload'
        """,
        (raw_id,),
    ).fetchone()
    if (
        blob_ref is None
        or _bytes_value(blob_ref["blob_hash"]) != blob_hash
        or str(blob_ref["source_path"] or "") != source_path
        or int(blob_ref["size_bytes"]) != blob_size
    ):
        return _browser_origin_ineligible(raw_id, "raw payload blob reference does not match the source envelope")
    store = BlobStore(archive_root / "blob")
    blob_path = store.blob_path(blob_hash_hex)
    if not blob_path.is_file() or blob_path.stat().st_size != blob_size:
        return _browser_origin_ineligible(raw_id, "retained raw blob is missing or has the wrong size")
    payload = store.read_all(blob_hash_hex)
    if hashlib.sha256(payload).digest() != blob_hash:
        return _browser_origin_ineligible(raw_id, "retained raw blob digest does not match durable source evidence")
    try:
        decoded = json.loads(payload)
        provider = detect_provider(decoded)
        if provider is None or provider is Provider.UNKNOWN:
            return _browser_origin_ineligible(raw_id, "complete browser envelope has no canonical provider identity")
        from polylogue.sources.revision_backfill import _parse_one

        sessions = _parse_one(provider, payload, source_path)
    except Exception as exc:
        logger.warning(
            "browser capture origin repair normalization failed",
            raw_id=raw_id,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return _browser_origin_ineligible(
            raw_id, f"retained browser capture did not parse cleanly: {type(exc).__name__}"
        )
    if len(sessions) != 1:
        return _browser_origin_ineligible(raw_id, f"retained browser capture normalized to {len(sessions)} sessions")
    session = sessions[0]
    canonical_origin = origin_from_provider(provider)
    canonical_key = f"{provider.value}:{session.provider_session_id}"
    session_id = str(make_session_id(provider, session.provider_session_id))
    old_key = str(raw["logical_source_key"] or "")
    null_native_id_mode = allow_legacy_null_native_id or allow_byte_proven_null_native_id_rekey
    expected_old_native_id = None if null_native_id_mode else session.provider_session_id
    if null_native_id_mode and raw["native_id"] is not None:
        mode = "legacy-native-id" if allow_legacy_null_native_id else "byte-proven browser rekey"
        return _browser_origin_ineligible(raw_id, f"{mode} actuator requires a NULL durable native identity")
    if (
        _browser_origin_source_envelope_is_exact(
            conn,
            schema="source",
            raw_id=raw_id,
            origin=Origin.UNKNOWN_EXPORT.value,
            capture_mode=Provider.UNKNOWN.value,
            native_id=expected_old_native_id,
            source_path=source_path,
            source_index=0,
            blob_hash=blob_hash,
            blob_size=blob_size,
            logical_source_key=old_key,
            revision_kind=RawRevisionKind.FULL,
            source_revision=blob_hash_hex,
            baseline_raw_id=raw_id if allow_byte_proven_null_native_id_rekey else None,
            acquisition_generation=0,
            revision_authority=expected_source_authority,
        )
        is None
        or int(raw["source_index"]) != 0
    ):
        return _browser_origin_ineligible(raw_id, "source envelope does not exactly bind the normalized session")
    if old_key != f"{Provider.UNKNOWN.value}:{session.provider_session_id}":
        return _browser_origin_ineligible(raw_id, "unknown source key does not match the normalized session identity")
    if canonical_origin is Origin.UNKNOWN_EXPORT or canonical_key == old_key:
        return _browser_origin_ineligible(raw_id, "normalized provider does not establish a different canonical origin")
    accepted_hash = bytes.fromhex(session_content_hash(session))
    head = conn.execute(
        """
        SELECT session_id, accepted_source_revision, accepted_content_hash,
               accepted_frontier_kind, accepted_frontier, acquisition_generation,
               append_end_offset, decided_at_ms
        FROM raw_revision_heads WHERE logical_source_key = ? AND accepted_raw_id = ?
        """,
        (old_key, raw_id),
    ).fetchone()
    indexed = conn.execute(
        "SELECT raw_id, content_hash FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    indexed_evidence = conn.execute(
        "SELECT session_id, origin, native_id, content_hash FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    old_applications = conn.execute(
        """
        SELECT decision_id, raw_id, session_id, logical_source_key, source_revision, acquisition_generation,
               decision, accepted_raw_id, accepted_source_revision,
               accepted_content_hash, baseline_raw_id, predecessor_raw_id,
               append_end_offset, detail, decided_at_ms
        FROM raw_revision_applications
        WHERE raw_id = ? AND logical_source_key = ?
        ORDER BY decision_id
        """,
        (raw_id, old_key),
    ).fetchall()
    old_key_application_count = conn.execute(
        "SELECT COUNT(*) FROM raw_revision_applications WHERE logical_source_key = ?", (old_key,)
    ).fetchone()
    if (
        head is None
        or indexed is None
        or indexed_evidence is None
        or str(head["session_id"]) != session_id
        or str(head["accepted_source_revision"]) != blob_hash_hex
        or _bytes_value(head["accepted_content_hash"]) != accepted_hash
        or str(head["accepted_frontier_kind"]) != "byte"
        or int(head["accepted_frontier"]) != blob_size
        or int(head["acquisition_generation"]) != int(raw["acquisition_generation"])
        or head["append_end_offset"] is not None
        or _bytes_value(indexed["content_hash"]) != accepted_hash
        or len(old_applications) != 1
        or old_key_application_count is None
        or int(old_key_application_count[0]) != 1
        or not _revision_application_decision_id_is_exact(old_applications[0])
        or str(old_applications[0]["raw_id"]) != raw_id
        or str(old_applications[0]["session_id"]) != session_id
        or str(old_applications[0]["logical_source_key"]) != old_key
        or str(old_applications[0]["source_revision"]) != blob_hash_hex
        or int(old_applications[0]["acquisition_generation"]) != int(raw["acquisition_generation"])
        or str(old_applications[0]["decision"]) != ApplicationDecision.SELECTED_BASELINE.value
        or str(old_applications[0]["accepted_raw_id"]) != raw_id
        or str(old_applications[0]["accepted_source_revision"]) != blob_hash_hex
        or _bytes_value(old_applications[0]["accepted_content_hash"]) != accepted_hash
        or str(old_applications[0]["baseline_raw_id"]) != raw_id
        or old_applications[0]["predecessor_raw_id"] is not None
        or old_applications[0]["append_end_offset"] is not None
        or int(old_applications[0]["decided_at_ms"]) < 0
        or int(head["decided_at_ms"]) != int(old_applications[0]["decided_at_ms"])
    ):
        return _browser_origin_ineligible(raw_id, "current accepted head does not exactly prove the normalized session")
    membership = conn.execute(
        """
        SELECT provider_session_id, source_revision, normalized_content_hash,
               message_count, acquisition_generation, revision_authority, decision
        FROM source.raw_session_memberships
        WHERE raw_id = ? AND logical_source_key = ?
        """,
        (raw_id, canonical_key),
    ).fetchone()
    membership_count = conn.execute(
        "SELECT COUNT(*) FROM source.raw_session_memberships WHERE raw_id = ?", (raw_id,)
    ).fetchone()
    census = conn.execute(
        "SELECT status, member_count FROM source.raw_membership_census WHERE raw_id = ?",
        (raw_id,),
    ).fetchone()
    projection = session_revision_projection(session)
    if allow_byte_proven_null_native_id_rekey:
        if membership is not None or membership_count is None or int(membership_count[0]) != 0 or census is not None:
            return _browser_origin_ineligible(
                raw_id, "byte-proven browser rekey requires no retained membership census"
            )
    elif (
        membership is None
        or membership_count is None
        or int(membership_count[0]) != 1
        or census is None
        or str(membership["provider_session_id"]) != session.provider_session_id
        or str(membership["source_revision"]) != accepted_hash.hex()
        or _bytes_value(membership["normalized_content_hash"]) != projection.session_hash
        or int(membership["message_count"]) != len(projection.message_hashes)
        or int(membership["acquisition_generation"]) != int(raw["acquisition_generation"])
        or str(membership["revision_authority"]) != RawRevisionAuthority.QUARANTINED.value
        # A quarantined source raw has not been admitted to replay, so its
        # singleton census witness must remain undecided.  The old
        # selected-baseline receipt above is the separate, immutable authority
        # witness for the current unknown-key head.  A non-null membership
        # decision would be incompatible with this narrow recovery shape.
        or membership["decision"] is not None
        or str(census["status"]) != "complete"
        or int(census["member_count"]) != 1
    ):
        return _browser_origin_ineligible(raw_id, "membership census does not exactly reproduce the accepted session")
    copy_path = f"browser-capture-origin-copy-forward/{raw_id}.json"
    copy_raw_id = _browser_origin_copy_raw_id(
        canonical_origin,
        copy_path,
        int(raw["source_index"]),
        blob_hash,
        session.provider_session_id,
    )
    canonical_head = conn.execute(
        """
        SELECT session_id, accepted_raw_id, accepted_source_revision,
               accepted_content_hash, accepted_frontier_kind, accepted_frontier,
               acquisition_generation, append_end_offset, decided_at_ms
        FROM raw_revision_heads WHERE logical_source_key = ?
        """,
        (canonical_key,),
    ).fetchone()
    copy_columns = {str(row[1]) for row in conn.execute("PRAGMA source.table_info(raw_sessions)")}
    copy_capture_mode_projection = "capture_mode" if "capture_mode" in copy_columns else "NULL AS capture_mode"
    copy_raw = conn.execute(
        f"""
        SELECT origin, {copy_capture_mode_projection}, native_id, source_path, source_index, blob_hash, blob_size,
               logical_source_key, revision_kind, source_revision, baseline_raw_id,
               predecessor_source_revision, predecessor_raw_id, append_start_offset,
               append_end_offset, acquisition_generation, revision_authority
        FROM source.raw_sessions WHERE raw_id = ?
        """,
        (copy_raw_id,),
    ).fetchone()
    copy_blob_ref = conn.execute(
        """
        SELECT blob_hash, source_path, size_bytes FROM source.blob_refs
        WHERE ref_id = ? AND ref_type = 'raw_payload'
        """,
        (copy_raw_id,),
    ).fetchone()
    copy_membership = conn.execute(
        """
        SELECT provider_session_id, source_revision, normalized_content_hash,
               message_count, acquisition_generation, revision_authority, decision
        FROM source.raw_session_memberships
        WHERE raw_id = ? AND logical_source_key = ?
        """,
        (copy_raw_id, canonical_key),
    ).fetchone()
    copy_census = conn.execute(
        "SELECT status, member_count FROM source.raw_membership_census WHERE raw_id = ?",
        (copy_raw_id,),
    ).fetchone()
    copy_applications = conn.execute(
        """
        SELECT decision_id, raw_id, session_id, logical_source_key, source_revision, acquisition_generation,
               decision, accepted_raw_id, accepted_source_revision, accepted_content_hash,
               baseline_raw_id, predecessor_raw_id, append_end_offset, detail, decided_at_ms
        FROM raw_revision_applications
        WHERE raw_id = ? AND logical_source_key = ?
        """,
        (copy_raw_id, canonical_key),
    ).fetchall()
    copy_application = copy_applications[0] if len(copy_applications) == 1 else None
    copy_raw_exact = copy_raw is not None and (
        str(copy_raw["origin"]),
        str(copy_raw["capture_mode"]) if copy_raw["capture_mode"] is not None else None,
        str(copy_raw["native_id"]),
        str(copy_raw["source_path"]),
        int(copy_raw["source_index"]),
        _bytes_value(copy_raw["blob_hash"]),
        int(copy_raw["blob_size"]),
        str(copy_raw["logical_source_key"]),
        str(copy_raw["revision_kind"]),
        str(copy_raw["source_revision"]),
        copy_raw["predecessor_source_revision"],
        copy_raw["predecessor_raw_id"],
        str(copy_raw["baseline_raw_id"]),
        copy_raw["append_start_offset"],
        copy_raw["append_end_offset"],
        int(copy_raw["acquisition_generation"]),
        str(copy_raw["revision_authority"]),
    ) == (
        canonical_origin.value,
        provider.value if "capture_mode" in copy_columns else None,
        session.provider_session_id,
        copy_path,
        int(raw["source_index"]),
        blob_hash,
        blob_size,
        canonical_key,
        RawRevisionKind.FULL.value,
        blob_hash_hex,
        None,
        None,
        copy_raw_id,
        None,
        None,
        0,
        RawRevisionAuthority.BYTE_PROVEN.value,
    )
    copy_forward_source_complete = (
        copy_raw_exact
        and copy_blob_ref is not None
        and tuple(copy_blob_ref) == (blob_hash, copy_path, blob_size)
        and copy_membership is not None
        and str(copy_membership["provider_session_id"]) == session.provider_session_id
        and str(copy_membership["source_revision"]) == accepted_hash.hex()
        and _bytes_value(copy_membership["normalized_content_hash"]) == accepted_hash
        and int(copy_membership["message_count"]) == len(projection.message_hashes)
        and int(copy_membership["acquisition_generation"]) == 0
        and str(copy_membership["revision_authority"]) == RawRevisionAuthority.BYTE_PROVEN.value
        and str(copy_membership["decision"]) == "applied"
        and copy_census is not None
        and tuple(copy_census) == ("complete", 1)
    )
    copy_forward_terminal = (
        copy_forward_source_complete
        and canonical_head is not None
        and str(canonical_head["session_id"]) == session_id
        and str(canonical_head["accepted_raw_id"]) == copy_raw_id
        and str(canonical_head["accepted_source_revision"]) == blob_hash_hex
        and _bytes_value(canonical_head["accepted_content_hash"]) == accepted_hash
        and str(canonical_head["accepted_frontier_kind"]) == "byte"
        and int(canonical_head["accepted_frontier"]) == blob_size
        and int(canonical_head["acquisition_generation"]) == 0
        and canonical_head["append_end_offset"] is None
        and str(indexed["raw_id"]) == copy_raw_id
        and copy_application is not None
        and _revision_application_decision_id_is_exact(copy_application)
        and str(copy_application["session_id"]) == session_id
        and str(copy_application["logical_source_key"]) == canonical_key
        and str(copy_application["source_revision"]) == blob_hash_hex
        and int(copy_application["acquisition_generation"]) == 0
        and str(copy_application["decision"]) == ApplicationDecision.SELECTED_BASELINE.value
        and str(copy_application["accepted_raw_id"]) == copy_raw_id
        and str(copy_application["accepted_source_revision"]) == blob_hash_hex
        and _bytes_value(copy_application["accepted_content_hash"]) == accepted_hash
        and str(copy_application["baseline_raw_id"]) == copy_raw_id
        and copy_application["predecessor_raw_id"] is None
        and copy_application["append_end_offset"] is None
        and int(copy_application["decided_at_ms"]) >= 0
        and int(canonical_head["decided_at_ms"]) == int(copy_application["decided_at_ms"])
    )
    repair_strategy = "copy_forward"
    replacement_raw_id = copy_raw_id
    replacement_source_revision = blob_hash_hex
    replacement_frontier_kind = "byte"
    replacement_frontier = blob_size
    already_repaired = copy_forward_terminal
    semantic_canonical_raw_id: str | None = None
    semantic_historical_raw_ids: tuple[str, ...] = ()
    semantic_head_snapshot: dict[str, object] | None = None
    semantic_witness_digest: str | None = None
    terminal_byte_witness_digest: str | None = None
    if (
        copy_raw is not None
        and copy_forward_source_complete
        and str(indexed["raw_id"]) == copy_raw_id
        and not copy_forward_terminal
    ):
        return _browser_origin_ineligible(raw_id, "copy-forward terminal byte authority is not exact")
    if (
        (allow_legacy_null_native_id or allow_byte_proven_null_native_id_rekey)
        and canonical_head is not None
        and not copy_forward_terminal
        and str(canonical_head["accepted_frontier_kind"]) != "semantic"
    ):
        return _browser_origin_ineligible(
            raw_id,
            (
                "legacy-native-id copy-forward refuses pre-existing canonical head authority"
                if allow_legacy_null_native_id
                else "byte-proven browser rekey refuses pre-existing canonical head authority"
            ),
        )
    if canonical_head is not None and not copy_forward_terminal:
        if _canonical_browser_origin_head_is_exact(
            conn,
            archive_root=archive_root,
            canonical_head=canonical_head,
            canonical_key=canonical_key,
            canonical_origin=canonical_origin,
            session_id=session_id,
            accepted_hash=accepted_hash,
            message_count=len(projection.message_hashes),
        ):
            replacement_raw_id = str(canonical_head["accepted_raw_id"])
            replacement_source_revision = str(canonical_head["accepted_source_revision"])
            replacement_frontier_kind = str(canonical_head["accepted_frontier_kind"])
            replacement_frontier = int(canonical_head["accepted_frontier"])
            supersession = conn.execute(
                """
                SELECT accepted_raw_id, accepted_source_revision, accepted_content_hash,
                       detail FROM raw_revision_applications
                WHERE raw_id = ? AND logical_source_key = ? AND decision = 'superseded'
                """,
                (raw_id, canonical_key),
            ).fetchone()
            repair_strategy = "restore_canonical_head"
            already_repaired = (
                str(indexed["raw_id"]) == replacement_raw_id
                and supersession is not None
                and tuple(supersession)
                == (
                    replacement_raw_id,
                    replacement_source_revision,
                    accepted_hash,
                    f"browser_capture_origin_supersession:{raw_id}",
                )
            )
        else:
            semantic_witness = _canonical_browser_origin_head_is_semantically_equivalent(
                conn,
                archive_root=archive_root,
                canonical_head=canonical_head,
                canonical_key=canonical_key,
                canonical_origin=canonical_origin,
                session_id=session_id,
                accepted_hash=accepted_hash,
                message_count=len(projection.message_hashes),
                indexed_raw_id=str(indexed["raw_id"]),
            )
            if semantic_witness is None:
                return _browser_origin_ineligible(raw_id, "canonical logical source has an incompatible accepted head")
            semantic_canonical_raw_id = semantic_witness.raw_id
            semantic_historical_raw_ids = semantic_witness.historical_raw_ids
            semantic_head_snapshot = semantic_witness.head_snapshot
            semantic_witness_digest = semantic_witness.digest
    if copy_raw is not None and repair_strategy == "copy_forward" and not copy_forward_source_complete:
        return _browser_origin_ineligible(
            raw_id, "copy-forward raw id exists but its durable source stage is not exact"
        )
    if str(indexed["raw_id"]) not in {raw_id, replacement_raw_id}:
        return _browser_origin_ineligible(raw_id, "indexed session points at unrelated raw evidence")
    if copy_forward_terminal:
        assert copy_application is not None
        detail_is_exact, terminal_snapshot = _browser_origin_semantic_head_from_copy_detail(
            copy_application["detail"],
            raw_id=raw_id,
            parser_derived_native_id=session.provider_session_id if null_native_id_mode else None,
            byte_proven_null_native_id_rekey=allow_byte_proven_null_native_id_rekey,
        )
        if not detail_is_exact:
            return _browser_origin_ineligible(raw_id, "copy-forward receipt detail is not exact")
        if terminal_snapshot is not None:
            terminal_witness = _canonical_browser_origin_head_is_semantically_equivalent(
                conn,
                archive_root=archive_root,
                canonical_head=terminal_snapshot,
                canonical_key=canonical_key,
                canonical_origin=canonical_origin,
                session_id=session_id,
                accepted_hash=accepted_hash,
                message_count=len(projection.message_hashes),
                indexed_raw_id=str(indexed["raw_id"]),
                excluded_application_decision_id=str(copy_application["decision_id"]),
            )
            if terminal_witness is None or terminal_witness.head_snapshot != terminal_snapshot:
                return _browser_origin_ineligible(raw_id, "copy-forward semantic witness no longer reproves")
            semantic_canonical_raw_id = terminal_witness.raw_id
            semantic_historical_raw_ids = terminal_witness.historical_raw_ids
            semantic_head_snapshot = terminal_witness.head_snapshot
            semantic_witness_digest = terminal_witness.digest
        terminal_byte_witness_digest = _authority_rows_digest([canonical_head], [copy_application])
    evidence_digest = _authority_rows_digest(
        [raw],
        [head],
        [indexed_evidence],
        [] if membership is None else [membership],
        [] if census is None else [census],
        old_applications,
    )
    item = BrowserCaptureOriginRepairItem(
        raw_id=raw_id,
        status="already_repaired" if already_repaired else "eligible",
        reason=(
            "canonical replacement head already supersedes the mismatched current pointer"
            if already_repaired
            else "exact retained browser capture can be repaired without rewriting historical evidence"
        ),
        origin=str(raw["origin"]),
        source_path=source_path,
        source_index=int(raw["source_index"]),
        blob_hash=blob_hash_hex,
        blob_size=blob_size,
        old_logical_source_key=old_key,
        canonical_provider=provider.value,
        canonical_origin=canonical_origin.value,
        canonical_logical_source_key=canonical_key,
        session_id=session_id,
        accepted_content_hash=accepted_hash.hex(),
        parsed_message_count=len(projection.message_hashes),
        accepted_frontier=blob_size,
        repair_strategy=repair_strategy,
        replacement_raw_id=replacement_raw_id,
        replacement_source_revision=replacement_source_revision,
        replacement_content_hash=accepted_hash.hex(),
        replacement_frontier_kind=replacement_frontier_kind,
        replacement_frontier=replacement_frontier,
        copy_forward_raw_id=copy_raw_id if repair_strategy == "copy_forward" else None,
        copy_forward_source_path=copy_path if repair_strategy == "copy_forward" else None,
        copy_forward_source_complete=copy_forward_source_complete,
        semantic_canonical_raw_id=semantic_canonical_raw_id,
        semantic_historical_raw_ids=semantic_historical_raw_ids,
        semantic_head_snapshot=semantic_head_snapshot,
        semantic_witness_digest=semantic_witness_digest,
        terminal_byte_witness_digest=terminal_byte_witness_digest,
        legacy_null_native_id=allow_legacy_null_native_id,
        parser_derived_native_id=session.provider_session_id if null_native_id_mode else None,
        byte_proven_null_native_id_rekey=allow_byte_proven_null_native_id_rekey,
        evidence_digest=evidence_digest,
    )
    return dataclasses.replace(item, proof_digest=_browser_origin_item_digest(item))


def _stage_browser_origin_copy_forward_source(
    conn: sqlite3.Connection,
    item: BrowserCaptureOriginRepairItem,
    *,
    source_schema: str = "main",
) -> None:
    if source_schema not in {"main", "source"}:
        raise ValueError(f"unsupported source schema: {source_schema}")
    assert item.copy_forward_raw_id is not None
    assert item.copy_forward_source_path is not None
    assert item.canonical_origin is not None
    assert item.canonical_provider is not None
    assert item.canonical_logical_source_key is not None
    assert item.session_id is not None
    assert item.blob_hash is not None
    assert item.blob_size is not None
    assert item.source_index is not None
    assert item.accepted_content_hash is not None
    assert item.accepted_frontier is not None
    native_id = item.session_id.split(":", 1)[1]
    blob_hash = bytes.fromhex(item.blob_hash)
    columns = {str(row[1]) for row in conn.execute(f"PRAGMA {source_schema}.table_info(raw_sessions)")}
    names = [
        "raw_id",
        "origin",
        "native_id",
        "source_path",
        "source_index",
        "blob_hash",
        "blob_size",
        "acquired_at_ms",
        "logical_source_key",
        "revision_kind",
        "source_revision",
        "baseline_raw_id",
        "acquisition_generation",
        "revision_authority",
    ]
    values: list[object] = [
        item.copy_forward_raw_id,
        item.canonical_origin,
        native_id,
        item.copy_forward_source_path,
        item.source_index,
        blob_hash,
        item.blob_size,
        int(time.time() * 1000),
        item.canonical_logical_source_key,
        RawRevisionKind.FULL.value,
        item.blob_hash,
        item.copy_forward_raw_id,
        0,
        RawRevisionAuthority.BYTE_PROVEN.value,
    ]
    if "capture_mode" in columns:
        names.insert(2, "capture_mode")
        values.insert(2, item.canonical_provider)
    conn.execute(
        f"INSERT INTO {source_schema}.raw_sessions ({', '.join(names)}) VALUES ({', '.join('?' for _ in names)})",
        values,
    )
    acquired_at_ms = int(time.time() * 1000)
    conn.execute(
        f"""
        INSERT INTO {source_schema}.blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
        VALUES (?, ?, 'raw_payload', ?, ?, ?)
        """,
        (blob_hash, item.copy_forward_raw_id, item.copy_forward_source_path, item.blob_size, acquired_at_ms),
    )
    if item.byte_proven_null_native_id_rekey:
        assert item.parsed_message_count is not None
        conn.execute(
            f"""
            INSERT INTO {source_schema}.raw_session_memberships (
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count, acquisition_generation,
                revision_authority, decision, decided_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, 0, 'byte_proven', 'applied', ?)
            """,
            (
                item.copy_forward_raw_id,
                item.canonical_logical_source_key,
                native_id,
                item.accepted_content_hash,
                bytes.fromhex(item.accepted_content_hash),
                item.parsed_message_count,
                acquired_at_ms,
            ),
        )
    else:
        conn.execute(
            f"""
            INSERT INTO {source_schema}.raw_session_memberships (
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count, acquisition_generation,
                revision_authority, decision, decided_at_ms
            )
            SELECT ?, ?, ?, ?, ?, message_count, 0, 'byte_proven', 'applied', ?
            FROM {source_schema}.raw_session_memberships
            WHERE raw_id = ? AND logical_source_key = ?
            """,
            (
                item.copy_forward_raw_id,
                item.canonical_logical_source_key,
                native_id,
                item.accepted_content_hash,
                bytes.fromhex(item.accepted_content_hash),
                acquired_at_ms,
                item.raw_id,
                item.canonical_logical_source_key,
            ),
        )
    conn.execute(
        f"""
        INSERT INTO {source_schema}.raw_membership_census (
            raw_id, parser_fingerprint, status, member_count, censused_at_ms, detail
        ) VALUES (?, 'browser-origin-copy-forward-v1', 'complete', 1, ?, ?)
        """,
        (
            item.copy_forward_raw_id,
            acquired_at_ms,
            f"origin copy-forward from immutable raw {item.raw_id}",
        ),
    )


def _finalize_browser_origin_copy_forward_index(conn: sqlite3.Connection, item: BrowserCaptureOriginRepairItem) -> None:
    from polylogue.storage.sqlite.archive_tiers.revision_application import (
        RevisionApplicationReceipt,
        record_revision_application_sync,
    )

    assert item.copy_forward_raw_id is not None
    assert item.canonical_logical_source_key is not None
    assert item.session_id is not None
    assert item.blob_hash is not None
    assert item.accepted_content_hash is not None
    assert item.accepted_frontier is not None
    acquired_at_ms = int(time.time() * 1000)
    if item.semantic_canonical_raw_id is not None:
        cursor = conn.execute(
            """
            UPDATE raw_revision_heads
            SET accepted_raw_id = ?, accepted_source_revision = ?,
                accepted_frontier_kind = 'byte', accepted_frontier = ?,
                acquisition_generation = 0, append_end_offset = NULL,
                decided_at_ms = ?
            WHERE logical_source_key = ? AND session_id = ? AND accepted_raw_id = ?
              AND accepted_content_hash = ? AND accepted_frontier_kind = 'semantic'
            """,
            (
                item.copy_forward_raw_id,
                item.blob_hash,
                item.accepted_frontier,
                acquired_at_ms,
                item.canonical_logical_source_key,
                item.session_id,
                item.semantic_canonical_raw_id,
                bytes.fromhex(item.accepted_content_hash),
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(f"semantic canonical-head CAS failed for {item.raw_id}")
    cursor = conn.execute(
        "UPDATE sessions SET raw_id = ? WHERE session_id = ? AND raw_id = ?",
        (item.copy_forward_raw_id, item.session_id, item.raw_id),
    )
    if cursor.rowcount != 1:
        raise RuntimeError(f"session raw pointer CAS failed for {item.raw_id}")
    record_revision_application_sync(
        conn,
        RevisionApplicationReceipt(
            raw_id=item.copy_forward_raw_id,
            session_id=item.session_id,
            logical_source_key=item.canonical_logical_source_key,
            source_revision=item.blob_hash,
            acquisition_generation=0,
            decision=ApplicationDecision.SELECTED_BASELINE,
            accepted_raw_id=item.copy_forward_raw_id,
            accepted_source_revision=item.blob_hash,
            accepted_content_hash=bytes.fromhex(item.accepted_content_hash),
            accepted_frontier_kind="byte",
            accepted_frontier=item.accepted_frontier,
            baseline_raw_id=item.copy_forward_raw_id,
            detail=_browser_origin_copy_forward_detail(item),
        ),
        decided_at_ms=acquired_at_ms,
    )
    _retire_browser_origin_legacy_head(
        conn,
        item,
        accepted_raw_id=item.copy_forward_raw_id,
        accepted_source_revision=item.blob_hash,
        accepted_content_hash=item.replacement_content_hash or item.accepted_content_hash,
        accepted_frontier_kind="byte",
        accepted_frontier=item.accepted_frontier,
        decided_at_ms=acquired_at_ms,
    )


def _retire_browser_origin_legacy_head(
    conn: sqlite3.Connection,
    item: BrowserCaptureOriginRepairItem,
    *,
    accepted_raw_id: str,
    accepted_source_revision: str,
    accepted_content_hash: str,
    accepted_frontier_kind: str,
    accepted_frontier: int,
    decided_at_ms: int,
) -> None:
    """Terminally receipt and remove the exact obsolete unknown-key head."""
    from polylogue.storage.sqlite.archive_tiers.revision_application import (
        RevisionApplicationReceipt,
        record_revision_application_sync,
    )

    assert item.session_id is not None
    assert item.old_logical_source_key is not None
    assert item.blob_hash is not None
    assert item.accepted_content_hash is not None
    assert item.accepted_frontier is not None
    record_revision_application_sync(
        conn,
        RevisionApplicationReceipt(
            raw_id=item.raw_id,
            session_id=item.session_id,
            logical_source_key=item.old_logical_source_key,
            source_revision=item.blob_hash,
            acquisition_generation=0,
            decision=ApplicationDecision.SUPERSEDED,
            accepted_raw_id=accepted_raw_id,
            accepted_source_revision=accepted_source_revision,
            accepted_content_hash=bytes.fromhex(accepted_content_hash),
            accepted_frontier_kind=accepted_frontier_kind,
            accepted_frontier=accepted_frontier,
            detail=f"browser_capture_origin_supersession:{item.raw_id}",
        ),
        decided_at_ms=decided_at_ms,
    )
    deleted = conn.execute(
        """
        DELETE FROM raw_revision_heads
        WHERE logical_source_key = ? AND session_id = ? AND accepted_raw_id = ?
          AND accepted_source_revision = ? AND accepted_content_hash = ?
          AND accepted_frontier_kind = 'byte' AND accepted_frontier = ?
          AND acquisition_generation = 0 AND append_end_offset IS NULL
        """,
        (
            item.old_logical_source_key,
            item.session_id,
            item.raw_id,
            item.blob_hash,
            bytes.fromhex(item.accepted_content_hash),
            item.accepted_frontier,
        ),
    ).rowcount
    if deleted != 1:
        raise RuntimeError(f"obsolete browser-origin head CAS failed for {item.raw_id}")


def _restore_browser_origin_canonical_head(conn: sqlite3.Connection, item: BrowserCaptureOriginRepairItem) -> None:
    assert item.session_id is not None
    assert item.canonical_logical_source_key is not None
    assert item.blob_hash is not None
    assert item.accepted_content_hash is not None
    assert item.replacement_raw_id is not None
    assert item.replacement_source_revision is not None
    assert item.replacement_frontier_kind is not None
    assert item.replacement_frontier is not None
    decided_at_ms = int(time.time() * 1000)
    cursor = conn.execute(
        "UPDATE sessions SET raw_id = ? WHERE session_id = ? AND raw_id = ?",
        (item.replacement_raw_id, item.session_id, item.raw_id),
    )
    if cursor.rowcount != 1:
        raise RuntimeError(f"session raw pointer CAS failed for {item.raw_id}")
    _retire_browser_origin_legacy_head(
        conn,
        item,
        accepted_raw_id=item.replacement_raw_id,
        accepted_source_revision=item.replacement_source_revision,
        accepted_content_hash=item.replacement_content_hash or item.accepted_content_hash,
        accepted_frontier_kind=item.replacement_frontier_kind,
        accepted_frontier=item.replacement_frontier,
        decided_at_ms=decided_at_ms,
    )


def _apply_browser_origin_repair_item(conn: sqlite3.Connection, item: BrowserCaptureOriginRepairItem) -> None:
    if item.repair_strategy == "copy_forward":
        _finalize_browser_origin_copy_forward_index(conn, item)
        return
    if item.repair_strategy == "restore_canonical_head":
        _restore_browser_origin_canonical_head(conn, item)
        return
    raise RuntimeError(f"unsupported browser-capture origin repair strategy: {item.repair_strategy}")


def _browser_origin_strategy_terminal(conn: sqlite3.Connection, item: BrowserCaptureOriginRepairItem) -> bool:
    """Prove the shared strategy retired the legacy head and selected its successor."""
    replacement_raw_id = item.copy_forward_raw_id or item.replacement_raw_id
    if (
        item.session_id is None
        or item.old_logical_source_key is None
        or item.canonical_logical_source_key is None
        or replacement_raw_id is None
        or item.replacement_source_revision is None
        or item.replacement_frontier_kind is None
        or item.replacement_frontier is None
    ):
        return False
    session = conn.execute("SELECT raw_id FROM sessions WHERE session_id = ?", (item.session_id,)).fetchone()
    canonical = conn.execute(
        """
        SELECT accepted_raw_id, accepted_source_revision, accepted_content_hash,
               accepted_frontier_kind, accepted_frontier
        FROM raw_revision_heads WHERE logical_source_key = ? AND session_id = ?
        """,
        (item.canonical_logical_source_key, item.session_id),
    ).fetchone()
    legacy = conn.execute(
        "SELECT 1 FROM raw_revision_heads WHERE logical_source_key = ?",
        (item.old_logical_source_key,),
    ).fetchone()
    superseded = conn.execute(
        """
        SELECT accepted_raw_id, accepted_source_revision, accepted_content_hash
        FROM raw_revision_applications
        WHERE raw_id = ? AND session_id = ? AND logical_source_key = ?
          AND decision = 'superseded'
        """,
        (item.raw_id, item.session_id, item.old_logical_source_key),
    ).fetchone()
    expected = (
        replacement_raw_id,
        item.replacement_source_revision,
        bytes.fromhex(item.replacement_content_hash or item.accepted_content_hash or ""),
        item.replacement_frontier_kind,
        item.replacement_frontier,
    )
    expected_supersession = expected[:3]
    return (
        session is not None
        and str(session[0]) == replacement_raw_id
        and canonical is not None
        and tuple(canonical) == expected
        and legacy is None
        and superseded is not None
        and tuple(superseded) == expected_supersession
    )


def _apply_browser_conflict_canonical_resolution(
    archive_root: Path,
    conn: sqlite3.Connection,
    raw_id: str,
    expected_witness: Mapping[str, object],
) -> None:
    """Apply one explicit retain-canonical judgment against an exact conflict witness."""
    base = _inspect_browser_capture_origin_strategy(archive_root, raw_id, conn=conn)
    if base.status != "ineligible":
        raise RuntimeError("conflict resolution no longer observes a conflicting browser authority")
    current = _browser_canonical_authority_conflict_witness(archive_root, conn, raw_id, base.reason)
    current_witness = dataclasses.asdict(current)
    # ``reason`` is the inspector's human-readable rejection path.  It can
    # legitimately become more specific as the shared classifier evolves and
    # is deliberately excluded from the evidence digest.  Every structural
    # witness field remains an exact CAS precondition.
    comparable_current = {key: value for key, value in current_witness.items() if key != "reason"}
    comparable_expected = {key: value for key, value in expected_witness.items() if key != "reason"}
    if comparable_current != comparable_expected:
        changed_fields = sorted(
            key
            for key in comparable_current.keys() | comparable_expected.keys()
            if comparable_current.get(key) != comparable_expected.get(key)
        )
        raise RuntimeError(
            f"browser conflict evidence changed after operator judgment: fields={','.join(changed_fields)}"
        )
    required = (
        current.session_id,
        current.old_logical_source_key,
        current.canonical_logical_source_key,
        current.unknown_raw_content_hash,
        current.unknown_source_revision,
        current.unknown_frontier,
        current.competing_raw_id,
        current.competing_content_hash,
        current.competing_source_revision,
        current.competing_frontier_kind,
        current.competing_frontier,
    )
    if any(value is None for value in required):
        raise RuntimeError("retain-canonical judgment lacks a complete typed competing-head witness")
    repair_item = BrowserCaptureOriginRepairItem(
        raw_id=raw_id,
        status="eligible",
        reason="accepted operator judgment retained canonical authority",
        session_id=current.session_id,
        old_logical_source_key=current.old_logical_source_key,
        canonical_logical_source_key=current.canonical_logical_source_key,
        blob_hash=current.unknown_source_revision,
        accepted_content_hash=current.unknown_raw_content_hash,
        accepted_frontier=current.unknown_frontier,
        repair_strategy="restore_canonical_head",
        replacement_raw_id=current.competing_raw_id,
        replacement_source_revision=current.competing_source_revision,
        replacement_content_hash=current.competing_content_hash,
        replacement_frontier_kind=current.competing_frontier_kind,
        replacement_frontier=current.competing_frontier,
    )
    _restore_browser_origin_canonical_head(conn, repair_item)
    if not _browser_origin_strategy_terminal(conn, repair_item):
        raise RuntimeError("retain-canonical judgment did not reach its typed terminal postcondition")


def _inspect_browser_capture_origin_strategy(
    archive_root: Path,
    raw_id: str,
    *,
    conn: sqlite3.Connection,
) -> BrowserCaptureOriginRepairItem:
    """Select one admitted strategy shape from its durable source envelope."""
    conn.row_factory = sqlite3.Row
    envelope = conn.execute(
        "SELECT native_id, revision_authority FROM source.raw_sessions WHERE raw_id = ?",
        (raw_id,),
    ).fetchone()
    legacy_null = bool(
        envelope is not None
        and envelope["native_id"] is None
        and envelope["revision_authority"] == RawRevisionAuthority.QUARANTINED.value
    )
    byte_proven_null = bool(
        envelope is not None
        and envelope["native_id"] is None
        and envelope["revision_authority"] == RawRevisionAuthority.BYTE_PROVEN.value
    )
    return _inspect_browser_capture_origin_mismatch(
        archive_root,
        raw_id,
        conn=conn,
        allow_legacy_null_native_id=legacy_null,
        allow_byte_proven_null_native_id_rekey=byte_proven_null,
    )


def inspect_browser_capture_origin_mismatches(
    config: Config,
    raw_ids: list[str],
) -> tuple[BrowserCaptureOriginRepairItem, ...]:
    """Return the exact admitted browser-origin strategy for each raw.

    The durable source envelope selects the applicable strategy shape.  This
    keeps historical null-native-id variants behind the same inspector and
    plan contract instead of exposing mode-specific repair entrypoints.
    """
    if len(set(raw_ids)) != len(raw_ids):
        raise ValueError("duplicate raw ids are not allowed")
    if not raw_ids or len(raw_ids) > _QUARANTINED_ACCEPTED_RAW_REPAIR_LIMIT:
        raise ValueError("raw-id list must contain 1..100 entries")
    if any(re.fullmatch(r"[0-9a-f]{64}", raw_id) is None for raw_id in raw_ids):
        raise ValueError("raw ids must be lowercase SHA-256 identifiers")
    archive_root = _raw_materialization_archive_root(config)
    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    if not source_db.exists() or not index_db.exists():
        raise RuntimeError("source or index tier is missing")
    with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
        conn.execute("ATTACH DATABASE ? AS source", (str(source_db),))
        return tuple(_inspect_browser_capture_origin_strategy(archive_root, raw_id, conn=conn) for raw_id in raw_ids)


def _browser_canonical_authority_conflict_witness(
    archive_root: Path, conn: sqlite3.Connection, raw_id: str, base_reason: str
) -> BrowserCanonicalAuthorityConflictWitness:
    """Re-derive competing-authority evidence for one ineligible byte-proven-rekey raw.

    Kept independent of ``_inspect_browser_capture_origin_mismatch`` (rather than
    threading extra return fields through it): that function's job is to prove
    or refuse a repair, and its many early-return branches deliberately discard
    partial state once refused. This function's only job is to look, so it
    re-derives the handful of facts a human adjudicator needs and fails soft
    (``divergence_note`` explains any gap) instead of failing closed.
    """

    def ineligible(reason: str) -> BrowserCanonicalAuthorityConflictWitness:
        return BrowserCanonicalAuthorityConflictWitness(
            raw_id=raw_id, status="ineligible", reason=base_reason, divergence_note=reason
        )

    raw = conn.execute(
        """
        SELECT origin, source_path, blob_hash, blob_size, source_revision
        FROM source.raw_sessions WHERE raw_id = ?
        """,
        (raw_id,),
    ).fetchone()
    if raw is None or str(raw["origin"]) != Origin.UNKNOWN_EXPORT.value:
        return ineligible("raw is missing or not typed unknown-export; cannot re-derive evidence")
    source_path = str(raw["source_path"])
    blob_hash = _bytes_value(raw["blob_hash"])
    blob_size = int(raw["blob_size"])
    try:
        store = BlobStore(archive_root / "blob")
        payload = store.read_all(blob_hash.hex())
        if len(payload) != blob_size or hashlib.sha256(payload).digest() != blob_hash:
            return ineligible("retained blob no longer matches the declared source envelope")
        decoded = json.loads(payload)
        provider = detect_provider(decoded)
        if provider is None or provider is Provider.UNKNOWN:
            return ineligible("retained envelope has no canonical provider identity")
        from polylogue.sources.revision_backfill import _parse_one

        sessions = _parse_one(provider, payload, source_path)
        if len(sessions) != 1:
            return ineligible(f"retained envelope normalized to {len(sessions)} sessions, expected 1")
        session = sessions[0]
        canonical_key = f"{provider.value}:{session.provider_session_id}"
        session_id = str(make_session_id(provider, session.provider_session_id))
        accepted_hash = bytes.fromhex(session_content_hash(session))
        projection = session_revision_projection(session)
    except Exception as exc:
        logger.warning(
            "browser canonical authority conflict evidence build failed",
            raw_id=raw_id,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return ineligible(f"could not re-parse the retained envelope: {type(exc).__name__}")

    head = conn.execute(
        """
        SELECT accepted_raw_id, accepted_source_revision, accepted_content_hash,
               accepted_frontier_kind, accepted_frontier, decided_at_ms
        FROM raw_revision_heads WHERE logical_source_key = ?
        """,
        (canonical_key,),
    ).fetchone()
    old_key = f"{Provider.UNKNOWN.value}:{session.provider_session_id}"
    unknown_head = conn.execute(
        """
        SELECT accepted_frontier_kind, accepted_frontier, decided_at_ms
        FROM raw_revision_heads
        WHERE logical_source_key = ? AND session_id = ? AND accepted_raw_id = ?
        """,
        (old_key, session_id, raw_id),
    ).fetchone()
    # Deliberately not scoped to ``canonical_key``: the byte-proven-rekey
    # actuator's precondition rejects on *any* retained membership row for
    # this raw id (``COUNT(*) ... WHERE raw_id = ?`` with no key filter,
    # ``_inspect_browser_capture_origin_mismatch``), including a stale row
    # still keyed under the historical unknown-origin key. Matching that
    # broader predicate here is what lets this function explain a
    # ``superseded_equivalent``-shaped conflict instead of reporting a false
    # "no competing evidence found".
    membership = conn.execute(
        """
        SELECT decision, normalized_content_hash, message_count, logical_source_key
        FROM source.raw_session_memberships
        WHERE raw_id = ?
        ORDER BY logical_source_key
        LIMIT 1
        """,
        (raw_id,),
    ).fetchone()

    competing_raw_id: str | None = None
    competing_content_hash: str | None = None
    competing_source_revision: str | None = None
    competing_frontier_kind: str | None = None
    competing_frontier: int | None = None
    competing_decided_at_ms: int | None = None
    competing_decision: str | None = None
    competing_message_count: int | None = None
    divergent_message_index: int | None = None
    divergence_note: str | None = None

    if head is not None:
        competing_raw_id = str(head["accepted_raw_id"])
        competing_content_hash = _bytes_value(head["accepted_content_hash"]).hex()
        competing_source_revision = str(head["accepted_source_revision"])
        competing_frontier_kind = str(head["accepted_frontier_kind"])
        competing_frontier = int(head["accepted_frontier"])
        competing_decided_at_ms = int(head["decided_at_ms"])
        application = conn.execute(
            """
            SELECT decision FROM raw_revision_applications
            WHERE logical_source_key = ? AND accepted_raw_id = ?
            ORDER BY decision_id DESC LIMIT 1
            """,
            (canonical_key, competing_raw_id),
        ).fetchone()
        if application is not None:
            competing_decision = str(application["decision"])
        if competing_content_hash == accepted_hash.hex():
            divergence_note = (
                "competing head content hash matches; conflict is membership/precondition-shaped, not a hash divergence"
            )
        elif competing_frontier_kind == "byte" and competing_raw_id != raw_id:
            try:
                competing_source = conn.execute(
                    "SELECT source_path FROM source.raw_sessions WHERE raw_id = ?", (competing_raw_id,)
                ).fetchone()
                competing_source_revision = str(head["accepted_source_revision"])
                competing_payload = store.read_all(competing_source_revision)
                competing_provider = detect_provider(json.loads(competing_payload))
                if competing_provider is not None and competing_source is not None:
                    competing_sessions = _parse_one(
                        competing_provider, competing_payload, str(competing_source["source_path"])
                    )
                    if len(competing_sessions) == 1:
                        competing_projection = session_revision_projection(competing_sessions[0])
                        competing_message_count = len(competing_projection.message_hashes)
                        divergent_message_index = next(
                            (
                                index
                                for index, (left, right) in enumerate(
                                    zip(projection.message_hashes, competing_projection.message_hashes, strict=False)
                                )
                                if left != right
                            ),
                            min(len(projection.message_hashes), len(competing_projection.message_hashes)),
                        )
            except Exception as exc:
                logger.warning(
                    "browser canonical authority conflict competing-head diff unavailable",
                    raw_id=raw_id,
                    competing_raw_id=competing_raw_id,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                divergence_note = (
                    f"competing head content hash differs; message-level diff unavailable: {type(exc).__name__}"
                )
        else:
            divergence_note = (
                "competing head content hash differs (semantic frontier; no single-raw message diff available)"
            )
    elif membership is not None:
        divergence_note = (
            f"no competing byte/semantic head at {canonical_key}; a retained membership row "
            f"(decision={membership['decision']!r}) blocks the null-membership precondition instead"
        )
    else:
        divergence_note = "no competing head or membership row found; re-run the ordinary rekey actuator"

    packet = {
        "raw_id": raw_id,
        "session_id": session_id,
        "canonical_logical_source_key": canonical_key,
        "unknown_raw_content_hash": accepted_hash.hex(),
        "unknown_source_revision": str(raw["source_revision"]),
        "unknown_frontier_kind": None if unknown_head is None else str(unknown_head["accepted_frontier_kind"]),
        "unknown_frontier": None if unknown_head is None else int(unknown_head["accepted_frontier"]),
        "unknown_decided_at_ms": None if unknown_head is None else int(unknown_head["decided_at_ms"]),
        "competing_raw_id": competing_raw_id,
        "competing_content_hash": competing_content_hash,
        "competing_source_revision": competing_source_revision,
        "competing_frontier_kind": competing_frontier_kind,
        "competing_frontier": competing_frontier,
        "competing_decided_at_ms": competing_decided_at_ms,
        "competing_decision": competing_decision,
        "divergent_message_index": divergent_message_index,
    }
    evidence_digest = hashlib.sha256(json.dumps(packet, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return BrowserCanonicalAuthorityConflictWitness(
        raw_id=raw_id,
        status="ineligible",
        reason=base_reason,
        session_id=session_id,
        old_logical_source_key=old_key,
        canonical_logical_source_key=canonical_key,
        unknown_raw_content_hash=accepted_hash.hex(),
        unknown_source_revision=str(raw["source_revision"]),
        unknown_frontier_kind=None if unknown_head is None else str(unknown_head["accepted_frontier_kind"]),
        unknown_frontier=None if unknown_head is None else int(unknown_head["accepted_frontier"]),
        unknown_decided_at_ms=None if unknown_head is None else int(unknown_head["decided_at_ms"]),
        unknown_raw_message_count=len(projection.message_hashes),
        competing_raw_id=competing_raw_id,
        competing_content_hash=competing_content_hash,
        competing_source_revision=competing_source_revision,
        competing_frontier_kind=competing_frontier_kind,
        competing_frontier=competing_frontier,
        competing_decided_at_ms=competing_decided_at_ms,
        competing_decision=competing_decision,
        competing_message_count=competing_message_count,
        divergent_message_index=divergent_message_index,
        divergence_note=divergence_note,
        evidence_digest=evidence_digest,
    )


def inspect_browser_canonical_authority_conflicts(
    config: Config, raw_ids: list[str]
) -> BrowserCanonicalAuthorityConflictReport:
    """Build read-only evidence packets for browser-capture raws a safe rekey refuses.

    Companion to :func:`inspect_browser_capture_origin_mismatches`: re-runs the
    shared actuator strategy's exact eligibility proof for each
    raw id. A raw that comes back ``eligible``/``already_repaired`` is not a
    conflict -- the ordinary actuator already owns it -- and is only counted in
    ``resolved_count``. A raw that stays ``ineligible`` gets an enriched
    :class:`BrowserCanonicalAuthorityConflictWitness`: the competing canonical
    head's content hash/frontier kind/decision, any blocking membership row, and
    -- when both sides are single-session byte-frontier raws -- the first
    diverging message index.

    Never mutates state and never selects an authority between the two
    histories; see :func:`record_browser_canonical_authority_conflict_blockers`
    for the separate, explicit step that persists this as a durable,
    operator-reviewable candidate blocker in ``user.db``.
    """
    if len(set(raw_ids)) != len(raw_ids):
        raise ValueError("duplicate raw ids are not allowed")
    if not raw_ids or len(raw_ids) > _QUARANTINED_ACCEPTED_RAW_REPAIR_LIMIT:
        raise ValueError("raw-id list must contain 1..100 entries")
    if any(re.fullmatch(r"[0-9a-f]{64}", raw_id) is None for raw_id in raw_ids):
        raise ValueError("raw ids must be lowercase SHA-256 identifiers")
    archive_root = _raw_materialization_archive_root(config)
    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    if not source_db.exists() or not index_db.exists():
        raise RuntimeError("source or index tier is missing")

    items: list[BrowserCanonicalAuthorityConflictWitness] = []
    resolved_count = 0
    with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("ATTACH DATABASE ? AS source", (str(source_db),))
        for raw_id in raw_ids:
            # Hold one read transaction across the base eligibility proof and
            # the witness's own re-reads of the same tables (raw_revision_heads,
            # raw_session_memberships, raw_revision_applications) so both see
            # one consistent snapshot. Without this, ``base.reason`` (the
            # eligibility proof) and the witness's competing-head evidence
            # are independent point-in-time reads on an autocommit
            # connection -- a concurrent writer landing between the two
            # calls could make the exported evidence packet describe a
            # state inconsistent with the reason it's explaining. This is
            # cheap read-snapshot isolation, not the apply-time CAS/receipt
            # ceremony: this function never mutates state, so there is
            # nothing to roll back -- COMMIT just releases the read lock.
            conn.execute("BEGIN DEFERRED")
            try:
                base = _inspect_browser_capture_origin_mismatch(
                    archive_root, raw_id, conn=conn, allow_byte_proven_null_native_id_rekey=True
                )
                if base.status != "ineligible":
                    resolved_count += 1
                    continue
                items.append(_browser_canonical_authority_conflict_witness(archive_root, conn, raw_id, base.reason))
            finally:
                conn.execute("COMMIT")
    return BrowserCanonicalAuthorityConflictReport(
        requested_count=len(raw_ids),
        conflict_count=len(items),
        resolved_count=resolved_count,
        items=tuple(items),
    )


def record_browser_canonical_authority_conflict_blockers(
    config: Config, raw_ids: list[str], *, now_ms: int | None = None
) -> tuple[BrowserCanonicalAuthorityConflictReport, tuple[str, ...]]:
    """Persist each unresolved conflict as a durable, non-injected ``BLOCKER`` candidate.

    This is the explicit, separate step the lkrc.3 design requires: it never
    picks an authority itself.  Each conflict becomes one
    ``AssertionKind.BLOCKER`` candidate assertion in ``user.db``, keyed by a
    deterministic id over the raw id and its evidence digest so re-running the
    census after new evidence appears creates a new row instead of silently
    overwriting the old one, while re-running it unchanged is idempotent. Like
    every other automated writer through ``upsert_assertion``, the row is
    written ``author_kind="detector"`` and is therefore always forced to
    ``status=candidate`` with ``inject: false`` -- it can never self-promote to
    an authoritative, context-injectable claim; an operator must explicitly
    judge it (mirrors ``upsert_pathology_findings_as_assertions``, #2383).

    This function never touches authoritative identity state -- it
    writes exactly one ``candidate``/non-injected/private assertion, through
    ``upsert_assertion``'s single write chokepoint, which already refuses to
    resurrect a judged-terminal row (accepted/rejected/deferred/superseded)
    back to candidate on a later automated write (see that function's
    docstring). Concretely: (1) the row can never be read as an authoritative
    claim (``inject: false``, ``promotion_required: true``); (2) an operator's
    judgment on a previously-recorded blocker is never silently clobbered by a
    re-run, even one racing this same function; (3) re-running with unchanged
    evidence is a no-op (deterministic id), and re-running after new evidence
    creates a new row rather than mutating the old one. This is the same
    write-safety posture as every other detector-authored candidate writer in
    this codebase (``upsert_pathology_findings_as_assertions``,
    ``digest``-derived ``TRANSFORM_CANDIDATE`` rows) -- none of which carry an
    apply flag either. Adding one here would be ceremony without a
    corresponding risk to gate.
    """
    from polylogue.storage.sqlite.archive_tiers.user_write import (
        AssertionKind,
        AssertionStatus,
        AssertionVisibility,
        read_assertion_envelope,
        upsert_assertion,
    )

    report = inspect_browser_canonical_authority_conflicts(config, raw_ids)
    archive_root = _raw_materialization_archive_root(config)
    user_db = archive_root / "user.db"
    if not user_db.exists():
        raise RuntimeError("user tier is not initialized")
    timestamp = now_ms if now_ms is not None else int(time.time() * 1000)
    scope_ref = "insight:browser-canonical-authority-conflict@v1"
    assertion_ids: list[str] = []
    conn = sqlite3.connect(user_db)
    conn.row_factory = sqlite3.Row
    try:
        for item in report.items:
            if item.session_id is None or item.evidence_digest is None:
                continue
            assertion_id = hashlib.sha256(
                f"assertion-blocker-browser-canonical-authority-conflict\0{item.raw_id}\0{item.evidence_digest}".encode()
            ).hexdigest()
            full_assertion_id = f"blocker:{assertion_id}"
            existing = read_assertion_envelope(conn, full_assertion_id)
            if existing is not None and existing.status != AssertionStatus.CANDIDATE:
                # Mirror ``upsert_pathology_findings_as_assertions``: once an
                # operator has judged a blocker (accepted/rejected/deferred/
                # superseded), a re-run over unchanged evidence must not
                # overwrite its display fields (value/body_text/evidence_refs
                # are plain ``ON CONFLICT DO UPDATE`` columns in
                # ``upsert_assertion`` -- only ``status`` itself is protected
                # by that function's terminal-judgment chokepoint). Leave the
                # judged row exactly as the operator left it.
                assertion_ids.append(existing.assertion_id)
                continue
            envelope = upsert_assertion(
                conn,
                assertion_id=full_assertion_id,
                scope_ref=scope_ref,
                target_ref=f"session:{item.session_id}",
                key=f"raw/{item.raw_id}",
                kind=AssertionKind.BLOCKER,
                value={
                    "raw_id": item.raw_id,
                    "canonical_logical_source_key": item.canonical_logical_source_key,
                    "unknown_raw_content_hash": item.unknown_raw_content_hash,
                    "unknown_raw_message_count": item.unknown_raw_message_count,
                    "competing_raw_id": item.competing_raw_id,
                    "competing_content_hash": item.competing_content_hash,
                    "competing_frontier_kind": item.competing_frontier_kind,
                    "competing_decision": item.competing_decision,
                    "competing_message_count": item.competing_message_count,
                    "divergent_message_index": item.divergent_message_index,
                    "reason": item.reason,
                },
                body_text=item.divergence_note or item.reason,
                author_ref=scope_ref,
                author_kind="detector",
                status=AssertionStatus.CANDIDATE,
                visibility=AssertionVisibility.PRIVATE,
                context_policy={"inject": False, "promotion_required": True},
                now_ms=timestamp,
            )
            assertion_ids.append(envelope.assertion_id)
        conn.commit()
    finally:
        conn.close()
    return report, tuple(assertion_ids)


# --- polylogue-t0dy: reconcile pre-#2729 duplicate-raw scheme ---


@dataclass(frozen=True, slots=True)
class DuplicateRawIdentityRepairItem:
    stale_raw_id: str
    canonical_raw_id: str
    status: str
    reason: str
    session_id: str | None = None
    logical_source_key: str | None = None
    origin: str | None = None
    source_path: str | None = None
    accepted_source_revision: str | None = None
    accepted_content_hash: str | None = None
    accepted_frontier_kind: str | None = None
    accepted_frontier: int | None = None
    accepted_decided_at_ms: int | None = None
    proof_digest: str | None = None
    repaired: bool = False


def _duplicate_raw_identity_ineligible(
    stale_raw_id: str, canonical_raw_id: str, reason: str
) -> DuplicateRawIdentityRepairItem:
    return DuplicateRawIdentityRepairItem(
        stale_raw_id=stale_raw_id, canonical_raw_id=canonical_raw_id, status="ineligible", reason=reason
    )


def _duplicate_raw_identity_proof_digest(item: DuplicateRawIdentityRepairItem) -> str:
    payload = {
        key: value
        for key, value in dataclasses.asdict(item).items()
        if key not in {"proof_digest", "reason", "repaired", "status"}
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _inspect_duplicate_raw_identity(
    conn: sqlite3.Connection, archive_root: Path, stale_raw_id: str, canonical_raw_id: str
) -> DuplicateRawIdentityRepairItem:
    """Prove one ``(stale, canonical)`` raw pair is the exact pre-/post-#2729 duplicate shape.

    Both raws must carry byte-identical content; each raw id must equal the
    deterministic id its own recorded fields predict under its own scheme
    (``native_id``-inclusive for the stale raw, ``native_id=NULL`` for the
    canonical raw -- see ``deterministic_raw_session_id``, #2729); the stale
    raw must be the *current* accepted head and session pointer; and the
    canonical raw must be a genuinely dangling duplicate -- not itself an
    accepted head or session pointer anywhere. Read-only; never mutates state.
    """
    from polylogue.storage.sqlite.archive_tiers.source_write import deterministic_raw_session_id

    def ineligible(reason: str) -> DuplicateRawIdentityRepairItem:
        return _duplicate_raw_identity_ineligible(stale_raw_id, canonical_raw_id, reason)

    if stale_raw_id == canonical_raw_id:
        return ineligible("stale and canonical raw ids must differ")
    fields = "origin, native_id, source_path, source_index, blob_hash, blob_size"
    stale = conn.execute(f"SELECT {fields} FROM source.raw_sessions WHERE raw_id = ?", (stale_raw_id,)).fetchone()
    canonical = conn.execute(
        f"SELECT {fields} FROM source.raw_sessions WHERE raw_id = ?", (canonical_raw_id,)
    ).fetchone()
    if stale is None or canonical is None:
        return ineligible("one or both raw rows are missing")
    if stale["native_id"] is None:
        return ineligible("stale raw must carry the pre-#2729 native_id-inclusive scheme")
    if canonical["native_id"] is not None:
        return ineligible("canonical raw must carry the post-#2729 native_id=NULL scheme")
    identity_fields = ("origin", "source_path", "source_index", "blob_hash", "blob_size")
    if tuple(stale[name] for name in identity_fields) != tuple(canonical[name] for name in identity_fields):
        return ineligible("stale and canonical raws do not share identical origin/source_path/blob content")
    stale_blob_hash = _bytes_value(stale["blob_hash"])
    blob_size = int(stale["blob_size"])
    expected_stale_id = deterministic_raw_session_id(
        str(stale["origin"]),
        str(stale["source_path"]),
        int(stale["source_index"]),
        stale_blob_hash,
        native_id=str(stale["native_id"]),
    )
    expected_canonical_id = deterministic_raw_session_id(
        str(canonical["origin"]),
        str(canonical["source_path"]),
        int(canonical["source_index"]),
        stale_blob_hash,
        native_id=None,
    )
    if expected_stale_id != stale_raw_id:
        return ineligible("stale raw id does not match the deterministic pre-#2729 id for its own fields")
    if expected_canonical_id != canonical_raw_id:
        return ineligible("canonical raw id does not match the deterministic post-#2729 id for its own fields")
    try:
        # BlobStore is content-addressed by ``blob_hash``, and the equality
        # check above already proved both raws declare the identical hash, so
        # a single read resolves to the one physical blob file either raw
        # would read -- there is no separate "canonical blob" to diverge from
        # it short of a SHA-256 collision. This still proves the retained
        # bytes genuinely match the declared digest/size rather than trusting
        # the ``raw_sessions`` columns blindly.
        store = BlobStore(archive_root / "blob")
        payload = store.read_all(stale_blob_hash.hex())
        if len(payload) != blob_size or hashlib.sha256(payload).digest() != stale_blob_hash:
            return ineligible("retained blob content does not match its declared digest/size")
    except (OSError, ValueError):
        return ineligible("retained blob for one or both raws is missing or unreadable")

    stale_head = conn.execute(
        """
        SELECT logical_source_key, session_id, accepted_source_revision, accepted_content_hash,
               accepted_frontier_kind, accepted_frontier, decided_at_ms
        FROM raw_revision_heads WHERE accepted_raw_id = ?
        """,
        (stale_raw_id,),
    ).fetchone()
    canonical_head = conn.execute(
        "SELECT logical_source_key FROM raw_revision_heads WHERE accepted_raw_id = ?", (canonical_raw_id,)
    ).fetchone()

    if stale_head is None:
        if canonical_head is not None:
            head_key = str(canonical_head["logical_source_key"])
            session = conn.execute(
                "SELECT session_id, raw_id FROM sessions WHERE raw_id = ?", (canonical_raw_id,)
            ).fetchone()
            # An immutable ``raw_revision_applications`` row is append-only and
            # its ``decision_id`` is a content hash, not a sequence -- ordering
            # by it to find the "latest" decision for this (raw_id, key) is
            # meaningless. The stale raw legitimately carries both an old
            # ``selected_baseline`` receipt (from before repair) and the new
            # ``superseded`` receipt (from repair); only presence of the
            # latter proves this exact pair was already repaired.
            stale_superseded = conn.execute(
                """
                SELECT 1 FROM raw_revision_applications
                WHERE raw_id = ? AND logical_source_key = ? AND decision = ? AND accepted_raw_id = ?
                """,
                (stale_raw_id, head_key, ApplicationDecision.SUPERSEDED.value, canonical_raw_id),
            ).fetchone()
            if session is not None and stale_superseded is not None:
                return DuplicateRawIdentityRepairItem(
                    stale_raw_id=stale_raw_id,
                    canonical_raw_id=canonical_raw_id,
                    status="already_repaired",
                    reason="",
                    session_id=str(session["session_id"]),
                    logical_source_key=head_key,
                )
        return ineligible("stale raw is not the currently accepted head of any logical source key")
    if canonical_head is not None:
        return ineligible("canonical raw is already an accepted head; not a dangling duplicate")
    session = conn.execute(
        "SELECT session_id, raw_id FROM sessions WHERE session_id = ?", (stale_head["session_id"],)
    ).fetchone()
    if session is None or str(session["raw_id"]) != stale_raw_id:
        return ineligible("session raw pointer does not match the accepted head")
    if conn.execute("SELECT 1 FROM sessions WHERE raw_id = ?", (canonical_raw_id,)).fetchone() is not None:
        return ineligible("canonical raw is already referenced by a different session pointer")

    item = DuplicateRawIdentityRepairItem(
        stale_raw_id=stale_raw_id,
        canonical_raw_id=canonical_raw_id,
        status="eligible",
        reason="",
        session_id=str(stale_head["session_id"]),
        logical_source_key=str(stale_head["logical_source_key"]),
        origin=str(stale["origin"]),
        source_path=str(stale["source_path"]),
        accepted_source_revision=str(stale_head["accepted_source_revision"]),
        accepted_content_hash=_bytes_value(stale_head["accepted_content_hash"]).hex(),
        accepted_frontier_kind=str(stale_head["accepted_frontier_kind"]),
        accepted_frontier=int(stale_head["accepted_frontier"]),
        accepted_decided_at_ms=int(stale_head["decided_at_ms"]),
    )
    return dataclasses.replace(item, proof_digest=_duplicate_raw_identity_proof_digest(item))


def _apply_duplicate_raw_identity_repair(conn: sqlite3.Connection, item: DuplicateRawIdentityRepairItem) -> None:
    from polylogue.storage.sqlite.archive_tiers.revision_application import (
        RevisionApplicationReceipt,
        record_revision_application_sync,
    )

    assert item.session_id is not None
    assert item.logical_source_key is not None
    assert item.accepted_source_revision is not None
    assert item.accepted_content_hash is not None
    assert item.accepted_frontier_kind is not None
    assert item.accepted_frontier is not None
    decided_at_ms = int(time.time() * 1000)
    accepted_hash = bytes.fromhex(item.accepted_content_hash)
    # Reuse the ordinary revision-application CAS (rather than a hand-written
    # ``UPDATE raw_revision_heads``, per the design note): session_id,
    # accepted_content_hash, accepted_frontier_kind, and accepted_frontier are
    # all unchanged from the current head (the two raws are byte-identical),
    # so this is exactly the "equivalent-content" acceptance path -- only
    # accepted_raw_id repoints, from the stale raw to its canonical twin.
    record_revision_application_sync(
        conn,
        RevisionApplicationReceipt(
            raw_id=item.canonical_raw_id,
            session_id=item.session_id,
            logical_source_key=item.logical_source_key,
            source_revision=item.accepted_source_revision,
            acquisition_generation=0,
            decision=ApplicationDecision.SELECTED_BASELINE,
            accepted_raw_id=item.canonical_raw_id,
            accepted_source_revision=item.accepted_source_revision,
            accepted_content_hash=accepted_hash,
            accepted_frontier_kind=item.accepted_frontier_kind,
            accepted_frontier=item.accepted_frontier,
            baseline_raw_id=item.canonical_raw_id if item.accepted_frontier_kind == "byte" else None,
            detail=f"duplicate_raw_identity_repair:{item.stale_raw_id}->{item.canonical_raw_id}",
        ),
        decided_at_ms=decided_at_ms,
    )
    cursor = conn.execute(
        "UPDATE sessions SET raw_id = ? WHERE session_id = ? AND raw_id = ?",
        (item.canonical_raw_id, item.session_id, item.stale_raw_id),
    )
    if cursor.rowcount != 1:
        raise RuntimeError(f"session raw pointer CAS failed for {item.stale_raw_id}")
    record_revision_application_sync(
        conn,
        RevisionApplicationReceipt(
            raw_id=item.stale_raw_id,
            session_id=item.session_id,
            logical_source_key=item.logical_source_key,
            source_revision=item.accepted_source_revision,
            acquisition_generation=0,
            decision=ApplicationDecision.SUPERSEDED,
            accepted_raw_id=item.canonical_raw_id,
            accepted_source_revision=item.accepted_source_revision,
            accepted_content_hash=accepted_hash,
            accepted_frontier_kind=item.accepted_frontier_kind,
            accepted_frontier=item.accepted_frontier,
            detail=f"duplicate_raw_identity_supersession:{item.stale_raw_id}",
        ),
        decided_at_ms=decided_at_ms,
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
    raw_acquired_at_ms: dict[str, int] = field(default_factory=dict)
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
    missing_blob_raw_ids: tuple[str, ...] = ()
    adoption_deferred_raw_ids: tuple[str, ...] = ()
    authority_quarantined_raw_ids: tuple[str, ...] = ()
    byte_authority_fragment_raw_ids: tuple[str, ...] = ()
    byte_authority_quarantined_raw_ids: tuple[str, ...] = ()
    byte_authority_pending_raw_ids: tuple[str, ...] = ()

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
    raw_acquired_at_ms: dict[str, int] = {}
    missing_blobs = 0
    missing_blob_source_available = 0
    missing_blob_source_missing = 0
    already_parsed = 0
    missing_blob_raw_ids: list[str] = []
    adoption_deferred_raw_ids: list[str] = []
    authority_quarantined_raw_ids: list[str] = []
    byte_authority_fragment_raw_ids: list[str] = []
    byte_authority_quarantined_raw_ids: list[str] = []
    byte_authority_pending_raw_ids: list[str] = []
    expanded_raw_ids: tuple[str, ...] = ()
    expanded_blob_bytes: dict[str, int] = {}
    authority_components: tuple[tuple[str, ...], ...] = ()
    with closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("ATTACH DATABASE ? AS index_tier", (str(index_db),))
        materialized_aliases = {
            (str(row[0]), str(row[1]))
            for row in conn.execute(
                """
                SELECT DISTINCT s.origin, s.native_id
                FROM index_tier.sessions AS s
                JOIN raw_sessions AS existing_raw ON existing_raw.raw_id = s.raw_id
                WHERE s.native_id IS NOT NULL
                """
            )
        }
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
            SELECT r.raw_id, r.origin, r.native_id, r.source_path, r.blob_hash, r.blob_size,
                   r.acquired_at_ms, r.parsed_at_ms,
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
            row_raw_id = str(row["raw_id"])
            if bool(row["adoption_deferred"]):
                adoption_deferred += 1
                adoption_deferred_raw_ids.append(row_raw_id)
                continue
            if bool(row["application_terminal"]):
                continue
            # Membership authority describes how to replay a shared raw; it is
            # not evidence that the rebuildable index still contains the
            # governed sessions.  The candidate query has already proved this
            # raw has no materialized index row, so retain complete censuses
            # as replay inputs after an index reset.
            if bool(row["membership_authority_quarantined"]):
                authority_quarantined += 1
                authority_quarantined_raw_ids.append(row_raw_id)
                continue
            if bool(row["byte_authority_fragment"]):
                byte_authority_fragments += 1
                byte_authority_fragment_raw_ids.append(row_raw_id)
                continue
            if bool(row["byte_authority_quarantined"]):
                byte_authority_quarantined += 1
                byte_authority_quarantined_raw_ids.append(row_raw_id)
                continue
            if bool(row["byte_authority_pending"]):
                byte_authority_pending += 1
                byte_authority_pending_raw_ids.append(row_raw_id)
                continue
            if row["parse_error"] and not _raw_materialization_retryable_missing_blob_error(row["parse_error"]):
                continue
            if _raw_materialized_by_source_path_native(materialized_aliases, row):
                continue
            if _raw_materialization_parsed_non_session_artifact(archive_root, row):
                continue
            blob_hash = row["blob_hash"].hex() if isinstance(row["blob_hash"], bytes) else str(row["blob_hash"])
            if blob_store.exists(blob_hash):
                raw_id = str(row["raw_id"])
                raw_ids.append(raw_id)
                raw_origins[raw_id] = str(row["origin"] or "")
                raw_source_paths[raw_id] = str(row["source_path"] or "")
                raw_acquired_at_ms[raw_id] = int(row["acquired_at_ms"] or 0)
                blob_size = row["blob_size"]
                if isinstance(blob_size, int):
                    raw_blob_bytes[raw_id] = blob_size
                if row["parsed_at_ms"] is not None:
                    already_parsed += 1
            else:
                missing_blobs += 1
                missing_blob_raw_ids.append(row_raw_id)
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
        raw_acquired_at_ms=raw_acquired_at_ms,
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
        missing_blob_raw_ids=tuple(sorted(missing_blob_raw_ids)),
        adoption_deferred_raw_ids=tuple(sorted(adoption_deferred_raw_ids)),
        authority_quarantined_raw_ids=tuple(sorted(authority_quarantined_raw_ids)),
        byte_authority_fragment_raw_ids=tuple(sorted(byte_authority_fragment_raw_ids)),
        byte_authority_quarantined_raw_ids=tuple(sorted(byte_authority_quarantined_raw_ids)),
        byte_authority_pending_raw_ids=tuple(sorted(byte_authority_pending_raw_ids)),
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


def _raw_materialization_ordered_components(
    candidates: RawMaterializationCandidates,
    *,
    archive_root: Path,
) -> list[tuple[str, ...]]:
    """Order complete components fairly without splitting authority cohorts."""
    candidate_ids = set(candidates.raw_ids)
    source_components = candidates.authority_components or tuple((raw_id,) for raw_id in candidates.raw_ids)
    components = [component for component in source_components if candidate_ids.intersection(component)]
    plans = build_raw_replay_plans(archive_root, components)
    plan_ids = {plan.input_raw_ids: plan.plan_id for plan in plans}
    last_attempts = raw_replay_plan_last_attempts(archive_root)

    def candidate_order(raw_id: str) -> tuple[int, int, str]:
        if raw_id in candidates.raw_acquired_at_ms:
            return (0, candidates.raw_acquired_at_ms[raw_id], raw_id)
        return (1, candidates.raw_blob_bytes.get(raw_id, 0), raw_id)

    return sorted(
        components,
        key=lambda component: (
            plan_ids[component] in last_attempts,
            last_attempts.get(plan_ids[component], 0),
            min(candidate_order(raw_id) for raw_id in component if raw_id in candidate_ids),
            component,
        ),
    )


def _raw_materialization_component_seed(
    candidates: RawMaterializationCandidates,
    component: tuple[str, ...],
) -> str:
    candidate_ids = set(candidates.raw_ids)
    return min(
        (raw_id for raw_id in component if raw_id in candidate_ids),
        key=lambda raw_id: (
            (0, candidates.raw_acquired_at_ms[raw_id], raw_id)
            if raw_id in candidates.raw_acquired_at_ms
            else (1, candidates.raw_blob_bytes.get(raw_id, 0), raw_id)
        ),
    )


def _raw_materialization_component_blob_bytes(candidates: RawMaterializationCandidates, raw_id: str) -> int:
    return candidates.expanded_blob_bytes.get(raw_id, candidates.raw_blob_bytes.get(raw_id, 0))


def _raw_authority_scope(
    *,
    raw_artifact_id: str | None,
    provider: str | None,
    source_family: str | None,
    source_root: Path | None,
    raw_artifact_limit: int | None,
    max_payload_bytes: int,
) -> dict[str, object]:
    return {
        "raw_artifact_id": raw_artifact_id,
        "provider": provider,
        "source_family": source_family,
        "source_root": str(source_root) if source_root is not None else None,
        "raw_artifact_limit": raw_artifact_limit,
        "max_payload_bytes": max_payload_bytes,
    }


def _raw_authority_residual(
    candidates: RawMaterializationCandidates,
    *,
    census_pending_raw_ids: tuple[str, ...] = (),
    resource_blocked_plan_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    """Return identity-sensitive residual debt for fixed-point comparison."""
    census_pending_digest = hashlib.sha256(
        json.dumps(list(census_pending_raw_ids), separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "missing_blob_raw_ids": list(candidates.missing_blob_raw_ids),
        "adoption_deferred_raw_ids": list(candidates.adoption_deferred_raw_ids),
        "authority_quarantined_raw_ids": list(candidates.authority_quarantined_raw_ids),
        "byte_authority_fragment_raw_ids": list(candidates.byte_authority_fragment_raw_ids),
        "byte_authority_quarantined_raw_ids": list(candidates.byte_authority_quarantined_raw_ids),
        "byte_authority_pending_raw_ids": list(candidates.byte_authority_pending_raw_ids),
        # A large initial catch-up may need hundreds of bounded census passes.
        # Keep every progress receipt identity-sensitive without copying the
        # entire shrinking raw-ID backlog into source.db on every pass.
        "census_pending_raw_count": len(census_pending_raw_ids),
        "census_pending_raw_digest": census_pending_digest,
        "resource_blocked_plan_ids": list(resource_blocked_plan_ids),
    }


def _raw_authority_postflight_snapshot(
    archive_root: Path,
    candidates: RawMaterializationCandidates,
    *,
    max_payload_bytes: int,
) -> tuple[tuple[RawReplayPlan, ...], dict[str, object]]:
    """Build the complete post-pass plan inventory and typed residual debt."""
    components = _raw_materialization_ordered_components(candidates, archive_root=archive_root)
    plans = build_raw_replay_plans(archive_root, components)
    blocked_plan_ids = tuple(
        sorted(
            plan.plan_id
            for component, plan in zip(components, plans, strict=True)
            if sum(_raw_materialization_component_blob_bytes(candidates, member) for member in component)
            > max_payload_bytes
        )
    )
    return plans, _raw_authority_residual(candidates, resource_blocked_plan_ids=blocked_plan_ids)


def _raw_authority_candidates_for_scope(
    config: Config,
    scope: Mapping[str, object],
) -> RawMaterializationCandidates:
    source_root_value = scope.get("source_root")
    return _raw_materialization_candidate_ids(
        config,
        raw_artifact_id=str(scope["raw_artifact_id"]) if scope.get("raw_artifact_id") is not None else None,
        provider=str(scope["provider"]) if scope.get("provider") is not None else None,
        source_family=str(scope["source_family"]) if scope.get("source_family") is not None else None,
        source_root=Path(str(source_root_value)) if source_root_value is not None else None,
    )


def _raw_materialization_base_metrics(
    candidates: RawMaterializationCandidates,
    *,
    recovered_census_count: int,
) -> dict[str, float]:
    metrics = {
        "raw_materialization_candidate_count": float(len(candidates.raw_ids)),
        "raw_materialization_missing_blob_count": float(candidates.missing_blobs),
        "raw_materialization_missing_blob_source_available_count": float(candidates.missing_blob_source_available),
        "raw_materialization_missing_blob_source_missing_count": float(candidates.missing_blob_source_missing),
        "raw_materialization_already_parsed_count": float(candidates.already_parsed),
        "raw_materialization_total_blob_bytes": float(candidates.total_blob_bytes),
        "raw_materialization_max_blob_bytes": float(candidates.max_blob_bytes),
        "raw_materialization_adoption_deferred_count": float(candidates.adoption_deferred),
        "raw_materialization_authority_quarantined_count": float(candidates.authority_quarantined),
        "raw_materialization_byte_authority_fragment_count": float(candidates.byte_authority_fragments),
        "raw_materialization_byte_authority_quarantined_count": float(candidates.byte_authority_quarantined),
        "raw_materialization_byte_authority_pending_count": float(candidates.byte_authority_pending),
    }
    if recovered_census_count:
        metrics["raw_materialization_recovered_census_count"] = float(recovered_census_count)
    return metrics


def _raw_replay_plan_outcome(
    conn: sqlite3.Connection,
    plan: RawReplayPlan,
    *,
    remaining: RawMaterializationCandidates,
) -> RawReplayPlanOutcome:
    """Conserve one selected component into an explicit post-pass state."""
    plan_id = plan.plan_id
    component = plan.input_raw_ids
    remaining_ids = set(remaining.expanded_raw_ids) | set(remaining.raw_ids)
    if remaining_ids.intersection(component):
        return RawReplayPlanOutcome(
            plan_id,
            component,
            RawReplayPlanStatus.RETRYABLE,
            "authority component remains executable after this pass",
            "retry the same plan after the current writer pass",
        )

    placeholders = ",".join("?" for _ in component)
    deferred = conn.execute(
        f"""
            SELECT 1
            FROM index_tier.raw_revision_applications
            WHERE raw_id IN ({placeholders}) AND decision = 'deferred'
            UNION ALL
            SELECT 1
            FROM raw_session_memberships
            WHERE raw_id IN ({placeholders}) AND decision = 'deferred'
            LIMIT 1
            """,
        (*component, *component),
    ).fetchone()
    if deferred is not None:
        return RawReplayPlanOutcome(
            plan_id,
            component,
            RawReplayPlanStatus.DEFERRED,
            "authority comparison produced a durable deferred decision",
            "resolve the recorded authority conflict before retry",
        )
    terminal = conn.execute(
        f"""
            SELECT 1
            FROM index_tier.raw_revision_applications
            WHERE raw_id IN ({placeholders}) AND decision = 'ambiguous'
            UNION ALL
            SELECT 1
            FROM raw_session_memberships
            WHERE raw_id IN ({placeholders}) AND decision = 'ambiguous'
            UNION ALL
            SELECT 1
            FROM raw_sessions
            WHERE raw_id IN ({placeholders})
              AND parse_error IS NOT NULL
              AND parse_error != ?
            UNION ALL
            SELECT 1
            FROM raw_membership_census
            WHERE raw_id IN ({placeholders}) AND status = 'failed'
            LIMIT 1
            """,
        (*component, *component, *component, _TRANSIENT_LOCK_PARSE_ERROR, *component),
    ).fetchone()
    if terminal is not None:
        return RawReplayPlanOutcome(
            plan_id,
            component,
            RawReplayPlanStatus.TERMINAL,
            "component ended in explicit ambiguous or parse-terminal authority state",
            "inspect durable authority debt; do not replay without new evidence",
        )
    executed = conn.execute(
        f"""
            SELECT COUNT(*) = ?
            FROM raw_sessions
            WHERE raw_id IN ({placeholders}) AND parsed_at_ms IS NOT NULL
            """,
        (len(component), *component),
    ).fetchone()
    if executed is not None and bool(executed[0]):
        return RawReplayPlanOutcome(
            plan_id,
            component,
            RawReplayPlanStatus.EXECUTED,
            "every retained raw in the selected component reached a terminal parse/application receipt",
            "none",
        )
    return RawReplayPlanOutcome(
        plan_id,
        component,
        RawReplayPlanStatus.REJECTED_STALE,
        "selected component disappeared without an executable or durable terminal outcome",
        "stop automatic convergence and investigate outcome conservation",
    )


def _raw_replay_plan_outcomes(
    archive_root: Path,
    plans: Sequence[RawReplayPlan],
    *,
    remaining: RawMaterializationCandidates,
) -> tuple[RawReplayPlanOutcome, ...]:
    if not plans:
        return ()
    with closing(sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("ATTACH DATABASE ? AS index_tier", (str(archive_root / "index.db"),))
        return tuple(_raw_replay_plan_outcome(conn, plan, remaining=remaining) for plan in plans)


def _raw_replay_conservation_metrics(
    plans: Sequence[RawReplayPlan],
    selected_plan_ids: set[str],
    outcomes: Sequence[RawReplayPlanOutcome],
) -> tuple[int, int, int]:
    """Return total plans, all carried inventory, and exact algebra errors."""
    outcome_ids = [outcome.plan_id for outcome in outcomes]
    outcome_id_set = set(outcome_ids)
    conservation_errors = (
        len(selected_plan_ids - outcome_id_set)
        + len(outcome_id_set - selected_plan_ids)
        + (len(outcome_ids) - len(outcome_id_set))
    )
    carried_forward = (
        len(plans)
        - len(selected_plan_ids)
        + sum(outcome.status is RawReplayPlanStatus.CARRIED_FORWARD for outcome in outcomes)
    )
    return len(plans), carried_forward, conservation_errors


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


def _unavailable_raw_materialization_backlog(reason: str) -> dict[str, object]:
    return {
        "available": False,
        "reason": reason,
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


def raw_materialization_replay_backlog(
    config: Config,
    *,
    limit: int = 10,
    _candidates: RawMaterializationCandidates | None = None,
) -> dict[str, object]:
    """Return a read-only weighted backlog for raw source-to-index replay.

    The report uses the same candidate selector as ``repair_raw_materialization``
    so diagnostics and actual replay agree about which raw rows are actionable.
    It does not parse raw blobs or mutate the archive.
    """

    archive_root = _raw_materialization_archive_root(config)
    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    if not source_db.exists() or not index_db.exists():
        return _unavailable_raw_materialization_backlog("source_or_index_tier_missing")
    with closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)) as source_conn:
        source_ready = source_conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'raw_sessions'"
        ).fetchone()
    if source_ready is None:
        return _unavailable_raw_materialization_backlog("source_tier_uninitialized")
    candidates = _candidates or _raw_materialization_candidate_ids(config)
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


def _histogram_upper_bound(value: int) -> int:
    """Return the inclusive power-of-two bucket for a non-negative value."""
    upper_bound = 1
    while upper_bound < max(value, 1):
        upper_bound *= 2
    return upper_bound


def _histogram(values: Sequence[int], *, field: str) -> list[dict[str, int]]:
    """Summarize numeric frontier shape without retaining identifying rows."""
    buckets: dict[int, int] = {}
    for value in values:
        upper_bound = _histogram_upper_bound(value)
        buckets[upper_bound] = buckets.get(upper_bound, 0) + 1
    return [{field: upper_bound, "count": count} for upper_bound, count in sorted(buckets.items())]


def _backlog_count(backlog: Mapping[str, object], field: str) -> int:
    """Read a bounded numeric backlog field without trusting an untyped dict."""
    value = backlog.get(field)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise RuntimeError(f"raw materialization backlog returned a non-integral {field!r}: {value!r}")


def raw_materialization_scale_profile(config: Config) -> dict[str, object]:
    """Return a private-free authority-frontier shape for synthetic proof input.

    The profile deliberately exposes counts and distributions only.  It is safe
    to retain with a synthetic workload receipt because it never includes raw
    ids, source paths, blob hashes, or payload-derived fields.
    """
    candidates = _raw_materialization_candidate_ids(config)
    backlog = raw_materialization_replay_backlog(config, limit=0, _candidates=candidates)
    if not bool(backlog["available"]):
        return {"available": False, "reason": backlog["reason"]}
    component_raw_counts = [len(component) for component in candidates.authority_components]
    candidate_ids = set(candidates.raw_ids)
    component_cohorts: dict[tuple[int, int], int] = {}
    component_byte_cohorts: dict[tuple[int, int, int], int] = {}
    for component in candidates.authority_components:
        raw_count = len(component)
        direct_candidate_count = len(candidate_ids.intersection(component))
        key = (raw_count, direct_candidate_count)
        component_cohorts[key] = component_cohorts.get(key, 0) + 1
    component_blob_bytes = [
        sum(_raw_materialization_component_blob_bytes(candidates, raw_id) for raw_id in component)
        for component in candidates.authority_components
    ]
    for component, blob_bytes in zip(candidates.authority_components, component_blob_bytes, strict=True):
        byte_key = (
            len(component),
            len(candidate_ids.intersection(component)),
            _histogram_upper_bound(blob_bytes),
        )
        component_byte_cohorts[byte_key] = component_byte_cohorts.get(byte_key, 0) + 1
    return {
        "available": True,
        "format": "raw-authority-scale-profile-v1",
        "candidate_count": _backlog_count(backlog, "candidate_count"),
        "expanded_candidate_count": _backlog_count(backlog, "expanded_candidate_count"),
        "authority_component_count": _backlog_count(backlog, "authority_component_count"),
        "executable_authority_component_count": _backlog_count(backlog, "executable_authority_component_count"),
        "blocked_authority_component_count": _backlog_count(backlog, "blocked_authority_component_count"),
        "total_blob_bytes": _backlog_count(backlog, "total_blob_bytes"),
        "expanded_total_blob_bytes": _backlog_count(backlog, "expanded_total_blob_bytes"),
        "component_raw_count_histogram": _histogram(component_raw_counts, field="upper_bound_raw_count"),
        "component_cohort_distribution": [
            {
                "component_raw_count": raw_count,
                "direct_candidate_count": direct_candidate_count,
                "component_count": count,
            }
            for (raw_count, direct_candidate_count), count in sorted(component_cohorts.items())
        ],
        "component_byte_cohort_distribution": [
            {
                "component_raw_count": raw_count,
                "direct_candidate_count": direct_candidate_count,
                "upper_bound_blob_bytes": upper_bound_blob_bytes,
                "component_count": count,
            }
            for (raw_count, direct_candidate_count, upper_bound_blob_bytes), count in sorted(
                component_byte_cohorts.items()
            )
        ],
        "component_blob_bytes_histogram": _histogram(component_blob_bytes, field="upper_bound_blob_bytes"),
        "residual_state_counts": {
            "missing_blob_count": _backlog_count(backlog, "missing_blob_count"),
            "authority_quarantined_count": _backlog_count(backlog, "authority_quarantined_count"),
            "byte_authority_fragment_count": _backlog_count(backlog, "byte_authority_fragment_count"),
            "byte_authority_quarantined_count": _backlog_count(backlog, "byte_authority_quarantined_count"),
            "byte_authority_pending_count": _backlog_count(backlog, "byte_authority_pending_count"),
            "adoption_deferred_count": _backlog_count(backlog, "adoption_deferred_count"),
            "blocked_candidate_count": _backlog_count(backlog, "blocked_candidate_count"),
        },
    }


def _raw_materialized_by_source_path_native(materialized_aliases: set[tuple[str, str]], row: sqlite3.Row) -> bool:
    origin = str(row["origin"] or "")
    if not origin:
        return False
    for native_id in _source_path_native_id_candidates(str(row["source_path"] or "")):
        if (origin, native_id) in materialized_aliases:
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
        with closing(sqlite3.connect(ops_db)) as conn:
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
        with closing(sqlite3.connect(ops_db)) as conn:
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
    profile_stale_predicate = session_profile_stale_predicate("s", "p")
    latency_stale_predicate = session_profile_stale_predicate("s", "lp")
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
               OR {profile_stale_predicate}
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
               OR {latency_stale_predicate}
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
        with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
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
    plan_outcomes: tuple[RawReplayPlanOutcome, ...] = ()
    census_receipt: RawAuthorityCensusReceipt | None = None

    def to_dict(self) -> JSONDocument:
        payload: dict[str, object] = {
            "name": self.name,
            "category": self.category.value,
            "destructive": self.destructive,
            "repaired_count": self.repaired_count,
            "success": self.success,
            "detail": self.detail,
            "metrics": dict(self.metrics),
        }
        if self.plan_outcomes:
            payload["plan_outcome_count"] = len(self.plan_outcomes)
            payload["plan_outcomes"] = [
                outcome.to_summary_dict() for outcome in self.plan_outcomes[:RAW_MATERIALIZATION_OUTCOME_SAMPLE_LIMIT]
            ]
            payload["plan_outcomes_truncated"] = len(self.plan_outcomes) > RAW_MATERIALIZATION_OUTCOME_SAMPLE_LIMIT
        if self.census_receipt is not None:
            payload["census"] = {
                "census_id": self.census_receipt.census_id,
                "sequence_no": self.census_receipt.sequence_no,
                "inventory_digest": self.census_receipt.inventory_digest,
                "residual_digest": self.census_receipt.residual_digest,
                "plan_count": self.census_receipt.plan_count,
                "post_inventory_digest": self.census_receipt.post_inventory_digest,
                "post_residual_digest": self.census_receipt.post_residual_digest,
                "post_plan_count": self.census_receipt.post_plan_count,
                "executable_plan_count": self.census_receipt.executable_plan_count,
                "residual_plan_count": self.census_receipt.residual_plan_count,
                "predecessor_census_id": self.census_receipt.predecessor_census_id,
                "mode": self.census_receipt.mode,
                "lifecycle_status": self.census_receipt.lifecycle_status,
                "quiescent": self.census_receipt.quiescent,
                "fixed_point": self.census_receipt.fixed_point,
                "query_handle": self.census_receipt.query_handle,
            }
        return json_document(payload)


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
    plan_outcomes: tuple[RawReplayPlanOutcome, ...] = (),
    census_receipt: RawAuthorityCensusReceipt | None = None,
) -> RepairResult:
    return RepairResult(
        name=name,
        category=MaintenanceCategory.DERIVED_REPAIR,
        destructive=False,
        repaired_count=repaired_count,
        success=success,
        detail=detail,
        metrics=dict(metrics or {}),
        plan_outcomes=plan_outcomes,
        census_receipt=census_receipt,
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
    max_payload_bytes: int = RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES,
    progress_callback: ProgressCallback | None = None,
) -> RepairResult:
    """Converge retained raws through typed per-session revision authority."""
    if max_payload_bytes < 1:
        raise ValueError("max_payload_bytes must be positive")
    archive_root = _raw_materialization_archive_root(config)
    recovered_censuses = recover_interrupted_raw_authority_censuses(archive_root)
    for recovered_census_id, recovered_scope in recovered_censuses:
        recovered_envelope = recovered_scope.get("max_payload_bytes")
        recovered_max_payload_bytes = (
            recovered_envelope if isinstance(recovered_envelope, int) and recovered_envelope > 0 else max_payload_bytes
        )
        recovered_candidates = _raw_authority_candidates_for_scope(config, recovered_scope)
        recovered_post_plans, recovered_post_residual = _raw_authority_postflight_snapshot(
            archive_root,
            recovered_candidates,
            max_payload_bytes=recovered_max_payload_bytes,
        )
        finalize_raw_authority_census(
            archive_root,
            recovered_census_id,
            post_plans=recovered_post_plans,
            post_residual=recovered_post_residual,
            interrupted=True,
        )
    recovered_census_count = len(recovered_censuses)
    from polylogue.storage.raw_authority import unresolved_raw_replay_blockers

    blocker_count = unresolved_raw_replay_blockers(archive_root)
    if blocker_count:
        return _internal_derived_repair_result(
            "raw_materialization",
            repaired_count=0,
            success=False,
            detail=(
                f"Raw materialization is fail-closed behind {blocker_count:,} unresolved durable stale-plan blocker(s)"
            ),
            metrics={"raw_materialization_unresolved_blocker_count": float(blocker_count)},
        )
    candidates = _raw_materialization_candidate_ids(
        config,
        raw_artifact_id=raw_artifact_id,
        provider=provider,
        source_family=source_family,
        source_root=source_root,
    )
    from polylogue.sources.revision_backfill import (
        RawRevisionReplayResourceBlockedError,
        census_historical_revision_evidence,
        uncensused_historical_revision_raw_ids,
    )

    relevant_raw_ids = list(candidates.expanded_raw_ids or tuple(candidates.raw_ids))
    uncensused_raw_ids = set(
        uncensused_historical_revision_raw_ids(archive_root, relevant_raw_ids, max_payload_bytes=max_payload_bytes)
    )
    census_failed_raw_ids: set[str] = set()
    census_resource_blocked_raw_ids: set[str] = set()
    census_component_limit = (
        raw_artifact_limit if raw_artifact_limit is not None else RAW_MATERIALIZATION_CENSUS_COMPONENT_LIMIT
    )
    if census_component_limit < 1:
        raise ValueError("raw_artifact_limit must be positive")
    census_components_attempted = 0
    if uncensused_raw_ids:
        preliminary_components = _raw_materialization_ordered_components(candidates, archive_root=archive_root)
        for component in preliminary_components:
            if not uncensused_raw_ids.intersection(component):
                continue
            if census_components_attempted >= census_component_limit:
                break
            census_components_attempted += 1
            seed = _raw_materialization_component_seed(candidates, component)
            try:
                census_historical_revision_evidence(
                    archive_root,
                    selected_raw_ids=[seed],
                    max_payload_bytes=max_payload_bytes,
                )
            except RawRevisionReplayResourceBlockedError as exc:
                logger.warning(
                    "raw authority census resource-blocked for component containing %s: %s",
                    seed,
                    exc,
                )
                census_resource_blocked_raw_ids.update(component)
                from polylogue.sources.revision_backfill import record_resource_blocked_revision_census

                record_resource_blocked_revision_census(
                    archive_root,
                    tuple(sorted(uncensused_raw_ids.intersection(component))),
                    max_payload_bytes=max_payload_bytes,
                    total_payload_bytes=exc.total_bytes,
                )
            except Exception:
                logger.exception("raw authority census failed for component containing %s", seed)
                census_failed_raw_ids.update(component)
        candidates = _raw_materialization_candidate_ids(
            config,
            raw_artifact_id=raw_artifact_id,
            provider=provider,
            source_family=source_family,
            source_root=source_root,
        )
        relevant_raw_ids = list(candidates.expanded_raw_ids or tuple(candidates.raw_ids))
        uncensused_raw_ids = set(
            uncensused_historical_revision_raw_ids(archive_root, relevant_raw_ids, max_payload_bytes=max_payload_bytes)
        )
    census_pending_raw_ids = tuple(sorted(uncensused_raw_ids | census_failed_raw_ids))
    if census_pending_raw_ids:
        residual = _raw_authority_residual(candidates, census_pending_raw_ids=census_pending_raw_ids)
        census_receipt = record_raw_authority_census(
            archive_root,
            (),
            selected_plan_ids=set(),
            executable_plan_ids=set(),
            mode="census",
            quiescent=False,
            scope=_raw_authority_scope(
                raw_artifact_id=raw_artifact_id,
                provider=provider,
                source_family=source_family,
                source_root=source_root,
                raw_artifact_limit=raw_artifact_limit,
                max_payload_bytes=max_payload_bytes,
            ),
            residual=residual,
        )
        metrics = _raw_materialization_base_metrics(
            candidates,
            recovered_census_count=recovered_census_count,
        )
        metrics.update(
            {
                "raw_materialization_selected_count": 0.0,
                "raw_materialization_selected_total_blob_bytes": 0.0,
                "raw_materialization_selected_max_blob_bytes": 0.0,
                "raw_materialization_executed_count": 0.0,
                "raw_materialization_census_incomplete_raw_count": float(len(census_pending_raw_ids)),
                "raw_materialization_census_component_limit": float(census_component_limit),
                "raw_materialization_census_components_attempted": float(census_components_attempted),
                "raw_materialization_census_sequence": float(census_receipt.sequence_no),
                "raw_materialization_census_fixed_point": 0.0,
            }
        )
        if census_resource_blocked_raw_ids:
            resource_blocked_candidate_raw_ids = set(candidates.raw_ids).intersection(census_resource_blocked_raw_ids)
            metrics["raw_materialization_executable_candidate_count"] = float(
                len(set(candidates.raw_ids) - resource_blocked_candidate_raw_ids)
            )
            metrics["raw_materialization_resource_blocked_candidate_count"] = float(
                len(resource_blocked_candidate_raw_ids)
            )
            metrics["raw_materialization_resource_blocked_count"] = float(len(census_resource_blocked_raw_ids))
            metrics["raw_materialization_execute_blob_limit_bytes"] = float(max_payload_bytes)
            oversized = {
                raw_id
                for raw_id in census_resource_blocked_raw_ids
                if _raw_materialization_component_blob_bytes(candidates, raw_id) > max_payload_bytes
            }
            if oversized:
                metrics["raw_materialization_oversized_count"] = float(len(oversized))
                stream_oversized = {
                    raw_id for raw_id in oversized if _raw_materialization_stream_safe(candidates, raw_id)
                }
                if stream_oversized:
                    metrics["raw_materialization_stream_oversized_count"] = float(len(stream_oversized))
        detail = (
            f"Raw replay planning paused until the persisted parser census completes for "
            f"{len(census_pending_raw_ids):,} relevant raw(s)"
        )
        if census_resource_blocked_raw_ids:
            detail += (
                f"; {len(census_resource_blocked_raw_ids):,} raw(s) belong to authority components whose "
                f"aggregate payload exceeds {_format_bytes(max_payload_bytes)}"
            )
        return _internal_derived_repair_result(
            "raw_materialization",
            repaired_count=0,
            success=False,
            detail=detail,
            metrics=metrics,
            census_receipt=census_receipt,
        )
    candidate_raw_ids = candidates.raw_ids
    ordered_components = _raw_materialization_ordered_components(candidates, archive_root=archive_root)
    plans = build_raw_replay_plans(archive_root, ordered_components)
    plan_by_component = {plan.input_raw_ids: plan for plan in plans}
    all_blocked_components = [
        component
        for component in ordered_components
        if sum(_raw_materialization_component_blob_bytes(candidates, member) for member in component)
        > max_payload_bytes
    ]
    all_blocked_component_raw_ids = {raw_id for component in all_blocked_components for raw_id in component}
    resource_blocked_candidate_raw_ids = set(candidate_raw_ids).intersection(all_blocked_component_raw_ids)
    deferred_plan_ids = raw_replay_plan_deferred_for_envelope(archive_root, max_payload_bytes=max_payload_bytes)
    admissible_components = [
        component for component in ordered_components if plan_by_component[component].plan_id not in deferred_plan_ids
    ]
    selected_components = (
        admissible_components[:raw_artifact_limit] if raw_artifact_limit is not None else admissible_components
    )
    blocked_components = [
        component for component in selected_components if all_blocked_component_raw_ids.intersection(component)
    ]
    blocked_component_raw_ids = {raw_id for component in blocked_components for raw_id in component}
    blocked_plan_outcomes = tuple(
        RawReplayPlanOutcome(
            plan_by_component[component].plan_id,
            component,
            RawReplayPlanStatus.DEFERRED,
            f"resource-envelope:{max_payload_bytes}",
            "retry only after a larger resource envelope or changed source/index preconditions",
        )
        for component in blocked_components
    )
    executable_components = [
        component for component in selected_components if not blocked_component_raw_ids.intersection(component)
    ]
    executable_plans = [plan_by_component[component] for component in executable_components]
    raw_ids = [_raw_materialization_component_seed(candidates, component) for component in executable_components]
    selected_component_raw_ids = {raw_id for component in selected_components for raw_id in component}
    selected_candidate_raw_ids = selected_component_raw_ids.intersection(candidate_raw_ids)
    missing_blobs = candidates.missing_blobs
    selected_total_bytes = sum(
        _raw_materialization_component_blob_bytes(candidates, raw_id) for raw_id in selected_component_raw_ids
    )
    selected_max_bytes = max(
        (_raw_materialization_component_blob_bytes(candidates, raw_id) for raw_id in selected_component_raw_ids),
        default=0,
    )
    metrics = _raw_materialization_base_metrics(candidates, recovered_census_count=recovered_census_count)
    metrics.update(
        {
            "raw_materialization_selected_count": float(len(selected_candidate_raw_ids)),
            "raw_materialization_selected_total_blob_bytes": float(selected_total_bytes),
            "raw_materialization_selected_max_blob_bytes": float(selected_max_bytes),
        }
    )
    if raw_artifact_limit is not None:
        metrics["raw_materialization_limit"] = float(raw_artifact_limit)
        metrics["raw_materialization_selected_component_count"] = float(len(selected_components))
    metrics["raw_materialization_before_component_count"] = float(len(ordered_components))
    metrics["raw_materialization_selected_executable_component_count"] = float(len(executable_components))
    metrics["raw_materialization_selected_blocked_component_count"] = float(len(blocked_components))
    if all_blocked_component_raw_ids:
        metrics["raw_materialization_executable_candidate_count"] = float(
            len(set(candidate_raw_ids) - resource_blocked_candidate_raw_ids)
        )
        metrics["raw_materialization_resource_blocked_candidate_count"] = float(len(resource_blocked_candidate_raw_ids))
        metrics["raw_materialization_resource_blocked_count"] = float(len(all_blocked_component_raw_ids))
        metrics["raw_materialization_execute_blob_limit_bytes"] = float(max_payload_bytes)
    diagnostic_component_raw_ids = selected_component_raw_ids | all_blocked_component_raw_ids
    oversized_candidate_raw_ids = [
        raw_id
        for raw_id in diagnostic_component_raw_ids
        if _raw_materialization_component_blob_bytes(candidates, raw_id) > max_payload_bytes
    ]
    oversized_stream_safe_raw_ids = [
        raw_id for raw_id in oversized_candidate_raw_ids if _raw_materialization_stream_safe(candidates, raw_id)
    ]
    oversized_raw_ids = oversized_candidate_raw_ids
    if oversized_raw_ids:
        metrics["raw_materialization_oversized_count"] = float(len(oversized_raw_ids))
        metrics["raw_materialization_resource_blocked_count"] = max(
            metrics.get("raw_materialization_resource_blocked_count", 0.0),
            float(len(oversized_raw_ids)),
        )
        metrics["raw_materialization_execute_blob_limit_bytes"] = float(max_payload_bytes)
    if oversized_stream_safe_raw_ids:
        metrics["raw_materialization_stream_oversized_count"] = float(len(oversized_stream_safe_raw_ids))
    if deferred_plan_ids:
        metrics["raw_materialization_deferred_plan_count"] = float(len(deferred_plan_ids))
    scope = _raw_authority_scope(
        raw_artifact_id=raw_artifact_id,
        provider=provider,
        source_family=source_family,
        source_root=source_root,
        raw_artifact_limit=raw_artifact_limit,
        max_payload_bytes=max_payload_bytes,
    )
    if not dry_run and not selected_components and deferred_plan_ids:
        retained_census_receipt = latest_raw_authority_census_receipt(archive_root, scope=scope)
        if retained_census_receipt is None:
            # v1 receipts predate envelope identity.  They are safe to reuse
            # only in this all-deferred branch: the persisted deferred-plan
            # reason was already matched against this active envelope above.
            legacy_scope = dict(scope)
            legacy_scope.pop("max_payload_bytes")
            retained_census_receipt = latest_raw_authority_census_receipt(archive_root, scope=legacy_scope)
        if retained_census_receipt is None:
            raise RuntimeError("resource-deferred raw replay lacks a completed durable census receipt")
        return _internal_derived_repair_result(
            "raw_materialization",
            repaired_count=0,
            success=True,
            detail=(
                "Raw materialization has no newly admissible authority components; "
                f"{len(deferred_plan_ids):,} unchanged plan(s) remain deferred for "
                f"the {_format_bytes(max_payload_bytes)} envelope"
            ),
            metrics=metrics,
            census_receipt=retained_census_receipt,
        )
    selected_plan_ids = {plan_by_component[component].plan_id for component in selected_components}
    executable_plan_ids = {
        plan_by_component[component].plan_id
        for component in ordered_components
        if not all_blocked_component_raw_ids.intersection(component)
    }
    residual = _raw_authority_residual(
        candidates,
        resource_blocked_plan_ids=tuple(
            sorted(plan_by_component[component].plan_id for component in all_blocked_components)
        ),
    )
    census_receipt = record_raw_authority_census(
        archive_root,
        plans,
        selected_plan_ids=set() if dry_run else selected_plan_ids,
        executable_plan_ids=executable_plan_ids,
        mode="dry_run" if dry_run else "apply",
        quiescent=True,
        scope=scope,
        residual=residual,
    )
    metrics["raw_materialization_census_sequence"] = float(census_receipt.sequence_no)
    metrics["raw_materialization_census_fixed_point"] = float(census_receipt.fixed_point)
    if not candidate_raw_ids:
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
            census_receipt=census_receipt,
        )
    if dry_run:
        plan_outcomes = (
            tuple(
                RawReplayPlanOutcome(
                    plan_by_component[component].plan_id,
                    component,
                    RawReplayPlanStatus.RETRYABLE,
                    "dry-run census selected this executable authority component",
                    "execute through the typed raw-materialization writer",
                )
                for component in executable_components
            )
            + blocked_plan_outcomes
        )
        detail = (
            f"Would: classify and replay {len(executable_components):,} selected authority component(s) "
            "through per-session revision authority; "
            f"selected raw payload bytes total={_format_bytes(selected_total_bytes)}, "
            f"largest={_format_bytes(selected_max_bytes)}"
        )
        if candidates.already_parsed:
            detail += f"; {candidates.already_parsed:,} already parsed but not materialized"
        if missing_blobs:
            detail += f"; {_raw_materialization_missing_blob_detail(candidates, final=True)}"
        if oversized_raw_ids:
            detail += (
                f"; {len(oversized_raw_ids):,} raw rows exceed replay size advisory {_format_bytes(max_payload_bytes)}"
            )
        if oversized_stream_safe_raw_ids:
            detail += f"; {len(oversized_stream_safe_raw_ids):,} oversized stream-record raw rows are stream-capable"
        return _internal_derived_repair_result(
            "raw_materialization",
            repaired_count=0,
            success=False,
            detail=detail,
            metrics=metrics,
            plan_outcomes=plan_outcomes,
            census_receipt=census_receipt,
        )

    for outcome in blocked_plan_outcomes:
        record_raw_replay_outcome(archive_root, census_receipt.census_id, outcome)

    stale_outcomes: list[RawReplayPlanOutcome] = []
    validated_plans: list[RawReplayPlan] = []
    for plan in executable_plans:
        valid, observed = validate_raw_replay_plan(archive_root, plan)
        if valid:
            validated_plans.append(plan)
        else:
            stale_outcomes.append(reject_stale_raw_replay_plan(archive_root, census_receipt.census_id, plan, observed))
    if stale_outcomes:
        carried = [
            RawReplayPlanOutcome(
                plan.plan_id,
                plan.input_raw_ids,
                RawReplayPlanStatus.CARRIED_FORWARD,
                "another selected plan failed immutable precondition validation",
                "resolve the durable stale-plan blocker before retrying this unchanged plan",
            )
            for plan in validated_plans
        ]
        for outcome in carried:
            record_raw_replay_outcome(archive_root, census_receipt.census_id, outcome)
        stale_candidates = _raw_materialization_candidate_ids(
            config,
            raw_artifact_id=raw_artifact_id,
            provider=provider,
            source_family=source_family,
            source_root=source_root,
        )
        stale_post_plans, stale_post_residual = _raw_authority_postflight_snapshot(
            archive_root,
            stale_candidates,
            max_payload_bytes=max_payload_bytes,
        )
        census_receipt = finalize_raw_authority_census(
            archive_root,
            census_receipt.census_id,
            post_plans=stale_post_plans,
            post_residual=stale_post_residual,
        )
        plan_outcomes = tuple(stale_outcomes + carried) + blocked_plan_outcomes
        plan_count, carried_forward_count, conservation_error_count = _raw_replay_conservation_metrics(
            plans,
            selected_plan_ids,
            plan_outcomes,
        )
        metrics["raw_materialization_plan_rejected_stale_count"] = float(len(stale_outcomes))
        metrics["raw_materialization_plan_carried_forward_count"] = float(carried_forward_count)
        metrics["raw_materialization_plan_outcome_count"] = float(plan_count)
        metrics["raw_materialization_plan_conservation_error_count"] = float(conservation_error_count)
        return _internal_derived_repair_result(
            "raw_materialization",
            repaired_count=0,
            success=False,
            detail=(
                f"Rejected {len(stale_outcomes):,} stale immutable replay plan(s); "
                "automatic convergence is now fail-closed behind durable blocker evidence"
            ),
            metrics=metrics,
            plan_outcomes=plan_outcomes,
            census_receipt=census_receipt,
        )

    from polylogue.sources.revision_backfill import RevisionBackfillResult, backfill_historical_revision_evidence

    executable_raw_ids = raw_ids
    metrics["raw_materialization_executed_count"] = float(len(executable_raw_ids))
    replay_parts: list[RevisionBackfillResult] = []
    execution_outcomes: list[RawReplayPlanOutcome] = []
    for plan, raw_id in zip(executable_plans, executable_raw_ids, strict=True):
        component = plan.input_raw_ids
        try:
            part = backfill_historical_revision_evidence(
                archive_root,
                selected_raw_ids=[raw_id],
                max_payload_bytes=max_payload_bytes,
            )
        except RawRevisionReplayResourceBlockedError as exc:
            metrics["raw_materialization_resource_blocked_count"] = max(
                metrics.get("raw_materialization_resource_blocked_count", 0.0), float(len(exc.raw_ids))
            )
            outcome = RawReplayPlanOutcome(
                plan.plan_id,
                component,
                RawReplayPlanStatus.DEFERRED,
                f"resource-envelope:{max_payload_bytes}",
                "retry only after a larger resource envelope or changed source/index preconditions",
            )
            record_raw_replay_outcome(archive_root, census_receipt.census_id, outcome)
            execution_outcomes.append(outcome)
            continue
        except Exception as exc:
            logger.exception("raw replay plan %s failed", plan.plan_id)
            application_receipt = raw_replay_application_receipt(archive_root, plan)
            receipt_valid, receipt_problems = validate_raw_replay_application_receipt(plan, application_receipt)
            if receipt_valid:
                outcome = RawReplayPlanOutcome(
                    plan.plan_id,
                    component,
                    RawReplayPlanStatus.EXECUTED,
                    f"component reached exact durable postconditions before {type(exc).__name__}",
                    "none",
                    application_receipt,
                )
                record_raw_replay_outcome(archive_root, census_receipt.census_id, outcome)
            else:
                plan_still_valid, _ = validate_raw_replay_plan(archive_root, plan)
                if plan_still_valid:
                    outcome = RawReplayPlanOutcome(
                        plan.plan_id,
                        component,
                        RawReplayPlanStatus.RETRYABLE,
                        f"component execution raised {type(exc).__name__}: {exc}",
                        "retry this unchanged plan after independent components have received a turn",
                        application_receipt,
                    )
                    record_raw_replay_outcome(archive_root, census_receipt.census_id, outcome)
                else:
                    outcome = reject_invalid_raw_replay_application(
                        archive_root,
                        census_receipt.census_id,
                        plan,
                        application_receipt,
                        (
                            f"component execution raised {type(exc).__name__}: {exc}",
                            *receipt_problems,
                        ),
                    )
            execution_outcomes.append(outcome)
            continue
        replay_parts.append(part)
        current = _raw_materialization_candidate_ids(
            config,
            raw_artifact_id=raw_artifact_id,
            provider=provider,
            source_family=source_family,
            source_root=source_root,
        )
        component_outcomes = _raw_replay_plan_outcomes(archive_root, [plan], remaining=current)
        for outcome in component_outcomes:
            application_receipt = raw_replay_application_receipt(archive_root, plan)
            receipted = dataclasses.replace(outcome, application_receipt=application_receipt)
            if outcome.status is RawReplayPlanStatus.EXECUTED:
                receipt_valid, receipt_problems = validate_raw_replay_application_receipt(plan, application_receipt)
                if not receipt_valid:
                    receipted = reject_invalid_raw_replay_application(
                        archive_root,
                        census_receipt.census_id,
                        plan,
                        application_receipt,
                        receipt_problems,
                    )
                else:
                    record_raw_replay_outcome(archive_root, census_receipt.census_id, receipted)
            elif outcome.status is RawReplayPlanStatus.REJECTED_STALE:
                receipted = reject_invalid_raw_replay_application(
                    archive_root,
                    census_receipt.census_id,
                    plan,
                    application_receipt,
                    (outcome.reason,),
                )
            else:
                record_raw_replay_outcome(archive_root, census_receipt.census_id, receipted)
            execution_outcomes.append(receipted)

    replay = RevisionBackfillResult(
        scanned=sum(part.scanned for part in replay_parts),
        classified_full=sum(part.classified_full for part in replay_parts),
        replayed_logical_sources=sum(part.replayed_logical_sources for part in replay_parts),
        quarantined=sum(part.quarantined for part in replay_parts),
        adoption_deferred=sum(part.adoption_deferred for part in replay_parts),
    )
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
    plan_outcomes = tuple(execution_outcomes) + blocked_plan_outcomes
    post_plans, post_residual = _raw_authority_postflight_snapshot(
        archive_root, remaining, max_payload_bytes=max_payload_bytes
    )
    census_receipt = finalize_raw_authority_census(
        archive_root,
        census_receipt.census_id,
        post_plans=post_plans,
        post_residual=post_residual,
    )
    for status in RawReplayPlanStatus:
        metrics[f"raw_materialization_plan_{status.value}_count"] = float(
            sum(outcome.status is status for outcome in plan_outcomes)
        )
    plan_count, carried_forward_count, conservation_error_count = _raw_replay_conservation_metrics(
        plans,
        selected_plan_ids,
        plan_outcomes,
    )
    metrics["raw_materialization_plan_carried_forward_count"] = float(carried_forward_count)
    metrics["raw_materialization_plan_outcome_count"] = float(plan_count)
    metrics["raw_materialization_plan_conservation_error_count"] = float(conservation_error_count)
    success = (
        not remaining.raw_ids
        and remaining.missing_blobs == 0
        and replay.adoption_deferred == 0
        and remaining.adoption_deferred == 0
        and remaining.byte_authority_pending == 0
        and conservation_error_count == 0
        and not any(outcome.status is RawReplayPlanStatus.REJECTED_STALE for outcome in plan_outcomes)
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
    expanded_component_blocked = any(len(component) > 1 for component in all_blocked_components)
    if all_blocked_component_raw_ids and expanded_component_blocked:
        detail += (
            f"; {len(all_blocked_component_raw_ids):,} raw row(s) belong to authority components whose aggregate "
            f"payload exceeds {_format_bytes(max_payload_bytes)}"
        )
    elif oversized_raw_ids:
        detail += (
            f"; {len(oversized_raw_ids):,} non-stream-safe raw row(s) exceed execution limit "
            f"{_format_bytes(max_payload_bytes)}"
        )
    elif all_blocked_component_raw_ids:
        detail += (
            f"; {len(all_blocked_component_raw_ids):,} raw row(s) belong to authority components whose aggregate "
            f"payload exceeds {_format_bytes(max_payload_bytes)}"
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
        plan_outcomes=plan_outcomes,
        census_receipt=census_receipt,
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
    "raw_materialization_scale_profile",
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
