"""Prepared byte-revision authority and replay publication.

The daemon owns scheduling and terminal source-state settlement.  This module
only prepares retained evidence on a pinned reader and publishes it after the
writer proves the complete byte cohort and head have not moved.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal

from polylogue.archive.revision_authority import (
    HistoricalRawRevisionStream,
    RawRevisionAuthority,
    RawRevisionKind,
    classify_historical_full_revision_streams,
)
from polylogue.archive.revision_replay import RevisionCandidate, RevisionReplayPlan, plan_revision_replay
from polylogue.core.timestamp_authority import normalize_session_timestamps
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.dispatch import merge_parsed_session_chunks
from polylogue.sources.revision_backfill import parse_retained_raw_sessions
from polylogue.storage.blob_store import BlobStore, PreparedBlob
from polylogue.storage.ingest_governance import (
    AcceptedHeadBinding,
    AppendFrontierBinding,
    RawDescriptorBinding,
    _append_frontier,
    _head_binding,
    _raw_binding,
)
from polylogue.storage.sqlite.archive_tiers.revision_governance import (
    _promote_contiguous_append_evidence,
    _raw_revision_candidates,
    raw_revision_file_mtime,
    raw_revision_replay_plan,
)
from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveSourceBlobRef
from polylogue.storage.sqlite.archive_tiers.write import (
    PreparedSessionWrite,
    PreparedSessionWriteRefusedError,
    prepare_session_write,
    prepared_lineage_bindings,
)

CheckStop = Callable[[], None]


@dataclass(frozen=True, slots=True)
class ByteAuthorityUpdate:
    raw_id: str
    authority: RawRevisionAuthority
    predecessor_raw_id: str | None
    baseline_raw_id: str | None
    acquisition_generation: int


@dataclass(frozen=True, slots=True)
class BytePreparationBudget:
    """Whole-cohort compute ceiling. Exhaustion defers, never truncates."""

    max_raws: int
    max_payload_bytes: int


@dataclass(frozen=True, slots=True)
class PreparedByteAuthority:
    logical_source_key: str
    accepted_generation_raw_ids: tuple[str, ...]
    candidates: tuple[RevisionCandidate, ...]
    bindings: tuple[RawDescriptorBinding, ...]
    head: AcceptedHeadBinding
    append_frontier: tuple[AppendFrontierBinding, ...]
    updates: tuple[ByteAuthorityUpdate, ...]


@dataclass(frozen=True, slots=True)
class AlreadyClassified:
    logical_source_key: str
    plan: RevisionReplayPlan


@dataclass(frozen=True, slots=True)
class Deferred:
    logical_source_key: str
    required_raw_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class PreparedByteAttachment:
    raw_id: str
    position: int
    prepared_blob: PreparedBlob | None
    precomputed_blob: tuple[str, int] | None


@dataclass(frozen=True, slots=True)
class PreparedByteReplay:
    logical_source_key: str
    blob_root: str
    required_raw_ids: tuple[str, ...]
    plan: RevisionReplayPlan
    candidates: tuple[RevisionCandidate, ...]
    bindings: tuple[RawDescriptorBinding, ...]
    head: AcceptedHeadBinding
    append_frontier: tuple[AppendFrontierBinding, ...]
    index_path: str
    index_epoch: int
    tip_authority: str
    tip_generation: int | None
    tip_file_mtime: str | None
    hook_parent_native_id: str | None
    resolved_parent_session_id: str | None
    parsed_by_raw_id: dict[str, Any]
    aggregate_session: Any
    aggregate_content_hash: bytes
    pending_session: Any
    prepared_write: PreparedSessionWrite
    attachments: tuple[PreparedByteAttachment, ...]
    acquired_at_ms: int


@dataclass(frozen=True, slots=True)
class ByteReplayPublication:
    disposition: Literal["applied", "binding_moved", "deferred"]
    accepted_raw_ids: tuple[str, ...] = ()
    requested_raw_ids: tuple[str, ...] = ()
    session_id: str | None = None
    reason: str | None = None


def _check(check_stop: CheckStop | None) -> None:
    if check_stop is not None:
        check_stop()


def _within_budget(candidates: Sequence[RevisionCandidate], budget: BytePreparationBudget | None) -> str | None:
    if budget is None:
        return None
    if budget.max_raws < 1 or budget.max_payload_bytes < 0:
        return "byte preparation budget is invalid"
    if len(candidates) > budget.max_raws:
        return f"byte cohort has {len(candidates)} raws, above prepared limit {budget.max_raws}"
    payload_bytes = sum(candidate.blob_size for candidate in candidates)
    if payload_bytes > budget.max_payload_bytes:
        return f"byte cohort has {payload_bytes} payload bytes, above prepared limit {budget.max_payload_bytes}"
    return None


def _prepare_attachments(
    archive: Any, parsed: dict[str, Any], *, check_stop: CheckStop | None = None
) -> tuple[PreparedByteAttachment, ...]:
    blob_store = BlobStore(Path(archive.archive_root) / "blob")
    staged: list[PreparedByteAttachment] = []
    try:
        for raw_id, session in parsed.items():
            for position, attachment in enumerate(session.attachments):
                _check(check_stop)
                if attachment.inline_bytes is None and attachment.precomputed_blob is None:
                    continue
                staged.append(
                    PreparedByteAttachment(
                        raw_id=raw_id,
                        position=position,
                        prepared_blob=(
                            None
                            if attachment.inline_bytes is None
                            else blob_store.prepare_from_bytes(attachment.inline_bytes)
                        ),
                        precomputed_blob=attachment.precomputed_blob,
                    )
                )
        _check(check_stop)
        return tuple(staged)
    except BaseException:
        for item in staged:
            if item.prepared_blob is not None:
                blob_store.discard_prepared(item.prepared_blob)
        raise


def _bindings(archive: Any, candidates: Sequence[RevisionCandidate]) -> tuple[RawDescriptorBinding, ...]:
    return tuple(
        _raw_binding(archive, candidate.raw_id, logical_source_key=candidate.logical_source_key)
        for candidate in candidates
    )


def _same_binding(archive: Any, prepared: PreparedByteAuthority | PreparedByteReplay) -> bool:
    candidates = tuple(_raw_revision_candidates(archive, prepared.logical_source_key))
    return (
        (not isinstance(prepared, PreparedByteReplay) or prepared.blob_root == str(archive.archive_root / "blob"))
        and candidates == prepared.candidates
        and _bindings(archive, candidates) == prepared.bindings
        and _head_binding(archive, prepared.logical_source_key) == prepared.head
        and _append_frontier(archive, prepared.logical_source_key) == prepared.append_frontier
    )


def _index_epoch(archive: Any) -> int:
    row = archive._conn.execute("SELECT epoch FROM query_unit_frame_state WHERE singleton = 1").fetchone()
    if row is None:
        raise RuntimeError("prepared byte replay requires index epoch state")
    return int(row[0])


def _replay_binding_matches(archive: Any, prepared: PreparedByteReplay) -> bool:
    if not _same_binding(archive, prepared):
        return False
    if str(archive.index_db_path.resolve()) != prepared.index_path or _index_epoch(archive) != prepared.index_epoch:
        return False
    tip = prepared.plan.accepted_raw_ids[-1]
    tip_binding = _raw_binding(archive, tip, logical_source_key=prepared.logical_source_key)
    if (
        tip_binding.revision_authority != prepared.tip_authority
        or tip_binding.acquisition_generation != prepared.tip_generation
        or archive.raw_revision_file_mtime(tip) != prepared.tip_file_mtime
    ):
        return False
    hook_parent, resolved_parent = prepared_lineage_bindings(
        archive._conn,
        prepared.pending_session,
        source_conn=archive._ensure_source_conn(),
    )
    return hook_parent == prepared.hook_parent_native_id and resolved_parent == prepared.resolved_parent_session_id


def prepare_ingest_byte_authority(
    read_archive: Any,
    logical_key: str,
    *,
    accepted_generation_raw_ids: Sequence[str],
    budget: BytePreparationBudget | None = None,
    check_stop: CheckStop | None = None,
) -> PreparedByteAuthority | AlreadyClassified | Deferred:
    """Classify every full candidate off-writer and bind all byte evidence."""
    _check(check_stop)
    candidates = tuple(_raw_revision_candidates(read_archive, logical_key))
    if reason := _within_budget(candidates, budget):
        return Deferred(logical_key, tuple(dict.fromkeys(accepted_generation_raw_ids)), reason)
    plan = plan_revision_replay(list(candidates))
    requested = tuple(dict.fromkeys(accepted_generation_raw_ids))
    if plan.accepted_raw_ids and set(requested).issubset(plan.accepted_raw_ids):
        return AlreadyClassified(logical_key, plan)
    fulls = [candidate for candidate in candidates if candidate.kind is RawRevisionKind.FULL]
    blob_store = BlobStore(Path(read_archive.archive_root) / "blob")
    streams: list[HistoricalRawRevisionStream] = []
    for candidate in fulls:
        blob_hash = _raw_binding(read_archive, candidate.raw_id).blob_hash

        def open_payload(blob_hash: str = blob_hash) -> BinaryIO:
            return blob_store.open(blob_hash)

        streams.append(
            HistoricalRawRevisionStream(
                raw_id=candidate.raw_id,
                payload_size=candidate.blob_size,
                open_payload=open_payload,
            )
        )
    decisions = {decision.raw_id: decision for decision in classify_historical_full_revision_streams(streams)}
    baseline = next((decision.raw_id for decision in decisions.values() if decision.relation == "baseline"), None)
    generation: dict[str, int] = {}
    current = baseline
    while current is not None:
        generation[current] = len(generation)
        current = next(
            (decision.raw_id for decision in decisions.values() if decision.predecessor_raw_id == current), None
        )
    updates = tuple(
        ByteAuthorityUpdate(
            raw_id=candidate.raw_id,
            authority=(
                decisions[candidate.raw_id].authority
                if candidate.raw_id in decisions
                else RawRevisionAuthority.QUARANTINED
            ),
            predecessor_raw_id=(
                decisions[candidate.raw_id].predecessor_raw_id if candidate.raw_id in decisions else None
            ),
            baseline_raw_id=(
                baseline
                if candidate.raw_id in decisions
                and decisions[candidate.raw_id].authority is RawRevisionAuthority.BYTE_PROVEN
                else None
            ),
            acquisition_generation=generation.get(candidate.raw_id, 0),
        )
        for candidate in fulls
    )
    return PreparedByteAuthority(
        logical_key,
        requested,
        candidates,
        _bindings(read_archive, candidates),
        _head_binding(read_archive, logical_key),
        _append_frontier(read_archive, logical_key),
        updates,
    )


def publish_ingest_byte_authority(
    writer_archive: Any, prepared: PreparedByteAuthority
) -> Literal["applied", "binding_moved"]:
    """Publish only precomputed full-authority updates, never inspect blobs."""
    if not _same_binding(writer_archive, prepared):
        return "binding_moved"
    conn = writer_archive._ensure_source_conn()
    with conn:
        for update in prepared.updates:
            conn.execute(
                """UPDATE raw_sessions SET revision_authority = ?, predecessor_raw_id = ?, baseline_raw_id = ?, acquisition_generation = ? WHERE raw_id = ?""",
                (
                    update.authority.value,
                    update.predecessor_raw_id,
                    update.baseline_raw_id,
                    update.acquisition_generation,
                    update.raw_id,
                ),
            )
        _promote_contiguous_append_evidence(conn, prepared.logical_source_key)
    return "applied"


def prepare_ingest_byte_replay(
    read_archive: Any,
    logical_key: str,
    *,
    required_raw_ids: Sequence[str],
    budget: BytePreparationBudget | None = None,
    check_stop: CheckStop | None = None,
) -> PreparedByteReplay | Deferred:
    """Decode, compose, and lower an accepted byte chain away from the writer."""
    _check(check_stop)
    candidates = tuple(_raw_revision_candidates(read_archive, logical_key))
    plan = raw_revision_replay_plan(read_archive, logical_key)
    bindings = _bindings(read_archive, candidates)
    head = _head_binding(read_archive, logical_key)
    append_frontier = _append_frontier(read_archive, logical_key)
    index_path = str(read_archive.index_db_path.resolve())
    index_epoch = _index_epoch(read_archive)
    required = tuple(dict.fromkeys(required_raw_ids))
    if reason := _within_budget(candidates, budget):
        return Deferred(logical_key, required, reason)
    if not plan.accepted_raw_ids or not set(required).issubset(plan.accepted_raw_ids):
        return Deferred(logical_key, required, "requested raw is not in the accepted byte chain")
    parsed: dict[str, Any] = {}
    for raw_id in plan.accepted_raw_ids:
        _check(check_stop)
        sessions = parse_retained_raw_sessions(read_archive, raw_id)
        if len(sessions) != 1:
            return Deferred(logical_key, required, f"accepted raw {raw_id} did not parse to one session")
        parsed[raw_id] = sessions[0]
    aggregate = merge_parsed_session_chunks(parsed[raw_id] for raw_id in plan.accepted_raw_ids)
    if len(aggregate) != 1:
        return Deferred(logical_key, required, "accepted byte chain did not compose to one session")
    head_raw_id = _head_binding(read_archive, logical_key).accepted_raw_id
    try:
        already_indexed = plan.accepted_raw_ids.index(head_raw_id) if head_raw_id is not None else -1
    except ValueError:
        already_indexed = -1
    pending = plan.accepted_raw_ids[already_indexed + 1 :]
    if not pending:
        return Deferred(logical_key, required, "accepted chain has no replay tail")
    composed_tail = merge_parsed_session_chunks(parsed[raw_id] for raw_id in pending)
    if len(composed_tail) != 1:
        return Deferred(logical_key, required, "accepted byte tail did not compose to one session")
    tip = plan.accepted_raw_ids[-1]
    fallback_timestamp = raw_revision_file_mtime(read_archive, tip)
    pending_session = normalize_session_timestamps(composed_tail[0], fallback_timestamp=fallback_timestamp)
    merge_append = already_indexed >= 0
    prepared_write = prepare_session_write(
        read_archive._conn,
        pending_session,
        merge_append=merge_append,
        source_conn=read_archive._ensure_source_conn(),
    )
    _check(check_stop)
    tip_binding = next(binding for binding in bindings if binding.raw_id == tip)
    # Stage filesystem bytes only once the complete write carrier exists.  A
    # staging error leaves no orphaned private attachment file behind.
    attachments = _prepare_attachments(read_archive, parsed, check_stop=check_stop)
    current_hook, current_parent = prepared_lineage_bindings(
        read_archive._conn,
        pending_session,
        source_conn=read_archive._ensure_source_conn(),
    )
    stable = (
        tuple(_raw_revision_candidates(read_archive, logical_key)) == candidates
        and _bindings(read_archive, candidates) == bindings
        and _head_binding(read_archive, logical_key) == head
        and _append_frontier(read_archive, logical_key) == append_frontier
        and str(read_archive.index_db_path.resolve()) == index_path
        and _index_epoch(read_archive) == index_epoch
        and current_hook == prepared_write.context.hook_parent_native_id
        and current_parent == prepared_write.context.parent_session_id
        and raw_revision_file_mtime(read_archive, tip) == fallback_timestamp
    )
    if not stable:
        provisional = PreparedByteReplay(
            logical_source_key=logical_key,
            blob_root=str(read_archive.archive_root / "blob"),
            required_raw_ids=required,
            plan=plan,
            candidates=candidates,
            bindings=bindings,
            head=head,
            append_frontier=append_frontier,
            index_path=index_path,
            index_epoch=index_epoch,
            tip_authority=tip_binding.revision_authority,
            tip_generation=tip_binding.acquisition_generation,
            tip_file_mtime=fallback_timestamp,
            hook_parent_native_id=prepared_write.context.hook_parent_native_id,
            resolved_parent_session_id=prepared_write.context.parent_session_id,
            parsed_by_raw_id=parsed,
            aggregate_session=aggregate[0],
            aggregate_content_hash=bytes.fromhex(session_content_hash(aggregate[0])),
            pending_session=pending_session,
            prepared_write=prepared_write,
            attachments=attachments,
            acquired_at_ms=int(read_archive.raw_revision_observation_order(tip)[0]),
        )
        discard_prepared_ingest_byte_replay(provisional)
        return Deferred(logical_key, required, "byte replay binding moved during preparation")
    return PreparedByteReplay(
        logical_source_key=logical_key,
        blob_root=str(read_archive.archive_root / "blob"),
        required_raw_ids=required,
        plan=plan,
        candidates=candidates,
        bindings=bindings,
        head=head,
        append_frontier=append_frontier,
        index_path=index_path,
        index_epoch=index_epoch,
        tip_authority=tip_binding.revision_authority,
        tip_generation=tip_binding.acquisition_generation,
        tip_file_mtime=fallback_timestamp,
        hook_parent_native_id=prepared_write.context.hook_parent_native_id,
        resolved_parent_session_id=prepared_write.context.parent_session_id,
        parsed_by_raw_id=parsed,
        aggregate_session=aggregate[0],
        aggregate_content_hash=bytes.fromhex(session_content_hash(aggregate[0])),
        pending_session=pending_session,
        prepared_write=prepared_write,
        attachments=attachments,
        acquired_at_ms=int(read_archive.raw_revision_observation_order(tip)[0]),
    )


def _writer_attachments(
    writer_archive: Any, prepared: PreparedByteReplay
) -> tuple[dict[str, dict[int, tuple[bytes | None, int, str]]], dict[str, tuple[ArchiveSourceBlobRef, ...]]]:
    publisher = writer_archive._blob_publisher
    if publisher is None:
        raise RuntimeError("byte replay publication requires a writable blob publisher")
    attachments: dict[str, dict[int, tuple[bytes | None, int, str]]] = {}
    refs: dict[str, list[ArchiveSourceBlobRef]] = {}
    bindings = {binding.raw_id: binding for binding in prepared.bindings}
    for item in prepared.attachments:
        attachment = prepared.parsed_by_raw_id[item.raw_id].attachments[item.position]
        if item.prepared_blob is None:
            if item.precomputed_blob is None:
                raise RuntimeError("prepared attachment has no retained blob evidence")
            hash_hex, size = item.precomputed_blob
            attachments.setdefault(item.raw_id, {})[id(attachment)] = (bytes.fromhex(hash_hex), size, "acquired")
            continue
        hash_hex, size = publisher.queue_prepared(item.prepared_blob)
        attachments.setdefault(item.raw_id, {})[id(attachment)] = (bytes.fromhex(hash_hex), size, "acquired")
        refs.setdefault(item.raw_id, []).append(
            ArchiveSourceBlobRef(
                blob_hash=bytes.fromhex(hash_hex),
                ref_type="attachment",
                source_path=bindings[item.raw_id].source_path,
                size_bytes=size,
                acquired_at_ms=prepared.acquired_at_ms,
                publication_receipt_id=publisher.receipt_id(hash_hex),
            )
        )
    return attachments, {raw_id: tuple(items) for raw_id, items in refs.items()}


def discard_prepared_ingest_byte_replay(prepared: PreparedByteReplay) -> None:
    """Remove compute-staged attachment bytes after cancellation or stale CAS."""
    blob_store = BlobStore(Path(prepared.blob_root))
    for attachment in prepared.attachments:
        if attachment.prepared_blob is not None:
            blob_store.discard_prepared(attachment.prepared_blob)


def publish_ingest_byte_replay(writer_archive: Any, prepared: PreparedByteReplay) -> ByteReplayPublication:
    """CAS-publish a fully prepared replay through the canonical publisher."""
    try:
        # This is deliberately before blob queueing.  The writer owns this
        # comparison-to-publication interval and may conservatively reject an
        # unrelated sibling write through the index epoch.
        if not _replay_binding_matches(writer_archive, prepared):
            return ByteReplayPublication(
                "binding_moved",
                requested_raw_ids=prepared.required_raw_ids,
                reason="prepared byte replay binding moved",
            )
        attachments, refs = _writer_attachments(writer_archive, prepared)
        session_id, accepted = writer_archive.apply_raw_revision_replay(
            prepared.plan,
            prepared.parsed_by_raw_id,
            acquired_at_ms=prepared.acquired_at_ms,
            skip_already_applied=True,
            prepared_aggregate_session=prepared.aggregate_session,
            prepared_aggregate_content_hash=prepared.aggregate_content_hash,
            prepared_pending_session=prepared.pending_session,
            prepared_write=prepared.prepared_write,
            preacquired_attachment_blobs_by_raw_id=attachments,
            preacquired_attachment_refs_by_raw_id=refs,
        )
    except PreparedSessionWriteRefusedError as exc:
        return ByteReplayPublication("binding_moved", requested_raw_ids=prepared.required_raw_ids, reason=str(exc))
    else:
        return ByteReplayPublication(
            "applied", accepted_raw_ids=accepted, requested_raw_ids=prepared.required_raw_ids, session_id=session_id
        )
    finally:
        discard_prepared_ingest_byte_replay(prepared)


__all__ = [
    "AlreadyClassified",
    "ByteReplayPublication",
    "BytePreparationBudget",
    "Deferred",
    "PreparedByteAuthority",
    "PreparedByteReplay",
    "prepare_ingest_byte_authority",
    "prepare_ingest_byte_replay",
    "discard_prepared_ingest_byte_replay",
    "publish_ingest_byte_authority",
    "publish_ingest_byte_replay",
]
