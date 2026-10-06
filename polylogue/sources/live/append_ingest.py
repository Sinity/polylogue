"""Append-only live-ingest persistence helpers."""

from __future__ import annotations

import sqlite3
import time
from builtins import BaseExceptionGroup
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Protocol, cast

from polylogue.archive.artifact_taxonomy import ArtifactKind, classify_artifact_path
from polylogue.archive.revision_authority import (
    RawRevisionAuthority,
    RawRevisionEnvelope,
    RawRevisionKind,
    append_source_revision,
)
from polylogue.core.compute import DaemonOperationCancelled
from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.enums import Provider
from polylogue.core.raw_failure_evidence import CohortMembershipRefusalError
from polylogue.core.sources import origin_from_provider
from polylogue.core.stage_admission import admit_stage_write
from polylogue.core.storage_faults import (
    ARCHIVE_SIDE_FAULTS,
    StorageFaultKind,
    raise_if_storage_fault,
    storage_fault_kind,
)
from polylogue.logging import get_logger
from polylogue.sources.live.archive_open import _open_archive_for_live_write, _source_tier_acquisition_required
from polylogue.sources.live.batch_support import _AppendPlan, _AppendResult, hook_carrier_logical_source_key
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.sqlite_locking import is_transient_sqlite_lock
from polylogue.sources.revision_backfill import RetainedPreparationNoProgressError
from polylogue.storage.archive_identity import resolve_active_index_path
from polylogue.storage.raw.models import RawSessionStateUpdate
from polylogue.storage.raw_authority import raw_authority_parser_fingerprint
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import read_raw_profile_identity

logger = get_logger(__name__)


def _add_timing(timings: dict[str, float], name: str, started_at: float) -> None:
    timings[name] = timings.get(name, 0.0) + (time.perf_counter() - started_at)


class _AppendIngestOwner(Protocol):
    _cursor: CursorStore
    _polylogue: Any


def _bind_append_revision(
    archive: Any,
    raw_id: str,
    *,
    logical_source_key: str,
    plan: _AppendPlan,
) -> tuple[str, RawRevisionAuthority]:
    """Persist an APPEND envelope from the append plan's durable identity."""
    if plan.cursor_fingerprint is None:
        raise ValueError("append payload did not prove cursor identity")
    parent = archive.raw_append_revision_parent(
        logical_source_key,
        plan.start_offset,
        plan.cursor_fingerprint,
    )
    predecessor_raw_id: str | None = None
    baseline_raw_id: str | None = None
    generation = archive.raw_full_revision_generation(logical_source_key)
    authority = RawRevisionAuthority.QUARANTINED
    if parent is not None:
        predecessor_raw_id, baseline_raw_id, generation = parent
        authority = RawRevisionAuthority.BYTE_PROVEN
    archive.bind_raw_revision(
        raw_id,
        RawRevisionEnvelope(
            logical_source_key=logical_source_key,
            kind=RawRevisionKind.APPEND,
            source_revision=append_source_revision(plan.cursor_fingerprint, plan.payload_hash),
            acquisition_generation=generation,
            predecessor_source_revision=plan.cursor_fingerprint,
            predecessor_raw_id=predecessor_raw_id,
            baseline_raw_id=baseline_raw_id,
            append_start_offset=plan.start_offset,
            append_end_offset=plan.last_complete_newline,
            authority=authority,
        ),
    )
    return logical_source_key, authority


def _bind_hook_carrier_append_revision(
    archive: Any,
    raw_id: str,
    *,
    provider: Provider,
    plan: _AppendPlan,
) -> tuple[str, RawRevisionAuthority]:
    """Bind one carrier tail without assigning its events a session identity."""
    logical_source_key = hook_carrier_logical_source_key(provider=provider, source_path=str(plan.path))
    return _bind_append_revision(
        archive,
        raw_id,
        logical_source_key=logical_source_key,
        plan=plan,
    )


def _write_append_raw_payload(
    archive: Any,
    *,
    provider: Provider,
    plan: _AppendPlan,
    acquired_at_ms: int,
) -> str:
    """Capture literal append bytes with their migration-stable raw identity."""
    try:
        file_mtime_ms = int(plan.path.stat().st_mtime * 1000)
    except FileNotFoundError:
        # The watcher can remove a completed append between planning and the
        # write.  The payload remains authoritative; absence of a filesystem
        # observation must stay unknown rather than aborting the append.
        file_mtime_ms = None
    return cast(
        str,
        archive.write_raw_payload(
            provider=provider,
            payload=plan.payload,
            source_path=str(plan.path),
            canonical_source_path=plan.canonical_source_path,
            source_index=plan.source_index,
            acquired_at_ms=acquired_at_ms,
            file_mtime_ms=file_mtime_ms,
            native_id=plan.acquisition_native_id_hint,
            post_parse=True,
        ),
    )


def reset_transient_raw_parse_state(
    archive: Any,
    raw_id: str,
    *,
    provider: Provider,
) -> None:
    """Leave acquired bytes pending when index persistence was unavailable."""
    archive.finalize_raw_parse_state(
        raw_id,
        state=RawSessionStateUpdate(
            parsed_at=None,
            parse_error=None,
            payload_provider=provider,
            detection_warnings=None,
        ),
    )


def _append_fault_kinds(exc: BaseException) -> frozenset[StorageFaultKind] | None:
    """The storage faults an append may escape with.

    SQLite reports archive-side results, so every storage kind applies. A bare
    ``OSError`` can come from the source file itself (``stat`` or a read
    reporting ``EIO``), which is this file's own failure; only capacity and
    read-only faults are archive-side by construction.
    """
    return ARCHIVE_SIDE_FAULTS if isinstance(exc, OSError) else None


def _append_storage_fault(exc: BaseException) -> bool:
    kind = storage_fault_kind(exc)
    kinds = _append_fault_kinds(exc)
    return kind is not None and (kinds is None or kind in kinds)


def ingest_append_plans(
    owner: _AppendIngestOwner,
    plans: list[_AppendPlan],
    *,
    converge_raw: Callable[[Path, str, _AppendPlan], None],
) -> _AppendResult:
    """Acquire exact append raws, then ask the canonical owner to prepare them off-gate."""
    if not plans:
        return _AppendResult(succeeded=[], failed=[], worker_count=0)
    archive_root = Path(getattr(owner._polylogue, "archive_root", owner._cursor._db_path.parent))
    timings: dict[str, float] = {}
    succeeded: list[_AppendPlan] = []
    failed: list[_AppendPlan] = []
    deferred: list[_AppendPlan] = []
    session_ids_by_path: dict[Path, str] = {}
    acquired_at_ms = int(datetime.now(UTC).timestamp() * 1000)
    source_only = _source_tier_acquisition_required()

    def acquire(plan: _AppendPlan, provider: Provider) -> tuple[str, RawRevisionAuthority, str | None]:
        if plan.native_id_hint is None:
            path_artifact = classify_artifact_path(str(plan.path), provider=provider)
            if path_artifact is None or path_artifact.kind is not ArtifactKind.HOOK_EVENT_CARRIER:
                raise ValueError("append plan lacks its declared session identity")
        source_db = archive_root / "source.db"
        archive_missing = not source_db.exists()
        if not source_only:
            archive_missing = archive_missing or not resolve_active_index_path(archive_root).exists()
        if archive_missing:
            initialize_active_archive_root(archive_root)
        with _open_archive_for_live_write(archive_root, cold_build=True) as archive:
            raw_id = _write_append_raw_payload(archive, provider=provider, plan=plan, acquired_at_ms=acquired_at_ms)
            source = archive.source_connection
            assert source is not None
            profile_key = read_raw_profile_identity(source, raw_id)
            if plan.native_id_hint is None:
                authority = _bind_hook_carrier_append_revision(archive, raw_id, provider=provider, plan=plan)[1]
            else:
                authority = _bind_append_revision(
                    archive,
                    raw_id,
                    logical_source_key=f"{origin_from_provider(provider).value}:{plan.native_id_hint}",
                    plan=plan,
                )[1]
            return raw_id, authority, profile_key

    def terminal_receipt(plan: _AppendPlan, raw_id: str, profile_key: str | None) -> tuple[bool, str | None]:
        """Acknowledge only the committed terminal proof of this raw's selected route."""
        with _open_archive_for_live_write(archive_root, cold_build=True) as archive:
            source = archive.source_connection
            assert source is not None
            row = source.execute(
                "SELECT r.logical_source_key, r.revision_authority, r.predecessor_raw_id, r.baseline_raw_id, "
                "r.acquisition_generation, r.source_revision, r.append_start_offset, r.append_end_offset, "
                "r.native_id, r.blob_hash, r.canonical_source_path, r.parse_error "
                "FROM raw_sessions r "
                "WHERE r.raw_id=?",
                (raw_id,),
            ).fetchone()
            if row is None or source.in_transaction:
                return False, None
            if read_raw_profile_identity(source, raw_id) != profile_key:
                return False, None
            if plan.cursor_fingerprint is None:
                # The APPEND bind refuses such a plan, so no terminal proof exists.
                return False, None
            parent = archive.raw_append_revision_parent(str(row[0]), plan.start_offset, plan.cursor_fingerprint)
            if (
                row[1] != RawRevisionAuthority.BYTE_PROVEN.value
                or parent != (row[2], row[3], row[4])
                or row[5] != append_source_revision(plan.cursor_fingerprint, plan.payload_hash)
                or row[6] != plan.start_offset
                or row[7] != plan.last_complete_newline
                or row[8] != plan.acquisition_native_id_hint
                or row[9] != bytes.fromhex(plan.payload_hash)
                or row[10] != plan.canonical_source_path
                or row[11] is not None
            ):
                return False, None
            if source_only:
                return True, None
            artifact = source.execute(
                "SELECT 1 FROM raw_artifacts a JOIN raw_authority_parser_census c ON c.raw_id=a.raw_id "
                "WHERE a.raw_id=? AND a.parse_as_session=0 AND c.parser_fingerprint=? AND c.status='complete'",
                (raw_id, raw_authority_parser_fingerprint()),
            ).fetchone()
            if artifact is not None:
                return True, None
            index = archive.index_connection
            if index is None:
                return False, None
            application = index.execute(
                "SELECT a.session_id FROM raw_revision_applications a "
                "JOIN raw_revision_heads h ON h.logical_source_key=a.logical_source_key AND h.session_id=a.session_id "
                "JOIN sessions s ON s.session_id=h.session_id AND s.raw_id=h.accepted_raw_id "
                "AND s.content_hash=h.accepted_content_hash "
                "WHERE a.raw_id=? AND a.decision='applied_append' AND a.source_revision=? "
                "AND a.acquisition_generation=? AND a.accepted_raw_id=? AND a.accepted_source_revision=? "
                "AND a.accepted_frontier_kind='byte' AND a.accepted_frontier=? "
                "AND h.accepted_raw_id=a.accepted_raw_id AND h.accepted_source_revision=a.accepted_source_revision "
                "AND h.accepted_content_hash=a.accepted_content_hash AND h.accepted_frontier_kind=a.accepted_frontier_kind "
                "AND h.accepted_frontier=a.accepted_frontier AND h.acquisition_generation=a.acquisition_generation",
                (raw_id, row[5], row[4], raw_id, row[5], plan.last_complete_newline),
            )
            try:
                first = application.fetchone()
                return (False, None) if first is None or application.fetchone() is not None else (True, str(first[0]))
            finally:
                application.close()

    for plan in plans:
        check_compute_cancelled()
        provider = Provider.from_string(plan.source_name)
        raw_id: str | None = None
        try:
            t0 = time.perf_counter()
            raw_id, authority, profile_key = admit_stage_write(
                "watcher.live_ingest.append.acquire", partial(acquire, plan, provider)
            )
            _add_timing(timings, "append.source_raw_write", t0)
            if authority is RawRevisionAuthority.QUARANTINED:
                deferred.append(plan)
                continue
            if not source_only:
                t0 = time.perf_counter()
                try:
                    converge_raw(archive_root, raw_id, plan)
                except RetainedPreparationNoProgressError:
                    # The append's chain has no accepted baseline to extend
                    # yet; its bytes are sound, so it waits rather than being
                    # settled as a refusal.
                    _add_timing(timings, "append.canonical_raw", t0)
                    deferred.append(plan)
                    continue
                _add_timing(timings, "append.canonical_raw", t0)
            check_compute_cancelled()
            terminal, session_id = admit_stage_write(
                "watcher.live_ingest.append.receipt", partial(terminal_receipt, plan, raw_id, profile_key)
            )
            if terminal:
                succeeded.append(plan)
                if session_id is not None:
                    session_ids_by_path[plan.path] = session_id
            else:
                deferred.append(plan)
        except DaemonOperationCancelled:
            raise
        except Exception as exc:
            from polylogue.sources.revision_backfill import RetainedPreparationRetryableError

            transient = isinstance(exc, sqlite3.OperationalError) and is_transient_sqlite_lock(exc)
            # A retryable preparation refusal (a lost worker, a moved input) is
            # not a verdict on the appended bytes: they stay pending for the
            # retry instead of carrying a parse failure.
            preparation_retry = isinstance(exc, RetainedPreparationRetryableError)

            def record_failure(
                raw_id: str | None = raw_id,
                provider: Provider = provider,
                transient: bool = transient,
                preparation_retry: bool = preparation_retry,
                error: Exception = exc,
            ) -> None:
                if raw_id is None:
                    return
                with _open_archive_for_live_write(archive_root, cold_build=True) as archive:
                    if transient or preparation_retry or _append_storage_fault(error):
                        reset_transient_raw_parse_state(archive, raw_id, provider=provider)
                    else:
                        archive.mark_raw_parse_failed(
                            raw_id, provider=provider, error=_append_refusal_cause(error, raw_id)
                        )

            if raw_id is not None:
                try:
                    admit_stage_write("watcher.live_ingest.append.failure", record_failure)
                except BaseException as cleanup:
                    raise BaseExceptionGroup(
                        "append publication and failure settlement failed", [exc, cleanup]
                    ) from exc
            if transient:
                raise
            raise_if_storage_fault(exc, kinds=_append_fault_kinds(exc))
            logger.warning("live.watcher: archive append ingest failed for %s", plan.path, exc_info=True)
            failed.append(plan)
    return _AppendResult(
        succeeded=succeeded,
        failed=failed,
        deferred=deferred,
        worker_count=1,
        stage_timings_s=timings,
        session_ids_by_path=session_ids_by_path,
    )


def _append_refusal_cause(error: Exception, raw_id: str) -> BaseException:
    """The append raw's own parse failure behind a refusal of its chain member.

    A refusal naming this raw wraps the failure of its own bytes; recording
    that failure lets a decode refusal carry the same terminal evidence the
    full route records. A refusal of another member stays the chain's
    refusal.
    """
    if isinstance(error, CohortMembershipRefusalError) and error.raw_id == raw_id and error.__cause__ is not None:
        return error.__cause__
    return error


__all__ = ["ingest_append_plans", "reset_transient_raw_parse_state"]
