"""Append-only live-ingest persistence helpers."""

from __future__ import annotations

import sqlite3
import time
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

from polylogue.archive.artifact_taxonomy import classify_artifact, classify_artifact_path
from polylogue.archive.raw_payload.decode import _sample_jsonl_payload_with_detail, jsonl_session_artifact
from polylogue.archive.revision_authority import (
    RawRevisionAuthority,
    RawRevisionEnvelope,
    RawRevisionKind,
    append_source_revision,
)
from polylogue.core.degraded import degraded_reason
from polylogue.core.enums import Provider
from polylogue.logging import get_logger
from polylogue.sources.live.archive_open import _open_archive_for_live_write, _source_tier_acquisition_required
from polylogue.sources.live.batch_support import _AppendPlan, _AppendResult
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.sqlite_locking import is_transient_sqlite_lock
from polylogue.storage.archive_identity import resolve_active_index_path
from polylogue.storage.raw.models import RawSessionStateUpdate
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.raw_admission import RawAdmissionArm

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
    provider: Provider,
    session_id: str,
    plan: _AppendPlan,
) -> tuple[str, RawRevisionAuthority]:
    """Persist an APPEND envelope from the append plan's durable identity."""
    if plan.cursor_fingerprint is None:
        raise ValueError("append payload did not prove cursor identity")
    logical_source_key = f"{provider.value}:{session_id}"
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


def ingest_append_plans(owner: _AppendIngestOwner, plans: list[_AppendPlan]) -> _AppendResult:
    """Persist and parse one bounded group of append plans."""
    if not plans:
        return _AppendResult(succeeded=[], failed=[], worker_count=0)
    archive_root = Path(getattr(owner._polylogue, "archive_root", owner._cursor._db_path.parent))
    return _ingest_append_plans_archive(owner, plans, archive_root)


def _ingest_append_plans_archive(
    owner: _AppendIngestOwner,
    plans: list[_AppendPlan],
    archive_root: Path,
) -> _AppendResult:
    timings: dict[str, float] = {}
    source_db = archive_root / "source.db"
    source_only = _source_tier_acquisition_required()
    archive_missing = not source_db.exists()
    if not source_only:
        archive_missing = archive_missing or not resolve_active_index_path(archive_root).exists()
    if archive_missing:
        t0 = time.perf_counter()
        initialize_active_archive_root(archive_root)
        _add_timing(timings, "append.archive_init", t0)

    t0 = time.perf_counter()
    from polylogue.sources.decoders import _iter_json_stream
    from polylogue.sources.dispatch import (
        STREAM_RECORD_PROVIDERS,
        parse_payload,
        parse_stream_payload,
        require_positive_conversational_evidence,
    )
    from polylogue.sources.revision_backfill import (
        _declared_non_session_artifact_classification,
        parse_retained_raw_sessions,
    )

    _add_timing(timings, "append.imports", t0)

    t0 = time.perf_counter()
    succeeded: list[_AppendPlan] = []
    failed: list[_AppendPlan] = []
    deferred: list[_AppendPlan] = []
    session_ids_by_path: dict[Path, str] = {}
    acquired_at_ms = int(datetime.now(UTC).timestamp() * 1000)
    try:
        t0 = time.perf_counter()
        with _open_archive_for_live_write(archive_root) as archive:
            _add_timing(timings, "append.archive_open", t0)
            for plan in plans:
                provider: Provider | None = None
                raw_id: str | None = None
                session_artifact = None
                try:
                    provider = Provider.from_string(plan.source_name)
                    path_artifact = classify_artifact_path(
                        str(plan.path),
                        provider=provider,
                    )
                    json_stream_started = time.perf_counter()
                    try:
                        payloads, _malformed_lines, _malformed_detail = _sample_jsonl_payload_with_detail(
                            plan.payload,
                            max_samples=64,
                            jsonl_dict_only=True,
                            scan_full=False,
                        )
                        session_artifact = jsonl_session_artifact(
                            plan.payload,
                            provider=provider,
                            jsonl_dict_only=True,
                        )
                    except Exception:
                        # Preserve the pre-parse raw capture for malformed input;
                        # the normal parser path below records the typed failure.
                        payloads = None
                    _add_timing(timings, "append.json_stream", json_stream_started)
                    if payloads is not None:
                        decoded_artifact = session_artifact or classify_artifact(payloads, provider=provider)
                        classification = (
                            _declared_non_session_artifact_classification(
                                provider,
                                str(plan.path),
                                sample=payloads[:64],
                            )
                            if session_artifact is None and not decoded_artifact.parse_as_session
                            else None
                        )
                        if classification is not None:
                            artifact_result = archive.admit_raw_artifact_payload(
                                provider=provider,
                                payload=plan.payload,
                                source_path=str(plan.path),
                                source_index=-1,
                                acquired_at_ms=acquired_at_ms,
                                classification=classification,
                            )
                            if artifact_result.arm is not RawAdmissionArm.ARTIFACT:
                                raise RuntimeError(f"unexpected append artifact admission arm: {artifact_result.arm!r}")
                            succeeded.append(plan)
                            continue
                    elif path_artifact is not None and not path_artifact.parse_as_session:
                        artifact_result = archive.admit_raw_artifact_payload(
                            provider=provider,
                            payload=plan.payload,
                            source_path=str(plan.path),
                            source_index=-1,
                            acquired_at_ms=acquired_at_ms,
                            classification=path_artifact,
                        )
                        if artifact_result.arm is not RawAdmissionArm.ARTIFACT:
                            raise RuntimeError(f"unexpected append artifact admission arm: {artifact_result.arm!r}")
                        raw_id = artifact_result.raw_id
                        succeeded.append(plan)
                        continue
                    t0 = time.perf_counter()
                    raw_id = archive.write_raw_payload(
                        provider=provider,
                        payload=plan.payload,
                        source_path=str(plan.path),
                        source_index=-1,
                        acquired_at_ms=acquired_at_ms,
                        # polylogue-u19l: persist the resolved provider
                        # session identity as sidecar metadata instead of
                        # splicing a synthetic session_meta record into the
                        # hashed/stored payload (batch.py's
                        # _append_payload_for_provider), so the stored blob
                        # stays a literal slice of the live file.
                        native_id=plan.native_id_hint,
                        post_parse=True,
                    )
                    _add_timing(timings, "append.source_raw_write", t0)
                    degraded = degraded_reason()
                    if degraded is not None and degraded.derived_only:
                        if plan.native_id_hint is None:
                            raise ValueError("source-only append has no durable session identity")
                        _bind_append_revision(
                            archive,
                            raw_id,
                            provider=provider,
                            session_id=plan.native_id_hint,
                            plan=plan,
                        )
                        # Keep the byte-contiguous append chain replayable
                        # even when this delta is runtime-only or cannot be
                        # parsed while the derived tier is unavailable.
                        succeeded.append(plan)
                        continue
                    t0 = time.perf_counter()
                    # polylogue-u19l: prefer the resolved provider session
                    # identity over the bare filename stem. For Codex this is
                    # the ONLY thing that made the synthetic session_meta
                    # header (formerly spliced into plan.payload) necessary
                    # in the first place -- the parser falls back to
                    # ``fallback_id`` exactly when its own record stream
                    # carries no session_meta of its own, which is always
                    # true for an append delta.
                    if provider in STREAM_RECORD_PROVIDERS:
                        parsed_sessions = parse_stream_payload(
                            provider,
                            _iter_json_stream(BytesIO(plan.payload), plan.path.name),
                            plan.native_id_hint or plan.path.stem,
                            source_path=str(plan.path),
                        )
                    else:
                        parsed_sessions = parse_payload(
                            provider,
                            payloads,
                            plan.native_id_hint or plan.path.stem,
                            source_path=str(plan.path),
                        )
                    sessions = require_positive_conversational_evidence(
                        parsed_sessions,
                        provider=provider,
                        source_path=str(plan.path),
                    )
                    _add_timing(timings, "append.provider_parse", t0)
                    if not sessions:
                        archive.mark_raw_parse_failed(
                            raw_id,
                            provider=provider,
                            error=ValueError(
                                "parsed raw payload produced no sessions with positive conversational evidence"
                            ),
                        )
                        failed.append(plan)
                        continue
                    if len(sessions) != 1 or plan.cursor_fingerprint is None:
                        archive.mark_raw_parse_failed(
                            raw_id,
                            provider=provider,
                            error=ValueError("append payload did not prove one session and cursor identity"),
                        )
                        failed.append(plan)
                        continue
                    session = sessions[0]
                    logical_source_key, authority = _bind_append_revision(
                        archive,
                        raw_id,
                        provider=provider,
                        session_id=session.provider_session_id,
                        plan=plan,
                    )
                    if authority is RawRevisionAuthority.QUARANTINED:
                        deferred.append(plan)
                        continue
                    # The append parent above is a durable byte-contiguous
                    # witness.  Once the prior full snapshot has been
                    # classified, its baseline and every accepted append are
                    # already represented in source-tier metadata.  Rebuild
                    # the replay plan from that metadata instead of reopening
                    # every retained historical full snapshot on each small
                    # append.  The classifier remains the conservative
                    # recovery path for a legacy/crash-interrupted cohort
                    # whose accepted metadata has not yet been established.
                    replay_plan = archive.raw_revision_replay_plan(logical_source_key)
                    if raw_id not in replay_plan.accepted_raw_ids:
                        replay_plan = archive.classify_raw_revision_cohort_for_live_watch(logical_source_key)
                    if raw_id not in replay_plan.accepted_raw_ids:
                        # A non-empty plan can still represent an older
                        # accepted chain while a newly observed full snapshot
                        # remains ambiguous.  Never acknowledge this append
                        # or advance its cursor until its own raw evidence is
                        # part of the accepted chain.
                        deferred.append(plan)
                        continue
                    parsed_by_raw_id: dict[str, Any] = {}
                    for replay_raw_id in replay_plan.accepted_raw_ids:
                        replay_sessions = parse_retained_raw_sessions(archive, replay_raw_id)
                        if len(replay_sessions) != 1:
                            raise RuntimeError(f"raw revision {replay_raw_id} did not replay to exactly one session")
                        parsed_by_raw_id[replay_raw_id] = replay_sessions[0]
                    t0 = time.perf_counter()
                    session_id, _applied_raw_ids = archive.apply_raw_revision_replay(
                        replay_plan,
                        parsed_by_raw_id,
                        acquired_at_ms=acquired_at_ms,
                        stage_timings_s=timings,
                        stage_timing_prefix="append",
                        # polylogue-de2a: this is the live watcher's hot
                        # per-append path -- called again for every tiny
                        # append a still-open session receives. Re-indexing
                        # (not re-parsing) every already-applied historical
                        # position on every call is what made the writer
                        # gate's hold time grow with the session's total
                        # accumulated append count instead of the size of
                        # just this append (see apply_raw_revision_replay's
                        # skip_already_applied docstring for the full
                        # evidence chain).
                        skip_already_applied=True,
                    )
                    _add_timing(timings, "append.raw_and_index_write", t0)
                    session_ids_by_path[plan.path] = session_id
                    succeeded.append(plan)
                except Exception as exc:
                    if isinstance(exc, sqlite3.OperationalError) and is_transient_sqlite_lock(exc):
                        # Contention is infrastructure state, not a poison
                        # payload. Let the watcher requeue without advancing
                        # the failure ledger toward exclusion.
                        if provider is not None and raw_id is not None:
                            reset_transient_raw_parse_state(archive, raw_id, provider=provider)
                        raise
                    if provider is not None and raw_id is not None:
                        archive.mark_raw_parse_failed(
                            raw_id,
                            provider=provider,
                            error=exc,
                        )
                    logger.warning("live.watcher: archive append ingest failed for %s", plan.path, exc_info=True)
                    failed.append(plan)
    except Exception as exc:
        if isinstance(exc, sqlite3.OperationalError) and is_transient_sqlite_lock(exc):
            raise
        logger.warning("live.watcher: archive append ingest failed: %s", exc)
        return _AppendResult(succeeded=[], failed=plans, worker_count=0, stage_timings_s=timings)
    return _AppendResult(
        succeeded=succeeded,
        failed=failed,
        deferred=deferred,
        worker_count=1,
        stage_timings_s=timings,
        session_ids_by_path=session_ids_by_path,
    )


__all__ = ["ingest_append_plans", "reset_transient_raw_parse_state"]
