"""Append-only live-ingest persistence helpers."""

from __future__ import annotations

import sqlite3
import time
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

from polylogue.archive.artifact_taxonomy import classify_artifact_path
from polylogue.archive.revision_authority import (
    RawRevisionAuthority,
    RawRevisionEnvelope,
    RawRevisionKind,
    append_source_revision,
)
from polylogue.core.degraded import degraded_reason
from polylogue.core.enums import Provider
from polylogue.logging import get_logger
from polylogue.sources.live.archive_open import _open_archive_for_live_write
from polylogue.sources.live.batch_support import _AppendPlan, _AppendResult
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.sqlite_locking import is_transient_sqlite_lock
from polylogue.storage.raw.models import RawSessionStateUpdate
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

logger = get_logger(__name__)


def _add_timing(timings: dict[str, float], name: str, started_at: float) -> None:
    timings[name] = timings.get(name, 0.0) + (time.perf_counter() - started_at)


class _AppendIngestOwner(Protocol):
    _cursor: CursorStore
    _polylogue: Any


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
    index_db = archive_root / "index.db"
    source_db = archive_root / "source.db"
    if not index_db.exists() or not source_db.exists():
        t0 = time.perf_counter()
        initialize_active_archive_root(archive_root)
        _add_timing(timings, "append.archive_init", t0)

    t0 = time.perf_counter()
    from polylogue.sources.decoders import _iter_json_stream
    from polylogue.sources.dispatch import parse_payload, require_positive_conversational_evidence
    from polylogue.sources.revision_backfill import (
        _is_declared_non_session_artifact,
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
                try:
                    provider = Provider.from_string(plan.source_name)
                    artifact_classification = classify_artifact_path(
                        str(plan.path),
                        provider=provider,
                    )
                    if artifact_classification is not None and not artifact_classification.parse_as_session:
                        artifact_result = archive.admit_raw_artifact_payload(
                            provider=provider,
                            payload=plan.payload,
                            source_path=str(plan.path),
                            source_index=-1,
                            acquired_at_ms=acquired_at_ms,
                            classification=artifact_classification,
                        )
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
                    )
                    _add_timing(timings, "append.source_raw_write", t0)
                    degraded = degraded_reason()
                    if degraded is not None and degraded.derived_only:
                        # polylogue-gbs02: the derived tier (index.db/
                        # embeddings.db) is behind the running code, but
                        # source.db just durably got this append range --
                        # stop here, before parsing or touching the stale
                        # derived tier. Treat as succeeded (not deferred):
                        # the acquire itself genuinely completed, so the
                        # cursor should advance normally rather than
                        # re-reading the same bytes on every tick. The raw
                        # row sits with parsed_at_ms=NULL exactly like any
                        # other not-yet-materialized raw, and ordinary
                        # convergence picks it up once the derived tier is
                        # current again -- no special resolution needed.
                        succeeded.append(plan)
                        continue
                    t0 = time.perf_counter()
                    payloads = list(_iter_json_stream(BytesIO(plan.payload), plan.path.name))
                    _add_timing(timings, "append.json_stream", t0)
                    # polylogue-xwkh: this was the third, ungated chokepoint.
                    # Live daemon ingest (ingest_worker.py) and rebuild replay
                    # (revision_backfill.py's _parse_one/_parse_stream) both
                    # already refuse to session-parse an OriginSpec-declared
                    # "fact" artifact (workflow_run_snapshot / workflow_journal
                    # / agent_sidecar_meta / adopt_manifest) or non-
                    # conversational content with no matching path rule, via
                    # this same classify_artifact-backed check. This path had
                    # none: parse_payload ran unconditionally below, so a
                    # growing fact-artifact file (e.g. a workflow
                    # journal.jsonl whose ops.db cursor was lost and
                    # resynthesized from a durable 'full' baseline in
                    # source.db -- see batch.py's
                    # _resynthesize_cursor_from_source) reached parse_payload
                    # ungated. Empirically it was NOT reaching
                    # write_parsed_session_to_archive: parse_retained_raw_sessions
                    # (used a few lines below, for replay) applies this same
                    # gate and returns no sessions, which trips the
                    # 'did not replay to exactly one session' RuntimeError and
                    # the whole plan fails -- but only after a wasted raw
                    # write, a wasted parse, and a crash-shaped failure log,
                    # repeated on every single observation of the file
                    # forever, since a failed append never advances its
                    # cursor. Refusing here, before any of that work, closes
                    # the third chokepoint on the same terms as the other two.
                    if _is_declared_non_session_artifact(provider, str(plan.path), sample=payloads[:64]):
                        archive.mark_raw_parse_failed(
                            raw_id,
                            provider=provider,
                            error=ValueError("append payload is a declared non-session artifact"),
                        )
                        failed.append(plan)
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
                    sessions = require_positive_conversational_evidence(
                        parse_payload(
                            provider,
                            payloads,
                            plan.native_id_hint or plan.path.stem,
                            source_path=str(plan.path),
                        ),
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
                    logical_source_key = f"{provider.value}:{session.provider_session_id}"
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
