"""Append-only live-ingest persistence helpers."""

from __future__ import annotations

import sqlite3
import time
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

from polylogue.archive.revision_authority import RawRevisionEnvelope, RawRevisionKind, append_source_revision
from polylogue.core.enums import Provider
from polylogue.logging import get_logger
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
    from polylogue.sources.dispatch import parse_payload
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    _add_timing(timings, "append.imports", t0)

    t0 = time.perf_counter()
    succeeded: list[_AppendPlan] = []
    failed: list[_AppendPlan] = []
    acquired_at_ms = int(datetime.now(UTC).timestamp() * 1000)
    try:
        t0 = time.perf_counter()
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            _add_timing(timings, "append.archive_open", t0)
            for plan in plans:
                provider: Provider | None = None
                raw_id: str | None = None
                try:
                    provider = Provider.from_string(plan.source_name)
                    t0 = time.perf_counter()
                    raw_id = archive.write_raw_payload(
                        provider=provider,
                        payload=plan.payload,
                        source_path=str(plan.path),
                        source_index=-1,
                        acquired_at_ms=acquired_at_ms,
                    )
                    _add_timing(timings, "append.source_raw_write", t0)
                    t0 = time.perf_counter()
                    payloads = list(_iter_json_stream(BytesIO(plan.payload), plan.path.name))
                    _add_timing(timings, "append.json_stream", t0)
                    t0 = time.perf_counter()
                    sessions = parse_payload(
                        provider,
                        payloads,
                        plan.path.stem,
                        source_path=str(plan.path),
                    )
                    _add_timing(timings, "append.provider_parse", t0)
                    if not sessions:
                        archive.mark_raw_parse_failed(
                            raw_id,
                            provider=provider,
                            error=ValueError("parsed raw payload produced no sessions"),
                        )
                        failed.append(plan)
                        continue
                    for session in sessions:
                        t0 = time.perf_counter()
                        archive.write_parsed_for_retained_raw(
                            session,
                            raw_id=raw_id,
                            source_path=str(plan.path),
                            source_index=-1,
                            acquired_at_ms=acquired_at_ms,
                            stage_timings_s=timings,
                            finalize_raw_parse=False,
                        )
                        _add_timing(timings, "append.raw_and_index_write", t0)
                    if len(sessions) == 1:
                        session = sessions[0]
                        logical_source_key = f"{provider.value}:{session.provider_session_id}"
                        parent = archive.raw_append_revision_parent(
                            logical_source_key,
                            plan.start_offset,
                            plan.cursor_fingerprint,
                        )
                        if parent is not None:
                            predecessor_raw_id, baseline_raw_id, generation = parent
                            assert plan.cursor_fingerprint is not None
                            archive.bind_raw_revision(
                                raw_id,
                                RawRevisionEnvelope(
                                    logical_source_key=logical_source_key,
                                    kind=RawRevisionKind.APPEND,
                                    source_revision=append_source_revision(plan.cursor_fingerprint, plan.payload_hash),
                                    acquisition_generation=generation,
                                    predecessor_raw_id=predecessor_raw_id,
                                    baseline_raw_id=baseline_raw_id,
                                    append_start_offset=plan.start_offset,
                                    append_end_offset=plan.last_complete_newline,
                                ),
                            )
                    archive.mark_raw_parse_succeeded(raw_id, provider=provider)
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
    return _AppendResult(succeeded=succeeded, failed=failed, worker_count=1, stage_timings_s=timings)


__all__ = ["ingest_append_plans", "reset_transient_raw_parse_state"]
