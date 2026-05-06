"""In-process live batch convergence for daemon source ingestion."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from json import dumps as json_dumps
from json import loads as json_loads
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from polylogue.config import Source
from polylogue.core.metrics import read_current_rss_mb, read_peak_rss_children_mb, read_peak_rss_self_mb
from polylogue.core.provider_identity import canonical_acquisition_provider
from polylogue.logging import get_logger
from polylogue.paths import blob_store_root
from polylogue.pipeline.services.ingest_batch._core import _process_ingest_batch_sync
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.runtime import RawConversationRecord
from polylogue.types import Provider

if TYPE_CHECKING:
    from polylogue.api import Polylogue

logger = get_logger(__name__)
LiveBatchEventEmitter = Callable[[str, dict[str, object]], None]


class LiveSourceRoot(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def root(self) -> Path: ...


@dataclass(frozen=True, slots=True)
class LiveBatchMetrics:
    """Observable counters and timings for one live ingest batch."""

    queued_file_count: int
    needed_file_count: int
    skipped_file_count: int
    succeeded_file_count: int
    failed_file_count: int
    source_group_count: int
    input_bytes: int
    source_payload_read_bytes: int
    cursor_fingerprint_read_bytes: int
    append_file_count: int
    full_file_count: int
    archive_bytes_before: int
    archive_bytes_after: int
    archive_write_bytes_delta: int
    parse_time_s: float
    convergence_time_s: float
    total_time_s: float
    stage_timings_s: dict[str, float] = field(default_factory=dict)
    failed_paths: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, object]:
        return {
            "queued_file_count": self.queued_file_count,
            "needed_file_count": self.needed_file_count,
            "skipped_file_count": self.skipped_file_count,
            "succeeded_file_count": self.succeeded_file_count,
            "failed_file_count": self.failed_file_count,
            "source_group_count": self.source_group_count,
            "input_bytes": self.input_bytes,
            "source_payload_read_bytes": self.source_payload_read_bytes,
            "cursor_fingerprint_read_bytes": self.cursor_fingerprint_read_bytes,
            "append_file_count": self.append_file_count,
            "full_file_count": self.full_file_count,
            "archive_bytes_before": self.archive_bytes_before,
            "archive_bytes_after": self.archive_bytes_after,
            "archive_write_bytes_delta": self.archive_write_bytes_delta,
            "parse_time_s": self.parse_time_s,
            "convergence_time_s": self.convergence_time_s,
            "total_time_s": self.total_time_s,
            "stage_timings_s": self.stage_timings_s,
            "failed_paths": self.failed_paths,
        }


class LiveBatchProcessor:
    """Run the daemon live ingest batch path without filesystem watching."""

    def __init__(
        self,
        polylogue: Polylogue,
        sources: Iterable[LiveSourceRoot],
        *,
        cursor: CursorStore,
        parser_fingerprint: str | Callable[[], str],
        converger: object | None = None,
        stop_requested: Callable[[], bool] | None = None,
        event_emitter: LiveBatchEventEmitter | None = None,
    ) -> None:
        self._polylogue = polylogue
        self._sources = tuple(sources)
        self._cursor = cursor
        self._parser_fingerprint = parser_fingerprint
        self._converger = converger
        self._stop_requested = stop_requested or (lambda: False)
        self._event_emitter = event_emitter

    async def ingest_files(
        self,
        paths: list[Path],
        *,
        queued_file_count: int | None = None,
        skipped_file_count: int = 0,
        emit_event: bool = True,
    ) -> LiveBatchMetrics:
        """Ingest files in batch, run post-ingest convergence, and return metrics."""
        batch_started = time.perf_counter()
        db_bytes_before = _path_size(self._cursor._db_path) + _path_size(self._cursor._db_path.with_suffix(".db-wal"))
        input_bytes = sum(_path_size(path) for path in paths)
        attempt_id = self._cursor.begin_ingest_attempt(
            paths=paths,
            input_bytes=input_bytes,
            queued_file_count=queued_file_count if queued_file_count is not None else len(paths),
        )
        source_payload_read_bytes = 0
        cursor_fingerprint_read_bytes = 0
        parse_time_s = 0.0
        convergence_time_s = 0.0
        stage_timings: dict[str, float] = {}
        failed_paths: list[str] = []
        succeeded_paths: set[Path] = set()

        append_plans: list[_AppendPlan] = []
        full_paths: list[Path] = []
        for path in paths:
            append_plan = self._append_plan(path) if self._can_ingest_appends_directly() else None
            if append_plan is None:
                full_paths.append(path)
            else:
                append_plans.append(append_plan)
                source_payload_read_bytes += append_plan.bytes_read

        if append_plans:
            self._record_attempt_progress(
                attempt_id,
                phase="append_parse",
                succeeded_file_count=len(succeeded_paths),
                failed_file_count=len(failed_paths),
                source_payload_read_bytes=source_payload_read_bytes,
                cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
                parse_time_s=parse_time_s,
                current_source=append_plans[0].source_name,
                current_path=append_plans[0].path,
            )
            t0 = time.perf_counter()
            append_result = self._ingest_append_plans(append_plans)
            parse_time_s += time.perf_counter() - t0
            for plan in append_result.succeeded:
                succeeded_paths.add(plan.path)
                self._record_append_cursor(plan)
            for plan in append_result.failed:
                failed_paths.append(str(plan.path))
                cursor_fingerprint_read_bytes += self._record_failed_cursor(plan.path)

        by_source: dict[str, list[Path]] = {}
        for path in full_paths:
            by_source.setdefault(self._source_name_for(path), []).append(path)

        for source_name, source_paths in by_source.items():
            if self._stop_requested():
                break
            sources = [Source(name=f"{source_name}:{path.parent.name}", path=path) for path in source_paths]
            t0 = time.perf_counter()
            try:
                self._record_attempt_progress(
                    attempt_id,
                    phase="full_parse",
                    succeeded_file_count=len(succeeded_paths),
                    failed_file_count=len(failed_paths),
                    source_payload_read_bytes=source_payload_read_bytes,
                    cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
                    parse_time_s=parse_time_s,
                    current_source=source_name,
                    current_path=source_paths[0] if source_paths else None,
                )
                await self._polylogue.parse_sources(sources=sources, download_assets=False)
            except Exception as exc:
                logger.warning("live.watcher: batch failed for %s: %s", source_name, exc)
                for path in source_paths:
                    failed_paths.append(str(path))
                    cursor_fingerprint_read_bytes += self._record_failed_cursor(path)
                self._record_attempt_progress(
                    attempt_id,
                    phase="full_parse_failed",
                    succeeded_file_count=len(succeeded_paths),
                    failed_file_count=len(failed_paths),
                    source_payload_read_bytes=source_payload_read_bytes,
                    cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
                    parse_time_s=parse_time_s,
                    current_source=source_name,
                    current_path=source_paths[0] if source_paths else None,
                    error=str(exc),
                )
                continue
            parse_elapsed = time.perf_counter() - t0
            parse_time_s += parse_elapsed
            source_payload_read_bytes += sum(_path_size(path) for path in source_paths)
            succeeded_paths.update(source_paths)
            logger.info(
                "live.watcher: batch ingested %s — %d in %.1fs (%.1f/s)",
                source_name,
                len(source_paths),
                parse_elapsed,
                len(source_paths) / max(parse_elapsed, 0.01),
            )

        if self._converger is not None and succeeded_paths:
            try:
                self._record_attempt_progress(
                    attempt_id,
                    phase="convergence",
                    succeeded_file_count=len(succeeded_paths),
                    failed_file_count=len(failed_paths),
                    source_payload_read_bytes=source_payload_read_bytes,
                    cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
                    parse_time_s=parse_time_s,
                    current_path=next(iter(sorted(succeeded_paths))),
                )
                t0 = time.perf_counter()
                converge_batch = getattr(self._converger, "converge_batch", None)
                if callable(converge_batch):
                    _states, batch_stage_timings = converge_batch(sorted(succeeded_paths))
                    stage_timings.update(
                        {stage_name: float(elapsed) for stage_name, elapsed in batch_stage_timings.items()}
                    )
                else:
                    for path in sorted(succeeded_paths):
                        invalidate = getattr(self._converger, "invalidate_file", None)
                        if callable(invalidate):
                            invalidate(path)
                        state = self._converger.converge_file(path)  # type: ignore[attr-defined]
                        for stage_name, elapsed in getattr(state, "last_stage_times", {}).items():
                            stage_timings[stage_name] = stage_timings.get(stage_name, 0.0) + float(elapsed)
                convergence_time_s = time.perf_counter() - t0
            except Exception as exc:
                logger.warning("live.watcher: post-ingest converge failed: %s", exc)
                self._record_attempt_progress(
                    attempt_id,
                    phase="convergence_failed",
                    succeeded_file_count=len(succeeded_paths),
                    failed_file_count=len(failed_paths),
                    source_payload_read_bytes=source_payload_read_bytes,
                    cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
                    parse_time_s=parse_time_s,
                    convergence_time_s=convergence_time_s,
                    error=str(exc),
                )

        self._record_attempt_progress(
            attempt_id,
            phase="cursor_update",
            succeeded_file_count=len(succeeded_paths),
            failed_file_count=len(failed_paths),
            source_payload_read_bytes=source_payload_read_bytes,
            cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
            parse_time_s=parse_time_s,
            convergence_time_s=convergence_time_s,
        )
        for path in succeeded_paths - {plan.path for plan in append_plans}:
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            fp, last_nl, bytes_read = self._cursor_state_after_full_ingest(path, stat.st_size)
            cursor_fingerprint_read_bytes += bytes_read
            self._cursor.set(
                path,
                stat.st_size,
                byte_offset=last_nl,
                last_complete_newline=last_nl,
                parser_fingerprint=self._current_parser_fingerprint(),
                content_fingerprint=fp,
                source_name=self._source_name_for(path),
                st_dev=stat.st_dev,
                st_ino=stat.st_ino,
                mtime_ns=stat.st_mtime_ns,
            )
            self._cursor.reset_failures(path)

        db_bytes_after = _path_size(self._cursor._db_path) + _path_size(self._cursor._db_path.with_suffix(".db-wal"))
        metrics = LiveBatchMetrics(
            queued_file_count=queued_file_count if queued_file_count is not None else len(paths),
            needed_file_count=len(paths),
            skipped_file_count=skipped_file_count,
            succeeded_file_count=len(succeeded_paths),
            failed_file_count=len(failed_paths),
            source_group_count=len({self._source_name_for(path) for path in paths}),
            input_bytes=input_bytes,
            source_payload_read_bytes=source_payload_read_bytes,
            cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
            append_file_count=len(append_plans),
            full_file_count=len(full_paths),
            archive_bytes_before=db_bytes_before,
            archive_bytes_after=db_bytes_after,
            archive_write_bytes_delta=max(0, db_bytes_after - db_bytes_before),
            parse_time_s=round(parse_time_s, 6),
            convergence_time_s=round(convergence_time_s, 6),
            total_time_s=round(time.perf_counter() - batch_started, 6),
            stage_timings_s={name: round(elapsed, 6) for name, elapsed in stage_timings.items()},
            failed_paths=failed_paths,
        )
        if emit_event and self._event_emitter is not None:
            self._event_emitter("ingestion_batch", metrics.to_payload())
        self._record_attempt_progress(
            attempt_id,
            phase="completed",
            status="completed",
            succeeded_file_count=len(succeeded_paths),
            failed_file_count=len(failed_paths),
            source_payload_read_bytes=source_payload_read_bytes,
            cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
            parse_time_s=parse_time_s,
            convergence_time_s=convergence_time_s,
        )
        self._cursor.finish_ingest_attempt(
            attempt_id,
            status="completed" if not failed_paths else "completed_with_failures",
            phase="completed",
            error="; ".join(failed_paths[:3]) if failed_paths else None,
        )
        return metrics

    def _record_attempt_progress(
        self,
        attempt_id: str,
        *,
        phase: str,
        status: str = "running",
        succeeded_file_count: int,
        failed_file_count: int,
        source_payload_read_bytes: int,
        cursor_fingerprint_read_bytes: int,
        parse_time_s: float,
        convergence_time_s: float = 0.0,
        current_source: str | None = None,
        current_path: Path | None = None,
        error: str | None = None,
    ) -> None:
        self._cursor.update_ingest_attempt(
            attempt_id,
            phase=phase,
            status=status,
            succeeded_file_count=succeeded_file_count,
            failed_file_count=failed_file_count,
            source_payload_read_bytes=source_payload_read_bytes,
            cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
            parse_time_s=round(parse_time_s, 6),
            convergence_time_s=round(convergence_time_s, 6),
            current_source=current_source,
            current_path=current_path,
            error=error,
            rss_current_mb=read_current_rss_mb(),
            rss_peak_self_mb=read_peak_rss_self_mb(),
            rss_peak_children_mb=read_peak_rss_children_mb(),
        )

    def _record_failed_cursor(self, path: Path) -> int:
        try:
            stat = path.stat()
            fp, last_nl = fingerprint_file(path)
        except FileNotFoundError:
            self._cursor.mark_failed(path)
            return 0
        existing = self._cursor.get_record(path)
        self._cursor.set(
            path,
            stat.st_size,
            byte_offset=last_nl,
            last_complete_newline=last_nl,
            parser_fingerprint=self._current_parser_fingerprint(),
            content_fingerprint=fp,
            source_name=self._source_name_for(path),
            st_dev=stat.st_dev,
            st_ino=stat.st_ino,
            mtime_ns=stat.st_mtime_ns,
            failure_count=existing.failure_count if existing else 0,
            next_retry_at=existing.next_retry_at if existing else None,
            excluded=bool(existing.excluded) if existing else False,
        )
        self._cursor.mark_failed(path)
        return stat.st_size

    def _cursor_state_after_full_ingest(self, path: Path, byte_size: int) -> tuple[str, int, int]:
        raw_fingerprint = self._latest_raw_fingerprint(path)
        if raw_fingerprint is None:
            fp, last_nl = fingerprint_file(path)
            return fp, last_nl, byte_size
        last_nl, bytes_read = last_complete_newline_from_tail(path, byte_size)
        return raw_fingerprint, last_nl, bytes_read

    def _latest_raw_fingerprint(self, path: Path) -> str | None:
        try:
            with self._cursor._connect() as conn:
                row = conn.execute(
                    """
                    SELECT raw_id
                    FROM raw_conversations
                    WHERE source_path = ?
                      AND COALESCE(source_index, 0) >= 0
                    ORDER BY acquired_at DESC, raw_id DESC
                    LIMIT 1
                    """,
                    (str(path),),
                ).fetchone()
        except Exception:
            return None
        if row is None:
            return None
        raw_id = row[0]
        return raw_id if isinstance(raw_id, str) and raw_id else None

    def _current_parser_fingerprint(self) -> str:
        if callable(self._parser_fingerprint):
            return self._parser_fingerprint()
        return self._parser_fingerprint

    def _source_name_for(self, path: Path) -> str:
        resolved = path.resolve()
        for source in self._sources:
            try:
                if resolved.is_relative_to(source.root.resolve()):
                    return source.name
            except OSError:
                continue
        return path.parent.name

    def _can_ingest_appends_directly(self) -> bool:
        backend = getattr(self._polylogue, "backend", None)
        return isinstance(getattr(backend, "db_path", None), Path)

    def _append_plan(self, path: Path) -> _AppendPlan | None:
        cursor = self._cursor.get_record(path)
        if cursor is None or cursor.parser_fingerprint != self._current_parser_fingerprint():
            return None
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        if stat.st_size <= cursor.byte_offset:
            return None
        if cursor.st_dev is not None and cursor.st_dev != stat.st_dev:
            return None
        if cursor.st_ino is not None and cursor.st_ino != stat.st_ino:
            return None

        start_offset = max(cursor.byte_offset, 0)
        with path.open("rb") as handle:
            handle.seek(start_offset)
            payload = handle.read()
        newline_at = payload.rfind(b"\n")
        if newline_at < 0:
            return None
        complete_payload = payload[: newline_at + 1]
        if not complete_payload:
            return None
        append_payload = self._append_payload_for_provider(path, self._source_name_for(path), complete_payload)
        if append_payload is None:
            return None
        tail_hash = sha256(complete_payload).hexdigest()
        return _AppendPlan(
            path=path,
            source_name=self._source_name_for(path),
            start_offset=start_offset,
            last_complete_newline=start_offset + newline_at + 1,
            stat_size=stat.st_size,
            st_dev=stat.st_dev,
            st_ino=stat.st_ino,
            mtime_ns=stat.st_mtime_ns,
            payload=append_payload,
            payload_hash=tail_hash,
            cursor_fingerprint=cursor.content_fingerprint,
            bytes_read=len(payload),
        )

    def _append_payload_for_provider(self, path: Path, source_name: str, payload: bytes) -> bytes | None:
        provider = Provider.from_string(canonical_acquisition_provider(source_name, source_name=source_name))
        if provider is Provider.CODEX:
            identity = self._existing_provider_conversation_id(path)
            if identity is None:
                return None
            session_meta = json_dumps(
                {"type": "session_meta", "payload": {"id": identity}},
                separators=(",", ":"),
            ).encode()
            return session_meta + b"\n" + payload
        if provider is Provider.CLAUDE_CODE and not self._claude_code_tail_matches_existing_identity(path, payload):
            return None
        return payload

    def _existing_provider_conversation_id(self, path: Path) -> str | None:
        with self._cursor._connect() as conn:
            row = conn.execute(
                """
                SELECT c.provider_conversation_id
                FROM conversations AS c
                JOIN raw_conversations AS r ON r.raw_id = c.raw_id
                WHERE r.source_path = ?
                  AND COALESCE(r.source_index, 0) >= 0
                ORDER BY c.updated_at DESC, c.created_at DESC, c.conversation_id DESC
                LIMIT 1
                """,
                (str(path),),
            ).fetchone()
        if row is None:
            return None
        value = row[0]
        return value if isinstance(value, str) and value.strip() else None

    def _claude_code_tail_matches_existing_identity(self, path: Path, payload: bytes) -> bool:
        existing_id = self._existing_provider_conversation_id(path)
        if existing_id is None:
            return False
        session_ids: set[str] = set()
        for line in payload.splitlines():
            if not line.strip():
                continue
            try:
                record = json_loads(line)
            except ValueError:
                return False
            if not isinstance(record, dict):
                return False
            session_id = record.get("sessionId")
            if isinstance(session_id, str) and session_id.strip():
                session_ids.add(session_id)
        if not session_ids:
            return existing_id == path.stem
        return any(existing_id == session_id or existing_id.startswith(f"{session_id}:") for session_id in session_ids)

    def _ingest_append_plans(self, plans: list[_AppendPlan]) -> _AppendResult:
        if not plans:
            return _AppendResult(succeeded=[], failed=[])
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        blob_root = blob_store_root()
        blob_store = BlobStore(blob_root)
        raw_records: list[RawConversationRecord] = []
        raw_by_id: dict[str, _AppendPlan] = {}
        for plan in plans:
            raw_id, blob_size = blob_store.write_from_bytes(plan.payload)
            raw_records.append(
                RawConversationRecord(
                    raw_id=raw_id,
                    provider_name=canonical_acquisition_provider(plan.source_name, source_name=plan.source_name),
                    source_name=plan.source_name,
                    source_path=str(plan.path),
                    source_index=-1,
                    blob_size=blob_size,
                    acquired_at=datetime.now(UTC).isoformat(),
                    file_mtime=datetime.fromtimestamp(plan.mtime_ns / 1_000_000_000, UTC).isoformat(),
                )
            )
            raw_by_id[raw_id] = plan

        self._persist_append_raw_records(raw_records)
        try:
            summary = _process_ingest_batch_sync(
                raw_records,
                db_path=self._cursor._db_path,
                archive_root_str=str(archive_root),
                blob_root_str=str(blob_root),
                validation_mode=str(getattr(getattr(self._polylogue, "config", None), "validation_mode", "advisory")),
                ingest_workers=1,
                measure_ingest_result_size=False,
            )
        except Exception as exc:
            logger.warning("live.watcher: append ingest failed: %s", exc)
            return _AppendResult(succeeded=[], failed=plans)

        failed = [raw_by_id[raw_id] for raw_id in summary.failed_raw_ids if raw_id in raw_by_id]
        failed_paths = {plan.path for plan in failed}
        succeeded = [plan for plan in plans if plan.path not in failed_paths and summary.parse_failures == 0]
        if summary.parse_failures and not failed:
            return _AppendResult(succeeded=[], failed=plans)
        return _AppendResult(succeeded=succeeded, failed=failed)

    def _persist_append_raw_records(self, records: list[RawConversationRecord]) -> None:
        if not records:
            return
        with self._cursor._connect() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO raw_conversations (
                    raw_id,
                    provider_name,
                    payload_provider,
                    source_name,
                    source_path,
                    source_index,
                    blob_size,
                    acquired_at,
                    file_mtime
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.raw_id,
                        record.provider_name,
                        record.payload_provider,
                        record.source_name,
                        record.source_path,
                        record.source_index,
                        record.blob_size,
                        record.acquired_at,
                        record.file_mtime,
                    )
                    for record in records
                ],
            )
            conn.commit()

    def _record_append_cursor(self, plan: _AppendPlan) -> None:
        content_fingerprint = sha256(f"{plan.cursor_fingerprint or ''}\0{plan.payload_hash}".encode()).hexdigest()
        self._cursor.set(
            plan.path,
            plan.stat_size,
            byte_offset=plan.last_complete_newline,
            last_complete_newline=plan.last_complete_newline,
            parser_fingerprint=self._current_parser_fingerprint(),
            content_fingerprint=content_fingerprint,
            source_name=plan.source_name,
            st_dev=plan.st_dev,
            st_ino=plan.st_ino,
            mtime_ns=plan.mtime_ns,
        )
        self._cursor.reset_failures(plan.path)


@dataclass(frozen=True, slots=True)
class _AppendPlan:
    path: Path
    source_name: str
    start_offset: int
    last_complete_newline: int
    stat_size: int
    st_dev: int
    st_ino: int
    mtime_ns: int
    payload: bytes
    payload_hash: str
    cursor_fingerprint: str | None
    bytes_read: int


@dataclass(frozen=True, slots=True)
class _AppendResult:
    succeeded: list[_AppendPlan]
    failed: list[_AppendPlan]


def fingerprint_file(path: Path) -> tuple[str, int]:
    import hashlib

    content = path.read_bytes()
    newline_at = content.rfind(b"\n")
    last_complete_newline = 0 if newline_at < 0 else newline_at + 1
    return hashlib.sha256(content).hexdigest(), last_complete_newline


def last_complete_newline_from_tail(path: Path, byte_size: int, *, chunk_size: int = 64 * 1024) -> tuple[int, int]:
    if byte_size <= 0:
        return 0, 0
    bytes_read = 0
    end = byte_size
    with path.open("rb") as handle:
        while end > 0:
            start = max(0, end - chunk_size)
            handle.seek(start)
            chunk = handle.read(end - start)
            bytes_read += len(chunk)
            newline_at = chunk.rfind(b"\n")
            if newline_at >= 0:
                return start + newline_at + 1, bytes_read
            end = start
    return 0, bytes_read


def _path_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


__all__ = [
    "LiveBatchEventEmitter",
    "LiveBatchMetrics",
    "LiveBatchProcessor",
    "fingerprint_file",
    "last_complete_newline_from_tail",
]
