"""In-process live batch convergence for daemon source ingestion."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from polylogue.config import Source
from polylogue.logging import get_logger
from polylogue.sources.live.cursor import CursorStore

if TYPE_CHECKING:
    from polylogue.api import Polylogue

logger = get_logger(__name__)


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
    cursor_fingerprint_read_bytes: int
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
            "cursor_fingerprint_read_bytes": self.cursor_fingerprint_read_bytes,
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
    ) -> None:
        self._polylogue = polylogue
        self._sources = tuple(sources)
        self._cursor = cursor
        self._parser_fingerprint = parser_fingerprint
        self._converger = converger
        self._stop_requested = stop_requested or (lambda: False)

    async def ingest_files(
        self,
        paths: list[Path],
        *,
        queued_file_count: int | None = None,
        skipped_file_count: int = 0,
        emit_event: bool = True,
    ) -> LiveBatchMetrics:
        """Ingest files in batch, run post-ingest convergence, and return metrics."""
        from polylogue.daemon.events import emit_daemon_event

        batch_started = time.perf_counter()
        db_bytes_before = _path_size(self._cursor._db_path) + _path_size(self._cursor._db_path.with_suffix(".db-wal"))
        input_bytes = sum(_path_size(path) for path in paths)
        cursor_fingerprint_read_bytes = 0
        parse_time_s = 0.0
        convergence_time_s = 0.0
        stage_timings: dict[str, float] = {}
        failed_paths: list[str] = []

        by_source: dict[str, list[Path]] = {}
        for path in paths:
            by_source.setdefault(self._source_name_for(path), []).append(path)

        succeeded_paths: set[Path] = set()
        for source_name, source_paths in by_source.items():
            if self._stop_requested():
                break
            sources = [Source(name=f"{source_name}:{path.parent.name}", path=path) for path in source_paths]
            t0 = time.perf_counter()
            try:
                await self._polylogue.parse_sources(sources=sources, download_assets=False)
            except Exception as exc:
                logger.warning("live.watcher: batch failed for %s: %s", source_name, exc)
                for path in source_paths:
                    failed_paths.append(str(path))
                    cursor_fingerprint_read_bytes += self._record_failed_cursor(path)
                continue
            parse_elapsed = time.perf_counter() - t0
            parse_time_s += parse_elapsed
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
                hint_path = next(iter(succeeded_paths))
                t0 = time.perf_counter()
                state = self._converger.converge_file(hint_path)  # type: ignore[attr-defined]
                convergence_time_s = time.perf_counter() - t0
                stage_timings = dict(getattr(state, "stage_times", {}))
            except Exception as exc:
                logger.warning("live.watcher: post-ingest converge failed: %s", exc)

        for path in succeeded_paths:
            try:
                stat = path.stat()
                fp, last_nl = fingerprint_file(path)
            except FileNotFoundError:
                continue
            cursor_fingerprint_read_bytes += stat.st_size
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
            source_group_count=len(by_source),
            input_bytes=input_bytes,
            cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
            archive_bytes_before=db_bytes_before,
            archive_bytes_after=db_bytes_after,
            archive_write_bytes_delta=max(0, db_bytes_after - db_bytes_before),
            parse_time_s=round(parse_time_s, 6),
            convergence_time_s=round(convergence_time_s, 6),
            total_time_s=round(time.perf_counter() - batch_started, 6),
            stage_timings_s={name: round(elapsed, 6) for name, elapsed in stage_timings.items()},
            failed_paths=failed_paths,
        )
        if emit_event:
            emit_daemon_event("ingestion_batch", payload=metrics.to_payload())
        return metrics

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


def fingerprint_file(path: Path) -> tuple[str, int]:
    import hashlib

    content = path.read_bytes()
    newline_at = content.rfind(b"\n")
    last_complete_newline = 0 if newline_at < 0 else newline_at + 1
    return hashlib.sha256(content).hexdigest(), last_complete_newline


def _path_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


__all__ = ["LiveBatchMetrics", "LiveBatchProcessor", "fingerprint_file"]
