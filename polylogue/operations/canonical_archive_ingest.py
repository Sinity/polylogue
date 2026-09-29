"""One-shot source discovery through the live acquisition and convergence owner."""

from __future__ import annotations

import asyncio
import contextvars
import threading
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.pipeline.services.parsing_models import ParseResult

_OWNED_ROOT: contextvars.ContextVar[tuple[Path, int, int] | None] = contextvars.ContextVar(
    "canonical_one_shot_archive_owner", default=None
)


class _ShutdownCoordinator(Protocol):
    async def shutdown(self, *, timeout: float) -> bool: ...


async def _wait_for_coordinator_idle(coordinator: _ShutdownCoordinator) -> None:
    """Do not let an archive owner unwind while its synchronous writer runs.

    A bounded shutdown returning ``False`` means the admitted writer still
    owns the archive. Callers may remove or replace the archive as soon as
    this function returns, so keep waiting until that writer has settled.
    """
    while not await coordinator.shutdown(timeout=30.0):
        await asyncio.sleep(0)


async def _ingest_selected_paths(
    paths: list[Path],
    ingest_pass: Callable[[list[Path]], Awaitable[Any]],
) -> tuple[Any, ...]:
    """Drain bounded unattempted paths, refusing any other incomplete result."""
    from polylogue.sources.live.metrics import REFUSED_NO_SESSIONS, REFUSED_UNATTEMPTED_TIME_BUDGET

    pending = list(dict.fromkeys(paths))
    receipts: list[Any] = []
    # A productive pass must settle at least one path. This permits the
    # slowest possible one-file progress while making a stuck route finite.
    for _ in range(max(1, len(pending))):
        offered = list(pending)
        metrics = await ingest_pass(offered)
        receipts.append(metrics)

        if metrics.failed_file_count or metrics.deferred_file_count:
            raise RuntimeError(
                "canonical ingestion did not settle every selected source file: "
                f"failed={metrics.failed_file_count}, deferred={metrics.deferred_file_count}"
            )

        succeeded = set(metrics.succeeded_paths)
        excluded = {Path(path): reason for path, reason in metrics.excluded_paths.items()}
        if len(succeeded) != metrics.succeeded_file_count:
            raise RuntimeError(
                "canonical ingestion returned incomplete succeeded-path evidence: "
                f"count={metrics.succeeded_file_count}, paths={len(succeeded)}"
            )
        if len(excluded) != metrics.excluded_file_count:
            raise RuntimeError(
                "canonical ingestion returned incomplete excluded-path evidence: "
                f"count={metrics.excluded_file_count}, paths={len(excluded)}"
            )

        offered_set = set(offered)
        if not succeeded <= offered_set or not set(excluded) <= offered_set:
            raise RuntimeError("canonical ingestion reported a path that was not offered in this pass")
        if succeeded & set(excluded):
            raise RuntimeError("canonical ingestion both succeeded and excluded a selected source file")
        accounted = succeeded | set(excluded)
        if accounted != offered_set:
            raise RuntimeError(
                "canonical ingestion left selected source files without terminal evidence: "
                f"unaccounted={len(offered_set - accounted)}"
            )

        # A source that parsed to no session is settled: its raw carries the
        # terminal outcome and its cursor advanced.
        excluded = {path: reason for path, reason in excluded.items() if reason != REFUSED_NO_SESSIONS}
        if not excluded:
            return tuple(receipts)

        settled_exclusions = {REFUSED_UNATTEMPTED_TIME_BUDGET, "durably_excised"}
        non_retryable = {path: reason for path, reason in excluded.items() if reason not in settled_exclusions}
        if non_retryable:
            reasons = ", ".join(sorted(set(non_retryable.values())))
            raise RuntimeError(
                "canonical ingestion excluded selected source files for a non-retryable reason: "
                f"count={len(non_retryable)}, reasons={reasons}"
            )

        if not succeeded:
            if all(reason == "durably_excised" for reason in excluded.values()):
                return tuple(receipts)
            raise RuntimeError(
                f"canonical ingestion made no progress on selected source files: unattempted={len(excluded)}"
            )
        pending = [path for path in offered if path in excluded]

    raise RuntimeError(
        "canonical ingestion exhausted its bounded passes with selected source files remaining: "
        f"unattempted={len(pending)}"
    )


@contextmanager
def scoped_one_shot_archive_owner(archive_root: Path) -> Iterator[None]:
    """Exclude daemon startup and other offline writers for a whole one-shot mutation.

    The daemon holds an exclusive flock on ``daemon.pid``. A shared flock here
    prevents it from starting between a residency check and a later write;
    ``OwnedArchiveLocation`` excludes other offline maintenance on the durable
    tier set. Re-entry is limited to the same task/thread so independent
    ingestion calls cannot borrow one another's lock.
    """
    from polylogue.maintenance.offline_guard import scoped_offline_archive_writer

    root = archive_root.expanduser().resolve()
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    owner_id = id(task) if task is not None else 0
    thread_id = threading.get_ident()
    held = _OWNED_ROOT.get()
    if held is not None:
        if held != (root, owner_id, thread_id):
            raise RuntimeError("one-shot archive ownership belongs to another root or execution")
        yield
        return

    with scoped_offline_archive_writer(root, owner_id="one-shot-canonical-ingest"):
        token = _OWNED_ROOT.set((root, owner_id, thread_id))
        try:
            yield
        finally:
            _OWNED_ROOT.reset(token)


async def ingest_sources_archive(
    archive_root: Path,
    sources: list[Source],
    *,
    parse_workers: int | None = None,
) -> ParseResult:
    """Offer declared files to the same cursor, raw, and convergence route as the daemon.

    This is for an isolated one-shot archive owner such as the synthetic demo
    builder. A resident daemon owns its archive; callers targeting one must use
    the daemon operation instead.
    """
    from polylogue import Polylogue
    from polylogue.archive.query.execution_control import QueryExecutionContext
    from polylogue.daemon.convergence import DaemonConverger
    from polylogue.daemon.convergence_stages import make_default_convergence_stages
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator
    from polylogue.maintenance.offline_guard import ArchiveWriterOwnershipError, resident_daemon_pid
    from polylogue.operations.operation_context import open_operation_read
    from polylogue.sources.live.batch import LiveBatchProcessor
    from polylogue.sources.live.cursor import CursorStore
    from polylogue.sources.live.parse_prefetch import LiveParseStage
    from polylogue.sources.live.watcher import _PARSER_FINGERPRINT, WatchSource
    from polylogue.sources.source_root_admission import refuse_non_capture_source_root
    from polylogue.sources.source_walk import _resolve_source_paths
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = archive_root.expanduser().resolve()

    task = asyncio.current_task()
    if _OWNED_ROOT.get() != (root, id(task), threading.get_ident()):
        raise RuntimeError("canonical one-shot ingestion requires scoped archive ownership")

    def require_offline_owner() -> None:
        pid = resident_daemon_pid(root)
        if pid is not None:
            raise ArchiveWriterOwnershipError(
                f"polylogued PID {pid} owns {root}; submit ingestion to that daemon",
                archive_root=root,
                resident_writer=f"polylogued PID {pid}",
            )

    require_offline_owner()
    paths: list[Path] = []
    watch_sources: list[WatchSource] = []
    for source in sources:
        if source.path is None:
            continue
        # Resolve once, then bind admission, provider path rules, traversal,
        # and acquired provenance to that physical coordinate. Keep the
        # caller's Source untouched so its declared spelling remains available
        # to the caller while relative paths gain a durable resolution base.
        resolved_source_path = source.path.expanduser().resolve()
        refuse_non_capture_source_root(resolved_source_path, destination=root)
        is_individual_file = resolved_source_path.is_file()
        is_antigravity_pb_file = (
            is_individual_file
            and Provider.from_string(source.name) is Provider.ANTIGRAVITY
            and resolved_source_path.suffix.lower() == ".pb"
        )
        if is_antigravity_pb_file:
            from polylogue.sources.source_parsing import _antigravity_source_root

            watch_root = _antigravity_source_root(resolved_source_path)
        else:
            watch_root = resolved_source_path if resolved_source_path.is_dir() else resolved_source_path.parent
        watch_sources.append(
            WatchSource(
                name=source.name,
                root=watch_root,
                exact_paths=frozenset({resolved_source_path}) if is_individual_file else None,
            )
        )
        paths.extend(_resolve_source_paths(Source(name=source.name, path=resolved_source_path)))
    paths = list(dict.fromkeys(paths))
    result = ParseResult()
    if not paths:
        return result

    coordinator = DaemonWriteCoordinator(archive_root=root)

    def initialize() -> None:
        require_offline_owner()
        with ArchiveStore.open_existing(root, read_only=False):
            pass

    await coordinator.run_sync("demo.ingest.initialize", initialize)
    archive = Polylogue(archive_root=root)
    parse_stage = LiveParseStage(
        max_workers=max(1, parse_workers) if parse_workers is not None else None,
        shard_directory=root / "parse-shards",
        use_processes=parse_workers != 1,
    )
    try:
        cursor = await coordinator.run_sync("demo.ingest.cursor", lambda: CursorStore(root / "index.db"))

        async def run_writer(actor: str, function: object, *args: object, **kwargs: object) -> object:
            require_offline_owner()
            return await coordinator.run_sync(actor, function, *args, **kwargs)  # type: ignore[arg-type]

        processor = LiveBatchProcessor(
            archive,
            watch_sources,
            cursor=cursor,
            parser_fingerprint=_PARSER_FINGERPRINT,
            converger=DaemonConverger(stages=make_default_convergence_stages(root / "index.db")),
            sync_runner=run_writer,
            parse_stage=parse_stage,
            read_snapshot=lambda root: open_operation_read(
                root,
                execution_context=QueryExecutionContext.create(
                    query_text="live-existing-session-preparation", workload_class="scan"
                ),
            ),
        )
        metrics_by_pass = await _ingest_selected_paths(
            paths,
            lambda offered: processor.ingest_files(offered, emit_event=False),
        )
    finally:
        # An admitted writer may still be consuming its parse-stage carrier
        # (``pop_path``) after the caller was cancelled. Let it settle before
        # the stage terminates workers and discards prepared results.
        # Each step runs even when the one before it raises or is cancelled
        # again: a skipped shutdown would leak the worker pool and its scratch.
        try:
            await _wait_for_coordinator_idle(coordinator)
        finally:
            try:
                parse_stage.shutdown()
            finally:
                await archive.close()

    for metrics in metrics_by_pass:
        result.counts["sessions"] = result.counts.get("sessions", 0) + metrics.ingested_session_count
        result.counts["messages"] = result.counts.get("messages", 0) + metrics.ingested_message_count
        result.changed_counts["sessions"] = result.changed_counts.get("sessions", 0) + metrics.changed_session_count
        result.processed_ids.update(metrics.changed_session_ids)
        for name, elapsed in metrics.stage_timings_s.items():
            result.stage_timings_s[name] = result.stage_timings_s.get(name, 0.0) + elapsed
        result.excised_skips += metrics.excised_skips
    return result


__all__ = ["ingest_sources_archive", "scoped_one_shot_archive_owner"]
