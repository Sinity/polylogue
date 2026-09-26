"""One-shot source discovery through the live acquisition and convergence owner."""

from __future__ import annotations

import asyncio
import contextvars
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from polylogue.config import Source
from polylogue.pipeline.services.parsing_models import ParseResult

_OWNED_ROOT: contextvars.ContextVar[tuple[Path, int, int] | None] = contextvars.ContextVar(
    "canonical_one_shot_archive_owner", default=None
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
        source_path = source.path.expanduser().resolve()
        refuse_non_capture_source_root(source_path, destination=root)
        watch_sources.append(
            WatchSource(name=source.name, root=source_path if source_path.is_dir() else source_path.parent)
        )
        paths.extend(_resolve_source_paths(Source(name=source.name, path=source_path)))
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
        metrics = await processor.ingest_files(paths, emit_event=False)
    finally:
        parse_stage.shutdown()
        await archive.close()
        await coordinator.shutdown(timeout=30.0)

    if metrics.failed_file_count or metrics.deferred_file_count:
        raise RuntimeError(
            "canonical ingestion did not settle every selected source file: "
            f"failed={metrics.failed_file_count}, deferred={metrics.deferred_file_count}"
        )
    result.counts["sessions"] = metrics.ingested_session_count
    result.counts["messages"] = metrics.ingested_message_count
    result.changed_counts["sessions"] = metrics.changed_session_count
    result.processed_ids = {session_id for _, session_id in (*metrics.new_sessions, *metrics.updated_sessions)}
    result.stage_timings_s = dict(metrics.stage_timings_s)
    result.excised_skips = metrics.excised_skips
    return result


__all__ = ["ingest_sources_archive", "scoped_one_shot_archive_owner"]
