"""One-shot source discovery through the live acquisition and convergence owner."""

from __future__ import annotations

import asyncio
import contextvars
import os
import sqlite3
import threading
from builtins import BaseExceptionGroup
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, closing, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from polylogue.config import Source
from polylogue.core.compute import BoundedComputeAdapter
from polylogue.core.enums import Provider
from polylogue.pipeline.services.parsing_models import ParseResult
from polylogue.storage.sqlite.population_admission import assert_population_admitted

if TYPE_CHECKING:
    from polylogue.sources.live.sqlite_capture import LiveSQLiteCaptureStage

_ONE_SHOT_MARKER = ".one-shot-ingest-owner"

_OWNED_ROOT: contextvars.ContextVar[tuple[Path, int, int] | None] = contextvars.ContextVar(
    "canonical_one_shot_archive_owner", default=None
)


class _ShutdownCoordinator(Protocol):
    async def shutdown(self, *, timeout: float) -> bool: ...


async def _wait_for_coordinator_idle(coordinator: _ShutdownCoordinator) -> None:
    """Settle the original writer before its archive or compute owner unwinds."""

    async def drain() -> None:
        while not await coordinator.shutdown(timeout=30.0):
            await asyncio.sleep(0)

    settlement = asyncio.create_task(drain())
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            await asyncio.shield(settlement)
            break
        except asyncio.CancelledError as failure:
            cancellation = cancellation or failure
        except BaseException as failure:
            if cancellation is not None:
                raise BaseExceptionGroup(
                    "one-shot writer settlement failed after cancellation", [cancellation, failure]
                ) from failure
            raise
    if cancellation is not None:
        raise cancellation


@asynccontextmanager
async def one_shot_compute_owner(*, parse_workers: int | None = None) -> AsyncIterator[BoundedComputeAdapter]:
    """Own the one-shot execution kernel; callers settle their writer before leaving."""
    adapter = (
        BoundedComputeAdapter(thread_name_prefix="polylogue-one-shot")
        if parse_workers is None
        else BoundedComputeAdapter(max_workers=parse_workers, thread_name_prefix="polylogue-one-shot")
    )
    try:
        yield adapter
    finally:
        shutdown = asyncio.create_task(asyncio.to_thread(adapter.shutdown, wait=True))
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                await asyncio.shield(shutdown)
                break
            except asyncio.CancelledError as failure:
                cancellation = cancellation or failure
            except BaseException as failure:
                if cancellation is not None:
                    raise BaseExceptionGroup(
                        "one-shot compute settlement failed after cancellation", [cancellation, failure]
                    ) from failure
                raise
        if cancellation is not None:
            raise cancellation


async def _ingest_selected_paths(
    paths: list[Path],
    ingest_pass: Callable[[list[Path]], Awaitable[Any]],
) -> tuple[Any, ...]:
    """Drain bounded unattempted paths, refusing any other incomplete result."""
    from polylogue.sources.live.metrics import REFUSED_UNATTEMPTED_TIME_BUDGET, SETTLED_EXCLUSION_REASONS

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

        # A source that settled to nothing admissible (no session, corrupt
        # input) is settled: its raw carries the terminal outcome and its
        # cursor advanced.
        excluded = {path: reason for path, reason in excluded.items() if reason not in SETTLED_EXCLUSION_REASONS}
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
    compute_adapter: BoundedComputeAdapter,
    parse_workers: int | None = None,
) -> ParseResult:
    """Offer declared files to the same cursor, raw, and convergence route as the daemon.

    This is for an isolated one-shot archive owner such as the synthetic demo
    builder. A resident daemon owns its archive; callers targeting one must use
    the daemon operation instead.
    """
    from polylogue import Polylogue
    from polylogue.daemon.convergence import DaemonConverger
    from polylogue.daemon.convergence_stages import make_default_convergence_stages
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator
    from polylogue.maintenance.offline_guard import ArchiveWriterOwnershipError, resident_daemon_pid
    from polylogue.sources.live.batch import LiveBatchProcessor
    from polylogue.sources.live.cursor import CursorStore
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
        paths.extend(
            _resolve_source_paths(Source(name=source.name, path=resolved_source_path), destination=archive_root)
        )
    paths = list(dict.fromkeys(paths))
    result = ParseResult()
    if not paths:
        return result

    coordinator = DaemonWriteCoordinator(archive_root=root)

    def initialize() -> None:
        require_offline_owner()
        with ArchiveStore.open_existing(root, read_only=False):
            pass

    try:
        await coordinator.run_sync("demo.ingest.initialize", initialize)
        archive = Polylogue(archive_root=root)
        sqlite_capture_stage = live_sqlite_capture_stage(compute_adapter)
    except BaseException as primary:
        try:
            await _wait_for_coordinator_idle(coordinator)
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                "one-shot initialization and writer settlement failed", [primary, cleanup]
            ) from primary
        raise
    try:
        cursor = await coordinator.run_sync("demo.ingest.cursor", lambda: CursorStore(root / "index.db"))

        async def run_writer(actor: str, function: object, *args: object, **kwargs: object) -> object:
            require_offline_owner()
            return await coordinator.run_sync(actor, function, *args, **kwargs)  # type: ignore[arg-type]

        from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner
        from polylogue.daemon.write_coordinator import DaemonWriteThreadBridge

        service_compute = compute_adapter
        raw_owner = RawObservationConvergenceOwner(
            root,
            compute_adapter=service_compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
            write_coordinator=coordinator,
        )

        processor = LiveBatchProcessor(
            archive,
            watch_sources,
            cursor=cursor,
            parser_fingerprint=_PARSER_FINGERPRINT,
            converger=DaemonConverger(
                stages=make_default_convergence_stages(root / "index.db", compute_adapter=service_compute)
            ),
            sync_runner=run_writer,
            append_runner=raw_owner.ingest_append_plans,
            retained_runner=raw_owner.ingest_retained_raw_ids,
            convergence_runner=raw_owner.run_convergence_sync,
            sqlite_capture_stage=sqlite_capture_stage,
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
                sqlite_capture_stage.shutdown()
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


def admit_one_shot_root(root: Path) -> None:
    """Claim an empty root once; subsequent calls may only reuse that claim."""
    from polylogue.maintenance.offline_guard import ArchiveWriterOwnershipError, resident_daemon_pid
    from polylogue.storage.archive_identity import resolve_active_index_path

    assert_population_admitted(root)
    pid = resident_daemon_pid(root)
    if pid is not None:
        raise ArchiveWriterOwnershipError(
            f"polylogued PID {pid} owns {root}; submit ingestion to that daemon",
            archive_root=root,
            resident_writer=f"polylogued PID {pid}",
        )
    marker = root / _ONE_SHOT_MARKER
    root_stat = root.stat()
    claim = f"canonical-one-shot-v1 {root_stat.st_dev}:{root_stat.st_ino}\n"
    if marker.exists():
        try:
            if marker.read_text(encoding="utf-8") != claim:
                raise ArchiveWriterOwnershipError(
                    f"{root} has an invalid one-shot owner claim; refusing an offline write",
                    archive_root=root,
                )
        except OSError as exc:
            raise ArchiveWriterOwnershipError(
                f"cannot verify the one-shot owner claim for {root}", archive_root=root
            ) from exc
        return
    prior_tiers = tuple(
        name
        for name in (
            "source.db",
            "index.db",
            "user.db",
            "embeddings.db",
            "audit.db",
            "ops.db",
            ".index-active-pointer",
        )
        if (root / name).exists()
    )
    for db_path, table in (
        (root / "source.db", "raw_sessions"),
        (root / "user.db", "assertions"),
        (resolve_active_index_path(root), "sessions"),
    ):
        if not db_path.exists():
            continue
        try:
            assert_population_admitted(db_path)
            with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
                present = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
                if present and conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone():
                    raise ArchiveWriterOwnershipError(
                        f"{root} already contains archive content; submit ingestion to the resident daemon",
                        archive_root=root,
                    )
        except sqlite3.Error as exc:
            raise ArchiveWriterOwnershipError(
                f"cannot prove {root} is an empty one-shot archive",
                archive_root=root,
            ) from exc
    if prior_tiers:
        raise ArchiveWriterOwnershipError(
            f"{root} is an existing archive ({', '.join(prior_tiers)}); "
            "one-shot ingestion requires a newly isolated root",
            archive_root=root,
        )
    with marker.open("x", encoding="utf-8") as handle:
        handle.write(claim)
        handle.flush()
        os.fsync(handle.fileno())


async def ingest_one_shot_archive(
    archive_root: Path,
    sources: list[Source],
    *,
    parse_workers: int | None = None,
) -> ParseResult:
    """Claim an isolated root and ingest ``sources`` through :func:`ingest_sources_archive`."""
    if not any(source.path is not None for source in sources):
        return ParseResult()
    root = archive_root.expanduser().resolve()
    with scoped_one_shot_archive_owner(root):
        admit_one_shot_root(root)
        async with one_shot_compute_owner(parse_workers=parse_workers) as adapter:
            return await ingest_sources_archive(root, sources, compute_adapter=adapter, parse_workers=parse_workers)


def live_sqlite_capture_stage(compute_adapter: BoundedComputeAdapter) -> LiveSQLiteCaptureStage:
    """The live watcher's SQLite capture stage on the owner's compute adapter."""
    from polylogue.sources.live.sqlite_capture import LiveSQLiteCaptureStage

    return LiveSQLiteCaptureStage(compute_adapter=compute_adapter)


__all__ = [
    "admit_one_shot_root",
    "ingest_one_shot_archive",
    "ingest_sources_archive",
    "live_sqlite_capture_stage",
    "scoped_one_shot_archive_owner",
]
