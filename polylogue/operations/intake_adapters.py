"""Domain adapters for the fair daemon intake scheduler.

This module owns the source/storage-facing part of intake composition.  The
daemon scheduler stays storage- and source-independent; adapters retain only
disposable discovery hints and delegate writes to the daemon's existing writer
runner.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

from polylogue.daemon.intake import (
    DEFAULT_INTAKE_BYTE_BUDGET,
    UNMEASURABLE_INTAKE_COST_BYTES,
    AdmissionOutcome,
    AdmissionResult,
    FairIntakeDispatcher,
    IntakeAdapter,
    IntakeItem,
)
from polylogue.logging import WARNING, emit
from polylogue.sources.live.acquisition_log import log_unclaimed_file
from polylogue.sources.live.cold_build import (
    ColdBuildGeneration,
    active_index_generation_is_empty,
    clear_cold_build_generation,
    register_cold_build_generation,
)
from polylogue.sources.live.metrics import REFUSED_DAEMON_DEGRADED
from polylogue.sources.live.source_selection import deepest_source_for_path
from polylogue.sources.live.watcher import LiveWatcher, WatchSource, _log_ingest_metrics
from polylogue.sources.walk_faults import WalkFault, WalkRefusedError

_T = TypeVar("_T")

__all__ = [
    "ColdBuildGeneration",
    "DaemonIntakeContext",
    "DaemonIntakeService",
    "FileIntakeAdapter",
    "MultiplexIntakeAdapter",
    "CallbackIntakeAdapter",
    "RawMaterializationIntakeAdapter",
    "RawMaterializationDiscovery",
    "active_index_generation_is_empty",
    "build_intake_adapters",
    "clear_cold_build_generation",
    "discover_pending_raw_ids",
    "register_cold_build_generation",
]


_RAW_DISCOVERY_INSPECTION_LIMIT = 32


@dataclass(frozen=True, slots=True)
class DaemonIntakeContext:
    archive_root: Path
    watcher: LiveWatcher
    sources: tuple[WatchSource, ...]
    write_runner: Callable[..., Awaitable[Any]] | None = None

    async def run_write(self, actor: str, function: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
        if self.write_runner is None:
            return await asyncio.to_thread(function, *args, **kwargs)
        return cast(_T, await self.write_runner(actor, function, *args, **kwargs))


def _walk_entry_key(path: Path, *, is_dir: bool) -> str:
    """The order key a walk entry occupies among its siblings.

    A file's key is its path string -- exactly the key the resume cursor
    compares against. A directory's key is its path string plus the path
    separator, which is what makes descending at the keyed position exact:
    ``root/a.json`` sorts below ``root/a`` + separator because ``.`` (46)
    sorts below ``/`` (47), so ``root/a.json`` is emitted before every file
    under ``root/a/``. Both comparisons are plain code-point ordering, so
    unicode and mixed-case names order identically here and at the cursor.
    """

    text = str(path)
    return text + os.sep if is_dir else text


def _real_path(path: Path) -> str:
    """The resolved real path used as this walk's cycle-detection identity."""

    return os.path.realpath(path)


def _emit_discovery_fault(source: WatchSource, fault: WalkFault, *, reason: str) -> None:
    """Report one non-fatal discovery fault without losing the rest of the page.

    A symlink cycle and a dangling symlink are both permanent properties of
    the tree: refusing the whole class for them would strand every real file
    beside them forever. They are still evidence -- the same
    ``daemon.intake.discovery_failed`` shape the dispatcher emits -- so the
    entry is counted per path instead of silently vanishing.
    """

    emit(
        "daemon.intake.discovery_failed",
        level=WARNING,
        outcome="degraded",
        reason=reason,
        component=source.name,
        path=str(fault.path),
        error_detail=fault.detail,
    )


def _admit_linked_directory(
    source: WatchSource,
    path: Path,
    *,
    visited_real_paths: set[str] | None,
) -> bool:
    """Whether a directory symlink may be descended into on this walk.

    Containment is the standing rule the live watcher already enforces
    through ``deepest_source_for_path``: a link whose resolved target lies
    outside the source root is rejected, never admitted. Only a link that
    stays inside the root is followed, which is what makes a
    ``current -> 2026-09`` style export pointer discoverable without letting
    a link hand intake material the source was never configured to read.
    """

    try:
        real = _real_path(path)
        root_real = _real_path(source.root)
    except OSError as exc:
        _emit_discovery_fault(
            source,
            WalkFault(path, f"symlink target could not be resolved: {exc}"),
            reason="unresolvable_symlink",
        )
        return False
    if real != root_real and not real.startswith(root_real + os.sep):
        _emit_discovery_fault(
            source,
            WalkFault(path, f"symlink target {real} escapes the source root"),
            reason="escaping_symlink",
        )
        return False
    if visited_real_paths is not None:
        if real in visited_real_paths:
            _emit_discovery_fault(
                source,
                WalkFault(path, f"symlink cycle: {real} was already visited"),
                reason="symlink_cycle",
            )
            return False
        visited_real_paths.add(real)
    return True


def _ordered_children(
    source: WatchSource,
    directory: Path,
    after: str | None,
    scandir: Callable[[Path], Any] = os.scandir,
    *,
    visited_real_paths: set[str] | None = None,
) -> list[tuple[str, Path, bool]]:
    """Siblings of ``directory``, reverse-sorted so a stack pops them in order.

    Directory symlinks are followed: an operator who mounts an export tree
    through a symlink configured a real source root, and refusing to enter it
    made the source silently unacquired. Following links needs the two guards
    below -- ``visited_real_paths`` (resolved real paths already entered on
    this walk) terminates cycles, and a resolved target outside the source
    root is rejected rather than followed.
    """

    children: list[tuple[str, Path, bool]] = []
    try:
        entries = scandir(directory)
    except OSError as exc:
        # Refuse rather than return an empty sibling list. Returning ``[]``
        # here removed every file beneath ``directory`` from the walk while
        # the caller saw a result indistinguishable from "that subtree is
        # empty"; the daemon's dispatcher already turns a raising ``discover``
        # into a counted ``daemon.intake.discovery_failed`` event plus a class
        # report reason, and retries the class on the next pass.
        raise WalkRefusedError(
            "intake discovery could not read a source directory",
            [WalkFault(directory, f"scandir failed: {exc}")],
        ) from exc
    with entries:
        for entry in entries:
            path = Path(entry.path)
            try:
                is_link = entry.is_symlink()
                if entry.is_dir(follow_symlinks=False) or (is_link and entry.is_dir()):
                    if source.ignores_directory(path):
                        continue
                    if is_link and not _admit_linked_directory(
                        source,
                        path,
                        visited_real_paths=visited_real_paths,
                    ):
                        continue
                    key = _walk_entry_key(path, is_dir=True)
                    # Every descendant path begins with ``key``. When the
                    # cursor sorts above ``key`` without having it as a
                    # prefix, it sorts above every such descendant too, so
                    # the whole subtree is already behind the cursor.
                    if after is not None and after > key and not after.startswith(key):
                        continue
                    children.append((key, path, True))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    if is_link and not entry.is_file():
                        # A dangling link is a fault, not an absence: the
                        # export it named is missing. Counting it keeps the
                        # walk alive over the rest of the directory.
                        _emit_discovery_fault(
                            source,
                            WalkFault(path, "symlink target does not exist"),
                            reason="broken_symlink",
                        )
                    continue
            except FileNotFoundError:
                # Ordinary producer churn: the entry vanished between the
                # listing and the type probe. Nothing was hidden.
                continue
            except OSError as exc:
                raise WalkRefusedError(
                    "intake discovery could not inspect a source entry",
                    [WalkFault(path, f"stat failed: {exc}")],
                ) from exc
            children.append((_walk_entry_key(path, is_dir=False), path, False))
    children.sort(key=lambda child: child[0], reverse=True)
    return children


def _log_unclaimed_intake_candidate(path: Path, *, source_name: str, suffixes: tuple[str, ...]) -> None:
    """Log one discovered file no configured suffix accepts.

    Best-effort ``stat``: a file that vanished between the listing and this
    call was still seen and unclaimed, just without size/mtime detail.
    """
    try:
        stat_result = path.stat()
        size: int | None = stat_result.st_size
        mtime: float | None = stat_result.st_mtime
    except OSError:
        size, mtime = None, None
    log_unclaimed_file(
        path=path,
        size=size,
        mtime=mtime,
        reason=f"suffix not in watched set {suffixes} for source {source_name!r}",
        source_name=source_name,
    )


def _bounded_source_paths(
    source: WatchSource,
    all_sources: tuple[WatchSource, ...],
    *,
    limit: int,
    after: str | None,
    scandir: Callable[[Path], Any] = os.scandir,
) -> list[Path]:
    """Collect at most ``limit`` files, stopping as soon as it is full.

    Files are emitted in exact lexicographic order of their path strings.
    That order is authoritative because it is the order the resume cursor
    compares in (``after``, advanced in ``acknowledge`` over consumed
    items): a producer that emitted ``os.scandir`` order instead let a
    high-water mark skip files it had never emitted, which was permanent
    loss rather than delay. Emitting in cursor order also makes the
    ``limit`` early exit safe -- the next pass resumes at exactly the key
    the previous one stopped on, mid-directory or not.

    Directory symlinks whose target stays inside the source root are
    followed, so an export tree mounted behind a link is discovered. Every
    directory entered on this walk records its resolved real path, which is
    what terminates a cycle.
    """

    if limit <= 0:
        return []
    if not source.root.is_dir():
        # A missing or unmounted root is a refusal, not an empty backlog.
        # ``operations/raw_sessions/sessions.py`` already raises for exactly
        # this condition; returning ``[]`` reported a fully ingested source
        # when the export drive was simply not mounted.
        raise WalkRefusedError(
            "intake discovery could not read a source root",
            [WalkFault(source.root, "source root is unavailable")],
        )
    found: list[Path] = []
    visited_real_paths: set[str] = {_real_path(source.root)}
    stack: list[list[tuple[str, Path, bool]]] = [
        _ordered_children(
            source,
            source.root,
            after,
            scandir,
            visited_real_paths=visited_real_paths,
        )
    ]
    while stack and len(found) < limit:
        level = stack[-1]
        if not level:
            stack.pop()
            continue
        key, path, is_dir = level.pop()
        if is_dir:
            visited_real_paths.add(_real_path(path))
            stack.append(
                _ordered_children(
                    source,
                    path,
                    after,
                    scandir,
                    visited_real_paths=visited_real_paths,
                )
            )
            continue
        if after is not None and key <= after:
            continue
        try:
            if deepest_source_for_path(path, all_sources) is not source:
                continue
            if not source.accepts(path):
                # A file this source's own walk reached but whose suffix no
                # detector is configured to accept. The record exists whether
                # or not an operator runs the standalone sweep, and discovery
                # is the only production walk left that reaches it.
                _log_unclaimed_intake_candidate(path, source_name=source.name, suffixes=source.suffixes)
                continue
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise WalkRefusedError(
                "intake discovery could not resolve a source file's owner",
                [WalkFault(path, f"ownership resolution failed: {exc}")],
            ) from exc
        found.append(path)
    return found


class FileIntakeAdapter(IntakeAdapter):
    """Browser/configured-local adapter; durable cursors are its ack state."""

    def __init__(self, context: DaemonIntakeContext, source: WatchSource, *, class_name: str | None = None) -> None:
        self.context = context
        self.source = source
        self.class_name = class_name or source.name
        self._after: str | None = None
        self._last_root_mtime_ns: int | None = None
        self._last_hint_revision = context.watcher.intake_revision(source)

    async def discover(self, *, limit: int) -> Sequence[IntakeItem]:
        hint_revision = self.context.watcher.intake_revision(self.source)
        if hint_revision != self._last_hint_revision:
            self._after = None
            self._last_hint_revision = hint_revision
        # A filesystem cursor is only a scheduling hint.  A producer may add
        # an item lexicographically before the previous position; restart the
        # bounded walk when the root changed so that insertion is revisited.
        # Durable cursor/raw identity makes revisiting already admitted files
        # harmless and avoids treating this hint as queue authority.
        try:
            root_mtime_ns = self.source.root.stat().st_mtime_ns
        except OSError:
            root_mtime_ns = None
        if root_mtime_ns is not None and root_mtime_ns != self._last_root_mtime_ns:
            self._after = None
            self._last_root_mtime_ns = root_mtime_ns
        # The cursor advances in ``acknowledge``, over items the dispatcher
        # actually consumed -- never here, over everything merely discovered.
        # A page is routinely truncated by the class deficit, so advancing on
        # discovery skipped every file past the first admitted one and, because
        # the walk only ever moves forward, those files were never revisited:
        # a static root silently retained just its lexicographically-first
        # session. Revisiting an already-admitted file is explicitly harmless
        # (durable cursor/raw identity, see above), so the conservative
        # direction here is to re-discover, never to skip.
        paths = _bounded_source_paths(self.source, self.context.sources, limit=limit, after=self._after)
        items: list[IntakeItem] = []
        for path in paths:
            try:
                size = path.stat().st_size
            except OSError:
                size = 1
            items.append(
                IntakeItem(
                    item_id=f"file:{path.resolve()}",
                    class_name=self.class_name,
                    payload=path,
                    estimated_cost=max(1, size),
                )
            )
        return tuple(items)

    async def admit(self, item: IntakeItem) -> AdmissionResult:
        outcomes = await self.admit_page((item,))
        return outcomes[item.item_id]

    async def admit_page(self, items: Sequence[IntakeItem]) -> dict[str, AdmissionResult]:
        """Admit a whole discovery page as one ingest batch, one hold.

        Every fixed cost of a live batch -- the writer hold, the six-tier
        bootstrap, the retention scan, the archive-wide convergence pass and
        the parse stage's own warm -- is paid once per call. Admitting one
        file per call made each of those a per-file cost over a corpus of
        tens of thousands of files, and handed ``LiveParseStage`` a single
        path per batch, which is no parallelism at all.

        The batch is one call; the *outcomes* stay per item, read back from
        ``LiveBatchMetrics`` by path, so the dispatcher's deficit,
        ``retry_after`` and isolation accounting are exactly what they were
        under per-file admission.
        """
        outcomes: dict[str, AdmissionResult] = {}
        batch: list[IntakeItem] = []
        for item in items:
            path = Path(item.payload) if isinstance(item.payload, (str, Path)) else None
            if path is None:
                outcomes[item.item_id] = AdmissionResult(
                    AdmissionOutcome.TERMINAL, reason="file intake item has no path"
                )
                continue
            if not path.is_file():
                outcomes[item.item_id] = AdmissionResult(
                    AdmissionOutcome.RETRYABLE, reason=f"source carrier vanished: {path}"
                )
                continue
            batch.append(item)
        if not batch:
            return outcomes

        try:
            # The source-selection/cursor authority gate runs before anything
            # in this page can mutate cursor state -- before initialization,
            # before the needs-work selection reads a row. A page whose
            # authority is refused must leave the archive exactly as it was.
            processor = getattr(self.context.watcher, "_batch_processor", None)
            if processor is not None:
                processor.require_cursor_authority([Path(cast(Any, item.payload)) for item in batch])
            # Cursor initialization precedes every read of cursor state, and
            # takes the writer admission to do it: the selection below reads
            # the cursor rows, so doing it first would touch (and create) the
            # store outside the writer lease.
            cursor = getattr(self.context.watcher, "_cursor", None)
            run_writer_sync = getattr(self.context.watcher, "_run_writer_sync", None)
            if cursor is not None and callable(run_writer_sync):
                await run_writer_sync("watcher.intake.cursor_initialize", cursor.initialize)
            # Narrow the page before it costs anything more: a bounded walk
            # re-offers files whose cursor already accounts for them, and
            # handing those to the batch buys a planning pass per file per
            # pass for no admission.
            paths = [Path(cast(Any, item.payload)) for item in batch]
            select = getattr(self.context.watcher, "select_ingest_candidates", None)
            if not callable(select):
                needed = set(paths)
            elif callable(run_writer_sync):
                # The selection is a read that can decide to write: an
                # incomplete-append deferral, an archived-cursor
                # reconciliation and a device-drift rebase all correct cursor
                # rows in place. Those are ordinary archive writes, so they
                # run through the writer admission like every other one --
                # under process-wide lease enforcement an unadmitted cursor
                # write is refused, which turned the whole page retryable.
                needed = set(await run_writer_sync("watcher.intake.select", select, paths))
            else:
                needed = set(select(paths))
            skipped = [item for item in batch if Path(cast(Any, item.payload)) not in needed]
            for item in skipped:
                outcomes[item.item_id] = AdmissionResult(
                    AdmissionOutcome.DUPLICATE, actual_cost=max(1, int(item.estimated_cost))
                )
            batch = [item for item in batch if Path(cast(Any, item.payload)) in needed]
            if not batch:
                return outcomes
            paths = [Path(cast(Any, item.payload)) for item in batch]
            metrics = await self.context.watcher._ingest_files(
                paths,
                queued_file_count=len(paths) + len(skipped),
                skipped_file_count=len(skipped),
                whole_archive_convergence=False,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            # A cursor-authority refusal lands here too: it is retryable for
            # every item in the page, and nothing in the archive changed. Say
            # so once per page: a class that reports only ``retried`` counts
            # is otherwise a silent refusal with no reason anywhere.
            reason = f"{type(exc).__name__}: {exc}"
            emit(
                "daemon.intake.page_refused",
                level=WARNING,
                outcome="degraded",
                reason="page_admission_failed",
                component=self.class_name,
                files=len(batch),
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            for item in batch:
                outcomes[item.item_id] = AdmissionResult(AdmissionOutcome.RETRYABLE, reason=reason)
            return outcomes

        stale_cursor_writes = int(getattr(metrics, "stale_cursor_write_count", 0) or 0)
        if stale_cursor_writes:
            # A stale cursor write means this batch raced another authority
            # for the same source rows; the whole page is retried rather than
            # acknowledged, even where some files reported success, because a
            # cursor advanced under a losing write is not evidence about any
            # item in the page.
            for item in batch:
                outcomes[item.item_id] = AdmissionResult(
                    AdmissionOutcome.RETRYABLE, reason="source cursor write was stale"
                )
            return outcomes

        succeeded = {str(path) for path in (getattr(metrics, "succeeded_paths", ()) or ())}
        failed = set(getattr(metrics, "failed_paths", ()) or ())
        deferred = set(getattr(metrics, "deferred_paths", ()) or ())
        # ``failed_paths`` carries the retry projection, deferrals included.
        # A deferral is its own outcome, so it must not be reported as a
        # failure here.
        failed -= deferred
        excluded_by_path = dict(getattr(metrics, "excluded_paths", {}) or {})
        if not succeeded:
            # This route calls ``_ingest_files`` directly, so the watcher's
            # own ``_log_ingest_metrics`` never runs for it and the
            # "admitted nothing" line was invisible on the intake path.
            _log_ingest_metrics(f"live.intake: {self.class_name}", metrics)

        # Reconcile the batch's measured read against the items that actually
        # produced it. An append-mode file reads far less than its size, so
        # charging every item its full estimate would overstate the class's
        # spend; distributing the measured total over the admitted items in
        # proportion to their estimates keeps the class budget denominated in
        # bytes actually read.
        refused_reasons = dict(getattr(metrics, "refused_bytes_by_reason", {}) or {})
        unattempted_is_retryable = bool(getattr(metrics, "time_budget_exceeded", False)) or (
            REFUSED_DAEMON_DEGRADED in refused_reasons
        )
        read_bytes = int(getattr(metrics, "source_payload_read_bytes", 0) or 0)
        estimated_total = sum(max(1, int(item.estimated_cost)) for item in batch)
        for item in batch:
            key = str(Path(cast(Any, item.payload)))
            item_estimate = max(1, int(item.estimated_cost))
            actual_cost = max(1, round(read_bytes * item_estimate / estimated_total)) if read_bytes else item_estimate
            if key in succeeded:
                outcomes[item.item_id] = AdmissionResult(AdmissionOutcome.ADMITTED, actual_cost=actual_cost)
            elif key in excluded_by_path:
                # polylogue-onbz3: a durable refusal is not "already admitted
                # under this identity". Reporting DUPLICATE here advanced the
                # cursor and counted the pass as progress.
                outcomes[item.item_id] = AdmissionResult(
                    AdmissionOutcome.EXCLUDED,
                    reason=f"source admission excluded {key}: {excluded_by_path[key]}",
                    actual_cost=item_estimate,
                )
            elif key in deferred:
                outcomes[item.item_id] = AdmissionResult(
                    AdmissionOutcome.DEFERRED,
                    reason=f"source admission deferred {key}",
                    actual_cost=item_estimate,
                )
            elif key in failed:
                outcomes[item.item_id] = AdmissionResult(
                    AdmissionOutcome.RETRYABLE, reason=f"source admission failed: {key}"
                )
            elif unattempted_is_retryable:
                # The pass ran out of its declared time budget, or refused
                # the whole batch while degraded: this item was never
                # attempted, so it is ordinary backlog, not a re-seen one.
                outcomes[item.item_id] = AdmissionResult(
                    AdmissionOutcome.RETRYABLE,
                    reason=f"source admission left {key} unattempted",
                )
            else:
                # Offered and attempted, with nothing new to admit under this
                # identity: the ordinary re-discovery of an already-ingested
                # file. Acknowledgeable, never progress.
                outcomes[item.item_id] = AdmissionResult(AdmissionOutcome.DUPLICATE, actual_cost=item_estimate)

        admitted_paths = [path for path in paths if str(path) in succeeded]
        if admitted_paths:
            converge_embeddings = getattr(self.context.watcher, "_converge_embeddings_off_writer", None)
            if callable(converge_embeddings):
                await converge_embeddings(admitted_paths)
            converge_profiles = getattr(self.context.watcher, "_converge_session_profiles_off_writer", None)
            if callable(converge_profiles):
                await converge_profiles(tuple(getattr(metrics, "changed_session_ids", ()) or ()))
        return outcomes

    async def acknowledge(self, item: IntakeItem) -> None:
        # Files remain retained source carriers.  The live batch's durable
        # cursor/raw commit is the acknowledgement projection.  The scheduling
        # cursor advances here, monotonically, so an item dropped by the class
        # deficit is rediscovered on the next pass instead of being skipped.
        payload = item.payload
        if isinstance(payload, (str, Path)):
            position = str(payload)
            if self._after is None or position > self._after:
                self._after = position


class MultiplexIntakeAdapter(IntakeAdapter):
    """Bounded round-robin over several configured local file roots.

    The dispatcher has one logical ``configured_local`` class.  Keeping one
    adapter per root would duplicate that class identity and make construction
    order accidentally authoritative, so this adapter owns only a disposable
    root position and delegates each item to its source adapter.
    """

    def __init__(self, adapters: Sequence[IntakeAdapter]) -> None:
        if not adapters:
            raise ValueError("at least one file adapter is required")
        self.adapters = tuple(adapters)
        self._next = 0
        self._by_item: dict[str, IntakeAdapter] = {}

    async def discover(self, *, limit: int) -> Sequence[IntakeItem]:
        if limit <= 0:
            return ()
        result: list[IntakeItem] = []
        owners: list[IntakeAdapter] = []
        start = self._next
        for offset in range(len(self.adapters)):
            adapter = self.adapters[(start + offset) % len(self.adapters)]
            remaining = limit - len(result)
            if remaining <= 0:
                break
            page = await adapter.discover(limit=remaining)
            result.extend(page)
            owners.extend([adapter] * len(page))
        self._next = (start + 1) % len(self.adapters)
        self._by_item.update({item.item_id: adapter for item, adapter in zip(result, owners, strict=True)})
        return tuple(result[:limit])

    async def admit(self, item: IntakeItem) -> AdmissionResult:
        adapter = self._by_item.get(item.item_id)
        if adapter is None:
            return AdmissionResult(AdmissionOutcome.RETRYABLE, reason="configured source item lost its adapter")
        return await adapter.admit(item)

    async def admit_page(self, items: Sequence[IntakeItem]) -> dict[str, AdmissionResult]:
        """Split one page along source ownership and admit each part as a page.

        One root's files are one ingest batch. Splitting here rather than
        admitting item by item is what keeps the per-batch fixed cost paid
        once per root per pass, and the parts are disjoint, so the per-item
        outcomes recombine without ambiguity.
        """
        outcomes: dict[str, AdmissionResult] = {}
        groups: dict[int, tuple[IntakeAdapter, list[IntakeItem]]] = {}
        for item in items:
            adapter = self._by_item.get(item.item_id)
            if adapter is None:
                outcomes[item.item_id] = AdmissionResult(
                    AdmissionOutcome.RETRYABLE, reason="configured source item lost its adapter"
                )
                continue
            groups.setdefault(id(adapter), (adapter, []))[1].append(item)
        for adapter, group in groups.values():
            admit_page = getattr(adapter, "admit_page", None)
            if admit_page is None:
                for item in group:
                    outcomes[item.item_id] = await adapter.admit(item)
                continue
            outcomes.update(await admit_page(tuple(group)))
        return outcomes

    async def acknowledge(self, item: IntakeItem) -> None:
        adapter = self._by_item.pop(item.item_id, None)
        if adapter is not None:
            await adapter.acknowledge(item)


class CallbackIntakeAdapter(IntakeAdapter):
    """One-shot bounded remote/raw adapter around an existing domain route.

    The wrapped route reports a changed-row count, not payload bytes, so it
    has no honest estimate in the dispatcher's unit. It therefore charges
    ``UNMEASURABLE_INTAKE_COST_BYTES`` -- a full budget share -- rather than
    the literal one byte that let a large remote sync starve its siblings
    (polylogue-swicx).
    """

    def __init__(
        self,
        class_name: str,
        callback: Callable[[], Awaitable[AdmissionResult | int] | AdmissionResult | int],
        *,
        estimated_cost: int = UNMEASURABLE_INTAKE_COST_BYTES,
        persistent: bool = True,
    ) -> None:
        self.class_name = class_name
        self.callback = callback
        self.estimated_cost = max(1, estimated_cost)
        self.persistent = persistent
        self._pending = True

    async def discover(self, *, limit: int) -> Sequence[IntakeItem]:
        if limit <= 0 or not self._pending:
            return ()
        return (IntakeItem(self.class_name, self.class_name, estimated_cost=self.estimated_cost),)

    async def admit(self, item: IntakeItem) -> AdmissionResult:
        try:
            changed = self.callback()
            if isinstance(changed, Awaitable):
                changed = await changed
            if isinstance(changed, AdmissionResult):
                return changed
            self._pending = self.persistent
            return AdmissionResult(
                AdmissionOutcome.ADMITTED if int(changed) else AdmissionOutcome.DUPLICATE,
                actual_cost=self.estimated_cost,
            )
        except Exception as exc:
            return AdmissionResult(AdmissionOutcome.RETRYABLE, reason=f"{self.class_name}: {exc}")

    async def acknowledge(self, item: IntakeItem) -> None:
        if not self.persistent:
            self._pending = False


class RawMaterializationIntakeAdapter(IntakeAdapter):
    """Bounded raw-id discovery delegated to the canonical derivation route."""

    def __init__(
        self,
        discover_ids: Callable[[int], Awaitable[Sequence[tuple[str, int]]] | Sequence[tuple[str, int]]],
        admit_id: Callable[[str], Awaitable[AdmissionResult | int] | AdmissionResult | int],
    ) -> None:
        self._discover_ids = discover_ids
        self._admit_id = admit_id

    async def discover(self, *, limit: int) -> Sequence[IntakeItem]:
        raw_ids = self._discover_ids(limit)
        if isinstance(raw_ids, Awaitable):
            raw_ids = await raw_ids
        return tuple(
            IntakeItem(raw_id, "raw_materialization", estimated_cost=max(1, int(cost))) for raw_id, cost in raw_ids
        )

    async def admit(self, item: IntakeItem) -> AdmissionResult:
        try:
            changed = self._admit_id(item.item_id)
            if isinstance(changed, Awaitable):
                changed = await changed
            if isinstance(changed, AdmissionResult):
                return changed
            return AdmissionResult(
                AdmissionOutcome.ADMITTED if int(changed) else AdmissionOutcome.DUPLICATE,
                actual_cost=item.estimated_cost,
            )
        except Exception as exc:
            return AdmissionResult(AdmissionOutcome.RETRYABLE, reason=f"raw materialization: {exc}")

    async def acknowledge(self, item: IntakeItem) -> None:
        return None


@dataclass(frozen=True, slots=True)
class _RawDiscoveryBinding:
    """The canonical frame identity that makes a continuation reusable."""

    archive_root: str
    source_revision: str
    recipe_version: str


class RawMaterializationDiscovery:
    """One process-local, bounded traversal of canonical raw obligations.

    The cursor is an intake scheduling hint. It never records validity or an
    admission result, so losing it merely starts a new bounded traversal.

    Two lanes feed it. The *sweep* lane pages through the canonical required-key
    space and retains its continuation. The *arrival* lane serves raws admitted
    since this process last looked, read straight off the append-only
    ``raw_sessions`` rowid frontier. A new raw can sort lexically before the
    sweep cursor, so without the arrival lane it would wait for a whole
    traversal to wrap; but restarting the sweep for it -- what this class used
    to do -- meant that under a sustained arrival rate the cursor was reset
    before it ever advanced past its first page, and every obligation behind
    that page was never reached. Serving the new raws directly reaches them
    without discarding the sweep's position, and the lanes alternate so an
    unbroken stream of arrivals cannot stall the sweep either.
    """

    def __init__(self, archive_root: Path, *, max_payload_bytes: int) -> None:
        self._archive_root = archive_root
        self._max_payload_bytes = max_payload_bytes
        self._binding: _RawDiscoveryBinding | None = None
        self._cursor: str | None = None
        #: The continuation start of the page currently being offered and the
        #: keys it still owed when it was last inspected. A page at the same
        #: cursor that now owes a key it did not owe before is new work, not a
        #: stalled head.
        self._held_page: tuple[str | None, tuple[str, ...]] | None = None
        self._frontier: int = 0
        self._arrivals_first = False

    def _raw_frontier(self) -> int:
        """Return the durable high-water mark for admitted raw observations."""
        from polylogue.storage.sqlite.connection_profile import open_readonly_connection

        source_db = self._archive_root / "source.db"
        with open_readonly_connection(source_db, timeout=5.0) as conn:
            row = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM raw_sessions").fetchone()
        return int(row[0]) if row is not None else 0

    def _arrived_since_frontier(self, limit: int) -> tuple[tuple[str, ...], int]:
        """Raw ids admitted after the recorded frontier, oldest first.

        ``raw_sessions`` is append-only for admitted observations, so a bounded
        ``rowid >`` page is the whole arrival set and its last rowid is the new
        frontier. Publishing changes to an existing raw does not advance the
        rowid and therefore produces no arrival.
        """
        from polylogue.storage.sqlite.connection_profile import open_readonly_connection

        source_db = self._archive_root / "source.db"
        with open_readonly_connection(source_db, timeout=5.0) as conn:
            rows = conn.execute(
                "SELECT rowid, raw_id FROM raw_sessions WHERE rowid > ? ORDER BY rowid LIMIT ?",
                (self._frontier, limit),
            ).fetchall()
        if not rows:
            return (), self._frontier
        return tuple(str(row[1]) for row in rows), int(rows[-1][0])

    def discover_pending_raw_ids(self, limit: int) -> tuple[tuple[str, int], ...]:
        """Inspect one bounded canonical page and retain its continuation.

        Raw discovery is deliberately a read-only derivation traversal.  The old
        census preview was a second backlog authority and could miss observations
        whose parser census was complete but whose derived output had been lost.
        A discovery call inspects one bounded page before admitting only non-valid
        observations. The next call resumes after that page, even if it contained
        only valid or already-isolated observations. The
        payload size is the scheduler's cost estimate, read from the durable raw
        rows rather than guessed by the adapter.
        """
        if limit <= 0:
            return ()
        if not (self._archive_root / "source.db").exists():
            # polylogue-f7pdm: a fresh archive root simply has no raw tier
            # yet. That is an empty page, not a failure: the tier appears as
            # soon as the first acquisition commits and the next pass
            # discovers it. Reporting it as an error here is what made the
            # daemon latch raw materialization off for its whole lifetime and
            # log a whale-schedule warning every 30 s on an empty root.
            return ()
        from polylogue.operations.raw_observation_derivation import raw_observation_frame
        from polylogue.storage.derived.raw import RAW_OBSERVATION_DOMAIN, RawObservationDerivation

        frame = raw_observation_frame(self._archive_root)
        binding = _RawDiscoveryBinding(
            archive_root=frame.archive_root,
            source_revision=frame.source_revision,
            recipe_version=frame.recipe_version(RAW_OBSERVATION_DOMAIN),
        )
        if binding != self._binding:
            # A replacement generation invalidates the traversal itself, not
            # merely its position: start a fresh sweep and treat everything
            # already durable as swept rather than as an arrival.
            self._binding = binding
            self._cursor = None
            self._held_page = None
            self._frontier = self._raw_frontier()
            self._arrivals_first = False

        inspected_limit = min(limit, _RAW_DISCOVERY_INSPECTION_LIMIT)
        adapter = RawObservationDerivation(self._archive_root, max_payload_bytes=self._max_payload_bytes)
        self._arrivals_first = not self._arrivals_first
        lanes: tuple[Callable[[Any, Any, int], tuple[str, ...]], ...] = (
            (self._arrival_selected, self._sweep_selected)
            if self._arrivals_first
            else (self._sweep_selected, self._arrival_selected)
        )
        for lane in lanes:
            selected = lane(frame, adapter, inspected_limit)
            if selected:
                return self._with_costs(selected)
        return ()

    def _arrival_selected(self, frame: Any, adapter: Any, limit: int) -> tuple[str, ...]:
        page, frontier = self._arrived_since_frontier(limit)
        self._frontier = frontier
        if not page:
            return ()
        statuses = adapter.inspect(frame, page)
        return tuple(raw_id for raw_id in page if statuses.get(raw_id) != "valid")

    def _sweep_selected(self, frame: Any, adapter: Any, limit: int) -> tuple[str, ...]:
        # At most one released page is skipped per call, so a stalled head
        # costs one extra bounded page read rather than the whole cycle.
        for _attempt in range(2):
            page_cursor = self._cursor
            page, next_cursor = adapter.required_page(frame, cursor=page_cursor, limit=limit)
            # An empty or fully valid page is progress through the required-key
            # space. ``None`` is the completed-traversal marker; the following
            # pass starts a fresh sweep so new work before this cursor is
            # eventually revisited.
            selected: tuple[str, ...] = ()
            if page:
                statuses = adapter.inspect(frame, page)
                selected = tuple(raw_id for raw_id in page if statuses.get(raw_id) != "valid")
            if not selected:
                self._cursor = next_cursor
                self._held_page = None
                return ()
            held = self._held_page
            if held is not None and held[0] == page_cursor and set(held[1]) == set(selected):
                # Re-inspected the same page and nothing moved: the blockage is
                # not budget pressure, so stop pinning the traversal behind it
                # and go on to the next page in this same call.
                self._cursor = next_cursor
                self._held_page = None
                continue
            # Hold the continuation over keys this page still owes. The
            # dispatcher admits the offered items only until its class budget
            # is spent, so advancing over the whole inspected page moves the
            # cursor past obligations nobody took -- the same
            # producer-advances-past-the-consumer shape
            # ``DerivationRunner.run_domain`` avoids with its ``stopped_at``
            # offset. The authority for "processed" is the output relation
            # re-inspected on the next pass.
            self._held_page = (page_cursor, selected)
            self._cursor = page_cursor
            return selected
        return ()

    def _with_costs(self, selected: tuple[str, ...]) -> tuple[tuple[str, int], ...]:
        from polylogue.operations.operation_context import open_operation_read

        with open_operation_read(self._archive_root) as pinned:
            sizes = pinned.archive.raw_payload_sizes(selected)
        return tuple((raw_id, max(1, int(sizes.get(raw_id, 1)))) for raw_id in selected)


def discover_pending_raw_ids(archive_root: Path, limit: int, max_payload_bytes: int) -> tuple[tuple[str, int], ...]:
    """Discover one bounded raw page for callers without an intake lifetime.

    The daemon owns a ``RawMaterializationDiscovery`` instance for its whole
    fair-intake lifetime. This compatibility helper deliberately has no
    continuation because it cannot retain one across independent callers.
    """
    return RawMaterializationDiscovery(
        archive_root,
        max_payload_bytes=max_payload_bytes,
    ).discover_pending_raw_ids(limit)


class DaemonIntakeService:
    """Supervisor-owned bounded loop; scheduler owns all policy."""

    def __init__(
        self,
        dispatcher: FairIntakeDispatcher,
        *,
        budget: int = DEFAULT_INTAKE_BYTE_BUDGET,
        idle_delay_s: float = 5.0,
        wakeup: asyncio.Event | None = None,
        on_backlog_drained: Callable[[], Awaitable[None] | None] | None = None,
    ) -> None:
        self.dispatcher = dispatcher
        # A count-scale budget (the previous literal 64) left a class's
        # per-pass share three to four orders of magnitude below one ordinary
        # session file, so a single admission drove the deficit deeply
        # negative and the class did no work for hundreds of passes.
        # ``DEFAULT_INTAKE_BYTE_BUDGET`` declares the unit.
        self.budget = max(1, budget)
        self.idle_delay_s = max(0.05, idle_delay_s)
        self._wakeup = wakeup if wakeup is not None else asyncio.Event()
        # polylogue-b7dkb: the one moment a cold build can be declared
        # finished is a pass that found nothing to do AFTER a pass that did
        # something. Fired once; a later backlog is ordinary live ingest.
        self._on_backlog_drained = on_backlog_drained
        self._progressed_once = False

    async def run(self) -> None:
        while True:
            self._wakeup.clear()
            result = await self.dispatcher.run_once(budget=self.budget)
            if result.progressed:
                self._progressed_once = True
            elif self._progressed_once and self._on_backlog_drained is not None:
                drained, self._on_backlog_drained = self._on_backlog_drained, None
                outcome = drained()
                if isinstance(outcome, Awaitable):
                    await outcome
            try:
                async with asyncio.timeout(0.05 if result.progressed else self.idle_delay_s):
                    await self._wakeup.wait()
            except TimeoutError:
                pass


def build_intake_adapters(
    context: DaemonIntakeContext,
    *,
    remote_callback: Callable[[], Awaitable[int] | int] | None = None,
    raw_callback: Callable[..., Awaitable[AdmissionResult | int] | AdmissionResult | int] | None = None,
    raw_discover: Callable[[int], Awaitable[Sequence[tuple[str, int]]] | Sequence[tuple[str, int]]] | None = None,
    hook_events_callback: Callable[..., Awaitable[AdmissionResult | int] | AdmissionResult | int] | None = None,
    hook_events_discover: Callable[[int], Awaitable[Sequence[tuple[str, int]]] | Sequence[tuple[str, int]]]
    | None = None,
) -> tuple[tuple[str, IntakeAdapter], ...]:
    """Compose browser, hook, local, remote, admitted-raw and hook-event classes."""

    result: list[tuple[str, IntakeAdapter]] = []
    local: list[FileIntakeAdapter] = []
    hook_carriers: list[FileIntakeAdapter] = []
    for source in context.sources:
        if source.role in {"primary-writable", "legacy-read-only"}:
            # Hook carriers are acquired by the ordinary file route -- they are
            # append-only JSONL like every other live source. They keep their
            # own intake class so a burst of hook traffic cannot crowd out
            # session files under the dispatcher's per-class fair share.
            hook_carriers.append(FileIntakeAdapter(context, source, class_name="hook_carrier"))
            continue
        if source.name == "browser-capture":
            result.append(("browser_capture", FileIntakeAdapter(context, source, class_name="browser_capture")))
        else:
            local.append(FileIntakeAdapter(context, source, class_name="configured_local"))
    if local:
        result.append(("configured_local", MultiplexIntakeAdapter(local)))
    if hook_carriers:
        result.append(("hook_carrier", MultiplexIntakeAdapter(hook_carriers)))
    if remote_callback is not None:
        result.append(("configured_remote", CallbackIntakeAdapter("configured_remote", remote_callback)))
    if raw_callback is not None:
        if raw_discover is None:
            result.append(("raw_materialization", CallbackIntakeAdapter("raw_materialization", raw_callback)))
        else:

            async def admit_raw(raw_id: str) -> AdmissionResult | int:
                value = raw_callback(raw_id)
                if isinstance(value, Awaitable):
                    value = await value
                return value

            result.append(("raw_materialization", RawMaterializationIntakeAdapter(raw_discover, admit_raw)))
    if hook_events_callback is not None and hook_events_discover is not None:
        # Hook-event materialization is keyed by carrier raw id exactly as raw
        # materialization is keyed by session raw id, so it reuses the same
        # adapter rather than a second one that would have to re-derive the
        # same retry, isolation and cost accounting.

        async def admit_hook_carrier(raw_id: str) -> AdmissionResult | int:
            value = hook_events_callback(raw_id)
            if isinstance(value, Awaitable):
                value = await value
            return value

        result.append(("hook_events", RawMaterializationIntakeAdapter(hook_events_discover, admit_hook_carrier)))
    return tuple(result)
