"""Domain adapters for the fair daemon intake scheduler.

This module owns the source/storage-facing part of intake composition.  The
daemon scheduler stays storage- and source-independent; adapters retain only
disposable discovery hints and delegate writes to the daemon's existing writer
runner.
"""

from __future__ import annotations

import asyncio
import os
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

from polylogue.daemon.intake import (
    AdmissionOutcome,
    AdmissionResult,
    FairIntakeDispatcher,
    IntakeAdapter,
    IntakeItem,
)
from polylogue.sources.hooks import (
    HookSpoolSourceSpec,
    _acknowledge,
    _iter_pending_event_paths,
    _persist_record,
    hook_spool_sources,
    pending_hook_spool_dir,
    read_hook_spool_record,
)
from polylogue.sources.live.source_selection import deepest_source_for_path
from polylogue.sources.live.watcher import LiveWatcher, WatchSource

_T = TypeVar("_T")

__all__ = [
    "DaemonIntakeContext",
    "DaemonIntakeService",
    "FileIntakeAdapter",
    "MultiplexIntakeAdapter",
    "HookSpoolIntakeAdapter",
    "CallbackIntakeAdapter",
    "RawMaterializationIntakeAdapter",
    "discover_pending_raw_ids",
    "build_intake_adapters",
]


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


def _bounded_source_paths(
    source: WatchSource,
    all_sources: tuple[WatchSource, ...],
    *,
    limit: int,
    after: str | None,
) -> list[Path]:
    """Collect at most ``limit`` files, stopping as soon as it is full."""

    if limit <= 0 or not source.root.is_dir():
        return []
    pending: deque[Path] = deque([source.root])
    found: list[Path] = []
    while pending and len(found) < limit:
        directory = pending.popleft()
        try:
            entries = os.scandir(directory)
        except OSError:
            continue
        with entries:
            for entry in entries:
                path = Path(entry.path)
                try:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                if after is not None and str(path) <= after:
                    continue
                try:
                    if deepest_source_for_path(path, all_sources) is not source or not source.accepts(path):
                        continue
                except OSError:
                    continue
                found.append(path)
                if len(found) >= limit:
                    break
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
        paths = _bounded_source_paths(self.source, self.context.sources, limit=limit, after=self._after)
        if paths:
            self._after = str(paths[-1])
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
        path = Path(item.payload) if isinstance(item.payload, (str, Path)) else None
        if path is None:
            return AdmissionResult(AdmissionOutcome.TERMINAL, reason="file intake item has no path")
        if not path.is_file():
            return AdmissionResult(AdmissionOutcome.RETRYABLE, reason=f"source carrier vanished: {path}")
        try:
            cursor = getattr(self.context.watcher, "_cursor", None)
            run_writer_sync = getattr(self.context.watcher, "_run_writer_sync", None)
            if cursor is not None and callable(run_writer_sync):
                await run_writer_sync("watcher.intake.cursor_initialize", cursor.initialize)
            metrics = await self.context.watcher._ingest_files(
                [path], queued_file_count=1, whole_archive_convergence=False
            )
            converge_embeddings = getattr(self.context.watcher, "_converge_embeddings_off_writer", None)
            if callable(converge_embeddings):
                await converge_embeddings([path])
            converge_profiles = getattr(self.context.watcher, "_converge_session_profiles_off_writer", None)
            if callable(converge_profiles):
                await converge_profiles(tuple(getattr(metrics, "changed_session_ids", ()) or ()))
        except (OSError, ValueError, RuntimeError) as exc:
            return AdmissionResult(AdmissionOutcome.RETRYABLE, reason=f"{type(exc).__name__}: {exc}")
        succeeded = int(getattr(metrics, "succeeded_file_count", 0) or 0)
        failed = int(getattr(metrics, "failed_file_count", 0) or 0)
        if succeeded:
            actual = int(getattr(metrics, "source_payload_read_bytes", 0) or item.estimated_cost)
            return AdmissionResult(AdmissionOutcome.ADMITTED, actual_cost=max(1, actual))
        if failed:
            return AdmissionResult(AdmissionOutcome.RETRYABLE, reason=f"source admission failed: {path}")
        return AdmissionResult(AdmissionOutcome.DUPLICATE, actual_cost=item.estimated_cost)

    async def acknowledge(self, item: IntakeItem) -> None:
        # Files remain retained source carriers.  The live batch's durable
        # cursor/raw commit is the acknowledgement projection.
        return None


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

    async def acknowledge(self, item: IntakeItem) -> None:
        adapter = self._by_item.pop(item.item_id, None)
        if adapter is not None:
            await adapter.acknowledge(item)


class HookSpoolIntakeAdapter(IntakeAdapter):
    """Hook adapter with an explicit commit-before-carrier-move boundary."""

    def __init__(
        self, context: DaemonIntakeContext, spec: HookSpoolSourceSpec, *, class_name: str = "hook_spool"
    ) -> None:
        self.context = context
        self.spec = spec
        self.class_name = class_name

    async def discover(self, *, limit: int) -> Sequence[IntakeItem]:
        paths = _iter_pending_event_paths(pending_hook_spool_dir(self.spec.root), limit=limit)
        items: list[IntakeItem] = []
        for path in paths:
            try:
                size = path.stat().st_size
            except OSError:
                size = 1
            items.append(
                IntakeItem(
                    item_id=f"hook:{self.spec.source_id}:{path.name}",
                    class_name=self.class_name,
                    payload=path,
                    estimated_cost=max(1, size),
                )
            )
        return tuple(items)

    async def admit(self, item: IntakeItem) -> AdmissionResult:
        path = Path(item.payload) if isinstance(item.payload, (str, Path)) else None
        if path is None or not path.is_file():
            return AdmissionResult(AdmissionOutcome.RETRYABLE, reason="hook carrier vanished")
        try:
            record = read_hook_spool_record(path)
            await self.context.run_write(
                f"intake.hook.admit.{self.spec.source_id}",
                _persist_hook_record,
                self.context.archive_root,
                path,
                record,
                self.spec,
            )
        except (OSError, ValueError) as exc:
            return AdmissionResult(AdmissionOutcome.TERMINAL, reason=f"invalid hook record: {exc}")
        except Exception as exc:
            return AdmissionResult(AdmissionOutcome.RETRYABLE, reason=f"hook admission failed: {exc}")
        return AdmissionResult(AdmissionOutcome.ADMITTED, actual_cost=max(1, path.stat().st_size))

    async def acknowledge(self, item: IntakeItem) -> None:
        path = Path(item.payload) if isinstance(item.payload, (str, Path)) else None
        if path is not None:
            await self.context.run_write(
                f"intake.hook.ack.{self.spec.source_id}", _acknowledge, path, root=self.spec.root
            )


def _persist_hook_record(archive_root: Path, path: Path, record: dict[str, object], spec: HookSpoolSourceSpec) -> None:
    from polylogue.sources.live.archive_open import _open_archive_for_live_write, _source_tier_acquisition_required

    if not _source_tier_acquisition_required():
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

        initialize_active_archive_root(archive_root)
    store = _open_archive_for_live_write(archive_root)
    with store as archive:
        _persist_record(
            archive,
            path,
            record,
            source_id=spec.source_id,
            source_root=pending_hook_spool_dir(spec.root),
            role=spec.role,
        )
        archive.commit()


class CallbackIntakeAdapter(IntakeAdapter):
    """One-shot bounded remote/raw adapter around an existing domain route."""

    def __init__(
        self,
        class_name: str,
        callback: Callable[[], Awaitable[int] | int],
        *,
        estimated_cost: int = 1,
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
        discover_ids: Callable[[int], Sequence[tuple[str, int]]],
        admit_id: Callable[[str], Awaitable[int] | int],
    ) -> None:
        self._discover_ids = discover_ids
        self._admit_id = admit_id

    async def discover(self, *, limit: int) -> Sequence[IntakeItem]:
        return tuple(
            IntakeItem(raw_id, "raw_materialization", estimated_cost=max(1, int(cost)))
            for raw_id, cost in self._discover_ids(limit)
        )

    async def admit(self, item: IntakeItem) -> AdmissionResult:
        try:
            changed = self._admit_id(item.item_id)
            if isinstance(changed, Awaitable):
                changed = await changed
            return AdmissionResult(
                AdmissionOutcome.ADMITTED if int(changed) else AdmissionOutcome.DUPLICATE,
                actual_cost=item.estimated_cost,
            )
        except Exception as exc:
            return AdmissionResult(AdmissionOutcome.RETRYABLE, reason=f"raw materialization: {exc}")

    async def acknowledge(self, item: IntakeItem) -> None:
        return None


def discover_pending_raw_ids(archive_root: Path, limit: int, max_payload_bytes: int) -> tuple[tuple[str, int], ...]:
    """Return a bounded, stable raw frontier page for the intake adapter.

    Raw discovery is deliberately a read-only derivation traversal.  The old
    census preview was a second backlog authority and could miss observations
    whose parser census was complete but whose derived output had been lost.
    Page through the canonical raw-observation adapter instead, inspecting
    each bounded page before admitting only non-valid observations.  The
    payload size is the scheduler's cost estimate, read from the durable raw
    rows rather than guessed by the adapter.
    """
    if limit <= 0:
        return ()
    from polylogue.operations.raw_observation_derivation import raw_observation_frame
    from polylogue.storage.derived.raw import RawObservationDerivation
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    adapter = RawObservationDerivation(archive_root, max_payload_bytes=max_payload_bytes)
    frame = raw_observation_frame(archive_root)
    pending: list[str] = []
    cursor: str | None = None
    while len(pending) < limit:
        page, next_cursor = adapter.required_page(frame, cursor=cursor, limit=limit - len(pending))
        if not page:
            break
        statuses = adapter.inspect(frame, page)
        pending.extend(raw_id for raw_id in page if statuses.get(raw_id) != "valid")
        if next_cursor is None:
            break
        cursor = next_cursor

    selected = tuple(pending[:limit])
    if not selected:
        return ()
    with ArchiveStore.open_existing(archive_root, read_only=True) as archive:
        sizes = archive.raw_payload_sizes(selected)
    return tuple((raw_id, max(1, int(sizes.get(raw_id, 1)))) for raw_id in selected)


class DaemonIntakeService:
    """Supervisor-owned bounded loop; scheduler owns all policy."""

    def __init__(
        self,
        dispatcher: FairIntakeDispatcher,
        *,
        budget: int = 64,
        idle_delay_s: float = 5.0,
        wakeup: asyncio.Event | None = None,
    ) -> None:
        self.dispatcher = dispatcher
        self.budget = max(1, budget)
        self.idle_delay_s = max(0.05, idle_delay_s)
        self._wakeup = wakeup if wakeup is not None else asyncio.Event()

    async def run(self) -> None:
        while True:
            self._wakeup.clear()
            result = await self.dispatcher.run_once(budget=self.budget)
            try:
                async with asyncio.timeout(0.05 if result.progressed else self.idle_delay_s):
                    await self._wakeup.wait()
            except TimeoutError:
                pass


def build_intake_adapters(
    context: DaemonIntakeContext,
    *,
    remote_callback: Callable[[], Awaitable[int] | int] | None = None,
    raw_callback: Callable[..., Awaitable[int] | int] | None = None,
    raw_discover: Callable[[int], Sequence[tuple[str, int]]] | None = None,
) -> tuple[tuple[str, IntakeAdapter], ...]:
    """Compose browser, hook, local, remote, and admitted-raw classes."""

    result: list[tuple[str, IntakeAdapter]] = []
    local: list[FileIntakeAdapter] = []
    for source in context.sources:
        if source.role in {"primary-writable", "legacy-read-only"}:
            continue
        if source.name == "browser-capture":
            result.append(("browser_capture", FileIntakeAdapter(context, source, class_name="browser_capture")))
        else:
            local.append(FileIntakeAdapter(context, source, class_name="configured_local"))
    if local:
        result.append(("configured_local", MultiplexIntakeAdapter(local)))
    hooks = [HookSpoolIntakeAdapter(context, spec) for spec in hook_spool_sources()]
    if hooks:
        result.append(("hook_spool", MultiplexIntakeAdapter(hooks)))
    if remote_callback is not None:
        result.append(("configured_remote", CallbackIntakeAdapter("configured_remote", remote_callback)))
    if raw_callback is not None:
        if raw_discover is None:
            result.append(("raw_materialization", CallbackIntakeAdapter("raw_materialization", raw_callback)))
        else:

            async def admit_raw(raw_id: str) -> int:
                value = raw_callback(raw_id)
                if isinstance(value, Awaitable):
                    value = await value
                return int(value)

            result.append(("raw_materialization", RawMaterializationIntakeAdapter(raw_discover, admit_raw)))
    return tuple(result)
