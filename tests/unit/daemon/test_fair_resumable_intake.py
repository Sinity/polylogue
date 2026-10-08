"""Fair, resumable intake: no class starves, no item wedges its siblings.

The regression this replaces gave one spool class absolute priority: while
any browser-capture file existed, raw materialization was skipped entirely.
Four undrainable files stopped unrelated work for as long as they sat
there. These tests are written so that restoring class-global preemption,
disabling weighted fairness, or losing the isolation of a poison item makes
one of them red.
"""

from __future__ import annotations

import asyncio
import os
import random
import sqlite3
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import polylogue.operations.raw_observation_derivation as _raw_observation_derivation  # noqa: F401
from polylogue.core.enums import Provider
from polylogue.core.raw_failure_evidence import RetainedRawDecodeRefusalError
from polylogue.daemon.derivation import DerivationFrame
from polylogue.daemon.intake import (
    DEFAULT_INTAKE_BYTE_BUDGET,
    UNMEASURABLE_INTAKE_COST_BYTES,
    AdmissionOutcome,
    AdmissionResult,
    FairIntakeDispatcher,
    IntakeClassSpec,
    IntakeItem,
)
from polylogue.daemon.observation import ObservationBoard, ObservationState
from polylogue.daemon.service_halt import HaltReason, HaltRegistry, UnitKind, unit_id
from polylogue.logging import capture
from polylogue.operations.intake_adapters import (
    CallbackIntakeAdapter,
    DaemonIntakeContext,
    DaemonIntakeService,
    FileIntakeAdapter,
    MultiplexIntakeAdapter,
    RawMaterializationDiscovery,
    RawMaterializationIntakeAdapter,
    _bounded_source_paths,
    discover_pending_raw_ids,
)
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.discovery import _source_path_steps as real_source_path_steps
from polylogue.sources.live.metrics import REFUSED_UNATTEMPTED, REFUSED_UNATTEMPTED_TIME_BUDGET
from polylogue.sources.live.watcher import WatchSource

# Raw-discovery tests patch ``RawObservationDerivation`` where
# ``make_raw_observation_derivation`` reads it: the name bound in
# ``operations.raw_observation_derivation``. That module reads
# ``RawObservationDerivation.recipe_version`` at import, so it is imported here,
# before any test can patch the class it binds.
from polylogue.sources.source_layout import LayoutEntry, SourceLayout, export_drop_layout
from polylogue.sources.walk_faults import WalkFault, WalkRefusedError
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root, run_off_event_loop
from tests.infra.cursor_authority import fixture_cursor_authority


async def _inline_writer_sync(_actor: str, function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """The fakes' declared write coordinator: the test's own loop is the writer."""
    return function(*args, **kwargs)


def _async_selection(
    classify: Callable[[Sequence[Path]], tuple[tuple[Path, ...], tuple[Path, ...]]],
) -> Callable[[Sequence[Path]], Any]:
    """The off-writer selection route over a synchronous fake classifier."""

    async def select(paths: Sequence[Path]) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
        return classify(paths)

    return select


class _InlineWriterWatcher:
    """A watcher fake that declares a write coordinator.

    ``FileIntakeAdapter.admit_page`` runs cursor initialization and candidate
    selection on the watcher's declared writer and refuses cursor writes
    without one (55bdd84105), so every fake that carries a cursor or a
    candidate classifier declares the coordinator the daemon would supply.
    """

    has_write_coordinator = True

    async def _run_writer_sync(self, actor: str, function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        return await _inline_writer_sync(actor, function, *args, **kwargs)

    async def classify_ingest_candidates_off_writer(
        self, paths: Sequence[Path]
    ) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
        selection: tuple[tuple[Path, ...], tuple[Path, ...]] = self.classify_ingest_candidates(paths)  # type: ignore[attr-defined]
        return selection


class FakeAdapter:
    """A spool whose queue authority is its own pending list."""

    def __init__(
        self,
        class_name: str,
        pending: Sequence[str],
        *,
        outcome_for: Callable[[IntakeItem], AdmissionResult] | None = None,
    ) -> None:
        self.class_name = class_name
        self.pending = list(pending)
        self.acknowledged: list[str] = []
        self.admitted: list[str] = []
        self.discover_calls: list[int] = []
        self._outcome_for = outcome_for or (lambda _item: AdmissionResult(AdmissionOutcome.ADMITTED))

    async def discover(self, *, limit: int) -> Sequence[IntakeItem]:
        self.discover_calls.append(limit)
        return [IntakeItem(item_id=name, class_name=self.class_name) for name in self.pending[:limit]]

    async def admit(self, item: IntakeItem) -> AdmissionResult:
        result = self._outcome_for(item)
        if result.outcome is AdmissionOutcome.ADMITTED:
            self.admitted.append(item.item_id)
        return result

    async def acknowledge(self, item: IntakeItem) -> None:
        self.acknowledged.append(item.item_id)
        # Atomic and idempotent: acknowledging twice releases one entry.
        if item.item_id in self.pending:
            self.pending.remove(item.item_id)


@pytest.mark.asyncio
async def test_cold_build_file_pages_drain_before_raw_materialization_resumes() -> None:
    """The candidate must reach its final file page before raw work can occupy the pass."""
    files = FakeAdapter("configured_local", [f"file-{index:02d}" for index in range(33)])
    cold_build_active = True
    raw_discoveries = 0
    raw_admissions: list[str] = []

    def discover_raw(limit: int) -> tuple[tuple[str, int], ...]:
        nonlocal raw_discoveries
        raw_discoveries += 1
        return tuple((f"raw-{index:02d}", 1) for index in range(min(32, limit)))

    def admit_raw(raw_id: str) -> int:
        raw_admissions.append(raw_id)
        return 1

    raw = RawMaterializationIntakeAdapter(
        discover_raw,
        admit_raw,
        suspended=lambda: cold_build_active,
    )
    dispatcher = FairIntakeDispatcher(
        (
            IntakeClassSpec("configured_local", files, page_size=32),
            IntakeClassSpec("raw_materialization", raw, page_size=32),
        )
    )

    first = await dispatcher.run_once()
    second = await dispatcher.run_once()
    drained = await dispatcher.run_once()
    assert first.require_report("configured_local").admitted == 32
    assert second.require_report("configured_local").admitted == 1
    assert drained.quiescent
    assert files.pending == []
    assert raw_discoveries == 0
    assert raw_admissions == []

    cold_build_active = False
    resumed = await dispatcher.run_once()
    assert resumed.require_report("raw_materialization").admitted == 32
    assert len(raw_admissions) == 32


def test_bounded_source_paths_prunes_ignored_subtrees_and_keeps_nested_sources(tmp_path: Path) -> None:
    """An ignored directory cannot consume a page or hide an accepted child."""
    ignored = tmp_path / "ignored"
    ignored.mkdir()
    (ignored / "not-an-intake.json").write_text("ignored")
    nested = tmp_path / "accepted" / "deeper"
    nested.mkdir(parents=True)
    accepted = nested / "intake.json"
    accepted.write_text("accepted")
    source = WatchSource(
        name="test",
        root=tmp_path,
        layout=SourceLayout(
            None,
            (LayoutEntry("intake", ("accepted", "deeper", r"[^/]+\.json"), "accepted/deeper/intake.json"),),
        ),
    )

    first_page = _bounded_source_paths(source, (source,), limit=1, after=None)
    assert first_page == [accepted]
    assert _bounded_source_paths(source, (source,), limit=8, after=str(accepted)) == []


ADVERSARIAL_RELATIVE_PATHS = (
    "s1.json",
    "s2.json",
    "s9.json",
    "s10.json",
    "s100.json",
    "s007.json",
    "S2.json",
    "\u00e9t\u00e9.json",
    "\u0161ok.json",
    "a.json",
    "a/b.json",
    "a/a.json",
    "a-b.json",
    "a/deeper/z.json",
    "ab.json",
)


class _ScandirHandle:
    """Iterable, closeable stand-in for a ``ScandirIterator``."""

    def __init__(self, entries: list[os.DirEntry[str]]) -> None:
        self._entries = iter(entries)

    def __iter__(self) -> Iterator[os.DirEntry[str]]:
        return self._entries

    def __enter__(self) -> _ScandirHandle:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


class _ShuffledScandir:
    """A ``scandir`` whose per-directory order is an arbitrary permutation."""

    def __init__(self, real: Callable[[Path], Any], seed: int) -> None:
        self._real = real
        self._rng = random.Random(seed)

    def __call__(self, directory: Path) -> _ScandirHandle:
        with self._real(directory) as handle:
            entries = list(handle)
        self._rng.shuffle(entries)
        return _ScandirHandle(entries)


def _seed_adversarial_root(root: Path) -> set[Path]:
    written: set[Path] = set()
    for relative in ADVERSARIAL_RELATIVE_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
        written.add(path)
    return written


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("page", [1, 2, 3])
def test_bounded_walk_ingests_every_file_under_any_scandir_order(tmp_path: Path, seed: int, page: int) -> None:
    """Any ``scandir`` permutation must still yield every file on disk.

    The producer's emission order and the resume cursor's comparison key are
    the same order, so a cursor set from a partially consumed page can never
    sort above a file that page never emitted.

    Anti-vacuity: restoring the previous ``os.scandir``-ordered BFS walk (or
    dropping the trailing-separator directory key, which emits ``a/b.json``
    before ``a.json``) leaves files on disk unreachable and turns this red.
    """
    on_disk = _seed_adversarial_root(tmp_path)
    source = WatchSource(name="test", root=tmp_path, layout=export_drop_layout((".json",)))
    # Injected rather than patched onto ``os``: a global ``scandir`` patch
    # also rewires importlib's finder and corrupts unrelated parallel tests.
    scandir = _ShuffledScandir(os.scandir, seed)

    ingested: set[Path] = set()
    after: str | None = None
    for _ in range(len(ADVERSARIAL_RELATIVE_PATHS) + 4):
        discovered = _bounded_source_paths(source, (source,), limit=page, after=after, scandir=scandir)
        if not discovered:
            break
        # The dispatcher admits only a prefix of a page; the cursor advances
        # in ``acknowledge`` over exactly those consumed items.
        consumed = discovered[: max(1, page - 1)]
        ingested.update(consumed)
        for path in consumed:
            position = str(path)
            if after is None or position > after:
                after = position
    assert ingested == on_disk


def test_bounded_walk_emits_exact_lexicographic_path_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Emission order equals sorted path strings, files before their sibling dirs."""
    on_disk = _seed_adversarial_root(tmp_path)
    source = WatchSource(name="test", root=tmp_path, layout=export_drop_layout((".json",)))
    scandir = _ShuffledScandir(os.scandir, 11)

    emitted = _bounded_source_paths(source, (source,), limit=len(on_disk) + 5, after=None, scandir=scandir)
    assert emitted == sorted(on_disk, key=str)
    assert str(tmp_path / "a.json") < str(tmp_path / "a" / "b.json")


@pytest.mark.asyncio
async def test_file_discovery_resumes_after_a_page_of_rejected_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    for index in range(300):
        (root / f"{index:04d}.txt").write_text("ignored")
    accepted = root / "z.json"
    accepted.write_text("{}")
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".json",)))
    watcher = SimpleNamespace(intake_revision=lambda _source: 0)
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    seen: list[Path] = []
    monkeypatch.setattr(
        "polylogue.sources.live.discovery._log_unclaimed_intake_candidate",
        lambda path, **_kwargs: seen.append(path),
    )

    assert await adapter.discover(limit=1) == ()
    assert len(seen) == 256
    page = await adapter.discover(limit=1)
    assert [item.payload for item in page] == [accepted]
    assert len(seen) == 300
    await adapter.acknowledge(page[0])
    assert await adapter.discover(limit=1) == ()
    assert len(seen) == 300


@pytest.mark.asyncio
async def test_file_discovery_keeps_its_page_under_repeated_watcher_hints(tmp_path: Path) -> None:
    """Resetting on each hint repeats the rejected prefix and hides z.json."""
    root = tmp_path / "source"
    root.mkdir()
    for index in range(300):
        (root / f"{index:04d}.txt").write_text("ignored")
    accepted = root / "z.json"
    accepted.write_text("{}")
    changed = root / "zz-hint.json"
    changed.write_text("{}")
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".json",)))

    class HintingWatcher(_InlineWriterWatcher):
        revision = 0

        def intake_revision(self, _source: WatchSource) -> int:
            return self.revision

    watcher = HintingWatcher()
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )

    assert await adapter.discover(limit=1) == ()
    changed.write_text('{"revision":1}')
    watcher.revision += 1
    page = await adapter.discover(limit=1)
    assert [item.payload for item in page] == [accepted]
    await adapter.acknowledge(page[0])

    inserted = root / "0000-new.json"
    inserted.write_text("{}")
    changed.write_text('{"revision":2}')
    watcher.revision += 1
    page = await adapter.discover(limit=1)
    assert [item.payload for item in page] == [changed]
    await adapter.acknowledge(page[0])
    # The lookahead already reached the walk's end, so the queued rescan
    # starts on the next discovery.
    page = await adapter.discover(limit=1)
    assert [item.payload for item in page] == [inserted]


@pytest.mark.asyncio
async def test_file_discovery_retries_queued_rescan_after_walk_failure(tmp_path: Path) -> None:
    """A failed continuation must not erase the hint that found an earlier file."""
    root = tmp_path / "source"
    root.mkdir()
    first = root / "a.json"
    first.write_text("{}")
    (root / "z.json").write_text("{}")
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".json",)))

    class HintingWatcher(_InlineWriterWatcher):
        revision = 0

        def intake_revision(self, _source: WatchSource) -> int:
            return self.revision

    watcher = HintingWatcher()
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    page = await adapter.discover(limit=1)
    assert [item.payload for item in page] == [first]
    await adapter.acknowledge(page[0])

    inserted = root / "0.json"
    inserted.write_text("{}")
    watcher.revision += 1

    def failed_walk() -> Iterator[Path | None]:
        raise OSError("temporary source-walk failure")
        yield None

    adapter._fresh_walk = failed_walk()
    with pytest.raises(OSError, match="temporary source-walk failure"):
        await adapter.discover(limit=1)
    page = await adapter.discover(limit=1)
    assert [item.payload for item in page] == [inserted]


@pytest.mark.asyncio
async def test_vanished_pending_file_does_not_block_walk_or_queued_rescan(tmp_path: Path) -> None:
    """Keeping a vanished retryable page pins both later files and the rescan."""
    root = tmp_path / "source"
    root.mkdir()
    first = root / "a.json"
    vanished = root / "z.json"
    later = root / "zz.json"
    for path in (first, vanished, later):
        path.write_text("{}")
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".json",)))

    class HintingWatcher(_InlineWriterWatcher):
        revision = 0

        def intake_revision(self, _source: WatchSource) -> int:
            return self.revision

    watcher = HintingWatcher()
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    first_page = await adapter.discover(limit=1)
    assert [item.payload for item in first_page] == [first]
    await adapter.acknowledge(first_page[0])
    vanished_page = await adapter.discover(limit=1)
    assert [item.payload for item in vanished_page] == [vanished]

    vanished.unlink()
    inserted = root / "0.json"
    inserted.write_text("{}")
    watcher.revision += 1
    outcome = await adapter.admit_page(vanished_page)
    assert outcome[vanished_page[0].item_id].outcome is AdmissionOutcome.RETRYABLE

    later_page = await adapter.discover(limit=1)
    assert [item.payload for item in later_page] == [later]
    await adapter.acknowledge(later_page[0])
    # The lookahead already reached the walk's end, so the queued rescan
    # starts on the next discovery.
    rescan_page = await adapter.discover(limit=1)
    assert [item.payload for item in rescan_page] == [inserted]


@pytest.mark.asyncio
async def test_unavailable_source_root_keeps_pending_file_retryable(tmp_path: Path) -> None:
    """A missing mount must refuse discovery instead of draining its pending page."""
    root = tmp_path / "source"
    root.mkdir()
    carrier = root / "a.json"
    carrier.write_text("{}")
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".json",)))
    watcher = SimpleNamespace(intake_revision=lambda _source: 0)
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    page = await adapter.discover(limit=1)
    assert [item.payload for item in page] == [carrier]
    assert adapter.discovery_pending

    parked = tmp_path / "parked"
    root.rename(parked)
    with pytest.raises(WalkRefusedError, match="source root"):
        await adapter.discover(limit=1)
    assert not adapter.discovery_pending
    assert adapter._fresh_page_paths == (carrier,)

    parked.rename(root)
    recovered = await adapter.discover(limit=1)
    assert [item.payload for item in recovered] == [carrier]
    assert adapter.discovery_pending


@pytest.mark.asyncio
async def test_unavailable_root_backs_off_due_local_retry(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    carrier = root / "capture.json"
    carrier.write_text("{}")
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".json",)))
    now = [5.0]
    watcher = SimpleNamespace(intake_revision=lambda _source: 0)
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
        clock=lambda: now[0],
    )
    adapter._fresh_retry_debt[carrier] = 5.0
    adapter._fresh_exhausted = True
    adapter._fresh_exhausted_at = 5.0
    root.rename(tmp_path / "parked")

    with pytest.raises(WalkRefusedError):
        await adapter.discover(limit=1)
    assert adapter.retry_due_in_s == 5.0
    now[0] = 9.0
    assert adapter.retry_due_in_s == 1.0
    now[0] = 10.0
    with pytest.raises(WalkRefusedError):
        await adapter.discover(limit=1)
    assert adapter.retry_due_in_s == 5.0


@pytest.mark.asyncio
async def test_root_outage_keeps_failed_sibling_after_later_ack(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    failed, accepted = (root / name for name in ("a.json", "b.json"))
    failed.write_text("{}")
    accepted.write_text("{}")
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".json",)))
    watcher = SimpleNamespace(intake_revision=lambda _source: 0)
    now = [0.0]
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
        clock=lambda: now[0],
    )
    page = await adapter.discover(limit=2)
    assert [item.payload for item in page] == [failed, accepted]
    await adapter.acknowledge(page[1])

    parked = tmp_path / "parked"
    root.rename(parked)
    with pytest.raises(WalkRefusedError, match="source root"):
        await adapter.discover(limit=2)
    parked.rename(root)
    retry = await adapter.discover(limit=2)
    assert [item.payload for item in retry] == [failed]


@pytest.mark.asyncio
async def test_live_retryable_pending_file_yields_to_queued_rescan(tmp_path: Path) -> None:
    """An unacknowledged live page must not pin a newer file before its cursor."""
    root = tmp_path / "source"
    root.mkdir()
    poison = root / "z.json"
    poison.write_text("{}")
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".json",)))

    class RetryWatcher(_InlineWriterWatcher):
        revision = 0

        def intake_revision(self, _source: WatchSource) -> int:
            return self.revision

        async def _ingest_files(self, _paths: Sequence[Path], **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(succeeded_paths=(), failed_paths=(), source_payload_read_bytes=0)

    watcher = RetryWatcher()
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    poison_page = await adapter.discover(limit=1)
    assert [item.payload for item in poison_page] == [poison]
    result = await adapter.admit_page(poison_page)
    assert result[poison_page[0].item_id].outcome is AdmissionOutcome.RETRYABLE

    inserted = root / "0.json"
    inserted.write_text("{}")
    watcher.revision += 1
    emitted: list[Path] = []
    for _ in range(4):
        page = await adapter.discover(limit=1)
        emitted.extend(cast(Path, item.payload) for item in page)
        if inserted in emitted:
            await adapter.acknowledge(next(item for item in page if item.payload == inserted))
            break
    assert inserted in emitted
    retry_page = await adapter.discover(limit=1)
    assert [item.payload for item in retry_page] == [poison]


@pytest.mark.asyncio
@pytest.mark.parametrize("durable_retry", [False, True])
async def test_retry_cooldown_does_not_restart_large_file_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, durable_retry: bool
) -> None:
    """A stable poison retries when due without a 50 ms full-walk loop."""
    root = tmp_path / "source"
    root.mkdir()
    for index in range(300):
        (root / f"{index:04d}.json").write_text("{}")
    poison = root / "z.json"
    poison.write_text("{}")
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".json",)))
    now = [0.0]
    walk_starts: list[str | None] = []

    def counted_steps(*args: Any, **kwargs: Any) -> Iterator[Path | None]:
        walk_starts.append(kwargs.get("after"))
        return real_source_path_steps(*args, **kwargs)

    monkeypatch.setattr("polylogue.operations.intake_adapters._source_path_steps", counted_steps)

    class RetryCursor:
        due = False

        def initialize(self) -> None:
            """Admission initializes a watcher's cursor on its writer first (55bdd84105)."""

        def get_records(self, paths: Sequence[Path]) -> dict[Path, SimpleNamespace]:
            return {path: SimpleNamespace(failure_count=1, next_retry_at="later") for path in paths if path == poison}

        def due_retry_high_water(self, _root: Path) -> str:
            return str(poison)

        def list_due_retry_paths(self, _root: Path, **_kwargs: object) -> tuple[Path, ...]:
            return (poison,) if self.due else ()

    class RetryWatcher(_InlineWriterWatcher):
        revision = 0
        poison_attempts = 0
        admitted: list[Path] = []

        def intake_revision(self, _source: WatchSource) -> int:
            return self.revision

        def classify_ingest_candidates(self, paths: Sequence[Path]) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
            return tuple(path for path in paths if path == poison or path.name == "0.json"), ()

        async def _ingest_files(self, paths: Sequence[Path], **_kwargs: object) -> SimpleNamespace:
            if paths == [poison]:
                self.poison_attempts += 1
                return SimpleNamespace(succeeded_paths=(), failed_paths=(str(poison),), source_payload_read_bytes=0)
            self.admitted.extend(paths)
            return SimpleNamespace(
                succeeded_paths=tuple(str(path) for path in paths),
                failed_paths=(),
                source_payload_read_bytes=sum(path.stat().st_size for path in paths),
            )

    watcher = RetryWatcher()
    cursor = RetryCursor()
    if durable_retry:
        watcher._cursor = cursor  # type: ignore[attr-defined]
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
        clock=lambda: now[0],
    )
    dispatcher = FairIntakeDispatcher(
        [IntakeClassSpec(name="capture", adapter=adapter, page_size=32, max_attempts=1, retry_cooldown_s=5.0)],
        clock=lambda: now[0],
    )

    for _ in range(16):
        await dispatcher.run_once()
        if watcher.poison_attempts:
            break
    assert watcher.poison_attempts == 1
    assert len(walk_starts) == 1
    for _ in range(20):
        await dispatcher.run_once()
    assert len(walk_starts) == 1
    assert watcher.poison_attempts == 1
    assert not adapter.discovery_pending

    now[0] = 5.1
    await dispatcher.run_once()
    if durable_retry:
        assert watcher.poison_attempts == 1
        cursor.due = True
        await dispatcher.run_once()
    assert watcher.poison_attempts == 2
    assert len(walk_starts) == 1

    if not durable_retry:
        now[0] = 10.2
        for _ in range(2):
            await dispatcher.run_once()
        assert watcher.poison_attempts == 3
        assert len(walk_starts) == 1
        watcher._cursor = cursor  # type: ignore[attr-defined]
        now[0] = 15.3
        await dispatcher.run_once()
        assert watcher.poison_attempts == 3
        cursor.due = True
        await dispatcher.run_once()
        assert watcher.poison_attempts == 4

    inserted = root / "0.json"
    inserted.write_text("{}")
    watcher.revision += 1
    for _ in range(3):
        await dispatcher.run_once()
        if inserted in watcher.admitted:
            break
    assert inserted in watcher.admitted
    assert len(walk_starts) == 2


@pytest.mark.asyncio
async def test_retry_debt_overflow_revisits_evicted_file_after_cooldown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bounded retry cache cannot hide its 257th live carrier for ten minutes."""
    root = tmp_path / "source"
    root.mkdir()
    files = tuple(root / f"{index:04d}.json" for index in range(300))
    for path in files:
        path.write_text("{}")
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".json",)))
    now = [0.0]
    walk_starts = 0

    def counted_steps(*args: Any, **kwargs: Any) -> Iterator[Path | None]:
        nonlocal walk_starts
        walk_starts += 1
        return real_source_path_steps(*args, **kwargs)

    monkeypatch.setattr("polylogue.operations.intake_adapters._source_path_steps", counted_steps)

    class BurstWatcher(_InlineWriterWatcher):
        admitted: list[Path] = []

        def intake_revision(self, _source: WatchSource) -> int:
            return 0

        def classify_ingest_candidates(self, paths: Sequence[Path]) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
            return tuple(paths), ()

        async def _ingest_files(self, paths: Sequence[Path], **_kwargs: object) -> SimpleNamespace:
            if now[0] < 5.0:
                return SimpleNamespace(
                    succeeded_paths=(),
                    failed_paths=tuple(str(path) for path in paths),
                    source_payload_read_bytes=0,
                )
            self.admitted.extend(paths)
            return SimpleNamespace(
                succeeded_paths=tuple(str(path) for path in paths),
                failed_paths=(),
                source_payload_read_bytes=sum(path.stat().st_size for path in paths),
            )

    watcher = BurstWatcher()
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
        clock=lambda: now[0],
    )
    dispatcher = FairIntakeDispatcher(
        [IntakeClassSpec(name="capture", adapter=adapter, page_size=32, max_attempts=1, retry_cooldown_s=5.0)],
        clock=lambda: now[0],
    )

    for _ in range(24):
        await dispatcher.run_once()
    assert walk_starts == 1
    assert len(adapter._fresh_retry_debt) == 256
    assert files[0] not in adapter._fresh_retry_debt
    assert not adapter.discovery_pending

    now[0] = 5.1
    for _ in range(40):
        await dispatcher.run_once()
        if files[0] in watcher.admitted:
            break
    assert files[0] in watcher.admitted
    assert walk_starts == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("vanish_without_hint", [False, True])
async def test_mixed_fresh_page_keeps_failed_sibling_retryable(tmp_path: Path, vanish_without_hint: bool) -> None:
    root = tmp_path / "source"
    root.mkdir()
    failed = root / "a" / "failed.json" if vanish_without_hint else root / "a.json"
    failed.parent.mkdir(exist_ok=True)
    accepted = root / "b.json"
    failed.write_text("{}")
    accepted.write_text("{}")
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".json",)))
    now = [0.0]

    class MixedWatcher(_InlineWriterWatcher):
        admitted: list[Path] = []

        def intake_revision(self, _source: WatchSource) -> int:
            return 0

        def classify_ingest_candidates(self, paths: Sequence[Path]) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
            return tuple(paths), ()

        async def _ingest_files(self, paths: Sequence[Path], **_kwargs: object) -> SimpleNamespace:
            if failed in paths and now[0] < 5.0 and vanish_without_hint and failed.exists():
                failed.unlink()
            succeeded = tuple(path for path in paths if path != failed or now[0] >= 5.0)
            self.admitted.extend(succeeded)
            return SimpleNamespace(
                succeeded_paths=tuple(str(path) for path in succeeded),
                failed_paths=(str(failed),) if failed in paths and now[0] < 5.0 else (),
                source_payload_read_bytes=2 * len(succeeded),
            )

    watcher = MixedWatcher()
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
        clock=lambda: now[0],
    )
    dispatcher = FairIntakeDispatcher(
        [IntakeClassSpec(name="capture", adapter=adapter, page_size=2, retry_cooldown_s=5.0)],
        clock=lambda: now[0],
    )
    await dispatcher.run_once()
    assert accepted in watcher.admitted
    assert failed not in watcher.admitted
    await dispatcher.run_once()
    assert adapter.retry_due_in_s == 5.0

    if vanish_without_hint:
        failed.write_text("{}")
    now[0] = 5.1
    for _ in range(5):
        await dispatcher.run_once()
        if failed in watcher.admitted:
            break
    assert failed in watcher.admitted


@pytest.mark.asyncio
async def test_fresh_ack_clears_obsolete_local_retry_debt(tmp_path: Path) -> None:
    carrier = tmp_path / "capture.json"
    carrier.write_text("{}")
    source = WatchSource(name="capture", root=tmp_path, layout=export_drop_layout((".json",)))
    watcher = SimpleNamespace(intake_revision=lambda _source: 0)
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
        clock=lambda: 0.0,
    )
    adapter._fresh_retry_debt[carrier] = 5.0
    page = await adapter.discover(limit=1)
    assert [item.payload for item in page] == [carrier]
    await adapter.acknowledge(page[0])
    assert adapter.retry_due_in_s is None


def test_symlink_alias_releases_local_retry_debt(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    target = root / "target.json"
    target.write_text("{}")
    alias = root / "alias.json"
    alias.write_text("{}")
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".json",)))
    now = [0.0]
    watcher = SimpleNamespace(intake_revision=lambda _source: 0)
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
        clock=lambda: now[0],
    )
    adapter._fresh_retry_debt[alias] = 5.0
    alias.unlink()
    alias.symlink_to(target)

    now[0] = 5.1
    assert adapter._due_retry_paths(2) == []
    assert alias not in adapter._fresh_retry_debt
    assert adapter.retry_due_in_s is None


def test_inaccessible_nested_carrier_keeps_local_retry_debt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "source"
    root.mkdir()
    nested = root / "nested" / "capture.json"
    nested.parent.mkdir()
    nested.write_text("{}")
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".json",)))
    watcher = SimpleNamespace(intake_revision=lambda _source: 0)
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
        clock=lambda: 5.0,
    )
    adapter._fresh_retry_debt[nested] = 5.0
    original_lstat = Path.lstat

    def inaccessible_lstat(path: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        if path == nested:
            raise PermissionError(path)
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", inaccessible_lstat)
    assert adapter._due_retry_paths(1) == [nested]
    assert adapter.retry_due_in_s == 5.0


@pytest.mark.asyncio
async def test_vanished_local_retry_gets_one_due_rescan(tmp_path: Path) -> None:
    root = tmp_path / "source"
    nested = root / "nested"
    nested.mkdir(parents=True)
    carrier = nested / "capture.json"
    carrier.write_text("{}")
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".json",)))
    watcher = SimpleNamespace(intake_revision=lambda _source: 0)
    now = [5.0]
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
        clock=lambda: now[0],
    )
    adapter._fresh_retry_debt[carrier] = 5.0
    adapter._fresh_exhausted = True
    adapter._fresh_exhausted_at = 5.0
    carrier.unlink()
    assert adapter._due_retry_paths(1) == []
    assert carrier not in adapter._fresh_retry_debt
    assert adapter.retry_due_in_s == 5.0

    carrier.write_text("{}")
    now[0] = 10.1
    page = await adapter.discover(limit=1)
    assert [item.payload for item in page] == [carrier]


@pytest.mark.asyncio
@pytest.mark.parametrize("cursor_state", ["excluded", "dormant"])
async def test_cursor_row_without_due_authority_does_not_replace_local_retry_debt(
    tmp_path: Path, cursor_state: str
) -> None:
    carrier = tmp_path / "capture.json"
    carrier.write_text("{}")
    source = WatchSource(name="capture", root=tmp_path, layout=export_drop_layout((".json",)))
    now = [0.0]

    class ExcludedCursor:
        def get_records(self, paths: Sequence[Path]) -> dict[Path, SimpleNamespace]:
            return {
                path: SimpleNamespace(
                    failure_count=1 if cursor_state == "excluded" else 0,
                    content_fingerprint=None if cursor_state == "excluded" else "old",
                    next_retry_at="later",
                    excluded=cursor_state == "excluded",
                )
                for path in paths
            }

        def list_due_retry_paths(self, _root: Path, **_kwargs: object) -> tuple[Path, ...]:
            return ()

    class RetryWatcher(_InlineWriterWatcher):
        _cursor = ExcludedCursor()

        def intake_revision(self, _source: WatchSource) -> int:
            return 0

        def classify_ingest_candidates(self, paths: Sequence[Path]) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
            return tuple(paths), ()

        async def _ingest_files(self, _paths: Sequence[Path], **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(succeeded_paths=(), failed_paths=(str(carrier),), source_payload_read_bytes=0)

    watcher = RetryWatcher()
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
        clock=lambda: now[0],
    )
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="capture", adapter=adapter)], clock=lambda: now[0])
    await dispatcher.run_once()
    await dispatcher.run_once()
    assert adapter.retry_due_in_s == 5.0

    now[0] = 5.1
    assert [item.payload for item in await adapter.discover(limit=1)] == [carrier]


@pytest.mark.asyncio
@pytest.mark.parametrize("escape", [False, True])
async def test_durable_retry_alias_is_retired_after_symlink_swap(tmp_path: Path, escape: bool) -> None:
    root = tmp_path / "source"
    root.mkdir()
    carrier = root / "alias.json"
    target = (tmp_path if escape else root) / "target.json"
    carrier.write_text("{}")
    target.write_text("{}")
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".json",)))
    cursor = CursorStore(tmp_path / "index.db")
    cursor.set(carrier, 2, next_retry_at="1970-01-01T00:00:00+00:00", authority=fixture_cursor_authority(carrier))
    parked = root / "parked.json"
    carrier.rename(parked)
    carrier.symlink_to(target)
    alias_stat = carrier.lstat()
    watcher = SimpleNamespace(
        has_write_coordinator=True,
        _run_writer_sync=_inline_writer_sync,
        _cursor=cursor,
        intake_revision=lambda _source: 0,
    )
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    adapter._retry_turn = True
    page = await adapter.discover(limit=1)
    assert [item.payload for item in page] == [carrier]
    outcomes = await adapter.admit_page(page)
    assert outcomes[page[0].item_id].outcome is AdmissionOutcome.RETRYABLE
    assert cursor.has_pending_retries((root,)) is False
    assert cursor.get_record(carrier).st_ino == alias_stat.st_ino  # type: ignore[union-attr]
    carrier.unlink()
    parked.rename(carrier)
    restored = carrier.stat()
    cursor.revive_replaced_exclusion(
        carrier,
        byte_size=restored.st_size,
        st_dev=restored.st_dev,
        st_ino=restored.st_ino,
        mtime_ns=restored.st_mtime_ns,
    )
    assert cursor.get_record(carrier).excluded is False  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_escaping_retry_alias_has_bounded_cost_and_distinct_source_identity(tmp_path: Path) -> None:
    alias_root = tmp_path / "alias-source"
    target_root = tmp_path / "target-source"
    alias_root.mkdir()
    target_root.mkdir()
    alias = alias_root / "capture.json"
    target = target_root / "capture.json"
    with target.open("wb") as target_file:
        target_file.truncate(1 << 40)
    alias.symlink_to(target)
    sources = (
        WatchSource(name="alias", root=alias_root, layout=export_drop_layout((".json",))),
        WatchSource(name="target", root=target_root, layout=export_drop_layout((".json",))),
    )
    cursor = CursorStore(tmp_path / "index.db")
    cursor.set(alias, 2, next_retry_at="1970-01-01T00:00:00+00:00", authority=fixture_cursor_authority(alias))
    watcher = SimpleNamespace(
        has_write_coordinator=True,
        _run_writer_sync=_inline_writer_sync,
        _cursor=cursor,
        intake_revision=lambda _source: 0,
    )
    context = DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=sources)  # type: ignore[arg-type]
    alias_adapter = FileIntakeAdapter(context, sources[0])
    target_adapter = FileIntakeAdapter(context, sources[1])
    alias_adapter._retry_turn = True
    multiplex = MultiplexIntakeAdapter((alias_adapter, target_adapter))

    page = await multiplex.discover(limit=2)
    assert [item.payload for item in page] == [alias, target]
    assert [item.estimated_cost for item in page] == [1, 1 << 40]
    assert len({item.item_id for item in page}) == 2
    assert multiplex._by_item[page[0].item_id] is alias_adapter
    assert multiplex._by_item[page[1].item_id] is target_adapter


@pytest.mark.asyncio
async def test_durable_alias_retirement_respects_cursor_authority(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    carrier = root / "alias.json"
    target = root / "target.json"
    carrier.write_text("{}")
    target.write_text("{}")
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".json",)))
    cursor = CursorStore(tmp_path / "index.db")
    cursor.set(carrier, 2, next_retry_at="1970-01-01T00:00:00+00:00", authority=fixture_cursor_authority(carrier))
    carrier.unlink()
    carrier.symlink_to(target)

    class RefusingProcessor:
        def require_cursor_authority(self, _paths: Sequence[Path]) -> None:
            raise RuntimeError("cursor authority refused")

    watcher = SimpleNamespace(
        has_write_coordinator=True,
        _run_writer_sync=_inline_writer_sync,
        _cursor=cursor,
        _batch_processor=RefusingProcessor(),
        intake_revision=lambda _source: 0,
    )
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    adapter._retry_turn = True
    page = await adapter.discover(limit=1)
    outcomes = await adapter.admit_page(page)
    assert outcomes[page[0].item_id].outcome is AdmissionOutcome.RETRYABLE
    assert cursor.get_record(carrier).excluded is False  # type: ignore[union-attr]
    assert cursor.has_pending_retries((root,)) is True


@pytest.mark.asyncio
async def test_durable_alias_retirement_skips_path_scoped_refusal(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    alias = root / "a.json"
    sibling = root / "b.json"
    sibling.write_text("{}")
    alias.symlink_to(sibling)
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".json",)))
    cursor = CursorStore(tmp_path / "index.db")
    cursor.set(alias, 2, next_retry_at="1970-01-01T00:00:00+00:00", authority=fixture_cursor_authority(alias))

    class PartialRefusalProcessor:
        _refused_paths: frozenset[Path] = frozenset()

        def require_cursor_authority(self, paths: Sequence[Path]) -> None:
            assert paths == [alias, sibling]
            self._refused_paths = frozenset((alias,))

    async def ingest(paths: Sequence[Path], **_kwargs: object) -> SimpleNamespace:
        assert paths == [sibling]
        return SimpleNamespace(succeeded_paths=(str(sibling),), source_payload_read_bytes=2)

    watcher = SimpleNamespace(
        has_write_coordinator=True,
        _run_writer_sync=_inline_writer_sync,
        _cursor=cursor,
        _batch_processor=PartialRefusalProcessor(),
        intake_revision=lambda _source: 0,
        _ingest_files=ingest,
    )
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    alias_item = IntakeItem(item_id=f"file:{alias}", class_name="capture", payload=alias)
    sibling_item = IntakeItem(item_id=f"file:{sibling}", class_name="capture", payload=sibling)
    outcomes = await adapter.admit_page((alias_item, sibling_item))
    assert outcomes[alias_item.item_id].outcome is AdmissionOutcome.RETRYABLE
    assert outcomes[sibling_item.item_id].outcome is AdmissionOutcome.ADMITTED
    assert watcher._batch_processor._refused_paths == frozenset()
    assert cursor.get_record(alias).excluded is False  # type: ignore[union-attr]
    assert cursor.has_pending_retries((root,)) is True


@pytest.mark.asyncio
async def test_path_scoped_refusal_skips_regular_candidate_selection(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    blocked, allowed = (root / name for name in ("a.json", "b.json"))
    blocked.write_text("{}")
    allowed.write_text("{}")
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".json",)))
    cursor = CursorStore(tmp_path / "index.db")
    cursor.set(blocked, 2, next_retry_at="1970-01-01T00:00:00+00:00", authority=fixture_cursor_authority(blocked))

    class PartialRefusalProcessor:
        _refused_paths: frozenset[Path] = frozenset()

        def require_cursor_authority(self, paths: Sequence[Path]) -> None:
            assert paths == [blocked, allowed]
            self._refused_paths = frozenset((blocked,))

    selected: list[Path] = []

    def select(paths: Sequence[Path]) -> tuple[Path, ...]:
        selected.extend(paths)
        return tuple(paths)

    async def ingest(paths: Sequence[Path], **_kwargs: object) -> SimpleNamespace:
        assert paths == [allowed]
        return SimpleNamespace(succeeded_paths=(str(allowed),), source_payload_read_bytes=2)

    watcher = SimpleNamespace(
        has_write_coordinator=True,
        _run_writer_sync=_inline_writer_sync,
        _cursor=cursor,
        _batch_processor=PartialRefusalProcessor(),
        intake_revision=lambda _source: 0,
        classify_ingest_candidates=lambda paths: (select(paths), ()),
        classify_ingest_candidates_off_writer=_async_selection(lambda paths: (select(paths), ())),
        _ingest_files=ingest,
    )
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    page = await adapter.discover(limit=2)
    outcomes = await adapter.admit_page(page)
    assert selected == [allowed]
    assert outcomes[page[0].item_id].outcome is AdmissionOutcome.RETRYABLE
    assert outcomes[page[1].item_id].outcome is AdmissionOutcome.ADMITTED
    assert watcher._batch_processor._refused_paths == frozenset()
    assert cursor.has_pending_retries((root,)) is True


@pytest.mark.asyncio
async def test_a_scheduled_retry_is_deferred_not_acknowledged_as_a_duplicate(tmp_path: Path) -> None:
    """A path whose cursor retry is not yet due stays owed work.

    Anti-vacuity (polylogue-b8of0): reporting every unselected path as a
    DUPLICATE acknowledges the page and drops the scheduled retry.
    """
    root = tmp_path / "source"
    root.mkdir()
    owed, current = (root / name for name in ("a.json", "b.json"))
    owed.write_text("{}")
    current.write_text("{}")
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".json",)))
    ingested: list[Path] = []

    class RefusingNothing:
        _refused_paths: frozenset[Path] = frozenset()

        def require_cursor_authority(self, paths: Sequence[Path]) -> None:
            return None

    async def ingest(paths: Sequence[Path], **_kwargs: object) -> SimpleNamespace:
        ingested.extend(paths)
        return SimpleNamespace(succeeded_paths=(), source_payload_read_bytes=0)

    watcher = SimpleNamespace(
        has_write_coordinator=True,
        _run_writer_sync=_inline_writer_sync,
        _cursor=CursorStore(tmp_path / "index.db"),
        _batch_processor=RefusingNothing(),
        intake_revision=lambda _source: 0,
        classify_ingest_candidates=lambda paths: ((), (owed,)),
        classify_ingest_candidates_off_writer=_async_selection(lambda paths: ((), (owed,))),
        _ingest_files=ingest,
    )
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    page = await adapter.discover(limit=2)
    outcomes = await adapter.admit_page(page)
    by_path = {Path(cast(Any, item.payload)): outcomes[item.item_id] for item in page}
    assert by_path[owed].outcome is AdmissionOutcome.DEFERRED
    assert by_path[owed].actual_cost == 0
    assert by_path[current].outcome is AdmissionOutcome.DUPLICATE
    assert ingested == []


@pytest.mark.asyncio
async def test_partially_planned_local_retry_rotates_past_poison(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    poison, healthy, later = (root / name for name in ("a.json", "b.json", "c.json"))
    for path in (poison, healthy, later):
        path.write_text("{}")
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".json",)))
    now = [5.0]

    class RetryWatcher(_InlineWriterWatcher):
        admitted: list[Path] = []

        def intake_revision(self, _source: WatchSource) -> int:
            return 0

        def classify_ingest_candidates(self, paths: Sequence[Path]) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
            return tuple(paths), ()

        async def _ingest_files(self, paths: Sequence[Path], **_kwargs: object) -> SimpleNamespace:
            succeeded = tuple(path for path in paths if path != poison)
            self.admitted.extend(succeeded)
            return SimpleNamespace(
                succeeded_paths=tuple(str(path) for path in succeeded),
                failed_paths=(str(poison),) if poison in paths else (),
                source_payload_read_bytes=2 * len(succeeded),
            )

    watcher = RetryWatcher()
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
        clock=lambda: now[0],
    )
    adapter._fresh_retry_debt.update({poison: 5.0, healthy: 5.0, later: 5.0})
    adapter._fresh_exhausted = True
    adapter._fresh_exhausted_at = 5.0
    adapter._retry_turn = True
    dispatcher = FairIntakeDispatcher(
        [IntakeClassSpec(name="capture", adapter=adapter, page_size=3, retry_cooldown_s=5.0)],
        clock=lambda: now[0],
    )

    await dispatcher.run_once(budget=1)
    assert healthy not in watcher.admitted
    now[0] = 10.1
    for _ in range(3):
        await dispatcher.run_once(budget=1)
        if healthy in watcher.admitted:
            break
    assert healthy in watcher.admitted


@pytest.mark.asyncio
async def test_multiplex_ownership_does_not_keep_released_pages() -> None:
    source = FakeAdapter("configured_local", ["file-0"])
    multiplex = MultiplexIntakeAdapter((source,))
    for index in range(300):
        source.pending = [f"file-{index}"]
        page = await multiplex.discover(limit=1)
        assert [item.item_id for item in page] == [f"file-{index}"]
        assert len(multiplex._by_item) == 1


@pytest.mark.asyncio
async def test_pending_path_replaced_by_escaping_symlink_is_not_admitted(tmp_path: Path) -> None:
    """A changed pending carrier must pass the source boundary again."""
    root = tmp_path / "source"
    root.mkdir()
    carrier = root / "a.json"
    carrier.write_text("{}")
    outside = tmp_path / "outside.json"
    outside.write_text("outside")
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".json",)))

    class GuardedWatcher(_InlineWriterWatcher):
        def intake_revision(self, _source: WatchSource) -> int:
            return 0

        async def _ingest_files(self, _paths: Sequence[Path], **_kwargs: object) -> None:
            raise AssertionError("escaping symlink reached ingest")

    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=GuardedWatcher(), sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    page = await adapter.discover(limit=1)
    carrier.unlink()
    carrier.symlink_to(outside)
    result = await adapter.admit_page(page)
    assert result[page[0].item_id].outcome is AdmissionOutcome.RETRYABLE


@pytest.mark.asyncio
async def test_exhausted_file_walk_recovers_a_missed_nested_change(tmp_path: Path) -> None:
    root = tmp_path / "source"
    nested = root / "a"
    nested.mkdir(parents=True)
    original = root / "z.json"
    original.write_text("{}")
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".json",)))
    watcher = SimpleNamespace(intake_revision=lambda _source: 0)
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    page = await adapter.discover(limit=1)
    assert [item.payload for item in page] == [original]
    await adapter.acknowledge(page[0])
    assert await adapter.discover(limit=1) == ()
    assert adapter._fresh_exhausted

    root_mtime = root.stat().st_mtime_ns
    inserted = nested / "new.json"
    inserted.write_text("{}")
    assert root.stat().st_mtime_ns == root_mtime
    adapter._fresh_exhausted_at = 0.0
    replay = await adapter.discover(limit=1)
    assert [item.payload for item in replay] == [inserted]


@pytest.mark.asyncio
async def test_intake_service_keeps_scanning_before_declaring_backlog_drained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    for index in range(300):
        (root / f"{index:04d}.txt").write_text("ignored")
    accepted = root / "z.json"
    accepted.write_text("{}")
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".json",)))
    watcher = SimpleNamespace(intake_revision=lambda _source: 0)
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    monkeypatch.setattr(
        "polylogue.sources.live.discovery._log_unclaimed_intake_candidate", lambda *_args, **_kwargs: None
    )

    async def admit_page(items: Sequence[IntakeItem]) -> dict[str, AdmissionResult]:
        return {item.item_id: AdmissionResult(AdmissionOutcome.DUPLICATE) for item in items}

    monkeypatch.setattr(adapter, "admit_page", admit_page)
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="capture", adapter=adapter, page_size=1)])
    drained = asyncio.Event()
    drained_after: list[str | None] = []

    def on_drained() -> None:
        drained_after.append(adapter._after)
        drained.set()

    service = DaemonIntakeService(dispatcher, idle_delay_s=5.0, on_backlog_drained=on_drained)
    task = asyncio.create_task(service.run())
    try:
        await asyncio.wait_for(drained.wait(), timeout=2.0)
        assert drained_after == [str(accepted)]
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
@pytest.mark.parametrize("multiplex", [False, True])
async def test_cold_build_waits_for_local_retry_debt_without_cursor_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, multiplex: bool
) -> None:
    """A failed file cannot settle the candidate before its local retry succeeds."""
    monkeypatch.setattr("polylogue.operations.intake_adapters._FILE_RETRY_DELAY_S", 0.1)
    path = tmp_path / "capture.json"
    path.write_text("{}")
    source = WatchSource(name="capture", root=tmp_path, layout=export_drop_layout((".json",)))

    class ColdBuildWatcher(_InlineWriterWatcher):
        attempts = 0
        admitted = False

        def intake_revision(self, _source: WatchSource) -> int:
            return 0

        def classify_ingest_candidates(self, paths: Sequence[Path]) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
            return tuple(paths), ()

        async def _ingest_files(self, paths: Sequence[Path], **_kwargs: object) -> SimpleNamespace:
            self.attempts += 1
            if self.attempts == 1:
                return SimpleNamespace(succeeded_paths=(), failed_paths=(str(path),), source_payload_read_bytes=0)
            self.admitted = True
            return SimpleNamespace(succeeded_paths=(str(path),), failed_paths=(), source_payload_read_bytes=2)

    watcher = ColdBuildWatcher()
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    scheduled_adapter = MultiplexIntakeAdapter((adapter,)) if multiplex else adapter
    dispatcher = FairIntakeDispatcher(
        [IntakeClassSpec(name="capture", adapter=scheduled_adapter, retry_cooldown_s=0.1, max_attempts=1)]
    )
    drained = asyncio.Event()
    settled_after_admission: list[bool] = []

    def on_drained() -> None:
        settled_after_admission.append(watcher.admitted)
        drained.set()

    service = DaemonIntakeService(
        dispatcher,
        idle_delay_s=5.0,
        on_backlog_drained=on_drained,
        has_pending_backlog=lambda: False,
    )
    task = asyncio.create_task(service.run())
    try:
        await asyncio.wait_for(drained.wait(), timeout=2.0)
        assert watcher.attempts == 2
        assert settled_after_admission == [True]
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_local_retry_deadline_does_not_block_on_filesystem_walk(tmp_path: Path) -> None:
    """A held discovery walk cannot stall the event-loop retry deadline read."""
    source = WatchSource(name="capture", root=tmp_path, layout=export_drop_layout((".json",)))
    watcher = SimpleNamespace(intake_revision=lambda _source: 0)
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
        clock=lambda: 0.0,
    )
    walking = threading.Event()
    release = threading.Event()
    adapter._fresh_retry_debt[tmp_path / "capture.json"] = 5.0

    def hold_discovery_lock() -> None:
        with adapter._discovery_lock:
            walking.set()
            release.wait(timeout=1.0)

    writer = threading.Thread(target=hold_discovery_lock)
    writer.start()
    assert await asyncio.to_thread(walking.wait, 1.0)
    try:
        started_at = asyncio.get_running_loop().time()
        assert adapter.retry_due_in_s == 5.0
        assert asyncio.get_running_loop().time() - started_at < 0.2
    finally:
        release.set()
        writer.join(timeout=2.0)


def test_raw_discovery_uses_canonical_adapter_and_returns_payload_costs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A census selector cannot hide a pending raw from fair intake."""
    bootstrap_archive_root(tmp_path)
    payloads = {
        "a.json": b"raw-a",
        "b.json": b"raw-b-longer",
        "c.json": b"raw-c",
    }
    raw_ids: dict[str, str] = {}
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        for path, payload in payloads.items():
            raw_ids[path] = archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=payload,
                source_path=path,
                canonical_source_path=path,
                acquired_at_ms=1,
            )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("legacy raw census selector was called")

    monkeypatch.setattr("polylogue.storage.archive_readiness.raw_materialization_readiness_snapshot", forbidden)
    result = discover_pending_raw_ids(tmp_path, limit=2)

    expected = tuple(
        (raw_id, len(payloads[path])) for path, raw_id in sorted(raw_ids.items(), key=lambda item: item[1])[:2]
    )
    assert result == expected


def test_raw_discovery_bounds_valid_prefix_and_resumes_after_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: scan until a pending raw is found, and the first call exceeds one page."""
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        valid = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"v",
            source_path="valid.json",
            canonical_source_path="valid.json",
            acquired_at_ms=1,
        )
        pending = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"pending",
            source_path="pending.json",
            canonical_source_path="pending.json",
            acquired_at_ms=1,
        )
    calls: list[tuple[str | None, int]] = []

    class FakeRawObservationDerivation:
        def terminal_decode_refusals(self, keys: Sequence[str]) -> dict[str, RetainedRawDecodeRefusalError]:
            return {}

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def required_page(
            self, _frame: object, *, cursor: str | None, limit: int
        ) -> tuple[tuple[str, ...], str | None]:
            calls.append((cursor, limit))
            return ((valid,), valid) if cursor is None else ((pending,), None)

        def inspect(self, _frame: object, keys: Sequence[str]) -> dict[str, str]:
            return {key: "valid" if key == valid else "missing" for key in keys}

    monkeypatch.setattr("polylogue.storage.derived.raw.RawObservationInspection", FakeRawObservationDerivation)
    discovery = RawMaterializationDiscovery(tmp_path)

    assert discovery.discover_pending_raw_ids(1) == ()
    assert calls == [(None, 1)]
    assert discovery.discover_pending_raw_ids(1) == ((pending, len(b"pending")),)
    assert calls == [(None, 1), (valid, 1)]


@pytest.mark.asyncio
async def test_raw_discovery_moves_past_a_cooled_down_poison_in_the_fair_dispatcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: reset discovery each pass, and the cooled-down head starves the healthy raw."""

    def seed() -> tuple[str, ...]:
        # Bootstrap and raw writes take the synchronous lease; keep them off the loop.
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            return tuple(
                archive.write_raw_payload(
                    provider=Provider.CHATGPT,
                    payload=payload,
                    source_path=source_path,
                    canonical_source_path=source_path,
                    acquired_at_ms=1,
                )
                for payload, source_path in ((b"v", "valid.json"), (b"p", "poison.json"), (b"h", "healthy.json"))
            )

    valid, poison, healthy = run_off_event_loop(seed)

    class FakeRawObservationDerivation:
        def terminal_decode_refusals(self, keys: Sequence[str]) -> dict[str, RetainedRawDecodeRefusalError]:
            return {}

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def required_page(
            self, _frame: object, *, cursor: str | None, limit: int
        ) -> tuple[tuple[str, ...], str | None]:
            assert limit <= 32
            if cursor is None:
                return (valid,), valid
            if cursor == valid:
                return (poison,), poison
            assert cursor == poison
            return (healthy,), None

        def inspect(self, _frame: object, keys: Sequence[str]) -> dict[str, str]:
            return {key: "valid" if key == valid else "missing" for key in keys}

    monkeypatch.setattr("polylogue.storage.derived.raw.RawObservationInspection", FakeRawObservationDerivation)
    admitted: list[str] = []

    async def admit(raw_id: str) -> AdmissionResult:
        if raw_id == poison:
            return AdmissionResult(AdmissionOutcome.RETRYABLE, reason="poison")
        admitted.append(raw_id)
        return AdmissionResult(AdmissionOutcome.ADMITTED)

    discovery = RawMaterializationDiscovery(tmp_path)
    dispatcher = FairIntakeDispatcher(
        [
            IntakeClassSpec(
                name="raw_materialization",
                adapter=RawMaterializationIntakeAdapter(discovery.discover_pending_raw_ids, admit),
                max_attempts=1,
            )
        ]
    )

    await dispatcher.run_once(budget=1)
    await dispatcher.run_once(budget=1)
    await dispatcher.run_once(budget=1)

    assert dispatcher.isolated_items("raw_materialization") == frozenset()
    assert admitted == [healthy]


def test_raw_discovery_resets_only_for_a_new_generation_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: reset the cursor every pass, or carry it into a replacement generation."""
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        first = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"first",
            source_path="first.json",
            canonical_source_path="first.json",
            acquired_at_ms=1,
        )
        second = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"second",
            source_path="second.json",
            canonical_source_path="second.json",
            acquired_at_ms=1,
        )
    frames = iter(
        (
            DerivationFrame(str(tmp_path), "index-v1", recipe_versions={"raw_observation": "recipe-v1"}),
            DerivationFrame(str(tmp_path), "index-v1", recipe_versions={"raw_observation": "recipe-v1"}),
            DerivationFrame(str(tmp_path), "index-v1", recipe_versions={"raw_observation": "recipe-v1"}),
            DerivationFrame(str(tmp_path), "index-v2", recipe_versions={"raw_observation": "recipe-v1"}),
        )
    )
    cursors: list[str | None] = []
    # The continuation only passes keys the output relation reports as done,
    # so the fake must publish a key once the pass that offered it is over.
    materialized: set[str] = set()

    class FakeRawObservationDerivation:
        def terminal_decode_refusals(self, keys: Sequence[str]) -> dict[str, RetainedRawDecodeRefusalError]:
            return {}

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def required_page(
            self, _frame: object, *, cursor: str | None, limit: int
        ) -> tuple[tuple[str, ...], str | None]:
            cursors.append(cursor)
            return ((first,), first) if cursor is None else ((second,), None)

        def inspect(self, _frame: object, keys: Sequence[str]) -> dict[str, str]:
            return {key: "valid" if key in materialized else "missing" for key in keys}

    monkeypatch.setattr(
        "polylogue.operations.raw_observation_derivation.raw_observation_frame",
        lambda _archive_root: next(frames),
    )
    monkeypatch.setattr("polylogue.storage.derived.raw.RawObservationInspection", FakeRawObservationDerivation)
    discovery = RawMaterializationDiscovery(tmp_path)

    assert discovery.discover_pending_raw_ids(1)[0][0] == first
    materialized.add(first)
    assert discovery.discover_pending_raw_ids(1) == ()
    assert discovery.discover_pending_raw_ids(1)[0][0] == second
    materialized.add(second)
    assert discovery.discover_pending_raw_ids(1) == ()
    assert cursors == [None, None, first, None]


def test_raw_discovery_cursor_stays_behind_ids_the_dispatcher_never_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partly-admitted page keeps its continuation until the page is drained.

    The dispatcher walks the offered items only while its class budget lasts,
    so a producer that advances over the whole inspected page drops the tail
    for a whole sweep -- the shape ``DerivationRunner.run_domain`` avoids with
    its ``stopped_at`` offset.

    Anti-vacuity: restore the unconditional ``self._cursor = next_cursor`` and
    the second pass resumes at ``page-1``: ``b`` and ``c`` are never offered
    again until the traversal wraps. Drop the no-progress release instead and
    the never-admitted ``c`` pins the cursor forever, so ``d`` is never
    reached.
    """
    bootstrap_archive_root(tmp_path)
    cursors: list[str | None] = []
    materialized: set[str] = set()

    class FakeRawObservationDerivation:
        def terminal_decode_refusals(self, keys: Sequence[str]) -> dict[str, RetainedRawDecodeRefusalError]:
            return {}

        # ``raw_observation_derivation`` binds this at import time.
        recipe_version = "recipe-v1"

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def required_page(
            self, _frame: object, *, cursor: str | None, limit: int
        ) -> tuple[tuple[str, ...], str | None]:
            cursors.append(cursor)
            if cursor is None:
                return (("a", "b", "c"), "page-1")
            return (("d",), None)

        def inspect(self, _frame: object, keys: Sequence[str]) -> dict[str, str]:
            return {key: "valid" if key in materialized else "missing" for key in keys}

    monkeypatch.setattr("polylogue.storage.derived.raw.RawObservationInspection", FakeRawObservationDerivation)
    discovery = RawMaterializationDiscovery(tmp_path)

    assert [raw_id for raw_id, _cost in discovery.discover_pending_raw_ids(8)] == ["a", "b", "c"]
    materialized.add("a")  # the class budget covered exactly one admission
    assert [raw_id for raw_id, _cost in discovery.discover_pending_raw_ids(8)] == ["b", "c"]
    materialized.add("b")
    assert [raw_id for raw_id, _cost in discovery.discover_pending_raw_ids(8)] == ["c"]
    # ``c`` is isolated by the dispatcher and never becomes valid. A pass that
    # makes no progress releases the hold and moves on within the same call,
    # rather than starving the rest of the traversal behind it.
    assert [raw_id for raw_id, _cost in discovery.discover_pending_raw_ids(8)] == ["d"]
    assert cursors == [None, None, None, None, "page-1"]


def test_raw_discovery_restarts_for_a_new_raw_before_its_cursor(tmp_path: Path) -> None:
    """A new durable raw cannot wait for an unrelated full cursor sweep.

    Anti-vacuity: omit the durable raw frontier from the discovery binding and
    the second page starts after ``first``; the newly admitted, lexically
    earlier raw is then invisible until a complete old traversal wraps.
    """
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        first = "z" * 64
        first_payload = b"high-frontier"
        assert (
            archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=first_payload,
                source_path="high.json",
                canonical_source_path="high.json",
                acquired_at_ms=1,
                raw_id=first,
            )
            == first
        )

    discovery = RawMaterializationDiscovery(tmp_path)
    assert discovery.discover_pending_raw_ids(1) == ((first, len(first_payload)),)

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        earlier = "a" * 64
        earlier_payload = b"earlier-frontier"
        assert (
            archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=earlier_payload,
                source_path="earlier.json",
                canonical_source_path="earlier.json",
                acquired_at_ms=2,
                raw_id=earlier,
            )
            == earlier
        )

    assert discovery.discover_pending_raw_ids(1) == ((earlier, len(earlier_payload)),)


def test_raw_discovery_second_idle_pass_stays_one_page_at_large_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: an all-valid census makes the second pass inspect every row."""
    bootstrap_archive_root(tmp_path)
    calls: list[tuple[str | None, int]] = []

    class FakeRawObservationDerivation:
        def terminal_decode_refusals(self, keys: Sequence[str]) -> dict[str, RetainedRawDecodeRefusalError]:
            return {}

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def required_page(
            self, _frame: object, *, cursor: str | None, limit: int
        ) -> tuple[tuple[str, ...], str | None]:
            calls.append((cursor, limit))
            # A synthetic million-row valid scope is represented by its page
            # boundary. Any scanner that walks it must call this more than once.
            return (("valid-page",), "valid-page")

        def inspect(self, _frame: object, keys: Sequence[str]) -> dict[str, str]:
            return dict.fromkeys(keys, "valid")

    monkeypatch.setattr("polylogue.storage.derived.raw.RawObservationInspection", FakeRawObservationDerivation)
    discovery = RawMaterializationDiscovery(tmp_path)

    assert discovery.discover_pending_raw_ids(32) == ()
    assert discovery.discover_pending_raw_ids(32) == ()
    assert calls == [(None, 32), ("valid-page", 32)]


@pytest.mark.asyncio
async def test_intake_coalesces_hints_received_during_discovery() -> None:
    """Clearing a wake after discovery loses the event and waits sixty seconds."""
    from polylogue.daemon.intake_adapters import DaemonIntakeService

    wakeup = asyncio.Event()
    rediscovered = asyncio.Event()

    class HintingAdapter(FakeAdapter):
        async def discover(self, *, limit: int) -> Sequence[IntakeItem]:
            self.discover_calls.append(limit)
            if len(self.discover_calls) == 1:
                for _ in range(20):
                    wakeup.set()
            else:
                rediscovered.set()
            return ()

    adapter = HintingAdapter("local", [])
    service = DaemonIntakeService(
        FairIntakeDispatcher([IntakeClassSpec(name="local", adapter=adapter)]),
        wakeup=wakeup,
        idle_delay_s=60,
    )
    task = asyncio.create_task(service.run())
    try:
        await asyncio.wait_for(rediscovered.wait(), timeout=1)
        await asyncio.sleep(0)
        assert len(adapter.discover_calls) == 2
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_a_huge_class_cannot_starve_its_siblings() -> None:
    """Mutation: give ``huge`` absolute priority and ``small`` admits nothing."""
    huge = FakeAdapter("huge", [f"h{index}" for index in range(10_000)])
    small = FakeAdapter("small", ["s0", "s1"])
    dispatcher = FairIntakeDispatcher(
        [
            IntakeClassSpec(name="huge", adapter=huge, page_size=8),
            IntakeClassSpec(name="small", adapter=small, page_size=8),
        ]
    )

    result = await dispatcher.run_once(budget=16)

    assert result.require_report("small").admitted == 2
    assert result.require_report("huge").admitted > 0
    assert small.pending == []


@pytest.mark.asyncio
async def test_no_pass_enumerates_a_whole_spool() -> None:
    """Discovery is paged; a per-tick full scan cannot converge at spool scale."""
    huge = FakeAdapter("huge", [f"h{index}" for index in range(830_789)])
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="huge", adapter=huge, page_size=32)])

    await dispatcher.run_once(budget=64)

    assert huge.discover_calls == [32]


@pytest.mark.asyncio
async def test_weight_decides_the_share_of_one_pass() -> None:
    """Mutation: ignore ``weight`` and both classes admit the same count."""
    heavy = FakeAdapter("heavy", [f"a{index}" for index in range(100)])
    light = FakeAdapter("light", [f"b{index}" for index in range(100)])
    dispatcher = FairIntakeDispatcher(
        [
            IntakeClassSpec(name="heavy", adapter=heavy, weight=3, page_size=64),
            IntakeClassSpec(name="light", adapter=light, weight=1, page_size=64),
        ]
    )

    result = await dispatcher.run_once(budget=40)

    assert result.require_report("heavy").admitted > result.require_report("light").admitted


@pytest.mark.asyncio
async def test_retryable_item_recovers_after_process_local_cooldown() -> None:
    """Mutation: isolate retryables permanently and a repaired item never returns."""

    clock = [0.0]
    poison_attempts = 0

    def outcome(item: IntakeItem) -> AdmissionResult:
        nonlocal poison_attempts
        if item.item_id == "poison":
            poison_attempts += 1
            if poison_attempts <= 3:
                return AdmissionResult(AdmissionOutcome.RETRYABLE, reason="temporary refusal")
        return AdmissionResult(AdmissionOutcome.ADMITTED)

    adapter = FakeAdapter("hooks", ["poison"], outcome_for=outcome)
    dispatcher = FairIntakeDispatcher(
        [IntakeClassSpec(name="hooks", adapter=adapter, max_attempts=3, retry_cooldown_s=10.0, page_size=8)],
        clock=lambda: clock[0],
    )

    for _ in range(3):
        await dispatcher.run_once(budget=1)
    assert dispatcher.isolated_items("hooks") == frozenset()
    assert poison_attempts == 3

    assert adapter.pending == ["poison"]
    await dispatcher.run_once(budget=1)
    assert poison_attempts == 3

    clock[0] = 10.0
    await dispatcher.run_once(budget=1)

    assert poison_attempts == 4
    assert adapter.acknowledged == ["poison"]


@pytest.mark.asyncio
async def test_file_intake_retries_a_stale_cursor_write_before_acknowledging_success(tmp_path: Path) -> None:
    """Mutation: count a stale cursor write as success and lose the retained source retry."""

    capture = tmp_path / "capture.json"
    capture.write_text("{}")
    source = WatchSource(name="capture", root=tmp_path, layout=export_drop_layout((".json",)))
    convergence_paths: list[tuple[Path, ...]] = []

    class StaleCursorWatcher(_InlineWriterWatcher):
        def intake_revision(self, _source: WatchSource) -> int:
            return 0

        async def _ingest_files(self, paths: Sequence[Path], **_kwargs: object) -> SimpleNamespace:
            assert paths == [capture]
            return SimpleNamespace(
                succeeded_file_count=1,
                failed_file_count=0,
                stale_cursor_write_count=1,
                source_payload_read_bytes=len("{}"),
            )

        async def _converge_embeddings_off_writer(self, paths: Sequence[Path]) -> None:
            convergence_paths.append(tuple(paths))

    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=StaleCursorWatcher(), sources=(source,)),  # type: ignore[arg-type]
        source,
    )

    result = await adapter.admit(
        IntakeItem(item_id=f"file:{capture.resolve()}", class_name="capture", payload=capture, estimated_cost=2)
    )

    assert result.outcome is AdmissionOutcome.RETRYABLE
    assert "stale" in (result.reason or "")
    assert convergence_paths == []


@pytest.mark.asyncio
async def test_an_adapter_that_raises_is_one_item_retried_not_a_dead_class() -> None:
    def outcome(item: IntakeItem) -> AdmissionResult:
        if item.item_id == "boom":
            raise RuntimeError("adapter exploded")
        return AdmissionResult(AdmissionOutcome.ADMITTED)

    adapter = FakeAdapter("hooks", ["boom", "fine"], outcome_for=outcome)
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="hooks", adapter=adapter, max_attempts=1, page_size=8)])

    result = await dispatcher.run_once(budget=8)

    assert result.require_report("hooks").admitted == 1
    assert dispatcher.isolated_items("hooks") == frozenset()


@pytest.mark.asyncio
async def test_a_terminal_item_is_set_aside_without_further_attempts() -> None:
    def outcome(item: IntakeItem) -> AdmissionResult:
        if item.item_id == "unparseable":
            return AdmissionResult(AdmissionOutcome.TERMINAL, reason="unknown envelope version")
        return AdmissionResult(AdmissionOutcome.ADMITTED)

    adapter = FakeAdapter("browser", ["unparseable", "ok"], outcome_for=outcome)
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="browser", adapter=adapter, page_size=8)])

    await dispatcher.run_once(budget=8)

    assert dispatcher.isolated_items("browser") == frozenset({"unparseable"})
    assert adapter.acknowledged == ["ok"]


@pytest.mark.asyncio
async def test_duplicate_delivery_is_acknowledged_without_double_admission() -> None:
    """Crash after admission but before acknowledgement replays the item."""
    adapter = FakeAdapter(
        "hooks",
        ["already"],
        outcome_for=lambda _item: AdmissionResult(AdmissionOutcome.DUPLICATE, reason="content hash present"),
    )
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="hooks", adapter=adapter, page_size=8)])

    result = await dispatcher.run_once(budget=8)

    assert result.require_report("hooks").duplicates == 1
    assert result.require_report("hooks").admitted == 0
    assert adapter.acknowledged == ["already"]
    assert adapter.admitted == []


@pytest.mark.asyncio
async def test_losing_every_scheduling_hint_preserves_correctness() -> None:
    """Deficits, attempts and the ready set are hints; the spool is authority."""
    adapter = FakeAdapter("hooks", [f"h{index}" for index in range(6)])
    first = FairIntakeDispatcher([IntakeClassSpec(name="hooks", adapter=adapter, page_size=2)])
    await first.run_once(budget=2)

    # A restart constructs a new dispatcher with no hints at all.
    second = FairIntakeDispatcher([IntakeClassSpec(name="hooks", adapter=adapter, page_size=2)])
    while adapter.pending:
        await second.run_once(budget=2)

    assert sorted(adapter.acknowledged) == [f"h{index}" for index in range(6)]
    assert len(adapter.acknowledged) == len(set(adapter.acknowledged))


@pytest.mark.asyncio
async def test_a_class_in_terminal_refusal_is_excluded_at_selection(tmp_path: Path) -> None:
    """The claude-code wedge: refusing late is the defect.

    Mutation: consult the halt inside ``admit`` instead of
    ``schedulable_classes`` and this reddens -- discovery runs again.
    """
    halts = HaltRegistry(tmp_path)
    dead = FakeAdapter(
        "claude-code",
        ["c0", "c1"],
        outcome_for=lambda _item: AdmissionResult(
            AdmissionOutcome.CLASS_TERMINAL, reason="refusing further ingest until restart"
        ),
    )
    live = FakeAdapter("codex", ["x0", "x1"])
    dispatcher = FairIntakeDispatcher(
        [
            IntakeClassSpec(name="claude-code", adapter=dead, page_size=8),
            IntakeClassSpec(name="codex", adapter=live, page_size=8),
        ],
        halts=halts,
        frame="daemon:1",
    )

    await dispatcher.run_once(budget=8)
    discover_calls_at_halt = len(dead.discover_calls)

    for _ in range(5):
        await dispatcher.run_once(budget=8)

    assert len(dead.discover_calls) == discover_calls_at_halt, "a halted class was planned again"
    assert live.acknowledged == ["x0", "x1"]
    assert halts.is_halted(unit_id(UnitKind.INTAKE_CLASS, "claude-code"))


@pytest.mark.asyncio
async def test_a_halted_class_is_named_in_its_status_observation(tmp_path: Path) -> None:
    halts = HaltRegistry(tmp_path)
    halts.halt(
        unit_id(UnitKind.INTAKE_CLASS, "claude-code"),
        reason=HaltReason.TERMINAL_REFUSAL,
        message="refusing further ingest until restart",
        frame="daemon:1",
    )
    board = ObservationBoard()
    adapter = FakeAdapter("claude-code", ["c0"])
    dispatcher = FairIntakeDispatcher(
        [IntakeClassSpec(name="claude-code", adapter=adapter, page_size=8)],
        halts=halts,
        board=board,
    )

    result = await dispatcher.run_once(budget=8)

    assert result.skipped_halted == ("claude-code",)
    observation = board.get_or_unavailable("intake.claude-code")
    assert observation.state is ObservationState.FAILED
    assert "refusing further ingest until restart" in (observation.reason or "")
    assert observation.value is None


@pytest.mark.asyncio
async def test_a_halted_class_survives_a_restart(tmp_path: Path) -> None:
    halts = HaltRegistry(tmp_path)
    halts.halt(
        unit_id(UnitKind.INTAKE_CLASS, "claude-code"),
        reason=HaltReason.TERMINAL_REFUSAL,
        message="refusing further ingest until restart",
        frame="daemon:1",
    )

    adapter = FakeAdapter("claude-code", ["c0"])
    restarted = FairIntakeDispatcher(
        [IntakeClassSpec(name="claude-code", adapter=adapter, page_size=8)],
        halts=HaltRegistry(tmp_path),
    )

    await restarted.run_once(budget=8)

    assert adapter.discover_calls == []
    assert adapter.acknowledged == []


@pytest.mark.asyncio
async def test_a_discovery_failure_reports_rather_than_raising() -> None:
    class BrokenAdapter(FakeAdapter):
        async def discover(self, *, limit: int) -> Sequence[IntakeItem]:
            raise OSError("spool directory vanished")

    live = FakeAdapter("codex", ["x0"])
    dispatcher = FairIntakeDispatcher(
        [
            IntakeClassSpec(name="broken", adapter=BrokenAdapter("broken", []), page_size=8),
            IntakeClassSpec(name="codex", adapter=live, page_size=8),
        ]
    )

    result = await dispatcher.run_once(budget=8)

    assert "spool directory vanished" in (result.require_report("broken").reason or "")
    assert result.require_report("codex").admitted == 1


def test_duplicate_class_names_are_refused() -> None:
    adapter = FakeAdapter("hooks", [])
    with pytest.raises(ValueError, match="duplicate intake class name"):
        FairIntakeDispatcher(
            [
                IntakeClassSpec(name="hooks", adapter=adapter),
                IntakeClassSpec(name="hooks", adapter=adapter),
            ]
        )


class ByteCostAdapter(FakeAdapter):
    """A spool whose items carry real payload sizes, as production adapters do."""

    def __init__(self, class_name: str, pending: Sequence[str], *, item_bytes: int) -> None:
        super().__init__(class_name, pending)
        self._item_bytes = item_bytes

    async def discover(self, *, limit: int) -> Sequence[IntakeItem]:
        self.discover_calls.append(limit)
        return [
            IntakeItem(item_id=name, class_name=self.class_name, estimated_cost=self._item_bytes)
            for name in self.pending[:limit]
        ]


@pytest.mark.asyncio
async def test_default_cycle_budget_admits_ordinary_sized_payloads() -> None:
    """The default budget is denominated in the bytes adapters actually charge.

    Red if ``run_once`` reverts to a count-scale default: a 2 MiB session file
    then exceeds one whole pass, so exactly one item is admitted and the class
    spends tens of thousands of passes climbing out of a negative deficit.
    """
    adapter = ByteCostAdapter("files", [f"f{index}" for index in range(32)], item_bytes=2 * 1024 * 1024)
    dispatcher = FairIntakeDispatcher((IntakeClassSpec(name="files", adapter=adapter),))

    result = await dispatcher.run_once()

    assert [report.admitted for report in result.classes] == [32]


@pytest.mark.asyncio
async def test_a_failed_whale_is_retried_on_the_next_pass() -> None:
    """An item many shares large takes one pass's share, not the next thousand passes'.

    A 1.6 GB raw whose publication refused a stale preparation was charged its
    whole size, so its class sat in debt for hundreds of passes while sibling
    classes ran, and the retryable refusal was never retried. Red if an
    oversized item is charged its full estimate again: the second pass is
    budget-blocked and admits nothing.
    """
    outcomes = iter((AdmissionResult(AdmissionOutcome.RETRYABLE, reason="ReferenceSealStaleError: stale"),))

    def outcome(_item: IntakeItem) -> AdmissionResult:
        return next(outcomes, AdmissionResult(AdmissionOutcome.ADMITTED))

    whale = ByteCostAdapter("raw", ["whale"], item_bytes=100 * DEFAULT_INTAKE_BYTE_BUDGET)
    whale._outcome_for = outcome
    sibling = ByteCostAdapter("files", ["f0"], item_bytes=1)
    dispatcher = FairIntakeDispatcher(
        (IntakeClassSpec(name="raw", adapter=whale), IntakeClassSpec(name="files", adapter=sibling))
    )

    first = await dispatcher.run_once()
    second = await dispatcher.run_once()

    assert first.require_report("raw").retried == 1
    assert second.require_report("raw").admitted == 1
    assert whale.admitted == ["whale"]


@pytest.mark.asyncio
async def test_discovery_page_is_bounded_by_rows_not_by_the_byte_deficit() -> None:
    """Discovery asks for ``page_size`` rows however few bytes remain.

    Red if ``limit`` is derived from ``runtime.deficit`` again: the deficit is
    payload bytes, so a partly spent one silently shrinks the row page — and a
    deficit of, say, three bytes would request three files.
    """
    adapter = ByteCostAdapter("files", [f"f{index}" for index in range(8)], item_bytes=4)
    dispatcher = FairIntakeDispatcher((IntakeClassSpec(name="files", adapter=adapter, page_size=8),))

    await dispatcher.run_once(budget=6)

    assert adapter.discover_calls == [8]


def _scandir_denying(blocked: Path) -> Callable[[Path], Any]:
    """``os.scandir`` that refuses exactly one directory.

    A tidy filesystem cannot prove anything here: the defect only appears when
    a directory the walk must descend cannot be read.
    """

    real = os.scandir

    def scandir(directory: Path) -> Any:
        if Path(directory) == blocked:
            raise PermissionError(13, "Permission denied", str(blocked))
        return real(directory)

    return scandir


def test_bounded_source_paths_refuses_an_unreadable_subtree(tmp_path: Path) -> None:
    """An unreadable subtree is a named refusal, never a shorter page.

    Anti-vacuity: restore ``except OSError: return children`` in
    ``_ordered_children`` and this returns ``[readable/kept.json]`` -- a result
    the caller cannot tell apart from "locked/ was empty" -- so the
    ``pytest.raises`` fails.
    """
    readable = tmp_path / "readable"
    readable.mkdir()
    kept = readable / "kept.json"
    kept.write_text("kept")
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "hidden.json").write_text("hidden")
    source = WatchSource(name="test", root=tmp_path, layout=export_drop_layout((".json",)))

    with pytest.raises(WalkRefusedError) as excinfo:
        _bounded_source_paths(
            source,
            (source,),
            limit=8,
            after=None,
            scandir=_scandir_denying(locked),
        )

    assert str(locked) in str(excinfo.value)
    assert [fault.path for fault in excinfo.value.faults] == [locked]


def test_bounded_source_paths_refuses_a_missing_source_root(tmp_path: Path) -> None:
    """A root that is not there is unavailable, not fully ingested.

    Anti-vacuity: restore ``not source.root.is_dir() -> []`` and the call
    returns an empty page, which the dispatcher reports as a healthy source
    with zero backlog.
    """
    missing = tmp_path / "unmounted"
    source = WatchSource(name="test", root=missing, layout=export_drop_layout((".json",)))

    with pytest.raises(WalkRefusedError) as excinfo:
        _bounded_source_paths(source, (source,), limit=8, after=None)

    assert str(missing) in str(excinfo.value)


def test_discovery_refusal_is_counted_on_the_class_report() -> None:
    """The refusal reaches a surface the caller reads, not just a log.

    Anti-vacuity: a dispatcher that dropped the exception would leave
    ``report.reason`` ``None`` and the class indistinguishable from an idle
    one.
    """

    class RefusingAdapter(FakeAdapter):
        async def discover(self, *, limit: int) -> Sequence[IntakeItem]:
            raise WalkRefusedError(
                "intake discovery could not read a source directory",
                [WalkFault(Path("/srv/locked"), "scandir failed")],
            )

    adapter = RefusingAdapter("configured_local", ())
    dispatcher = FairIntakeDispatcher(
        [IntakeClassSpec(name="configured_local", adapter=cast(Any, adapter))],
    )

    result = asyncio.run(dispatcher.run_once())

    report = result.require_report("configured_local")
    assert report.discovered == 0
    assert report.reason is not None
    assert "/srv/locked" in report.reason


@pytest.mark.asyncio
async def test_a_durably_excluded_file_is_not_reported_as_a_duplicate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """polylogue-onbz3: a refused intake file is EXCLUDED, never DUPLICATE progress.

    Anti-vacuity: restore the ``AdmissionOutcome.DUPLICATE`` return for a pass
    that admitted nothing and this goes red three ways -- the outcome is
    ``duplicate``, the class report counts a duplicate so ``IntakePass.progressed``
    claims progress for a pass that admitted nothing, and the "admitted nothing
    for N planned path(s)" line stays absent from the intake path.
    """

    capture = tmp_path / "capture.json"
    capture.write_text("{}")
    source = WatchSource(name="capture", root=tmp_path, layout=export_drop_layout((".json",)))

    class ExcludingWatcher(_InlineWriterWatcher):
        def intake_revision(self, _source: WatchSource) -> int:
            return 0

        async def _ingest_files(self, paths: Sequence[Path], **_kwargs: object) -> SimpleNamespace:
            assert paths == [capture]
            return SimpleNamespace(
                succeeded_file_count=0,
                failed_file_count=0,
                stale_cursor_write_count=0,
                excluded_file_count=1,
                excluded_reasons={"unsupported_shape": 1},
                excluded_paths={str(capture): "unsupported_shape"},
                source_payload_read_bytes=len("{}"),
            )

    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=ExcludingWatcher(), sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    item = IntakeItem(item_id=f"file:{capture.resolve()}", class_name="capture", payload=capture, estimated_cost=2)

    with caplog.at_level("INFO"):
        result = await adapter.admit(item)

    assert result.outcome is AdmissionOutcome.EXCLUDED
    assert result.acknowledgeable is True
    assert "admitted nothing for 1 planned path(s)" in caplog.text

    class OneFileAdapter:
        async def discover(self, *, limit: int) -> Sequence[IntakeItem]:
            return (item,) if limit else ()

        async def admit(self, _item: IntakeItem) -> AdmissionResult:
            return result

        async def acknowledge(self, _item: IntakeItem) -> None:
            return None

    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="capture", adapter=OneFileAdapter(), page_size=4)])
    intake_pass = await dispatcher.run_once(budget=8)

    report = intake_pass.require_report("capture")
    assert (report.excluded, report.duplicates, report.admitted) == (1, 0, 0)
    assert intake_pass.progressed is False


def test_raw_discovery_sweep_advances_under_a_sustained_arrival_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A steady stream of new raws must not pin the sweep on its first page.

    Anti-vacuity: put the durable raw frontier back into the discovery binding
    so an arrival resets ``self._cursor`` to ``None``, and every recorded sweep
    cursor below becomes ``None`` -- the obligations behind page one are then
    never reached however long the daemon runs.
    """
    bootstrap_archive_root(tmp_path)
    cursors: list[str | None] = []

    class FakeRawObservationDerivation:
        def terminal_decode_refusals(self, keys: Sequence[str]) -> dict[str, RetainedRawDecodeRefusalError]:
            return {}

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def required_page(
            self, _frame: object, *, cursor: str | None, limit: int
        ) -> tuple[tuple[str, ...], str | None]:
            cursors.append(cursor)
            nxt = "page1" if cursor is None else f"page{int(str(cursor)[4:]) + 1}"
            return (nxt,), nxt

        def inspect(self, _frame: object, keys: Sequence[str]) -> dict[str, str]:
            # The whole swept space is already materialized; only the arrivals
            # are outstanding, which is exactly the starvation condition.
            return {key: "valid" if key.startswith("page") else "missing" for key in keys}

    monkeypatch.setattr("polylogue.storage.derived.raw.RawObservationInspection", FakeRawObservationDerivation)
    discovery = RawMaterializationDiscovery(tmp_path)

    assert discovery.discover_pending_raw_ids(4) == ()
    for index in range(5):
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=f"arrival-{index}".encode(),
                source_path=f"arrival-{index}.json",
                canonical_source_path=f"arrival-{index}.json",
                acquired_at_ms=index + 1,
            )
        discovery.discover_pending_raw_ids(4)

    assert cursors == [None, "page1", "page2", "page3"]


def test_raw_discovery_is_an_empty_page_before_the_raw_tier_exists(tmp_path: Path) -> None:
    """A fresh archive root reports no pending raw work instead of raising.

    polylogue-f7pdm: the daemon used to decide raw-materialization
    availability once, at startup, from ``source.db`` existing. On the
    declared build route that file does not exist yet, so the class was never
    registered for the process lifetime and the whale pass logged a failure
    every thirty seconds. Discovery owns the absence now.

    Anti-vacuity: removing the missing-tier guard makes this raise
    ``sqlite3.OperationalError`` rather than return an empty page.
    """
    discovery = RawMaterializationDiscovery(tmp_path)

    assert not (tmp_path / "source.db").exists()
    assert discovery.discover_pending_raw_ids(4) == ()


def test_raw_discovery_admits_work_once_the_tier_appears_without_a_restart(tmp_path: Path) -> None:
    """One long-lived discovery re-evaluates availability every pass.

    Anti-vacuity: latching availability at construction -- the startup-only
    check this bead replaces -- keeps the second call empty and makes this
    test red.
    """
    discovery = RawMaterializationDiscovery(tmp_path)
    assert discovery.discover_pending_raw_ids(4) == ()

    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"raw-after-bootstrap",
            source_path="late.json",
            canonical_source_path="late.json",
            acquired_at_ms=1,
        )

    assert [found for found, _cost in discovery.discover_pending_raw_ids(4)] == [raw_id]


@pytest.mark.asyncio
async def test_an_unmeasurable_callback_class_reserves_a_budget_share() -> None:
    """A remote sync cannot buy a whole pass for one byte.

    polylogue-swicx: ``CallbackIntakeAdapter`` charged the literal ``1``
    while the deficit is denominated in payload bytes, so an arbitrarily
    large Drive sync starved its byte-denominated siblings.

    Anti-vacuity: restoring ``estimated_cost=1`` drops the charged estimate
    to one byte and makes this assertion red.
    """
    adapter = CallbackIntakeAdapter("configured_remote", lambda: 1)
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="configured_remote", adapter=cast(Any, adapter))])

    result = await dispatcher.run_once()

    report = result.require_report("configured_remote")
    assert report.admitted == 1
    assert report.estimated_cost == UNMEASURABLE_INTAKE_COST_BYTES


def test_callback_adapter_failure_signature_keys_on_the_exception_type() -> None:
    """A repeated distinct exception on one item must not share a signature.

    ``_failure_signature`` takes the text before the first colon in the
    reason. ``CallbackIntakeAdapter`` used to format its reason as
    ``"<class_name>: <ExceptionType>: <detail>"``, so every failure -- of
    whatever exception type -- shared the adapter's class name as its
    signature and could isolate the item after ``max_deterministic_cooldowns``
    even though the underlying defect never repeated.

    Anti-vacuity: reverting to ``f"{self.class_name}: {type(exc).__name__}: {exc}"``
    makes the two distinct exception types below compare equal.
    """
    from polylogue.daemon.intake import _failure_signature

    def callback_raising(exc: BaseException) -> int:
        raise exc

    async def admit(exc: BaseException) -> str | None:
        adapter = CallbackIntakeAdapter("configured_remote", lambda: callback_raising(exc))
        item = IntakeItem(item_id="x", class_name="configured_remote", payload=None, estimated_cost=1)
        outcome = await adapter.admit(item)
        return outcome.reason

    key_error_reason = asyncio.run(admit(KeyError("field")))
    type_error_reason = asyncio.run(admit(TypeError("shape")))
    assert _failure_signature(key_error_reason) != _failure_signature(type_error_reason)
    assert _failure_signature(key_error_reason) == "KeyError"
    assert _failure_signature(type_error_reason) == "TypeError"


def test_raw_materialization_failure_signature_keys_on_the_exception_type() -> None:
    """The raw-materialization adapter had the same class-label-first bug.

    ``RawMaterializationIntakeAdapter`` formatted its reason as
    ``"raw materialization: <ExceptionType>: <detail>"``, so every failure
    shared the constant "raw materialization" label as its signature
    instead of the exception type, the same defect fixed on
    ``CallbackIntakeAdapter`` above.

    Anti-vacuity: reverting to that format makes the two distinct exception
    types below compare equal.
    """
    from polylogue.daemon.intake import _failure_signature

    def admit_raising(exc: BaseException) -> Callable[[str], int]:
        def admit_id(_raw_id: str) -> int:
            raise exc

        return admit_id

    async def admit(exc: BaseException) -> str | None:
        adapter = RawMaterializationIntakeAdapter(lambda _limit: (), admit_raising(exc))
        outcome = await adapter.admit(IntakeItem(item_id="raw-1", class_name="raw_materialization"))
        return outcome.reason

    key_error_reason = asyncio.run(admit(KeyError("field")))
    type_error_reason = asyncio.run(admit(TypeError("shape")))
    assert _failure_signature(key_error_reason) != _failure_signature(type_error_reason)
    assert _failure_signature(key_error_reason) == "KeyError"
    assert _failure_signature(type_error_reason) == "TypeError"


@pytest.mark.asyncio
async def test_a_deferred_retry_placeholders_zero_cost_is_not_replaced() -> None:
    """A DEFERRED item reporting ``actual_cost=0`` must return its estimate.

    ``item_cost if result.actual_cost is None else max(0, ...)`` distinguishes
    an explicit zero (no work attempted; the retry is still pending) from a
    measurement the adapter never took. The acknowledgeable-result path once
    used ``result.actual_cost or item_cost``, which treats zero as falsy and
    charges the full estimate anyway, so a large file waiting on its retry
    never returns the deficit it reserved.

    Anti-vacuity: reverting to ``max(1, int(result.actual_cost or item_cost))``
    makes this assertion red.
    """

    class PendingRetryAdapter:
        def __init__(self) -> None:
            self.acknowledged: list[str] = []

        async def discover(self, *, limit: int) -> Sequence[IntakeItem]:
            return [IntakeItem(item_id="big-pending-retry", class_name="configured_local", estimated_cost=10_000_000)]

        async def admit(self, _item: IntakeItem) -> AdmissionResult:
            return AdmissionResult(AdmissionOutcome.DEFERRED, actual_cost=0)

        async def acknowledge(self, item: IntakeItem) -> None:
            self.acknowledged.append(item.item_id)

    dispatcher = FairIntakeDispatcher(
        (IntakeClassSpec("configured_local", cast(Any, PendingRetryAdapter()), page_size=1),)
    )

    await dispatcher.run_once()

    # Planning reserves the full 10,000,000-byte estimate against the pass's
    # share; an honored zero actual cost returns every reserved byte, so the
    # deficit lands back at the untouched per-class share.
    runtime = dispatcher._runtime["configured_local"]
    assert runtime.deficit == DEFAULT_INTAKE_BYTE_BUDGET


@pytest.mark.asyncio
async def test_a_pass_of_only_duplicates_is_not_progress() -> None:
    """A static source must back off to the idle delay, not spin.

    polylogue-swicx: ``DaemonIntakeService`` sleeps 0.05 s when a pass
    progressed, so counting re-recognised duplicates as progress re-ran
    discovery about twenty times a second on a source with nothing new.

    Anti-vacuity: restoring ``admitted or duplicates`` in ``progressed``
    makes this assertion red.
    """
    adapter = FakeAdapter(
        "codex",
        ["x0"],
        outcome_for=lambda _item: AdmissionResult(AdmissionOutcome.DUPLICATE),
    )
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="codex", adapter=cast(Any, adapter))])

    result = await dispatcher.run_once()

    assert result.require_report("codex").duplicates == 1
    assert result.progressed is False


@pytest.mark.asyncio
async def test_zero_success_file_intake_is_retryable_not_duplicate(tmp_path: Path) -> None:
    """A batch with no per-path verdict must remain queued for retry.

    polylogue-swicx: file intake acknowledged this shape as ``DUPLICATE`` even
    though the watcher treats the same zero-success, zero-failure batch as
    needing retry.  Drive the real adapter through the dispatcher so cursor
    acknowledgement is part of the assertion.

    Anti-vacuity: restoring ``AdmissionOutcome.DUPLICATE`` for the zero-success
    fall-through makes the retry count and duplicate count below red.
    """
    capture = tmp_path / "capture.json"
    capture.write_text("{}")
    source = WatchSource(name="capture", root=tmp_path, layout=export_drop_layout((".json",)))

    class ZeroSuccessWatcher(_InlineWriterWatcher):
        def intake_revision(self, _source: WatchSource) -> int:
            return 0

        async def _ingest_files(self, paths: Sequence[Path], **_kwargs: object) -> SimpleNamespace:
            assert list(paths) == [capture]
            return SimpleNamespace(
                succeeded_file_count=0,
                succeeded_paths=(),
                failed_file_count=0,
                failed_paths=(),
                stale_cursor_write_count=0,
                source_payload_read_bytes=0,
            )

    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=ZeroSuccessWatcher(), sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="capture", adapter=adapter, page_size=8)])

    intake_pass = await dispatcher.run_once()

    report = intake_pass.require_report("capture")
    assert (report.retried, report.duplicates) == (1, 0)
    assert adapter._after is None


@pytest.mark.asyncio
async def test_unattempted_file_refunds_estimate_so_next_pass_can_discover(tmp_path: Path) -> None:
    """A batch refusal costs no bytes for a file the watcher never attempted.

    Without the refund, the oversized page leaves a negative deficit and the
    next pass skips discovery even though its deferred file is ready.
    """
    first, deferred = (tmp_path / name for name in ("a.json", "b.json"))
    for path in (first, deferred):
        path.write_bytes(b"x" * 40)
    source = WatchSource(name="capture", root=tmp_path, layout=export_drop_layout((".json",)))

    class PartlyRefusedWatcher(_InlineWriterWatcher):
        def __init__(self) -> None:
            self.batches: list[list[Path]] = []

        def intake_revision(self, _source: WatchSource) -> int:
            return 0

        async def _ingest_files(self, paths: Sequence[Path], **_kwargs: object) -> SimpleNamespace:
            self.batches.append(list(paths))
            if len(self.batches) == 1:
                # The first file grew after discovery. Its measured charge
                # overdraws the share unless the unattempted file is refunded.
                first.write_bytes(b"x" * 120)
                return SimpleNamespace(
                    succeeded_paths=(str(first),),
                    excluded_paths={str(deferred): REFUSED_UNATTEMPTED},
                    source_payload_read_bytes=120,
                )
            return SimpleNamespace(succeeded_paths=(str(deferred),), source_payload_read_bytes=40)

    watcher = PartlyRefusedWatcher()
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="capture", adapter=adapter, page_size=2)])

    first_pass = await dispatcher.run_once(budget=80)
    report = first_pass.require_report("capture")
    assert (report.admitted, report.retried, report.estimated_cost, report.actual_cost) == (1, 1, 80, 60)

    second_pass = await dispatcher.run_once(budget=1)
    assert second_pass.require_report("capture").admitted == 1
    assert watcher.batches == [[first, deferred], [deferred]]


@pytest.mark.asyncio
async def test_attempted_retry_keeps_its_estimated_charge() -> None:
    """An ordinary retry cannot claim zero cost merely because admission failed.

    The failed attempt keeps the whole share it was charged (nothing is
    refunded), and, being larger than that share, it is offered again on the
    next pass rather than after its full size has been repaid.
    """
    item = IntakeItem(item_id="retry", class_name="files", estimated_cost=40)

    class RetryAdapter:
        def __init__(self) -> None:
            self.discover_calls = 0

        async def discover(self, *, limit: int) -> Sequence[IntakeItem]:
            self.discover_calls += 1
            return [item]

        async def admit(self, _item: IntakeItem) -> AdmissionResult:
            return AdmissionResult(AdmissionOutcome.RETRYABLE, reason="attempted")

        async def acknowledge(self, _item: IntakeItem) -> None:
            raise AssertionError("retry was acknowledged")

    adapter = RetryAdapter()
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="files", adapter=adapter)])
    first_pass = await dispatcher.run_once(budget=1)
    assert first_pass.require_report("files").actual_cost == 40
    assert dispatcher._runtime["files"].deficit == 0
    second_pass = await dispatcher.run_once(budget=1)
    assert second_pass.require_report("files").retried == 1
    assert adapter.discover_calls == 2


@pytest.mark.asyncio
async def test_deferred_file_is_reoffered_when_cursor_retry_is_due(tmp_path: Path) -> None:
    """A static root must not strand a deferred file behind the acknowledged walk position.

    Anti-vacuity: removing the due-cursor discovery leaves the second ingest
    batch absent even after its retry time has elapsed.
    """
    deferred_path = tmp_path / "a.json"
    sibling_path = tmp_path / "z.json"
    for path in (deferred_path, sibling_path):
        path.write_text("{}")
    source = WatchSource(name="capture", root=tmp_path, layout=export_drop_layout((".json",)))
    cursor = CursorStore(tmp_path / "index.db")

    class DeferredWatcher(_InlineWriterWatcher):
        def __init__(self) -> None:
            self._cursor = cursor
            self.batches: list[list[Path]] = []

        def intake_revision(self, _source: WatchSource) -> int:
            return 0

        async def _ingest_files(self, paths: Sequence[Path], **_kwargs: object) -> SimpleNamespace:
            self.batches.append(list(paths))
            if len(self.batches) == 1:
                cursor.set(
                    deferred_path,
                    2,
                    next_retry_at="2999-01-01T00:00:00+00:00",
                    authority=fixture_cursor_authority(deferred_path),
                )
                return SimpleNamespace(
                    succeeded_paths=(str(sibling_path),),
                    deferred_paths=(str(deferred_path),),
                    failed_paths=(str(deferred_path),),
                    source_payload_read_bytes=2,
                )
            cursor.set(
                deferred_path,
                2,
                content_fingerprint="admitted",
                next_retry_at=None,
                authority=fixture_cursor_authority(deferred_path),
            )
            return SimpleNamespace(succeeded_paths=(str(deferred_path),), source_payload_read_bytes=2)

    watcher = DeferredWatcher()
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="capture", adapter=adapter, page_size=2)])

    first = await dispatcher.run_once()
    assert first.require_report("capture").deferred == 1
    assert adapter._after == str(sibling_path)
    assert (await dispatcher.run_once()).require_report("capture").discovered == 0
    assert watcher.batches == [[deferred_path, sibling_path]]

    cursor.set(
        deferred_path, 2, next_retry_at="1970-01-01T00:00:00+00:00", authority=fixture_cursor_authority(deferred_path)
    )
    retried = await dispatcher.run_once()
    assert retried.require_report("capture").admitted == 1
    assert watcher.batches == [[deferred_path, sibling_path], [deferred_path]]
    assert (await dispatcher.run_once()).require_report("capture").discovered == 0


@pytest.mark.asyncio
async def test_archive_sidecars_do_not_restart_a_file_sweep_but_new_source_files_do(tmp_path: Path) -> None:
    paths = [tmp_path / name for name in ("a.json", "c.json")]
    for path in paths:
        path.write_text("{}")
    source = WatchSource(name="capture", root=tmp_path, layout=export_drop_layout((".json",)))
    cursor = CursorStore(tmp_path / "index.db")
    watcher = SimpleNamespace(
        has_write_coordinator=True,
        _run_writer_sync=_inline_writer_sync,
        _cursor=cursor,
        intake_revision=lambda _source: 0,
    )
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )

    first = await adapter.discover(limit=1)
    assert [item.payload for item in first] == [paths[0]]
    await adapter.acknowledge(first[0])
    Path(f"{cursor._ops_db_path}-wal").write_text("")
    second = await adapter.discover(limit=1)
    assert [item.payload for item in second] == [paths[1]]
    await adapter.acknowledge(second[0])

    # An insertion behind the current position needs a fresh source walk,
    # after the active continuation reaches its end.
    earlier = tmp_path / "b.json"
    earlier.write_text("{}")
    # The lookahead already reached the walk's end, so the restart begins on
    # the next discovery.
    restarted = await adapter.discover(limit=1)
    assert [item.payload for item in restarted] == [paths[0]]
    await adapter.acknowledge(restarted[0])
    inserted = await adapter.discover(limit=1)
    assert [item.payload for item in inserted] == [earlier]


@pytest.mark.asyncio
async def test_acquisition_budget_retains_unattempted_file_page_tail(tmp_path: Path) -> None:
    paths = [tmp_path / name for name in ("a.json", "b.json", "c.json")]
    for path in paths:
        path.write_text("{}")
    source = WatchSource(name="capture", root=tmp_path, layout=export_drop_layout((".json",)))

    class BudgetWatcher(_InlineWriterWatcher):
        def __init__(self) -> None:
            self.batches: list[list[Path]] = []

        def intake_revision(self, _source: WatchSource) -> int:
            return 0

        async def _ingest_files(self, batch: Sequence[Path], **_kwargs: object) -> SimpleNamespace:
            self.batches.append(list(batch))
            if len(self.batches) == 1:
                return SimpleNamespace(
                    succeeded_paths=(str(paths[0]),),
                    excluded_paths={str(path): REFUSED_UNATTEMPTED_TIME_BUDGET for path in paths[1:]},
                    time_budget_exceeded=True,
                    source_payload_read_bytes=2,
                )
            return SimpleNamespace(
                succeeded_paths=tuple(str(path) for path in batch),
                source_payload_read_bytes=2 * len(batch),
            )

    watcher = BudgetWatcher()
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    dispatcher = FairIntakeDispatcher(
        [IntakeClassSpec(name="capture", adapter=adapter, page_size=3, retry_cooldown_s=0)]
    )

    first = await dispatcher.run_once()
    assert (first.require_report("capture").admitted, first.require_report("capture").retried) == (1, 2)
    assert adapter._after == str(paths[0])
    second = await dispatcher.run_once()
    assert second.require_report("capture").admitted == 2
    assert watcher.batches == [paths, paths[1:]]
    assert adapter._after == str(paths[-1])


@pytest.mark.asyncio
async def test_acquisition_budget_retains_unattempted_file_sorting_before_an_acknowledged_sibling(
    tmp_path: Path,
) -> None:
    """A batch admits in its own order, not path order.

    When the budget runs out on ``a.json`` after ``b.json`` succeeded, the
    acknowledgement of ``b`` advances the walk cursor past ``a``. Filtering the
    continuation by that cursor alone dropped ``a`` until the exhausted-walk
    rescan; removing the unattempted carve-out makes the second pass admit
    nothing.
    """
    paths = [tmp_path / name for name in ("a.json", "b.json", "c.json")]
    for path in paths:
        path.write_text("{}")
    source = WatchSource(name="capture", root=tmp_path, layout=export_drop_layout((".json",)))

    class BudgetWatcher(_InlineWriterWatcher):
        def __init__(self) -> None:
            self.batches: list[list[Path]] = []

        def intake_revision(self, _source: WatchSource) -> int:
            return 0

        async def _ingest_files(self, batch: Sequence[Path], **_kwargs: object) -> SimpleNamespace:
            self.batches.append(list(batch))
            if len(self.batches) == 1:
                return SimpleNamespace(
                    succeeded_paths=(str(paths[1]),),
                    excluded_paths={
                        str(paths[0]): REFUSED_UNATTEMPTED_TIME_BUDGET,
                        str(paths[2]): REFUSED_UNATTEMPTED_TIME_BUDGET,
                    },
                    time_budget_exceeded=True,
                    source_payload_read_bytes=2,
                )
            return SimpleNamespace(
                succeeded_paths=tuple(str(path) for path in batch),
                source_payload_read_bytes=2 * len(batch),
            )

    watcher = BudgetWatcher()
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    dispatcher = FairIntakeDispatcher(
        [IntakeClassSpec(name="capture", adapter=adapter, page_size=3, retry_cooldown_s=0)]
    )

    first = await dispatcher.run_once()
    assert (first.require_report("capture").admitted, first.require_report("capture").retried) == (1, 2)
    assert adapter._after == str(paths[1])
    second = await dispatcher.run_once()
    assert second.require_report("capture").admitted == 2
    assert watcher.batches == [paths, [paths[0], paths[2]]]


@pytest.mark.asyncio
async def test_dispatcher_byte_budget_reoffers_unplanned_fresh_page_tail(tmp_path: Path) -> None:
    paths = [tmp_path / name for name in ("a.json", "b.json", "c.json")]
    for path in paths:
        path.write_text("data")
    source = WatchSource(name="capture", root=tmp_path, layout=export_drop_layout((".json",)))

    class BudgetWatcher(_InlineWriterWatcher):
        def __init__(self) -> None:
            self.batches: list[list[Path]] = []

        def intake_revision(self, _source: WatchSource) -> int:
            return 0

        async def _ingest_files(self, batch: Sequence[Path], **_kwargs: object) -> SimpleNamespace:
            self.batches.append(list(batch))
            return SimpleNamespace(
                succeeded_paths=tuple(str(path) for path in batch),
                source_payload_read_bytes=4 * len(batch),
            )

    watcher = BudgetWatcher()
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="capture", adapter=adapter, page_size=3)])
    assert (await dispatcher.run_once(budget=4)).require_report("capture").admitted == 1
    assert (await dispatcher.run_once(budget=4)).require_report("capture").admitted == 1
    assert (await dispatcher.run_once(budget=4)).require_report("capture").admitted == 1
    assert watcher.batches == [[path] for path in paths]
    assert adapter.retry_due_in_s is None


@pytest.mark.asyncio
async def test_retry_page_does_not_skip_fresh_files_or_starve_discovery(tmp_path: Path) -> None:
    """Acknowledging a late retry cannot move ordinary discovery past fresh files."""
    paths = [tmp_path / name for name in ("a.json", "b.json", "c.json", "z.json")]
    for path in paths:
        path.write_text("{}")
    source = WatchSource(name="capture", root=tmp_path, layout=export_drop_layout((".json",)))
    cursor = CursorStore(tmp_path / "index.db")
    cursor.set(paths[-1], 2, next_retry_at="1970-01-01T00:00:00+00:00", authority=fixture_cursor_authority(paths[-1]))
    watcher = SimpleNamespace(
        has_write_coordinator=True,
        _run_writer_sync=_inline_writer_sync,
        _cursor=cursor,
        intake_revision=lambda _source: 0,
    )
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    offered: list[Path] = []
    for _ in range(5):
        page = await adapter.discover(limit=1)
        assert len(page) == 1
        assert isinstance(page[0].payload, Path)
        offered.append(page[0].payload)
        await adapter.acknowledge(page[0])
    assert offered == [paths[0], paths[-1], paths[1], paths[2], paths[3]]


@pytest.mark.asyncio
async def test_retry_cursor_keeps_unoffered_page_tail_under_a_byte_budget(tmp_path: Path) -> None:
    """A due page may be larger than the dispatcher's planned admission."""
    paths = [tmp_path / name for name in ("a.json", "b.json", "c.json", "d.json")]
    for path in paths[:3]:
        path.write_text("{}")
    source = WatchSource(name="capture", root=tmp_path, layout=export_drop_layout((".json",)))
    cursor = CursorStore(tmp_path / "index.db")
    for path in paths[:3]:
        cursor.set(path, 2, next_retry_at="1970-01-01T00:00:00+00:00", authority=fixture_cursor_authority(path))
    watcher = SimpleNamespace(
        has_write_coordinator=True,
        _run_writer_sync=_inline_writer_sync,
        _cursor=cursor,
        intake_revision=lambda _source: 0,
    )
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    adapter._after = str(paths[2])
    adapter._retry_turn = True

    page = await adapter.discover(limit=3)
    assert [item.payload for item in page] == paths[:3]
    await adapter.acknowledge(page[0])  # only this item fit the byte plan
    assert adapter._retry_after == str(paths[0])

    paths[3].write_text("{}")
    cursor.set(paths[3], 2, next_retry_at="1970-01-01T00:00:00+00:00", authority=fixture_cursor_authority(paths[3]))
    adapter._retry_turn = True
    tail = await adapter.discover(limit=3)
    assert [item.payload for item in tail] == paths[1:3]
    assert paths[3] not in [item.payload for item in tail]


@pytest.mark.asyncio
async def test_cooldown_only_retry_page_rotates_within_a_finite_sweep(tmp_path: Path) -> None:
    paths = [tmp_path / name for name in ("a.json", "b.json", "c.json")]
    for path in paths:
        path.write_text("{}")
    source = WatchSource(name="capture", root=tmp_path, layout=export_drop_layout((".json",)))
    cursor = CursorStore(tmp_path / "index.db")
    for path in paths:
        cursor.set(path, 2, next_retry_at="1970-01-01T00:00:00+00:00", authority=fixture_cursor_authority(path))
    watcher = SimpleNamespace(
        has_write_coordinator=True,
        _run_writer_sync=_inline_writer_sync,
        _cursor=cursor,
        intake_revision=lambda _source: 0,
    )
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    adapter._after = str(paths[-1])
    adapter._last_root_mtime_ns = source.root.stat().st_mtime_ns
    adapter._retry_turn = True

    cooled_page = await adapter.discover(limit=2)
    assert [item.payload for item in cooled_page] == paths[:2]
    # The dispatcher planned nothing because these identities are cooling
    # down, so neither admit_page nor acknowledge is called.
    assert adapter._retry_after is None
    next_page = await adapter.discover(limit=2)
    assert [item.payload for item in next_page] == [paths[2]]
    assert adapter._retry_after is None
    await adapter.acknowledge(next_page[0])
    assert adapter._retry_after == str(paths[2])

    assert (await adapter.discover(limit=2)) == ()
    revisit = await adapter.discover(limit=2)
    assert [item.payload for item in revisit] == paths[:2]


@pytest.mark.asyncio
async def test_parent_retry_query_filters_nested_source_before_limit(tmp_path: Path) -> None:
    parent_root = tmp_path / "capture"
    child_root = parent_root / "child"
    child_root.mkdir(parents=True)
    child_paths = [child_root / f"{n}.json" for n in range(4)]
    parent_path = parent_root / "z.json"
    for path in (*child_paths, parent_path):
        path.write_text("{}")
    parent = WatchSource(name="parent", root=parent_root, layout=export_drop_layout((".json",)))
    child = WatchSource(name="child", root=child_root, layout=export_drop_layout((".json",)))
    cursor = CursorStore(tmp_path / "index.db")
    for path in (*child_paths, parent_path):
        cursor.set(path, 2, next_retry_at="1970-01-01T00:00:00+00:00", authority=fixture_cursor_authority(path))
    watcher = SimpleNamespace(
        has_write_coordinator=True,
        _run_writer_sync=_inline_writer_sync,
        _cursor=cursor,
        intake_revision=lambda _source: 0,
    )
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(parent, child)),  # type: ignore[arg-type]
        parent,
    )
    adapter._retry_turn = True
    page = await adapter.discover(limit=1)
    assert [item.payload for item in page] == [parent_path]


@pytest.mark.asyncio
async def test_missing_ops_ledger_resets_the_ordinary_walk(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    paths = [source_root / name for name in ("a.json", "z.json")]
    for path in paths:
        path.write_text("{}")
    source = WatchSource(name="capture", root=source_root, layout=export_drop_layout((".json",)))
    cursor = CursorStore(tmp_path / "index.db")
    watcher = SimpleNamespace(
        has_write_coordinator=True,
        _run_writer_sync=_inline_writer_sync,
        _cursor=cursor,
        intake_revision=lambda _source: 0,
    )
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    for path in paths:
        page = await adapter.discover(limit=1)
        assert [item.payload for item in page] == [path]
        await adapter.acknowledge(page[0])
    assert adapter._after == str(paths[-1])
    assert (await adapter.discover(limit=1)) == ()

    root_mtime_ns = source_root.stat().st_mtime_ns
    cursor._ops_db_path.unlink()
    assert source_root.stat().st_mtime_ns == root_mtime_ns
    page = await adapter.discover(limit=1)
    assert [item.payload for item in page] == [paths[0]]


def test_due_retry_discovery_is_bounded_scoped_and_read_only(tmp_path: Path) -> None:
    """Retry paging excludes future/quarantined rows and neighboring roots."""
    root = tmp_path / "source"
    cursor = CursorStore(tmp_path / "index.db")
    for name in ("a.json", "b.json"):
        cursor.set(
            root / name, 2, next_retry_at="1970-01-01T00:00:00+00:00", authority=fixture_cursor_authority(root / name)
        )
    cursor.set(
        root / "future.json",
        2,
        next_retry_at="2999-01-01T00:00:00+00:00",
        authority=fixture_cursor_authority(root / "future.json"),
    )
    cursor.set(
        root / "done.json", 2, content_fingerprint="done", authority=fixture_cursor_authority(root / "done.json")
    )
    cursor.set(
        root / "excluded.json",
        2,
        next_retry_at="1970-01-01T00:00:00+00:00",
        authority=fixture_cursor_authority(root / "excluded.json"),
    )
    cursor.mark_excluded(root / "excluded.json")
    cursor.set(
        tmp_path / "source-other" / "a.json",
        2,
        next_retry_at="1970-01-01T00:00:00+00:00",
        authority=fixture_cursor_authority(tmp_path / "source-other" / "a.json"),
    )
    assert cursor.list_due_retry_paths(root, after=None, limit=1) == (root / "a.json",)
    assert cursor.list_due_retry_paths(root, after=str(root / "a.json"), limit=2) == (root / "b.json",)
    assert cursor.list_due_retry_paths(root, after=None, limit=0) == ()
    assert cursor.has_pending_retries((root,)) is True
    assert cursor.has_pending_retries((tmp_path / "empty",)) is False
    absent = CursorStore(tmp_path / "absent" / "index.db", initialize=False)
    assert absent.list_due_retry_paths(root, after=None, limit=1) == ()
    assert absent.has_pending_retries((root,)) is None
    assert not (tmp_path / "absent").exists()


@pytest.mark.asyncio
async def test_a_raising_discovery_is_published_unmeasured_not_as_zeros() -> None:
    """A class that could not look is never a real reading of nothing.

    polylogue-swicx: the dispatcher published the failed class through the
    measured branch with admitted=duplicates=discovered=0, so status could
    not distinguish a broken spool from an idle one and the reason was lost.

    Anti-vacuity: publishing this report through ``Observation.measured``
    again leaves the state ``MEASURED`` and makes this test red.
    """

    class BrokenAdapter(FakeAdapter):
        async def discover(self, *, limit: int) -> Sequence[IntakeItem]:
            raise OSError("spool directory vanished")

    board = ObservationBoard()
    dispatcher = FairIntakeDispatcher(
        [IntakeClassSpec(name="broken", adapter=cast(Any, BrokenAdapter("broken", [])))],
        board=board,
    )

    result = await dispatcher.run_once()

    assert result.require_report("broken").discovery_failed is True
    observation = board.snapshot()["intake.broken"]
    assert observation.state is ObservationState.FAILED
    assert "spool directory vanished" in (observation.reason or "")


@pytest.mark.asyncio
async def test_a_dispatcher_pass_admits_its_whole_page_as_one_ingest_batch(tmp_path: Path) -> None:
    """polylogue-v4dcc: one page is one ingest batch, with per-item outcomes.

    Every per-batch fixed cost the live route pays -- the writer hold, the
    tier bootstrap, the convergence pass -- is paid once per page here. The
    outcomes still come back per path, so deficit, retry and isolation
    accounting are unchanged.

    Anti-vacuity: restore per-item admission (``admit`` once per file inside
    the dispatcher loop, or an ``admit_page`` that loops over ``admit``) and
    this goes red on the ingest-call count, the convergence call counts, and
    the per-path outcome split below.
    """

    paths = [tmp_path / f"session-{index}.json" for index in range(5)]
    for path in paths:
        path.write_text("{}")
    source = WatchSource(name="capture", root=tmp_path, layout=export_drop_layout((".json",)))
    admitted, excluded_path, deferred_path = paths[:3], paths[3], paths[4]

    class PageWatcher(_InlineWriterWatcher):
        def __init__(self) -> None:
            self.ingest_batches: list[list[Path]] = []
            self.embedding_calls: list[tuple[Path, ...]] = []
            self.profile_calls: list[tuple[str, ...]] = []

        def intake_revision(self, _source: WatchSource) -> int:
            return 0

        async def _ingest_files(self, batch: Sequence[Path], **_kwargs: object) -> SimpleNamespace:
            self.ingest_batches.append(list(batch))
            return SimpleNamespace(
                succeeded_file_count=len(admitted),
                succeeded_paths=tuple(admitted),
                failed_file_count=0,
                failed_paths=[str(deferred_path)],
                deferred_paths=(str(deferred_path),),
                excluded_file_count=1,
                excluded_reasons={"unsupported_shape": 1},
                excluded_paths={str(excluded_path): "unsupported_shape"},
                stale_cursor_write_count=0,
                source_payload_read_bytes=50,
                changed_session_ids=("s1", "s2"),
            )

        async def _converge_embeddings_off_writer(self, batch: Sequence[Path]) -> None:
            self.embedding_calls.append(tuple(batch))

        async def _converge_session_profiles_off_writer(self, session_ids: Sequence[str]) -> None:
            self.profile_calls.append(tuple(session_ids))

    watcher = PageWatcher()
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="capture", adapter=adapter, page_size=8)])

    result = await dispatcher.run_once(budget=1_000_000)

    assert watcher.ingest_batches == [paths]
    assert watcher.embedding_calls == [tuple(admitted)]
    assert watcher.profile_calls == [("s1", "s2")]
    report = result.require_report("capture")
    assert (report.admitted, report.excluded, report.deferred, report.retried) == (3, 1, 1, 0)
    assert result.progressed is True


@pytest.mark.asyncio
async def test_a_page_never_splits_below_one_file_and_stops_at_the_class_share(tmp_path: Path) -> None:
    """A page fills up to the class byte share and is never split below one file.

    Anti-vacuity: drop the byte-aware plan (admit the whole discovered page
    regardless of deficit) and the first assertion goes red; refuse an item
    larger than the share and the second does.
    """

    batches: list[list[Path]] = []

    class RecordingAdapter:
        def __init__(self, items: Sequence[IntakeItem]) -> None:
            self._items = tuple(items)

        async def discover(self, *, limit: int) -> Sequence[IntakeItem]:
            return self._items[:limit]

        async def admit(self, item: IntakeItem) -> AdmissionResult:
            # An adapter still owes the per-item entry point: the dispatcher
            # falls back to it whenever a page-shaped one is absent.
            return (await self.admit_page((item,)))[item.item_id]

        async def admit_page(self, items: Sequence[IntakeItem]) -> dict[str, AdmissionResult]:
            batches.append([cast(Path, item.payload) for item in items])
            return {item.item_id: AdmissionResult(AdmissionOutcome.ADMITTED, actual_cost=1) for item in items}

        async def acknowledge(self, _item: IntakeItem) -> None:
            return None

    def _item(name: str, cost: int) -> IntakeItem:
        return IntakeItem(item_id=name, class_name="capture", payload=tmp_path / name, estimated_cost=cost)

    page = [_item("a", 40), _item("b", 40), _item("c", 40)]
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="capture", adapter=RecordingAdapter(page), page_size=8)])
    await dispatcher.run_once(budget=100)
    assert [path.name for path in batches[0]] == ["a", "b"]

    batches.clear()
    whale = [_item("whale", 10_000)]
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="capture", adapter=RecordingAdapter(whale), page_size=8)])
    await dispatcher.run_once(budget=100)
    assert [path.name for path in batches[0]] == ["whale"]


def _linked_export_source(tmp_path: Path) -> tuple[WatchSource, Path]:
    """A source root whose export tree is reached through an in-root link."""

    root = tmp_path / "root"
    export = root / "store" / "2026-09"
    export.mkdir(parents=True)
    (export / "session.json").write_text("{}")
    (root / "current").symlink_to(export, target_is_directory=True)
    return WatchSource(
        name="capture", root=root, layout=export_drop_layout((".json",))
    ), root / "current" / "session.json"


@pytest.mark.asyncio
async def test_a_symlinked_export_tree_is_discovered_and_admitted(tmp_path: Path) -> None:
    """polylogue-lu1dk: an export tree behind an in-root link is acquired.

    Anti-vacuity: restore ``entry.is_dir(follow_symlinks=False)`` as the only
    directory test in ``_ordered_children`` and the walk never enters
    ``current/``, so the linked path is never discovered and never ingested.
    """

    source, linked_session = _linked_export_source(tmp_path)
    ingested: list[Path] = []

    class RecordingWatcher(_InlineWriterWatcher):
        def intake_revision(self, _source: WatchSource) -> int:
            return 0

        async def _ingest_files(self, paths: Sequence[Path], **_kwargs: object) -> SimpleNamespace:
            ingested.extend(paths)
            return SimpleNamespace(
                succeeded_file_count=len(paths),
                failed_file_count=0,
                stale_cursor_write_count=0,
                source_payload_read_bytes=2 * len(paths),
                succeeded_paths=[str(path) for path in paths],
            )

        async def _converge_embeddings_off_writer(self, paths: Sequence[Path]) -> None:
            return None

    adapter = FileIntakeAdapter(
        DaemonIntakeContext(
            archive_root=tmp_path / "archive",
            watcher=RecordingWatcher(),  # type: ignore[arg-type]
            sources=(source,),
        ),
        source,
    )
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="capture", adapter=adapter, page_size=8)])

    result = await dispatcher.run_once(budget=64)

    assert result.require_report("capture").discovered >= 1
    assert linked_session in ingested


def test_a_symlink_cycle_terminates_and_is_reported_once(tmp_path: Path) -> None:
    """A link back to an ancestor ends the walk instead of recursing forever.

    Anti-vacuity: drop the ``visited_real_paths`` check in
    ``_admit_linked_directory`` and this walk descends ``loop/loop/loop/...``
    until the recursion limit; drop the fault emission and the cycle is
    silent.
    """

    root = tmp_path / "root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    kept = nested / "session.json"
    kept.write_text("{}")
    (nested / "loop").symlink_to(root, target_is_directory=True)
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".json",)))

    with capture() as records:
        found = _bounded_source_paths(source, (source,), limit=32, after=None)

    assert found == [kept]
    cycles = [record for record in records if record.get("reason") == "symlink_cycle"]
    assert [record["path"] for record in cycles] == [str(nested / "loop")]
    assert cycles[0]["event"] == "daemon.intake.discovery_failed"


def test_a_dangling_symlink_is_a_fault_not_a_crash(tmp_path: Path) -> None:
    """A link whose target is gone is counted, and its siblings still discovered.

    Anti-vacuity: remove the broken-symlink branch in ``_ordered_children``
    and the missing export vanishes from the walk with no record at all.
    """

    root = tmp_path / "root"
    root.mkdir()
    kept = root / "session.json"
    kept.write_text("{}")
    (root / "gone.json").symlink_to(tmp_path / "never-existed.json")
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".json",)))

    with capture() as records:
        found = _bounded_source_paths(source, (source,), limit=32, after=None)

    assert found == [kept]
    faults = [record for record in records if record.get("reason") == "broken_symlink"]
    assert [record["path"] for record in faults] == [str(root / "gone.json")]
    assert faults[0]["event"] == "daemon.intake.discovery_failed"


def test_a_directory_symlink_escaping_the_source_root_is_refused(tmp_path: Path) -> None:
    """Containment outranks following: an escaping link is a fault, not material.

    Anti-vacuity: drop the containment check in ``_admit_linked_directory``
    and ``outside/secret.json`` -- a tree this source was never configured to
    read -- is discovered as intake material.
    """

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.json").write_text("{}")
    root = tmp_path / "root"
    root.mkdir()
    kept = root / "session.json"
    kept.write_text("{}")
    (root / "escape").symlink_to(outside, target_is_directory=True)
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".json",)))

    with capture() as records:
        found = _bounded_source_paths(source, (source,), limit=32, after=None)

    assert found == [kept]
    escapes = [record for record in records if record.get("reason") == "escaping_symlink"]
    assert [record["path"] for record in escapes] == [str(root / "escape")]


# -- a halted *source* inside one multiplexed class -------------------------


class _LeaseTakingSourceAdapter(FakeAdapter):
    """A configured-source adapter that takes the writer lease to admit.

    The lease is what the kqrbw incident actually wasted: 926 empty chunks
    each held it for 1.0-3.1 s on behalf of a source that had already
    refused. Counting acquisitions is what lets "zero subsequent lease
    acquisitions attributable to it" be asserted rather than inferred from
    the absence of discovery calls.

    ``endless`` models the shape that made the waste unbounded: the source
    had a 12,150-file backlog, so every pass found fresh work to select. A
    fixed pending list would exhaust itself and hide the cost behind the
    dispatcher's per-item isolation rather than behind the halt.
    """

    def __init__(
        self,
        source_name: str,
        pending: Sequence[str],
        *,
        archive_root: Path,
        endless: bool = False,
        outcome_for: Callable[[IntakeItem], AdmissionResult] | None = None,
    ) -> None:
        super().__init__("configured_local", pending, outcome_for=outcome_for)
        self.source = SimpleNamespace(name=source_name)
        self.archive_root = archive_root
        self.lease_acquisitions = 0
        self._endless = endless
        self._minted = 0

    async def discover(self, *, limit: int) -> Sequence[IntakeItem]:
        if not self._endless:
            return await super().discover(limit=limit)
        self.discover_calls.append(limit)
        items = []
        for _ in range(max(0, limit)):
            self._minted += 1
            items.append(IntakeItem(item_id=f"{self.source.name}-{self._minted}", class_name=self.class_name))
        return items

    async def admit(self, item: IntakeItem) -> AdmissionResult:
        from polylogue.core.write_lease import async_write_lease

        # Admission runs on the dispatcher's event loop: a synchronous lease
        # may not block it, so the configured adapter takes the async lease.
        async with async_write_lease(f"test.intake.{self.source.name}", archive_root=self.archive_root):
            self.lease_acquisitions += 1
            return await super().admit(item)


def _source_halt_policy(halts: HaltRegistry) -> Any:
    from polylogue.operations.intake_adapters import SubUnitHaltPolicy

    def _halt(name: str, message: str) -> None:
        halts.halt(
            unit_id(UnitKind.SOURCE, name),
            reason=HaltReason.TERMINAL_REFUSAL,
            message=message,
            frame="daemon:1",
        )

    return SubUnitHaltPolicy(
        is_halted=lambda name: halts.is_halted(unit_id(UnitKind.SOURCE, name)),
        halt=_halt,
    )


@pytest.mark.asyncio
async def test_a_source_in_terminal_refusal_stops_costing_the_writer_lease(tmp_path: Path) -> None:
    """One dead source is excluded at planning; its siblings keep draining.

    In production every configured source shares one ``configured_local``
    class, so a class-grain halt is the wrong instrument twice over: halting
    the class stops every healthy sibling, and not halting anything is
    polylogue-kqrbw -- the planner kept selecting a dead source's files and
    each empty chunk took the writer lease.

    The halt is consulted where work is *selected*, not where it is
    executed, so the page that carried the refusal is still admitted -- one
    bounded page, not a run's worth. Everything after it costs nothing.

    Anti-vacuity, both executed: remove the ``schedulable_adapters`` filter
    from ``MultiplexIntakeAdapter.discover`` and the dead source is planned
    again, so its lease count keeps climbing; make ``_isolate_sub_unit``
    return the CLASS_TERMINAL unchanged and the whole class halts, so the
    live sibling stops draining too.
    """
    from polylogue.operations.intake_adapters import MultiplexIntakeAdapter

    halts = HaltRegistry(tmp_path)
    dead = _LeaseTakingSourceAdapter(
        "claude-code",
        (),
        archive_root=tmp_path,
        endless=True,
        outcome_for=lambda _item: AdmissionResult(
            AdmissionOutcome.CLASS_TERMINAL, reason="refusing further ingest until restart"
        ),
    )
    live = _LeaseTakingSourceAdapter("codex", ["x0", "x1"], archive_root=tmp_path)
    multiplex = MultiplexIntakeAdapter((dead, live), halts=_source_halt_policy(halts))
    dispatcher = FairIntakeDispatcher(
        [IntakeClassSpec(name="configured_local", adapter=multiplex, page_size=8)],
        halts=halts,
        frame="daemon:1",
    )

    first = await dispatcher.run_once(budget=8)
    leases_at_halt = dead.lease_acquisitions
    discoveries_at_halt = len(dead.discover_calls)

    assert halts.is_halted(unit_id(UnitKind.SOURCE, "claude-code"))
    assert leases_at_halt >= 1, "the page that carried the refusal is admitted; everything after it is the defect"
    # The class itself survives: the refusal belonged to one source.
    assert not halts.is_halted(unit_id(UnitKind.INTAKE_CLASS, "configured_local"))
    assert first.skipped_halted == ()

    for _ in range(5):
        await dispatcher.run_once(budget=8)

    assert dead.lease_acquisitions == leases_at_halt, "a halted source kept taking the writer lease"
    assert len(dead.discover_calls) == discoveries_at_halt, "a halted source was planned again"
    assert dead.acknowledged == [], "a halted source's items were released as if handled"
    assert live.acknowledged == ["x0", "x1"], "a sibling source stopped draining behind the halt"


@pytest.mark.asyncio
async def test_a_halted_source_survives_a_restart(tmp_path: Path) -> None:
    """The halt is durable state, not this process' memory.

    Anti-vacuity: drop the `_write_records` call from `HaltRegistry.halt` and
    the reopened registry plans the dead source again. Executed.
    """
    from polylogue.operations.intake_adapters import MultiplexIntakeAdapter

    halts = HaltRegistry(tmp_path)
    halts.halt(
        unit_id(UnitKind.SOURCE, "claude-code"),
        reason=HaltReason.TERMINAL_REFUSAL,
        message="refusing further ingest until restart",
        frame="daemon:1",
    )

    dead = _LeaseTakingSourceAdapter("claude-code", ["c0"], archive_root=tmp_path)
    live = _LeaseTakingSourceAdapter("codex", ["x0"], archive_root=tmp_path)
    restarted = FairIntakeDispatcher(
        [
            IntakeClassSpec(
                name="configured_local",
                adapter=MultiplexIntakeAdapter((dead, live), halts=_source_halt_policy(HaltRegistry(tmp_path))),
                page_size=8,
            )
        ],
        halts=HaltRegistry(tmp_path),
    )

    await restarted.run_once(budget=8)

    assert dead.discover_calls == []
    assert dead.lease_acquisitions == 0
    assert live.acknowledged == ["x0"]

    record = HaltRegistry(tmp_path).record_for(unit_id(UnitKind.SOURCE, "claude-code"))
    assert record is not None
    assert record.reason is HaltReason.TERMINAL_REFUSAL
    assert record.frame == "daemon:1"


@pytest.mark.asyncio
async def test_a_structural_ingest_halt_stops_this_process_but_not_a_repaired_restart(tmp_path: Path) -> None:
    """A structural refusal stops selection without becoming an operator halt."""
    from polylogue.core.degraded import DegradedReason
    from polylogue.core.source_halts import clear_all_source_halts, set_source_halt
    from polylogue.operations.intake_adapters import MultiplexIntakeAdapter

    halts = HaltRegistry(tmp_path)
    dead = _LeaseTakingSourceAdapter("claude-code", (), archive_root=tmp_path, endless=True)
    live = _LeaseTakingSourceAdapter("codex", ["x0", "x1"], archive_root=tmp_path)
    multiplex = MultiplexIntakeAdapter((dead, live), halts=_source_halt_policy(halts))
    dispatcher = FairIntakeDispatcher(
        [IntakeClassSpec(name="configured_local", adapter=multiplex, page_size=8)],
        halts=halts,
        frame="daemon:1",
    )

    try:
        set_source_halt(
            "claude-code",
            DegradedReason(
                code="schema_version_mismatch",
                message="db schema v9, runtime expects v11",
                detail={"current_version": 9, "expected_version": 11},
            ),
        )

        for _ in range(3):
            await dispatcher.run_once(budget=8)
    finally:
        clear_all_source_halts()

    assert dead.discover_calls == [], "a structurally halted source was planned again"
    assert dead.lease_acquisitions == 0, "a structurally halted source kept taking the writer lease"
    assert live.acknowledged == ["x0", "x1"], "a sibling source stopped draining behind the halt"

    assert halts.record_for(unit_id(UnitKind.SOURCE, "claude-code")) is None
    # Reconstruct the durable registry after clearing process state, as a
    # restart does. The repaired source must be selected without editing JSON.
    repaired = _LeaseTakingSourceAdapter("claude-code", ["repaired"], archive_root=tmp_path)
    restarted_halts = HaltRegistry(tmp_path)
    restarted = FairIntakeDispatcher(
        [
            IntakeClassSpec(
                name="configured_local",
                adapter=MultiplexIntakeAdapter((repaired,), halts=_source_halt_policy(restarted_halts)),
            )
        ],
        halts=restarted_halts,
    )
    await restarted.run_once(budget=8)
    assert repaired.acknowledged == ["repaired"]


@pytest.mark.asyncio
async def test_an_unhalted_source_is_still_planned(tmp_path: Path) -> None:
    """The opposite direction: the bridge must not exclude a healthy source.

    Anti-vacuity: treat any recorded halt for *any* source as halting this one
    (or skip every adapter unconditionally) and this goes red with nothing
    drained.
    """
    from polylogue.core.degraded import DegradedReason
    from polylogue.core.source_halts import clear_all_source_halts, set_source_halt
    from polylogue.operations.intake_adapters import MultiplexIntakeAdapter

    halts = HaltRegistry(tmp_path)
    live = _LeaseTakingSourceAdapter("codex", ["x0"], archive_root=tmp_path)
    dispatcher = FairIntakeDispatcher(
        [
            IntakeClassSpec(
                name="configured_local",
                adapter=MultiplexIntakeAdapter((live,), halts=_source_halt_policy(halts)),
                page_size=8,
            )
        ],
        halts=halts,
        frame="daemon:1",
    )

    try:
        set_source_halt("claude-code", DegradedReason(code="schema_version_mismatch", message="other source"))
        await dispatcher.run_once(budget=8)
    finally:
        clear_all_source_halts()

    assert live.acknowledged == ["x0"]
    assert halts.record_for(unit_id(UnitKind.SOURCE, "codex")) is None


@pytest.mark.asyncio
async def test_a_deterministic_admission_failure_is_isolated_not_retried_forever() -> None:
    """An unchanged item failing the same non-transient way stops being retried.

    Anti-vacuity (polylogue-wyi9p): with the attempt counter reset after every
    cooldown, the defect is retried each cooldown for the daemon's life and
    ``isolated_items`` stays empty. A transient failure (a lock) must keep
    retrying, so the same loop over it never isolates.
    """

    now = [0.0]

    def broken(item: IntakeItem) -> AdmissionResult:
        if item.item_id != "poison":
            return AdmissionResult(AdmissionOutcome.ADMITTED)
        raise KeyError("missing_field")

    def locked(item: IntakeItem) -> AdmissionResult:
        if item.item_id != "poison":
            return AdmissionResult(AdmissionOutcome.ADMITTED)
        raise sqlite3.OperationalError("database is locked")

    for outcome_for, expect_isolated in ((broken, True), (locked, False)):
        adapter = FakeAdapter("capture", ["poison", "sibling"], outcome_for=outcome_for)
        dispatcher = FairIntakeDispatcher(
            [
                IntakeClassSpec(
                    name="capture",
                    adapter=adapter,
                    max_attempts=1,
                    retry_cooldown_s=1.0,
                    max_deterministic_cooldowns=3,
                )
            ],
            clock=lambda: now[0],
        )
        for _ in range(6):
            now[0] += 2.0
            await dispatcher.run_once()
        assert (dispatcher.isolated_items("capture") == frozenset({"poison"})) is expect_isolated
        # The failing item never blocks the rest of its class.
        assert adapter.acknowledged == ["sibling"]


@pytest.mark.asyncio
async def test_a_sqlite_error_escaping_page_admission_is_reported_once_per_page() -> None:
    """Anti-vacuity (polylogue-wyi9p): an error the adapter's page handler
    does not catch, such as ``sqlite3.IntegrityError``, produced no
    ``page_refused`` event; its reason appeared only per item."""

    class PageAdapter(FakeAdapter):
        async def admit_page(self, _items: Sequence[IntakeItem]) -> dict[str, AdmissionResult]:
            raise sqlite3.IntegrityError("UNIQUE constraint failed: raws.raw_id")

    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="capture", adapter=PageAdapter("capture", ["a", "b"]))])
    with capture() as records:
        report = await dispatcher.run_once()

    refused = [record for record in records if record["event"] == "daemon.intake.page_refused"]
    assert [(record["component"], record["error_type"], record["files"]) for record in refused] == [
        ("capture", "IntegrityError", 2)
    ]
    assert report.require_report("capture").retried == 2


def test_only_contention_and_storage_faults_are_transient() -> None:
    """Anti-vacuity: classifying by exception class alone made every
    ``sqlite3.OperationalError`` transient, so ``no such table`` was retried
    after every cooldown forever instead of being isolated."""
    from polylogue.daemon.intake import is_transient_admission_error

    locked = sqlite3.OperationalError("database is locked")
    locked.sqlite_errorcode = 5  # SQLITE_BUSY
    missing = sqlite3.OperationalError("no such table: raws")
    missing.sqlite_errorcode = 1  # SQLITE_ERROR
    assert is_transient_admission_error(locked) is True
    assert is_transient_admission_error(OSError("disk")) is True
    assert is_transient_admission_error(missing) is False
    assert is_transient_admission_error(KeyError("field")) is False


@pytest.mark.asyncio
async def test_new_content_under_an_isolated_identity_is_admitted_again() -> None:
    """Isolation binds to the revision that failed, not the path alone.

    Anti-vacuity (Codex): with a path-only isolated key, a poison capture
    replaced at the same path by a valid export stayed skipped until the
    daemon restarted, whatever its new content.
    """

    now = [0.0]
    revision = ["poison-bytes"]

    class RevisedAdapter(FakeAdapter):
        async def discover(self, *, limit: int) -> Sequence[IntakeItem]:
            return [
                IntakeItem(item_id=name, class_name=self.class_name, revision=revision[0])
                for name in self.pending[:limit]
            ]

    def by_revision(item: IntakeItem) -> AdmissionResult:
        if item.revision == "poison-bytes":
            raise KeyError("missing_field")
        return AdmissionResult(AdmissionOutcome.ADMITTED)

    adapter = RevisedAdapter("capture", ["export"], outcome_for=by_revision)
    dispatcher = FairIntakeDispatcher(
        [
            IntakeClassSpec(
                name="capture", adapter=adapter, max_attempts=1, retry_cooldown_s=1.0, max_deterministic_cooldowns=3
            )
        ],
        clock=lambda: now[0],
    )
    for _ in range(6):
        now[0] += 2.0
        await dispatcher.run_once()
    assert dispatcher.isolated_items("capture") == frozenset({"export"})
    assert adapter.admitted == []

    revision[0] = "valid-bytes"
    now[0] += 2.0
    await dispatcher.run_once()
    assert adapter.admitted == ["export"]
    assert dispatcher.isolated_items("capture") == frozenset()


@pytest.mark.asyncio
async def test_a_success_between_failures_restarts_the_isolation_streak() -> None:
    """Anti-vacuity: an exhaustion count kept across a successful admission
    isolates a persistent item after failures that were never consecutive."""

    now = [0.0]
    fail = [True]

    def flaky(_item: IntakeItem) -> AdmissionResult:
        if fail[0]:
            raise KeyError("missing_field")
        return AdmissionResult(AdmissionOutcome.DUPLICATE)

    class PersistentAdapter(FakeAdapter):
        async def acknowledge(self, item: IntakeItem) -> None:
            self.acknowledged.append(item.item_id)

    adapter = PersistentAdapter("capture", ["poison"], outcome_for=flaky)
    dispatcher = FairIntakeDispatcher(
        [
            IntakeClassSpec(
                name="capture", adapter=adapter, max_attempts=1, retry_cooldown_s=1.0, max_deterministic_cooldowns=3
            )
        ],
        clock=lambda: now[0],
    )
    for failing in (True, True, False, True, True):
        fail[0] = failing
        now[0] += 2.0
        await dispatcher.run_once()
    assert dispatcher.isolated_items("capture") == frozenset()


@pytest.mark.asyncio
async def test_degraded_intake_service_parks_without_passes_or_settlement() -> None:
    """A fully degraded daemon runs no intake pass and never settles a cold build.

    Anti-vacuity: drop the degraded park from ``DaemonIntakeService.run`` and
    the loop calls ``run_once`` on every tick; with a quiescent pass after
    earlier progress it then invokes ``on_backlog_drained`` for a backlog that
    is parked, not drained.
    """
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    class _Dispatcher:
        calls = 0

        async def run_once(self, *, budget: object) -> object:
            self.calls += 1
            raise AssertionError("a degraded intake service must not run a pass")

        def schedulable_classes(self) -> tuple[object, ...]:
            return ()

    drained: list[bool] = []
    dispatcher = _Dispatcher()
    service = DaemonIntakeService(
        cast(Any, dispatcher), idle_delay_s=0.05, on_backlog_drained=lambda: drained.append(True)
    )
    set_degraded(DegradedReason(code="schema_version_mismatch", message="v12 vs v9"))
    task = asyncio.create_task(service.run())
    try:
        await asyncio.sleep(0.3)
        assert not task.done()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        clear_degraded()
    assert dispatcher.calls == 0
    assert drained == []


@pytest.mark.asyncio
async def test_degradation_during_a_pass_stops_the_remaining_classes() -> None:
    """Anti-vacuity: drop the per-class degraded check from ``run_once`` and the
    second class is still discovered after the first degraded the daemon."""
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    def degrade(_item: IntakeItem) -> AdmissionResult:
        set_degraded(DegradedReason(code="database_layout_mismatch", message="structural error"))
        return AdmissionResult(AdmissionOutcome.RETRYABLE, reason="structural error")

    first = FakeAdapter("configured_local", ["file-00"], outcome_for=degrade)
    second = FakeAdapter("hook_events", ["event-00"])
    dispatcher = FairIntakeDispatcher(
        (
            IntakeClassSpec("configured_local", first, page_size=1),
            IntakeClassSpec("hook_events", second, page_size=1),
        )
    )
    try:
        await dispatcher.run_once()
    finally:
        clear_degraded()

    assert first.discover_calls
    assert second.discover_calls == []


@pytest.mark.asyncio
async def test_a_pass_that_degrades_the_daemon_runs_no_post_pass_callback() -> None:
    """Anti-vacuity: drop the post-pass degraded check from ``DaemonIntakeService.run``
    and ``on_pass_complete`` runs for a pass that degraded the daemon."""
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    def degrade(_item: IntakeItem) -> AdmissionResult:
        set_degraded(DegradedReason(code="database_layout_mismatch", message="structural error"))
        return AdmissionResult(AdmissionOutcome.ADMITTED)

    files = FakeAdapter("configured_local", ["file-00"], outcome_for=degrade)
    dispatcher = FairIntakeDispatcher((IntakeClassSpec("configured_local", files, page_size=1),))
    completed: list[object] = []
    service = DaemonIntakeService(dispatcher, idle_delay_s=0.05, on_pass_complete=completed.append)
    task = asyncio.create_task(service.run())
    try:
        await asyncio.sleep(0.3)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        clear_degraded()
    assert files.admitted == ["file-00"]
    assert completed == []
    # The committed progress is still recorded for later cold-build settlement.
    assert service._progress_since_blocked is True


@pytest.mark.asyncio
async def test_unattempted_retryable_results_count_no_attempt() -> None:
    """Anti-vacuity: count an unattempted result as an attempt and a
    ``max_attempts=1`` class puts the item into cooldown."""
    files = FakeAdapter(
        "configured_local",
        ["file-00"],
        outcome_for=lambda _item: AdmissionResult(AdmissionOutcome.RETRYABLE, actual_cost=0, unattempted=True),
    )
    dispatcher = FairIntakeDispatcher((IntakeClassSpec("configured_local", files, page_size=1, max_attempts=1),))
    await dispatcher.run_once()
    runtime = dispatcher._runtime["configured_local"]
    assert runtime.attempts == {}
    assert runtime.retry_after == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "transient"),
    [
        (ValueError("malformed selection"), False),
        (OSError("disk busy"), True),
    ],
)
async def test_a_caught_page_error_is_classified_like_an_escaped_one(
    workspace_env: dict[str, Path], error: Exception, transient: bool
) -> None:
    """Anti-vacuity (Codex): the file adapter's own handler returned every
    caught ``ValueError`` as transient, so a deterministic page failure reset
    its exhaustion streak each cooldown and was retried forever."""
    from polylogue import Polylogue
    from polylogue.operations.intake_adapters import DaemonIntakeContext, FileIntakeAdapter
    from polylogue.sources.live import LiveWatcher, WatchSource
    from polylogue.sources.live.cursor import CursorStore

    archive_root = workspace_env["archive_root"]
    source_root = workspace_env["data_root"] / "claude-projects"
    source_root.mkdir(parents=True)
    (source_root / "a.jsonl").write_text("{}\n", encoding="utf-8")
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator

    archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    # The daemon watcher writes its cursor through the writer it is given.
    coordinator = DaemonWriteCoordinator(archive_root=archive_root)
    watcher = LiveWatcher(
        archive,
        (WatchSource(name="claude-code", root=source_root, layout=export_drop_layout((".jsonl",))),),
        cursor=CursorStore(archive_root / "index.db"),
        write_coordinator=coordinator,
    )

    async def failing_ingest(*_args: object, **_kwargs: object) -> object:
        raise error

    watcher._ingest_files = failing_ingest  # type: ignore[method-assign,assignment]
    try:
        adapter = FileIntakeAdapter(
            DaemonIntakeContext(archive_root=archive_root, watcher=watcher, sources=watcher._sources),
            watcher._sources[0],
        )
        outcomes = await adapter.admit_page(await adapter.discover(limit=8))
    finally:
        watcher.stop()
        await archive.close()
        assert await coordinator.shutdown(timeout=float("inf"))
    assert outcomes
    assert {result.outcome for result in outcomes.values()} == {AdmissionOutcome.RETRYABLE}
    assert {result.transient for result in outcomes.values()} == {transient}


@pytest.mark.asyncio
async def test_stale_cursor_retries_only_its_path_and_settles_empty_sibling(tmp_path: Path) -> None:
    """A stale cursor in one source file must not strand a stable empty peer."""
    from polylogue.operations.intake_adapters import DaemonIntakeContext, FileIntakeAdapter

    root = tmp_path / "source"
    root.mkdir()
    empty = root / "empty.jsonl"
    stale = root / "stale.jsonl"
    empty.write_bytes(b"")
    stale.write_text("{}\n", encoding="utf-8")
    source = WatchSource(name="capture", root=root, layout=export_drop_layout((".jsonl",)))
    cursor = CursorStore(tmp_path / "index.db")

    async def ingest(paths: Sequence[Path], **_kwargs: object) -> SimpleNamespace:
        assert paths == [empty, stale]
        return SimpleNamespace(
            stale_cursor_write_count=1,
            stale_cursor_paths=(str(stale),),
            succeeded_paths=(str(empty), str(stale)),
            failed_paths=(),
            deferred_paths=(),
            excluded_paths={},
            settled_exclusion_paths={str(empty): "no_sessions"},
            partial_admission_paths={},
            daemon_degraded_skip=False,
            time_budget_exceeded=False,
            source_payload_read_bytes=1,
        )

    watcher = SimpleNamespace(
        has_write_coordinator=True,
        _run_writer_sync=_inline_writer_sync,
        _cursor=cursor,
        intake_revision=lambda _source: 0,
        _ingest_files=ingest,
    )
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    empty_item = IntakeItem(item_id=f"file:{empty}", class_name="capture", payload=empty)
    stale_item = IntakeItem(item_id=f"file:{stale}", class_name="capture", payload=stale)

    outcomes = await adapter.admit_page((empty_item, stale_item))

    assert outcomes[empty_item.item_id].outcome is AdmissionOutcome.EXCLUDED
    assert outcomes[stale_item.item_id].outcome is AdmissionOutcome.RETRYABLE
    assert outcomes[stale_item.item_id].reason == "source cursor write was stale"


@pytest.mark.asyncio
async def test_a_transient_failure_mid_window_breaks_the_isolation_streak() -> None:
    """Anti-vacuity (Codex): only the result on the cooldown boundary was
    examined, so two clean deterministic windows followed by a window of
    transient-then-deterministic results isolated the item."""

    now = [0.0]
    script: list[BaseException] = []

    def scripted(_item: IntakeItem) -> AdmissionResult:
        raise script.pop(0)

    adapter = FakeAdapter("capture", ["poison"], outcome_for=scripted)
    dispatcher = FairIntakeDispatcher(
        [
            IntakeClassSpec(
                name="capture", adapter=adapter, max_attempts=2, retry_cooldown_s=1.0, max_deterministic_cooldowns=3
            )
        ],
        clock=lambda: now[0],
    )
    locked = sqlite3.OperationalError("database is locked")
    locked.sqlite_errorcode = 5  # SQLITE_BUSY
    script.extend([KeyError("f"), KeyError("f"), KeyError("f"), KeyError("f"), locked, KeyError("f")])
    for _ in range(6):
        now[0] += 2.0
        await dispatcher.run_once()
    assert not script
    assert dispatcher.isolated_items("capture") == frozenset()
