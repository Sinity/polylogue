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
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

# Bound before any test monkeypatches ``storage.derived.raw.RawObservationDerivation``:
# ``operations.raw_observation_derivation`` reads ``RawObservationDerivation.recipe_version``
# at module scope, so a first import that happens under a patched fake class raises
# AttributeError. Which test imports it first depends on the pytest-randomly seed.
import polylogue.operations.raw_observation_derivation as _raw_observation_derivation  # noqa: F401
from polylogue.core.enums import Provider
from polylogue.daemon.derivation import DerivationFrame
from polylogue.daemon.intake import (
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
    FileIntakeAdapter,
    RawMaterializationDiscovery,
    RawMaterializationIntakeAdapter,
    _bounded_source_paths,
    discover_pending_raw_ids,
)
from polylogue.sources.live.watcher import WatchSource
from polylogue.sources.walk_faults import WalkFault, WalkRefusedError
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root


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
        suffixes=(".json",),
        ignored_dir_names=frozenset({"ignored"}),
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
    source = WatchSource(name="test", root=tmp_path, suffixes=(".json",))
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
    source = WatchSource(name="test", root=tmp_path, suffixes=(".json",))
    scandir = _ShuffledScandir(os.scandir, 11)

    emitted = _bounded_source_paths(source, (source,), limit=len(on_disk) + 5, after=None, scandir=scandir)
    assert emitted == sorted(on_disk, key=str)
    assert str(tmp_path / "a.json") < str(tmp_path / "a" / "b.json")


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
                acquired_at_ms=1,
            )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("legacy raw census selector was called")

    monkeypatch.setattr("polylogue.storage.archive_readiness.raw_materialization_readiness_snapshot", forbidden)
    result = discover_pending_raw_ids(tmp_path, limit=2, max_payload_bytes=1024)

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
            acquired_at_ms=1,
        )
        pending = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"pending",
            source_path="pending.json",
            acquired_at_ms=1,
        )
    calls: list[tuple[str | None, int]] = []

    class FakeRawObservationDerivation:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def required_page(
            self, _frame: object, *, cursor: str | None, limit: int
        ) -> tuple[tuple[str, ...], str | None]:
            calls.append((cursor, limit))
            return ((valid,), valid) if cursor is None else ((pending,), None)

        def inspect(self, _frame: object, keys: Sequence[str]) -> dict[str, str]:
            return {key: "valid" if key == valid else "missing" for key in keys}

    monkeypatch.setattr("polylogue.storage.derived.raw.RawObservationDerivation", FakeRawObservationDerivation)
    discovery = RawMaterializationDiscovery(tmp_path, max_payload_bytes=1024)

    assert discovery.discover_pending_raw_ids(1) == ()
    assert calls == [(None, 1)]
    assert discovery.discover_pending_raw_ids(1) == ((pending, len(b"pending")),)
    assert calls == [(None, 1), (valid, 1)]


@pytest.mark.asyncio
async def test_raw_discovery_moves_past_a_cooled_down_poison_in_the_fair_dispatcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: reset discovery each pass, and the cooled-down head starves the healthy raw."""
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        valid = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"v",
            source_path="valid.json",
            acquired_at_ms=1,
        )
        poison = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"p",
            source_path="poison.json",
            acquired_at_ms=1,
        )
        healthy = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"h",
            source_path="healthy.json",
            acquired_at_ms=1,
        )

    class FakeRawObservationDerivation:
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

    monkeypatch.setattr("polylogue.storage.derived.raw.RawObservationDerivation", FakeRawObservationDerivation)
    admitted: list[str] = []

    async def admit(raw_id: str) -> AdmissionResult:
        if raw_id == poison:
            return AdmissionResult(AdmissionOutcome.RETRYABLE, reason="poison")
        admitted.append(raw_id)
        return AdmissionResult(AdmissionOutcome.ADMITTED)

    discovery = RawMaterializationDiscovery(tmp_path, max_payload_bytes=1024)
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
            acquired_at_ms=1,
        )
        second = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"second",
            source_path="second.json",
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
    monkeypatch.setattr("polylogue.storage.derived.raw.RawObservationDerivation", FakeRawObservationDerivation)
    discovery = RawMaterializationDiscovery(tmp_path, max_payload_bytes=1024)

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

    monkeypatch.setattr("polylogue.storage.derived.raw.RawObservationDerivation", FakeRawObservationDerivation)
    discovery = RawMaterializationDiscovery(tmp_path, max_payload_bytes=1024)

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
                acquired_at_ms=1,
                raw_id=first,
            )
            == first
        )

    discovery = RawMaterializationDiscovery(tmp_path, max_payload_bytes=1024)
    assert discovery.discover_pending_raw_ids(1) == ((first, len(first_payload)),)

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        earlier = "a" * 64
        earlier_payload = b"earlier-frontier"
        assert (
            archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=earlier_payload,
                source_path="earlier.json",
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

    monkeypatch.setattr("polylogue.storage.derived.raw.RawObservationDerivation", FakeRawObservationDerivation)
    discovery = RawMaterializationDiscovery(tmp_path, max_payload_bytes=1024)

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
    source = WatchSource(name="capture", root=tmp_path, suffixes=(".json",))
    convergence_paths: list[tuple[Path, ...]] = []

    class StaleCursorWatcher:
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
    source = WatchSource(name="test", root=tmp_path, suffixes=(".json",))

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
    source = WatchSource(name="test", root=missing, suffixes=(".json",))

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
    source = WatchSource(name="capture", root=tmp_path, suffixes=(".json",))

    class ExcludingWatcher:
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

    monkeypatch.setattr("polylogue.storage.derived.raw.RawObservationDerivation", FakeRawObservationDerivation)
    discovery = RawMaterializationDiscovery(tmp_path, max_payload_bytes=1024)

    assert discovery.discover_pending_raw_ids(4) == ()
    for index in range(5):
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=f"arrival-{index}".encode(),
                source_path=f"arrival-{index}.json",
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
    discovery = RawMaterializationDiscovery(tmp_path, max_payload_bytes=1024)

    assert not (tmp_path / "source.db").exists()
    assert discovery.discover_pending_raw_ids(4) == ()


def test_raw_discovery_admits_work_once_the_tier_appears_without_a_restart(tmp_path: Path) -> None:
    """One long-lived discovery re-evaluates availability every pass.

    Anti-vacuity: latching availability at construction -- the startup-only
    check this bead replaces -- keeps the second call empty and makes this
    test red.
    """
    discovery = RawMaterializationDiscovery(tmp_path, max_payload_bytes=1024)
    assert discovery.discover_pending_raw_ids(4) == ()

    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"raw-after-bootstrap",
            source_path="late.json",
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
    source = WatchSource(name="capture", root=tmp_path, suffixes=(".json",))
    admitted, excluded_path, deferred_path = paths[:3], paths[3], paths[4]

    class PageWatcher:
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
    return WatchSource(name="capture", root=root, suffixes=(".json",)), root / "current" / "session.json"


@pytest.mark.asyncio
async def test_a_symlinked_export_tree_is_discovered_and_admitted(tmp_path: Path) -> None:
    """polylogue-lu1dk: an export tree behind an in-root link is acquired.

    Anti-vacuity: restore ``entry.is_dir(follow_symlinks=False)`` as the only
    directory test in ``_ordered_children`` and the walk never enters
    ``current/``, so the linked path is never discovered and never ingested.
    """

    source, linked_session = _linked_export_source(tmp_path)
    ingested: list[Path] = []

    class RecordingWatcher:
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
    source = WatchSource(name="capture", root=root, suffixes=(".json",))

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
    source = WatchSource(name="capture", root=root, suffixes=(".json",))

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
    source = WatchSource(name="capture", root=root, suffixes=(".json",))

    with capture() as records:
        found = _bounded_source_paths(source, (source,), limit=32, after=None)

    assert found == [kept]
    escapes = [record for record in records if record.get("reason") == "escaping_symlink"]
    assert [record["path"] for record in escapes] == [str(root / "escape")]
