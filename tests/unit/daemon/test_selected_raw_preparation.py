"""Selected independent Raw preparations use real shared compute admission."""

from __future__ import annotations

import asyncio
import gc
import threading
import weakref
from builtins import BaseExceptionGroup
from collections.abc import Callable, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from polylogue.core.compute import BoundedComputeAdapter
from polylogue.core.enums import Provider
from polylogue.core.write_lease import coordinator_write_lease_active
from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
from polylogue.schemas.runtime_registry import SchemaRegistry
from polylogue.sources.prepared_jsonl import PreparedJsonl
from polylogue.storage.derived import raw as raw_module
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead
from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.selected_raw_preparation import acquire_independent_codex_raws


@pytest.mark.asyncio
@pytest.mark.uses_real_clock
@pytest.mark.parametrize(
    "workers,capacity,expected_peak", [(1, 64 * 1024 * 1024, 1), (8, 64 * 1024 * 1024, 2), (8, 1, 1)]
)
async def test_selected_independent_preparations_overlap_with_real_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workers: int,
    capacity: int,
    expected_peak: int,
    record_property: Callable[[str, object], None],
) -> None:
    await _exercise_independent_preparation_admission(
        tmp_path, monkeypatch, workers, capacity, expected_peak, record_property
    )


@pytest.mark.asyncio
@pytest.mark.uses_real_clock
async def test_shared_page_context_allows_two_real_parsers_in_once_charged_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record_property: Callable[[str, object], None]
) -> None:
    """Repeated per-worker schema admission cannot fit this same complete page."""
    await _exercise_independent_preparation_admission(tmp_path, monkeypatch, 8, None, 2, record_property)


async def _exercise_independent_preparation_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workers: int,
    capacity: int | None,
    expected_peak: int,
    record_property: Callable[[str, object], None],
) -> None:
    """The ordinary route overlaps two closed-file parsers, never Source/writers.

    Reverting page preparation to nested/serial compute makes the 8-worker law
    fail; bypassing byte admission makes the oversized-input law fail.
    """
    await run_archive_fixture_write(tmp_path, lambda: bootstrap_archive_root(tmp_path))
    raws = await run_archive_fixture_write(tmp_path, lambda: acquire_independent_codex_raws(tmp_path, 2))
    constrained_shared_input = capacity is None
    if capacity is None:
        with SchemaRegistry().current_provider_snapshot(Provider.CODEX) as schema:
            unique_schema_bytes = sum(
                {id(payload): len(payload) for _root, files in schema.roots for _name, payload in files}.values()
            )
        with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
            private_bytes = sum(archive.raw_payload_sizes(raws).values())
        assert unique_schema_bytes > 0 and private_bytes > 0
        capacity = unique_schema_bytes + private_bytes
    kernel = BoundedComputeAdapter(max_workers=workers, queue_units=8, queue_bytes=capacity)
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    owner = RawObservationConvergenceOwner(
        tmp_path,
        compute_adapter=kernel,
        write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
        write_coordinator=coordinator,
    )
    both_started = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    active = peak = calls = 0
    pages: list[raw_module.NeutralRawPreparation] = []
    actual_private_bytes: dict[str, int] = {}
    original_capture = raw_module.RawObservationDerivation.capture_neutral_raws
    original = raw_module._parse_captured_neutral

    def capture(
        adapter: raw_module.RawObservationDerivation,
        raw_ids: Sequence[str],
        *,
        selection: Callable[[PreparedSessionSourceRead], Sequence[str]] | None,
    ) -> raw_module.NeutralRawPreparation | None:
        page = original_capture(adapter, raw_ids, selection=selection)
        if page is not None:
            pages.append(page)
        return page

    def prepare(
        raw_id: str,
        captured: raw_module._CapturedNeutralRaw,
        artifact_key: tuple[object, ...],
        context: raw_module.NeutralPreparationContext,
        scratch: Path,
    ) -> PreparedJsonl:
        nonlocal active, peak, calls
        assert not coordinator_write_lease_active()
        kernel.require_current_creator()
        assert kernel.snapshot().exclusive_byte_units == 0
        with lock:
            calls += 1
            active += 1
            actual_private_bytes[raw_id] = captured.descriptor[4]
            peak = max(peak, active)
            if active == expected_peak:
                both_started.set()
        try:
            assert release.wait(10), "preparations did not reach the admitted rendezvous"
            return original(raw_id, captured, artifact_key, context, scratch)
        finally:
            with lock:
                active -= 1
                actual_private_bytes.pop(raw_id)

    monkeypatch.setattr(raw_module.RawObservationDerivation, "capture_neutral_raws", capture)
    monkeypatch.setattr(raw_module, "_parse_captured_neutral", prepare)
    task = asyncio.create_task(owner.replay_retained_raw_ids(raws))
    rendezvous = asyncio.create_task(asyncio.to_thread(both_started.wait, 10))
    try:
        completed, _ = await asyncio.wait((task, rendezvous), return_when=asyncio.FIRST_COMPLETED)
        if task in completed:
            await task
            pytest.fail("ordinary retained replay ended before independent parsers overlapped")
        assert await rendezvous, f"admitted parser calls={calls}, peak={peak}, active={active}, captures={len(pages)}"
        page = pages[0]
        assert page.shared_input is not None
        snapshot = kernel.snapshot()
        assert snapshot.retained_input_count == 1
        assert snapshot.retained_input_bytes == page.shared_context_bytes > 0
        with lock:
            private_bytes = sum(actual_private_bytes.values())
        assert snapshot.active_input_bytes == private_bytes
        shared_charge = min(page.shared_context_bytes, capacity)
        assert snapshot.used_bytes == shared_charge + min(private_bytes, capacity - shared_charge)
        assert snapshot.to_dict()["retained_input_bytes"] == page.shared_context_bytes
        record_property("shared_input_bytes", page.shared_context_bytes)
        record_property("private_input_bytes", private_bytes)
        record_property("admitted_byte_charge", snapshot.used_bytes)
        record_property("simultaneous_parser_count", snapshot.active_units)
        if constrained_shared_input:
            assert snapshot.used_bytes == capacity
            repeated_schema_charge = private_bytes + expected_peak * page.shared_context_bytes
            assert repeated_schema_charge > capacity
            record_property("repeated_schema_byte_charge", repeated_schema_charge)
        release.set()
        (await task).require_complete()
        assert peak == expected_peak
        assert calls == 2, "fresh Source phases recopy/reparse unchanged neutral inputs"
        assert kernel.snapshot().used_units == kernel.snapshot().used_bytes == 0
        with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
            assert archive.index_connection is not None
            assert archive.source_connection is not None
            assert archive.index_connection is not None
            assert {
                row[0] for row in archive.index_connection.execute("SELECT accepted_raw_id FROM raw_revision_heads")
            } == set(raws)
            assert archive.index_connection.execute("SELECT count(*) FROM sessions").fetchone()[0] == 2
            assert archive.source_connection is not None
            assert (
                archive.source_connection.execute(
                    "SELECT count(*) FROM raw_authority_parser_census WHERE status='complete'"
                ).fetchone()[0]
                == 2
            )
        assert not tuple((tmp_path / "blob").glob("**/.raw-prepared-*"))
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, rendezvous, return_exceptions=True)
        kernel.shutdown(wait=True)
        await coordinator.shutdown(timeout=1)


@pytest.mark.asyncio
@pytest.mark.uses_real_clock
@pytest.mark.parametrize(
    "phase,cleanup_fault", [("capture", False), ("parse", False), ("capture", True), ("parse", True)]
)
async def test_selected_preparation_cancellation_joins_creators_before_scratch_retirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    cleanup_fault: bool,
) -> None:
    """A cancelled logical wait retains successful late-returned file custody."""
    await run_archive_fixture_write(tmp_path, lambda: bootstrap_archive_root(tmp_path))
    raws = await run_archive_fixture_write(tmp_path, lambda: acquire_independent_codex_raws(tmp_path, 2))
    kernel = BoundedComputeAdapter(max_workers=8, queue_units=8)
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    owner = RawObservationConvergenceOwner(
        tmp_path,
        compute_adapter=kernel,
        write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
        write_coordinator=coordinator,
    )
    started, release = threading.Event(), threading.Event()
    pages: list[raw_module.NeutralRawPreparation] = []
    original_capture = raw_module.RawObservationDerivation._capture_neutral_jsonl
    original_parse = raw_module._parse_captured_neutral
    original_close = raw_module.NeutralRawPreparation.close

    def capture(
        adapter: raw_module.RawObservationDerivation,
        key: str,
        *,
        selection: Callable[[PreparedSessionSourceRead], Sequence[str]] | None,
        seal: PreparedIndexMutation,
        carry: raw_module._PreparationCarry,
    ) -> tuple[raw_module.NeutralRawPreparation, tuple[tuple[str, ...], tuple[str, ...]]] | None:
        result = original_capture(adapter, key, selection=selection, seal=seal, carry=carry)
        if result is not None:
            pages.append(result[0])
        if phase == "capture":
            started.set()
            assert release.wait(10)
        return result

    def parse(
        raw_id: str,
        captured: raw_module._CapturedNeutralRaw,
        artifact_key: tuple[object, ...],
        context: raw_module.NeutralPreparationContext,
        scratch: Path,
    ) -> PreparedJsonl:
        artifact = original_parse(raw_id, captured, artifact_key, context, scratch)
        started.set()
        assert release.wait(10)
        if cleanup_fault and phase == "parse":
            raise BaseExceptionGroup(
                "native cancellation with failed cleanup",
                [asyncio.CancelledError(), ValueError("declared parser cleanup failure")],
            )
        return artifact

    def close(page: raw_module.NeutralRawPreparation) -> None:
        assert kernel.snapshot().used_units == 0, "scratch retired while creator still owns admission"
        assert release.is_set()
        if cleanup_fault and phase == "capture":
            raise ValueError("declared page cleanup failure")
        original_close(page)

    monkeypatch.setattr(raw_module.RawObservationDerivation, "_capture_neutral_jsonl", capture)
    monkeypatch.setattr(raw_module, "_parse_captured_neutral", parse)
    monkeypatch.setattr(raw_module.NeutralRawPreparation, "close", close)
    task = asyncio.create_task(owner.replay_retained_raw_ids(raws))
    try:
        assert await asyncio.to_thread(started.wait, 10)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done(), "cancellation abandoned the physical creator"
        release.set()
        if cleanup_fault:
            with pytest.raises(BaseExceptionGroup) as failure:
                await task
            assert failure.value.subgroup(asyncio.CancelledError) is not None
            assert failure.value.subgroup(ValueError) is not None
        else:
            with pytest.raises(asyncio.CancelledError):
                await task
        assert kernel.snapshot().used_units == 0
        if cleanup_fault and phase == "capture":
            assert kernel.snapshot().retained_input_count == 1
            assert kernel.snapshot().retained_input_bytes == pages[0].shared_context_bytes > 0
            assert pages[0].contexts
            original_close(pages[0])
        assert kernel.snapshot().retained_input_count == 0
        with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
            assert archive.index_connection is not None
            assert archive.source_connection is not None
            assert archive.index_connection.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
        if not cleanup_fault:
            monkeypatch.setattr(raw_module.RawObservationDerivation, "_capture_neutral_jsonl", original_capture)
            monkeypatch.setattr(raw_module, "_parse_captured_neutral", original_parse)
            monkeypatch.setattr(raw_module.NeutralRawPreparation, "close", original_close)
            # A fresh owner has no preparation page/carry. Durable acquired bytes
            # are the only input after the interrupted process-local attempt.
            restarted = RawObservationConvergenceOwner(
                tmp_path,
                compute_adapter=kernel,
                write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
                write_coordinator=coordinator,
            )
            (await restarted.replay_retained_raw_ids(raws)).require_complete()
            with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
                assert archive.index_connection is not None
                assert archive.source_connection is not None
                assert {
                    row[0] for row in archive.index_connection.execute("SELECT accepted_raw_id FROM raw_revision_heads")
                } == set(raws)
                assert archive.source_connection.execute("SELECT count(*) FROM raw_sessions").fetchone()[0] == len(raws)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for page in pages:
            original_close(page)
        kernel.shutdown(wait=True)
        await coordinator.shutdown(timeout=1)


@pytest.mark.asyncio
async def test_default_selected_pages_preserve_all_independent_source_classifications(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """More independent inputs than slots drain without full-scope capture repeats."""
    await run_archive_fixture_write(tmp_path, lambda: bootstrap_archive_root(tmp_path))
    kernel = BoundedComputeAdapter(max_workers=8, queue_units=8)
    width = kernel.snapshot().by_class("incremental-background").ceiling_slots
    raws = await run_archive_fixture_write(tmp_path, lambda: acquire_independent_codex_raws(tmp_path, width + 3))
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    owner = RawObservationConvergenceOwner(
        tmp_path,
        compute_adapter=kernel,
        write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
        write_coordinator=coordinator,
    )
    captured: list[tuple[str, ...]] = []
    parsed: list[str] = []
    capture_original = raw_module.RawObservationDerivation.capture_neutral_raws
    parse_original = raw_module._parse_captured_neutral

    def capture(
        adapter: raw_module.RawObservationDerivation,
        raw_ids: Sequence[str],
        *,
        selection: Callable[[PreparedSessionSourceRead], Sequence[str]] | None,
    ) -> raw_module.NeutralRawPreparation | None:
        result = capture_original(adapter, raw_ids, selection=selection)
        if result is not None:
            captured.append(tuple(result.captures))
        return result

    def parse(
        raw_id: str,
        captured: raw_module._CapturedNeutralRaw,
        artifact_key: tuple[object, ...],
        context: raw_module.NeutralPreparationContext,
        scratch: Path,
    ) -> PreparedJsonl:
        parsed.append(raw_id)
        return parse_original(raw_id, captured, artifact_key, context, scratch)

    monkeypatch.setattr(raw_module.RawObservationDerivation, "capture_neutral_raws", capture)
    monkeypatch.setattr(raw_module, "_parse_captured_neutral", parse)
    try:
        (await owner.materialize_retained_raw_ids(raws)).outcome.require_complete()
        width = kernel.snapshot().by_class("incremental-background").ceiling_slots
        assert [len(page) for page in captured] == [width, len(raws) - width]
        assert sorted(parsed) == sorted(raws)
        with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
            assert archive.index_connection is not None
            assert archive.source_connection is not None
            assert {
                row[0] for row in archive.index_connection.execute("SELECT accepted_raw_id FROM raw_revision_heads")
            } == set(raws)
            assert archive.source_connection.execute(
                "SELECT count(*) FROM raw_authority_parser_census WHERE status='complete'"
            ).fetchone()[0] == len(raws)
    finally:
        kernel.shutdown(wait=True)
        await coordinator.shutdown(timeout=1)


@pytest.mark.asyncio
@pytest.mark.uses_real_clock
async def test_selected_preparation_rebinds_changed_selection_without_reparsing_unchanged_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Captured bytes cannot make a removed dependency current or suppress its replacement."""
    await run_archive_fixture_write(tmp_path, lambda: bootstrap_archive_root(tmp_path))
    raws = await run_archive_fixture_write(tmp_path, lambda: acquire_independent_codex_raws(tmp_path, 2))
    kernel = BoundedComputeAdapter(max_workers=8, queue_units=8)
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    owner = RawObservationConvergenceOwner(
        tmp_path,
        compute_adapter=kernel,
        write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
        write_coordinator=coordinator,
    )
    selected = list(raws)
    original = raw_module._parse_captured_neutral
    parsed: list[str] = []
    started, release = threading.Event(), threading.Event()
    lock = threading.Lock()

    def parse(
        raw_id: str,
        captured: raw_module._CapturedNeutralRaw,
        artifact_key: tuple[object, ...],
        context: raw_module.NeutralPreparationContext,
        scratch: Path,
    ) -> PreparedJsonl:
        artifact = original(raw_id, captured, artifact_key, context, scratch)
        with lock:
            parsed.append(raw_id)
            if len(parsed) == 2:
                started.set()
        if raw_id in raws:
            assert release.wait(10)
        return artifact

    monkeypatch.setattr(raw_module, "_parse_captured_neutral", parse)
    task = asyncio.create_task(
        owner.replay_retained_raw_ids((raws[0],), select_retained_raw_ids=lambda _read: tuple(selected))
    )
    try:
        assert await asyncio.to_thread(started.wait, 10)
        replacement = (
            await coordinator.run_sync(
                "test.selected.changed-dependency",
                lambda: acquire_independent_codex_raws(tmp_path, 1, namespace="replacement"),
            )
        )[0]
        selected[1] = replacement
        release.set()
        (await task).require_complete()
        assert parsed.count(raws[0]) == parsed.count(raws[1]) == parsed.count(replacement) == 1
        with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
            assert archive.index_connection is not None
            assert archive.source_connection is not None
            heads = {
                row[0] for row in archive.index_connection.execute("SELECT accepted_raw_id FROM raw_revision_heads")
            }
            assert heads == {raws[0], replacement}
            assert (
                archive.index_connection.execute("SELECT count(*) FROM sessions WHERE raw_id=?", (raws[1],)).fetchone()[
                    0
                ]
                == 0
            )
        assert kernel.snapshot().used_units == 0
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        kernel.shutdown(wait=True)
        await coordinator.shutdown(timeout=1)


@pytest.mark.parametrize("state", ["publisher", "blob_retired", "blob_prepared", "thread_key"])
def test_sealed_read_transfer_refuses_publication_or_projection_state(tmp_path: Path, state: str) -> None:
    """A closed-file borrow cannot silently carry a producer's mutable state."""
    from dataclasses import replace

    from polylogue.storage.blob_publication import ArchiveBlobPublisher

    artifact = PreparedJsonl(None, None, None)
    if state == "publisher":
        artifact = replace(
            artifact, publication_publisher=ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
        )
    elif state == "blob_retired":
        artifact._blob_publication.retired = True
    elif state == "blob_prepared":
        artifact._blob_publication.prepared = True
    else:
        artifact._thread_projection.raw_id = "prepared-owner"
    with pytest.raises(ValueError):
        artifact.read_result()


@pytest.mark.asyncio
@pytest.mark.uses_real_clock
async def test_slow_first_parser_releases_completion_capacity_without_reordering_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The third actual parser starts while the first creator still owns its bytes."""
    await run_archive_fixture_write(tmp_path, lambda: bootstrap_archive_root(tmp_path))
    raws = await run_archive_fixture_write(tmp_path, lambda: acquire_independent_codex_raws(tmp_path, 3))
    kernel = BoundedComputeAdapter(max_workers=3, queue_units=8)
    assert kernel.snapshot().by_class("incremental-background").ceiling_slots == 2
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    owner = RawObservationConvergenceOwner(
        tmp_path,
        compute_adapter=kernel,
        write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
        write_coordinator=coordinator,
    )
    first_started, later_started, release = threading.Event(), threading.Event(), threading.Event()
    captured_pages: list[raw_module.NeutralRawPreparation] = []
    contexts: list[raw_module.NeutralPreparationContext] = []
    original_capture = raw_module.RawObservationDerivation._capture_neutral_jsonl
    original_parse = raw_module._parse_captured_neutral

    def capture(
        adapter: raw_module.RawObservationDerivation,
        key: str,
        *,
        selection: Callable[[PreparedSessionSourceRead], Sequence[str]] | None,
        seal: PreparedIndexMutation,
        carry: raw_module._PreparationCarry,
    ) -> tuple[raw_module.NeutralRawPreparation, tuple[tuple[str, ...], tuple[str, ...]]] | None:
        result = original_capture(adapter, key, selection=selection, seal=seal, carry=carry)
        if result is not None:
            captured_pages.append(result[0])
        return result

    def parse(
        raw_id: str,
        captured: raw_module._CapturedNeutralRaw,
        artifact_key: tuple[object, ...],
        context: raw_module.NeutralPreparationContext,
        scratch: Path,
    ) -> PreparedJsonl:
        contexts.append(context)
        ordered_ids = tuple(captured_pages[0].captures)
        if raw_id == ordered_ids[0]:
            first_started.set()
            assert release.wait(10)
        if raw_id == ordered_ids[2]:
            later_started.set()
        return original_parse(raw_id, captured, artifact_key, context, scratch)

    monkeypatch.setattr(raw_module.RawObservationDerivation, "_capture_neutral_jsonl", capture)
    monkeypatch.setattr(raw_module, "_parse_captured_neutral", parse)
    # Real declared dependency expansion may exceed the offered slot window.
    # Selecting the complete scope creates three jobs on the two-slot page.
    task = asyncio.create_task(owner.replay_retained_raw_ids(raws, select_retained_raw_ids=lambda _read: raws))
    try:
        assert await asyncio.to_thread(first_started.wait, 10)
        assert await asyncio.to_thread(later_started.wait, 10), "FIFO collection withheld released creator capacity"
        page = captured_pages[0]
        assert len({id(context) for context in contexts}) == 1
        assert page.shared_context_bytes == next(iter(page.contexts.values())).schema_bytes
        assert page.unique_backing_bytes == sum(item.descriptor[4] for item in page.captures.values())
        assert not page._backings[next(iter(page._backings))].retired
        for result in page.artifacts.values():
            assert not any(
                hasattr(result, name)
                for name in (
                    "discard",
                    "publish_blobs",
                    "prepare_thread_projection",
                    "apply_thread_projection",
                    "publication_publisher",
                    "custody",
                    "validation_reuse",
                    "prepared_writes",
                )
            )
        release.set()
        (await task).require_complete()
        with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
            assert archive.index_connection is not None
            assert archive.index_connection.execute("SELECT count(*) FROM sessions").fetchone()[0] == 3
            assert {
                row[0] for row in archive.index_connection.execute("SELECT accepted_raw_id FROM raw_revision_heads")
            } == set(raws)
        assert len(contexts) == 3
        assert not page._backings
        assert not page.contexts
        assert page.unique_backing_bytes == page.shared_context_bytes == 0
        assert kernel.snapshot().used_units == kernel.snapshot().used_bytes == 0
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        kernel.shutdown(wait=True)
        await coordinator.shutdown(timeout=1)


@pytest.mark.asyncio
async def test_failed_page_cleanup_retains_its_actual_owner_without_caller_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same ledger retains a failed page and releases it after physical close."""
    await run_archive_fixture_write(tmp_path, lambda: bootstrap_archive_root(tmp_path))
    raws = await run_archive_fixture_write(tmp_path, lambda: acquire_independent_codex_raws(tmp_path, 2))
    kernel = BoundedComputeAdapter(max_workers=4, queue_units=8)
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    owner = RawObservationConvergenceOwner(
        tmp_path,
        compute_adapter=kernel,
        write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
        write_coordinator=coordinator,
    )
    references: list[weakref.ReferenceType[raw_module.NeutralRawPreparation]] = []
    scratch_names: set[str] = set()
    original_capture = raw_module.RawObservationDerivation.capture_neutral_raws
    original_cleanup = raw_module._cleanup_scratch
    allow_cleanup = False

    def capture(
        adapter: raw_module.RawObservationDerivation,
        raw_ids: Sequence[str],
        *,
        selection: Callable[[PreparedSessionSourceRead], Sequence[str]] | None,
    ) -> raw_module.NeutralRawPreparation | None:
        page = original_capture(adapter, raw_ids, selection=selection)
        if page is not None:
            references.append(weakref.ref(page))
            scratch_names.add(page.scratch_owner.name)
        return page

    def cleanup(scratch: TemporaryDirectory[str]) -> None:
        if scratch.name in scratch_names and not allow_cleanup:
            raise OSError("synthetic page directory retirement refusal")
        original_cleanup(scratch)

    monkeypatch.setattr(raw_module.RawObservationDerivation, "capture_neutral_raws", capture)
    monkeypatch.setattr(raw_module, "_cleanup_scratch", cleanup)
    try:
        failed = False
        try:
            await owner.replay_retained_raw_ids(raws)
        except OSError:
            # Discard the exception and its traceback instead of keeping a
            # hidden page reference through pytest's exception-info object.
            failed = True
        assert failed
        gc.collect()
        assert len(references) == 1
        assert references[0]() is not None
        assert kernel.snapshot().used_units == 0
        assert kernel.snapshot().retained_input_count == 1
        with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
            assert archive.count_sessions() == 2
        retained = kernel.retained_shared_inputs()
        assert len(retained) == 1
        page = retained[0].owner
        assert isinstance(page, raw_module.NeutralRawPreparation)
        assert page is references[0]()
        assert page.contexts and Path(page.scratch_owner.name).is_dir()
        assert retained[0].byte_length == page.shared_context_bytes > 0
        allow_cleanup = True
        page.close()
        del page
        gc.collect()
        assert retained[0].owner is None
        assert references[0]() is None
        assert kernel.snapshot().retained_input_count == kernel.snapshot().used_bytes == 0
    finally:
        allow_cleanup = True
        for reservation in kernel.retained_shared_inputs():
            page = reservation.owner
            if isinstance(page, raw_module.NeutralRawPreparation):
                page.close()
        kernel.shutdown(wait=True)
        await coordinator.shutdown(timeout=1)


@pytest.mark.asyncio
@pytest.mark.uses_real_clock
@pytest.mark.parametrize("failed_close,cancelled", [(False, False), (True, False), (True, True)])
async def test_failed_prepared_operation_settles_its_actual_creator_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_close: bool, cancelled: bool
) -> None:
    from polylogue.daemon.write_coordinator import DaemonWriterSettlementError, _TerminalWriter

    await run_archive_fixture_write(tmp_path, lambda: bootstrap_archive_root(tmp_path))
    kernel = BoundedComputeAdapter(max_workers=1)
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    owner = RawObservationConvergenceOwner(
        tmp_path,
        compute_adapter=kernel,
        write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
        write_coordinator=coordinator,
    )
    seals: list[PreparedIndexMutation] = []
    original = ValueError("prepared operation failed before closing its seal")
    creator: list[int] = []
    close_threads: list[int] = []
    attempted_close = threading.Event()
    registered = threading.Event()
    allow_close = threading.Event()
    original_close = PreparedIndexMutation.close

    def close(seal: PreparedIndexMutation) -> None:
        if any(seal is current for current in seals):
            close_threads.append(threading.get_ident())
            if failed_close and not allow_close.is_set():
                attempted_close.set()
                raise OSError("original native close unavailable")
        return original_close(seal)

    monkeypatch.setattr(PreparedIndexMutation, "close", close)
    from polylogue.daemon import write_coordinator as coordinator_module

    observed_cleanup = asyncio.Event()
    original_completed = coordinator._terminal_worker_completed

    def completed(worker: _TerminalWriter) -> None:
        original_completed(worker)
        observed_cleanup.set()

    monkeypatch.setattr(coordinator, "_terminal_worker_completed", completed)
    original_retain = coordinator_module.retain_current_creator_settlement

    def retain(
        request: Callable[[], object], *, owner_count: int, failure_types: tuple[str, ...]
    ) -> Callable[[], None] | None:
        release = original_retain(request, owner_count=owner_count, failure_types=failure_types)
        registered.set()
        return release

    monkeypatch.setattr(coordinator_module, "retain_current_creator_settlement", retain)

    def failed() -> None:
        seal = PreparedIndexMutation.source_only(archive_root=tmp_path)
        seals.append(seal)
        creator.append(threading.get_ident())
        raise original

    pending = asyncio.create_task(owner.run_convergence_sync("test.prepared.failed", failed))
    try:
        if failed_close:
            assert await asyncio.to_thread(registered.wait, 10)
            assert await asyncio.to_thread(attempted_close.wait, 10)
            await observed_cleanup.wait()
            assert not pending.done()
            assert kernel.retained_sql_settlements()
            assert kernel.snapshot().used_units == 1
            if cancelled:
                pending.cancel()
                await asyncio.sleep(0)
                assert not pending.done()
                assert kernel.retained_sql_settlements()
            allow_close.set()
            kernel.retry_sql_settlement()
        expected = BaseExceptionGroup if cancelled else DaemonWriterSettlementError
        with pytest.raises(expected) as caught:
            await pending
        if cancelled:
            assert isinstance(caught.value, BaseExceptionGroup)
            errors: list[BaseException] = []

            def collect(error: BaseException) -> None:
                if isinstance(error, BaseExceptionGroup):
                    for child in error.exceptions:
                        collect(child)
                else:
                    errors.append(error)

            collect(caught.value)
            assert any(
                isinstance(error, DaemonWriterSettlementError) and error.__cause__ is original for error in errors
            )
            assert any(isinstance(error, asyncio.CancelledError) for error in errors)
        else:
            assert caught.value.__cause__ is original
        assert close_threads and all(thread == creator[0] for thread in close_threads)
        assert len(seals) == 1 and seals[0]._closed
        assert kernel.snapshot().used_units == 0
        assert kernel.retained_sql_settlements() == ()
        assert coordinator.snapshot().unsettled_writer_workers == 0
    finally:
        allow_close.set()
        kernel.retry_sql_settlement()
        assert await coordinator.shutdown(timeout=float("inf"))
        try:
            await pending
        except BaseException:
            pass
        kernel.shutdown(wait=True)
