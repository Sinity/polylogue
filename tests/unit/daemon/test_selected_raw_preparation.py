"""Selected independent Raw preparations use real shared compute admission."""

from __future__ import annotations

import asyncio
import threading
from builtins import BaseExceptionGroup
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest

from polylogue.core.compute import BoundedComputeAdapter
from polylogue.core.write_lease import coordinator_write_lease_active
from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
from polylogue.sources.prepared_jsonl import PreparedJsonl
from polylogue.sources.sidecar_evidence import RetainedSidecarScope
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
) -> None:
    """The ordinary route overlaps two closed-file parsers, never Source/writers.

    Reverting page preparation to nested/serial compute makes the 8-worker law
    fail; bypassing byte admission makes the oversized-input law fail.
    """
    await run_archive_fixture_write(tmp_path, lambda: bootstrap_archive_root(tmp_path))
    raws = await run_archive_fixture_write(tmp_path, lambda: acquire_independent_codex_raws(tmp_path, 2))
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
    original = raw_module._parse_captured_neutral

    def prepare(
        raw_id: str,
        captured: raw_module._CapturedNeutralRaw,
        artifact_key: tuple[object, ...],
        sidecar_scopes: Mapping[str, RetainedSidecarScope],
        scratch: Path,
    ) -> PreparedJsonl:
        nonlocal active, peak, calls
        assert not coordinator_write_lease_active()
        kernel.require_current_creator()
        assert kernel.snapshot().exclusive_byte_units == 0
        with lock:
            calls += 1
            active += 1
            peak = max(peak, active)
            if active == expected_peak:
                both_started.set()
        try:
            assert release.wait(10), "preparations did not reach the admitted rendezvous"
            return original(raw_id, captured, artifact_key, sidecar_scopes, scratch)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(raw_module, "_parse_captured_neutral", prepare)
    task = asyncio.create_task(owner.replay_retained_raw_ids(raws))
    rendezvous = asyncio.create_task(asyncio.to_thread(both_started.wait, 10))
    try:
        completed, _ = await asyncio.wait((task, rendezvous), return_when=asyncio.FIRST_COMPLETED)
        if task in completed:
            await task
            pytest.fail("ordinary retained replay ended before independent parsers overlapped")
        assert await rendezvous
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
        sidecar_scopes: Mapping[str, RetainedSidecarScope],
        scratch: Path,
    ) -> PreparedJsonl:
        artifact = original_parse(raw_id, captured, artifact_key, sidecar_scopes, scratch)
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
        sidecar_scopes: Mapping[str, RetainedSidecarScope],
        scratch: Path,
    ) -> PreparedJsonl:
        parsed.append(raw_id)
        return parse_original(raw_id, captured, artifact_key, sidecar_scopes, scratch)

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
        sidecar_scopes: Mapping[str, RetainedSidecarScope],
        scratch: Path,
    ) -> PreparedJsonl:
        artifact = original(raw_id, captured, artifact_key, sidecar_scopes, scratch)
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
def test_neutral_file_borrow_refuses_publication_or_projection_state(tmp_path: Path, state: str) -> None:
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
        artifact.borrow_sealed_files()
