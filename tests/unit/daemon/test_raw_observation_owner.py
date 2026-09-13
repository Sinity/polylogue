"""Daemon-owned exact raw admission stays outside the writer until publish."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.daemon.derivation import DerivationFrame, ReplacementLike
from polylogue.daemon.execution import BoundedComputeAdapter
from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner
from polylogue.daemon.write_coordinator import (
    DaemonWriteCoordinator,
    DaemonWriteThreadBridge,
    daemon_write_lease_active,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root


def _admit(root: Path, native_id: str = "owner") -> str:
    payload = [
        {
            "id": native_id,
            "title": native_id,
            "create_time": 1,
            "current_node": "m",
            "mapping": {
                "m": {
                    "id": "m",
                    "parent": None,
                    "children": [],
                    "message": {
                        "id": "m",
                        "author": {"role": "user"},
                        "create_time": 1,
                        "content": {"content_type": "text", "parts": [native_id]},
                    },
                }
            },
        }
    ]
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        return archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=json.dumps(payload).encode(),
            source_path=f"{native_id}.json",
            acquired_at_ms=1,
        )


async def _owner(root: Path) -> tuple[RawObservationConvergenceOwner, BoundedComputeAdapter, DaemonWriteCoordinator]:
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator()
    return (
        RawObservationConvergenceOwner(
            root,
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
            max_payload_bytes=1_000_000,
        ),
        compute,
        coordinator,
    )


async def _shutdown(compute: BoundedComputeAdapter, coordinator: DaemonWriteCoordinator) -> None:
    compute.shutdown(wait=True)
    await coordinator.shutdown(timeout=1.0)


@pytest.mark.asyncio
async def test_exact_raw_admission_uses_canonical_derivation_not_legacy_authority(tmp_path: Path) -> None:
    """The owner materializes an exact raw through the canonical derivation."""
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(tmp_path)
    owner, compute, coordinator = await _owner(tmp_path)
    try:
        report = await owner.converge_raw_id(raw_id)
        assert report.done == 1
        with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
            assert archive.raw_payload_sizes((raw_id,))[raw_id] > 0
    finally:
        await _shutdown(compute, coordinator)


@pytest.mark.asyncio
async def test_widened_owner_refuses_nonstream_before_blob_open_but_ordinary_owner_can_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(tmp_path, "nonstream")
    owner, compute, coordinator = await _owner(tmp_path)

    def forbidden_verify(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("widened nonstream refusal must precede blob verification")

    try:
        with monkeypatch.context() as widened:
            widened.setattr(BlobStore, "verify", forbidden_verify)
            for _ in range(2):
                refused = await owner.converge_raw_id(raw_id, max_payload_bytes=2_000_000)
                assert refused.done == 0 and refused.failed == 1
                assert any(outcome.error and "stream-safe" in outcome.error for outcome in refused.outcomes)
        ordinary = await owner.converge_raw_id(raw_id)
        assert ordinary.done == 1 and ordinary.failed == ordinary.pending == 0
    finally:
        await _shutdown(compute, coordinator)


@pytest.mark.asyncio
async def test_owner_refuses_a_preheld_writer_lease_before_preparation(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(tmp_path, "preheld")
    owner, compute, coordinator = await _owner(tmp_path)

    async def nested() -> None:
        with pytest.raises(RuntimeError, match="writer lease is released"):
            await owner.converge_raw_id(raw_id)

    try:
        await coordinator.run("raw-observation-test", nested)
        report = await owner.converge_raw_id(raw_id)
        assert report.done == 1
    finally:
        await _shutdown(compute, coordinator)


@pytest.mark.asyncio
async def test_paused_raw_preparation_does_not_hold_writer_for_unrelated_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compute is lease-free; only the final publication crosses the bridge."""
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(tmp_path, "paused")
    owner, compute, coordinator = await _owner(tmp_path)
    adapter = owner._converger._derivation_adapter("raw_observation")
    original_compute = adapter.compute
    started = threading.Event()
    release = threading.Event()

    def paused_compute(frame: DerivationFrame, key: str) -> ReplacementLike:
        assert not daemon_write_lease_active()
        started.set()
        assert release.wait(timeout=2.0)
        return original_compute(frame, key)

    monkeypatch.setattr(adapter, "compute", paused_compute)
    task = asyncio.create_task(owner.converge_raw_id(raw_id))
    try:
        await asyncio.wait_for(asyncio.to_thread(started.wait), timeout=1.0)
        assert (
            await asyncio.wait_for(coordinator.run_sync("unrelated.writer", lambda: "published"), timeout=1.0)
            == "published"
        )
        release.set()
        assert (await asyncio.wait_for(task, timeout=3.0)).done == 1
    finally:
        release.set()
        if not task.done():
            await task
        await _shutdown(compute, coordinator)
