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
async def test_retained_jsonl_converges_from_sealed_carrier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A retained JSONL raw publishes through its prepared carrier.

    The merge spy fails if this route reconstructs a whole session inline.
    """
    bootstrap_archive_root(tmp_path)
    payload = (
        b'{"type":"session_meta","payload":{"id":"above-cache-budget"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"m1",'
        b'"role":"user","content":[{"type":"input_text","text":"hello"}]}}\n'
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path="above-cache-budget.jsonl",
            acquired_at_ms=1,
        )
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator()
    owner = RawObservationConvergenceOwner(
        tmp_path,
        compute_adapter=compute,
        write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
    )

    def no_singleton_merge(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("single-raw replay reconstructed a whole session")

    monkeypatch.setattr("polylogue.sources.dispatch.merge_parsed_session_chunks", no_singleton_merge)
    try:
        report = await owner.converge_raw_id(raw_id)
        assert report.done == 1 and report.failed == report.pending == 0
        with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
            assert archive.index_connection is not None
            assert (
                archive.index_connection.execute("SELECT accepted_raw_id FROM raw_revision_heads").fetchone()[0]
                == raw_id
            )
            assert archive.source_connection is not None
            assert (
                archive.source_connection.execute(
                    "SELECT status FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,)
                ).fetchone()[0]
                == "complete"
            )
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
