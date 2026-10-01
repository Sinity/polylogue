"""Daemon-owned exact raw admission stays outside the writer until publish."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.core.write_lease import coordinator_write_lease_active
from polylogue.daemon.derivation import DerivationFrame, ReplacementLike
from polylogue.daemon.execution import BoundedComputeAdapter
from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner
from polylogue.daemon.write_coordinator import (
    DaemonWriteCoordinator,
    DaemonWriteThreadBridge,
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
    coordinator = DaemonWriteCoordinator(archive_root=root)
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
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
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
        assert not coordinator_write_lease_active()
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
        # The property is the unrelated writer above publishing within 1s
        # while compute is paused. Completion only has to happen: since #5565
        # retained preparation runs in a spawned worker process, whose start
        # and replay take several seconds on a loaded host, so this bound is
        # a hang guard, not a latency budget.
        assert (await asyncio.wait_for(task, timeout=60.0)).done == 1
    finally:
        release.set()
        if not task.done():
            await task
        await _shutdown(compute, coordinator)


@pytest.mark.asyncio
@pytest.mark.parametrize("envelope_byte_classified", [False, True])
async def test_multi_session_claude_code_raw_settles_every_session(
    tmp_path: Path, envelope_byte_classified: bool
) -> None:
    """One retained Claude Code file carrying two sessionIds settles, not retries.

    Anti-vacuity: requiring the pending-raw tip artifact to yield exactly one
    session raises a retryable preparation error, leaving the raw pending.
    ``envelope_byte_classified`` starts from the state an earlier derivation
    left behind (the pending envelope byte-proven with itself as baseline);
    checking the member applications against that envelope's chain columns
    reports the settled raw stale forever.
    """
    bootstrap_archive_root(tmp_path)
    records = [
        {
            "type": "user",
            "uuid": f"{session}-u1",
            "sessionId": session,
            "timestamp": f"2026-07-01T10:00:0{index}Z",
            "message": {"role": "user", "content": f"hello from {session}"},
        }
        for index, session in enumerate(("multi-alpha", "multi-beta"))
    ]
    payload = b"".join(json.dumps(record).encode() + b"\n" for record in records)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=payload,
            source_path="multi-session.jsonl",
            acquired_at_ms=1,
            post_parse=True,
        )
        if envelope_byte_classified:
            (pending_key,) = archive.pending_raw_revision_logical_keys()
            assert archive.classify_raw_revision_cohort_for_rebuild_repair(pending_key).accepted_raw_ids == (raw_id,)
    owner, compute, coordinator = await _owner(tmp_path)
    try:
        # A multi-session raw first publishes its parser census (pending,
        # binding moved), then settles; fair intake repeats exactly this call.
        reports = []
        for _attempt in range(4):
            report = await owner.converge_raw_id(raw_id)
            reports.append(report)
            assert report.failed == 0, report.outcomes
            if report.done:
                break
        assert reports[-1].done == 1, [item.outcomes for item in reports]
        settled = await owner.converge_raw_id(raw_id)
        assert settled.failed == settled.pending == 0, settled.outcomes
        with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
            assert archive.index_connection is not None
            assert sorted(
                str(row[0])
                for row in archive.index_connection.execute(
                    "SELECT session_id FROM sessions WHERE raw_id = ?", (raw_id,)
                )
            ) == ["claude-code-session:multi-alpha", "claude-code-session:multi-beta"]
            assert archive.source_connection is not None
            assert (
                archive.source_connection.execute(
                    "SELECT status FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,)
                ).fetchone()[0]
                == "complete"
            )
            # Each session is governed under its own identity, never the raw's
            # pending byte envelope.
            assert sorted(
                str(row[0])
                for row in archive.source_connection.execute(
                    "SELECT logical_source_key FROM raw_session_memberships WHERE raw_id = ?", (raw_id,)
                )
            ) == ["claude-code-session:multi-alpha", "claude-code-session:multi-beta"]
            assert (
                archive.source_connection.execute(
                    "SELECT parsed_at_ms FROM raw_sessions WHERE raw_id = ?", (raw_id,)
                ).fetchone()[0]
                is not None
            )
    finally:
        await _shutdown(compute, coordinator)


def _claude_code_payload(*sessions: str) -> bytes:
    return b"".join(
        json.dumps(
            {
                "type": "user",
                "uuid": f"{session}-u1",
                "sessionId": session,
                "timestamp": f"2026-07-01T10:00:0{index}Z",
                "message": {"role": "user", "content": f"hello from {session}"},
            }
        ).encode()
        + b"\n"
        for index, session in enumerate(sessions)
    )


async def _settle(owner: RawObservationConvergenceOwner, raw_id: str) -> None:
    reports = []
    for _attempt in range(4):
        report = await owner.converge_raw_id(raw_id)
        reports.append(report)
        assert report.failed == 0, report.outcomes
        if report.done:
            break
    assert reports[-1].done == 1, [item.outcomes for item in reports]


@pytest.mark.asyncio
@pytest.mark.parametrize("second_path", ["overlap.jsonl", "resumed.jsonl"])
async def test_multi_session_raw_overlapping_a_byte_chain_decides_every_member(
    tmp_path: Path, second_path: str
) -> None:
    """A later two-session file repeating a session another raw already governs.

    ``overlap.jsonl`` recaptures the first file's path; ``resumed.jsonl`` is a
    different file carrying the earlier session, as a resumed transcript does.
    Anti-vacuity: skipping membership replay for a key that byte replay already
    wrote leaves the later raw's member undecided (reported missing on every
    retry); a membership yield to a chain-governed head that the receipt
    validator cannot certify reports the settled raw stale forever.
    """
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        first = archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=_claude_code_payload("overlap-alpha"),
            source_path="overlap.jsonl",
            acquired_at_ms=1,
            post_parse=True,
        )
    owner, compute, coordinator = await _owner(tmp_path)
    try:
        await _settle(owner, first)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            second = archive.write_raw_payload(
                provider=Provider.CLAUDE_CODE,
                payload=_claude_code_payload("overlap-alpha", "overlap-beta"),
                source_path=second_path,
                acquired_at_ms=2,
                post_parse=True,
            )
        await _settle(owner, second)
        for raw_id in (first, second):
            settled = await owner.converge_raw_id(raw_id)
            assert settled.failed == settled.pending == 0, settled.outcomes
        with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
            assert archive.index_connection is not None
            assert archive.source_connection.execute(
                "SELECT logical_source_key, decision IS NOT NULL FROM raw_session_memberships "
                "WHERE raw_id = ? ORDER BY logical_source_key",
                (second,),
            ).fetchall() == [("claude-code-session:overlap-alpha", 1), ("claude-code-session:overlap-beta", 1)]
            assert {str(row[0]) for row in archive.index_connection.execute("SELECT session_id FROM sessions")} == {
                "claude-code-session:overlap-alpha",
                "claude-code-session:overlap-beta",
            }
    finally:
        await _shutdown(compute, coordinator)


@pytest.mark.asyncio
async def test_raw_parse_reserves_its_retained_payload_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The raw parse is admitted with its payload size, not as a zero-byte task.

    Anti-vacuity: submitting without ``estimated_bytes`` records 0 here.
    """
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(tmp_path)
    owner, compute, coordinator = await _owner(tmp_path)
    reserved: list[int] = []
    real_submit = compute.submit

    def recording_submit(function: object, **kwargs: object) -> object:
        reserved.append(int(kwargs.get("estimated_bytes", 0)))  # type: ignore[call-overload]
        return real_submit(function, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(compute, "submit", recording_submit)
    try:
        await owner.converge_raw_id(raw_id)
        with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
            size = archive.raw_payload_sizes((raw_id,))[raw_id]
        assert size > 0
        assert reserved == [size]
    finally:
        await _shutdown(compute, coordinator)
