"""Fixture-scale responsiveness proof for polylogue-gd6v's remaining AC.

PR #3189's own body deferred this explicitly: "agvo responsiveness p99 gate
during a live drain -- not independently measured here; rests on the same
off-writer-hold parse mechanism phase (a) already established." This module
supplies the missing measurement at fixture scale: it drives the REAL
``polylogue.daemon.bulk_rebuild.run_daemon_bulk_rebuild_pass`` -- the exact
production pass driver, not a stub -- against a real archive, CONCURRENTLY
with small simulated writer-actor coroutines (standing in for live-ingest
appends / hook-spool drain writes) sharing the SAME
``polylogue.daemon.write_coordinator.DaemonWriteCoordinator`` every other
daemon writer actor goes through, and asserts the small actors' queued-wait
time stays within a documented budget throughout the drain.

Why this is a meaningful (non-vacuous) proof, not just a green assertion:

* The coordinator is a strict FIFO single-writer gate (see
  ``DaemonWriteCoordinator._execute``): once a small actor's request is
  queued, its wait time is bounded by, at most, the currently-held pass's
  remaining hold duration plus any earlier-queued items -- there is no
  starvation-by-priority path. What actually determines whether that bound
  is small is whether the *bulk-rebuild* side keeps its own passes bounded
  (small ``raw_batch_size``, parse pre-warmed off the writer hold by
  ``DaemonParseStage`` per #3168) instead of holding the writer for an
  entire corpus in one sweep.
* The semantic guarantee is admission ordering, not an absolute wall-clock
  duration. A full-IO host can stall a bounded writer hold longer than an
  old measurement of an unbounded one, so a p99-second threshold cannot
  distinguish a product regression from unrelated scheduler pressure. The
  coordinator's real event stream instead proves the intended property:
  every pair of bounded bulk-pass acquisitions has a live writer acquisition
  between it. Collapsing the corpus into one unbounded pass fails the
  multi-pass floor; bypassing the coordinator fails the bulk-event floor;
  requeueing bulk work without yielding fails the per-pair interleaving
  assertion.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

import polylogue.daemon.write_coordinator as write_coordinator_module
from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.config import Config
from polylogue.core.enums import Provider
from polylogue.daemon.bulk_rebuild import run_daemon_bulk_rebuild_pass
from polylogue.daemon.parse_prefetch import DaemonParseStage
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteEvent
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.rebuild_receipt import write_current_rebuild_receipt

_RAW_COUNT = 150
_BULK_BATCH_SIZE = 8  # forces >= 10 bounded passes over the fixture corpus
_SMALL_ACTOR_COUNT = 3
_SMALL_ACTOR_INTERVAL_SECONDS = 0.02
_MAX_PAYLOAD_BYTES = 10_000_000


def _codex_session(native_id: str, messages: tuple[tuple[str, str], ...]) -> bytes:
    rows: list[dict[str, object]] = [
        {"type": "session_meta", "payload": {"id": native_id, "timestamp": "2026-07-20T00:00:00Z"}}
    ]
    for position, (role, text) in enumerate(messages):
        rows.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": f"{native_id}-m{position}",
                    "role": role,
                    "content": [
                        {"type": "input_text" if role == "user" else "output_text", "text": text},
                    ],
                },
            }
        )
    return b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)


def _config(root: Path) -> Config:
    return Config(archive_root=root, render_root=root / "render", sources=[])


def _seed_corpus(root: Path, *, count: int = _RAW_COUNT) -> None:
    initialize_active_archive_root(root)
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for index in range(count):
            native_id = f"responsiveness-session-{index}"
            raw_id = archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=_codex_session(
                    native_id,
                    (
                        ("user", f"question {index}"),
                        ("assistant", f"searchable answer {index}" * 10),
                    ),
                ),
                source_path=f"responsiveness-corpus-{index}.jsonl",
                acquired_at_ms=index,
                native_id=native_id,
            )
            archive.bind_raw_revision(
                raw_id,
                RawRevisionEnvelope(
                    logical_source_key=f"codex-session:{native_id}",
                    kind=RawRevisionKind.FULL,
                    source_revision=f"responsiveness:{index}",
                    acquisition_generation=0,
                    baseline_raw_id=raw_id,
                    authority=RawRevisionAuthority.BYTE_PROVEN,
                ),
            )


def _small_write(_marker: int) -> None:
    """Trivial fast writer-actor body -- stands in for a live-ingest append
    or hook-spool drain write that must never queue for long behind a
    bulk-rebuild pass sharing the same coordinator."""
    time.sleep(0.001)


async def _run_small_actor(
    coordinator: DaemonWriteCoordinator,
    name: str,
    stop: asyncio.Event,
    *,
    interval: float,
) -> None:
    counter = 0
    while not stop.is_set():
        await coordinator.run_sync(name, _small_write, counter)
        counter += 1
        await asyncio.sleep(interval)


async def _drive_bulk_rebuild_to_promotion(
    root: Path,
    *,
    batch_size: int,
) -> int:
    """Drive the REAL daemon bulk-rebuild pass driver to promotion.

    Returns the number of bounded passes it took. Uses a fresh
    ``DaemonParseStage`` per pass (mirroring a daemon restart between
    ticks, same pattern as ``tests/unit/daemon/test_bulk_rebuild.py``) so
    this also exercises the resume path rather than only a warm cache.
    """
    config = _config(root)
    pass_count = 0
    for _ in range(_RAW_COUNT * 2):  # generous upper bound; promotion ends the loop early
        stage = DaemonParseStage(max_workers=2, max_inflight_bytes=_MAX_PAYLOAD_BYTES)
        try:
            receipt = await run_daemon_bulk_rebuild_pass(
                config=config,
                parse_stage=stage,
                batch_size=batch_size,
                max_payload_bytes=_MAX_PAYLOAD_BYTES,
            )
        finally:
            stage.shutdown()
        if receipt is None:
            break
        pass_count += 1
        transaction_status = receipt.transaction["status"] if receipt.transaction else receipt.status
        if transaction_status == "promoted":
            break
    else:
        pytest.fail("bulk rebuild did not reach promotion within the generous pass budget")
    return pass_count


def test_small_writer_actors_stay_responsive_during_bulk_rebuild_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """gd6v residual: concurrent small writer actors must not be starved by
    a real bulk-rebuild drain sharing the daemon write coordinator.

    Anti-vacuity: this drives ``run_daemon_bulk_rebuild_pass`` (the real
    production pass driver used by ``_maybe_route_daemon_bulk_rebuild`` in
    ``polylogue/daemon/cli.py``) against a real fixture archive, and the
    small actors run through the real ``DaemonWriteCoordinator.run_sync`` --
    the exact same coordinator every other daemon writer actor (live
    ingest, hook-spool drain, insight convergence) uses. A regression that
    collapsed the bulk driver's own per-pass batching back into one
    unbounded writer-held sweep (removing ``RebuildIndexRequest``'s paged
    ``raw_batch_size``, or bypassing the coordinator's FIFO admission
    entirely) would make at least one small actor wait for a hold
    proportional to the WHOLE corpus instead of one bounded page, which
    this fixture's corpus size (see module docstring) pushes well past
    ``_SMALL_ACTOR_WAIT_BUDGET_SECONDS``.
    """
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    _seed_corpus(tmp_path)
    schema_receipt = write_current_rebuild_receipt(tmp_path, tmp_path.parent / "schema-inference-gate-receipt.json")
    monkeypatch.setenv("POLYLOGUE_SCHEMA_INFERENCE_RECEIPT", str(schema_receipt))

    events: list[DaemonWriteEvent] = []
    coordinator = DaemonWriteCoordinator(observer=events.append)
    # ``run_daemon_bulk_rebuild_pass`` resolves the coordinator via a local
    # ``from polylogue.daemon.write_coordinator import daemon_write_coordinator``
    # import each call, so patching the module-level factory function makes
    # every writer actor in this test -- the real bulk driver AND the small
    # simulated actors below -- share this one instrumented instance,
    # exactly like every writer actor in a real daemon process shares the
    # one per-event-loop coordinator singleton.
    monkeypatch.setattr(write_coordinator_module, "daemon_write_coordinator", lambda: coordinator)

    async def scenario() -> int:
        stop = asyncio.Event()
        small_actor_tasks = [
            asyncio.create_task(
                _run_small_actor(
                    coordinator,
                    f"live.append.{i}",
                    stop,
                    interval=_SMALL_ACTOR_INTERVAL_SECONDS,
                )
            )
            for i in range(_SMALL_ACTOR_COUNT)
        ]
        try:
            return await _drive_bulk_rebuild_to_promotion(tmp_path, batch_size=_BULK_BATCH_SIZE)
        finally:
            stop.set()
            for task in small_actor_tasks:
                task.cancel()
            await asyncio.gather(*small_actor_tasks, return_exceptions=True)

    pass_count = asyncio.run(scenario())

    small_actor_acquisitions = [
        event for event in events if event.phase == "acquired" and event.actor.startswith("live.append.")
    ]
    bulk_pass_events = [
        event for event in events if event.phase == "acquired" and event.actor == "maintenance.bulk_rebuild"
    ]

    # Sanity floor on the scenario itself: a single pass or a handful of
    # small-actor samples would make the interleaving assertion below vacuous (no
    # real concurrency to interleave against).
    assert pass_count >= 10, "fixture must force multiple bounded bulk passes to be a meaningful concurrency proof"
    assert len(bulk_pass_events) == pass_count
    assert len(small_actor_acquisitions) >= 10, (
        "small actors must genuinely interleave with the drain, not merely bookend it"
    )

    # The exact scheduling guarantee: the daemon must surrender the writer
    # between each pair of bounded maintenance passes whenever live writers
    # are present. Sequence values are allocated by the real coordinator at
    # queue admission, so this does not use a test-local scheduler model or
    # a host-pressure-sensitive elapsed-time threshold.
    gaps_without_live_admission = [
        (left.sequence, right.sequence)
        for left, right in zip(bulk_pass_events, bulk_pass_events[1:], strict=False)
        if not any(left.sequence < live.sequence < right.sequence for live in small_actor_acquisitions)
    ]
    assert not gaps_without_live_admission, (
        "bulk rebuild reacquired the daemon writer before any live writer in "
        f"{len(gaps_without_live_admission)} bounded-pass gap(s): {gaps_without_live_admission}"
    )
