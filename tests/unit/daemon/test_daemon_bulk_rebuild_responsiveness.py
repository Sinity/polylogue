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
* Manual measurement during development, at this exact fixture shape
  (150-raw corpus, batch=8, 3 concurrent small actors): bounded passes held
  the writer for ~0.04-0.34s each and small-actor queued wait had p99
  ~0.32s. Collapsing the SAME corpus into one UNBOUNDED pass (batch=150,
  the whole backlog in a single writer hold -- reproducing a regression
  that removed per-pass batching, e.g. dropping ``RebuildIndexRequest``'s
  paged ``raw_batch_size`` back to "whole backlog") measured a single
  ~1.28s writer hold and pushed small-actor p99 wait to ~1.26s -- a small
  actor queued behind that one giant hold waits for nearly the WHOLE
  drain, not a bounded fraction of it. This test's budget (see
  ``_SMALL_ACTOR_WAIT_BUDGET_SECONDS`` below) sits between those two
  measurements: comfortably above the bounded-pass p99 (headroom against
  host CPU contention -- this repo commonly runs concurrent agent/rebuild
  load) while still well below what the unbounded-pass regression produces
  at this exact fixture size, so a real regression to unbounded passes
  would fail this test, not just a hypothetical one at a different scale.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

import polylogue.daemon.write_coordinator as write_coordinator_module
from polylogue.config import Config
from polylogue.core.enums import Provider
from polylogue.daemon.bulk_rebuild import run_daemon_bulk_rebuild_pass
from polylogue.daemon.parse_prefetch import DaemonParseStage
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteEvent
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

_RAW_COUNT = 150
_BULK_BATCH_SIZE = 8  # forces >= 10 bounded passes over the fixture corpus
_SMALL_ACTOR_COUNT = 3
_SMALL_ACTOR_INTERVAL_SECONDS = 0.02
_MAX_PAYLOAD_BYTES = 10_000_000

# Manually measured at this exact fixture shape (150 raws, batch=8, 3
# concurrent small actors): the correct bounded-pass implementation saw p99
# queued wait ~0.32s (max single writer hold ~0.34s). Collapsing the SAME
# 150-raw corpus into one unbounded pass (batch=150) pushed p99 wait to
# ~1.26s. This budget sits between those two measurements: several times
# the bounded-pass p99 (headroom against host CPU contention -- this repo
# commonly runs concurrent agent/rebuild load) while staying below the
# unbounded-pass regression's measurement at this same fixture size, so
# this is a real (not merely hypothetical) regression detector, not just a
# loose ceiling nothing could ever hit.
_SMALL_ACTOR_WAIT_BUDGET_SECONDS = 1.0


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
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=_codex_session(
                    f"responsiveness-session-{index}",
                    (
                        ("user", f"question {index}"),
                        ("assistant", f"searchable answer {index}" * 10),
                    ),
                ),
                source_path=f"responsiveness-corpus-{index}.jsonl",
                acquired_at_ms=index,
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

    small_actor_waits = sorted(
        event.wait_seconds
        for event in events
        if event.phase == "acquired" and event.actor.startswith("live.append.") and event.wait_seconds is not None
    )
    bulk_pass_events = [
        event for event in events if event.phase == "acquired" and event.actor == "maintenance.bulk_rebuild"
    ]

    # Sanity floor on the scenario itself: a single pass or a handful of
    # small-actor samples would make the p99 assertion below vacuous (no
    # real concurrency to interleave against).
    assert pass_count >= 10, "fixture must force multiple bounded bulk passes to be a meaningful concurrency proof"
    assert len(bulk_pass_events) == pass_count
    assert len(small_actor_waits) >= 10, "small actors must genuinely interleave with the drain, not merely bookend it"

    p99_index = min(len(small_actor_waits) - 1, int(len(small_actor_waits) * 0.99))
    p99_wait = small_actor_waits[p99_index]
    assert p99_wait < _SMALL_ACTOR_WAIT_BUDGET_SECONDS, (
        f"small writer actor p99 queued-wait {p99_wait:.3f}s exceeded the "
        f"{_SMALL_ACTOR_WAIT_BUDGET_SECONDS}s budget while a real bulk-rebuild pass was draining "
        f"({len(small_actor_waits)} samples across {pass_count} bulk passes)"
    )
