"""Tests for the daemon-owned parse-stage warmer (polylogue-m6tp phase (a)).

Production dependencies exercised here:

* ``DaemonParseStage.warm`` -- the actual off-writer-hold pre-parse entry
  point the daemon conveyor calls.
* ``polylogue.storage.repair.raw_materialization_pending_census_raw_ids`` /
  ``raw_materialization_readonly_descriptors`` -- the read-only candidate
  and descriptor lookups ``warm`` uses.
* ``polylogue.sources.revision_backfill.census_parse_worker`` -- the same
  pure parse function the production census path dispatches; these tests
  prove it is genuinely reached via a background thread, not called inline.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from polylogue.config import Config
from polylogue.core.enums import Provider
from polylogue.daemon.parse_prefetch import DaemonParseStage
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def _codex_payload(native_id: str, text: str) -> bytes:
    return (
        b'{"type":"session_meta","payload":{"id":"' + native_id.encode() + b'"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"'
        + native_id.encode()
        + b'-m0","role":"user","content":[{"type":"input_text","text":"'
        + text.encode()
        + b'"}]}}\n'
    )


def _config(root: Path) -> Config:
    return Config(archive_root=root, render_root=root / "render", sources=[])


def _seed_raws(tmp_path: Path, payloads: dict[str, bytes]) -> None:
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        for index, (source_path, payload) in enumerate(payloads.items()):
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path=source_path,
                acquired_at_ms=index,
            )


def test_warm_parses_pending_candidates_off_writer_hold(tmp_path: Path) -> None:
    """Two never-censused raws are both discovered read-only and parsed."""
    _seed_raws(
        tmp_path,
        {
            "a.jsonl": _codex_payload("session-a", "hello from a"),
            "b.jsonl": _codex_payload("session-b", "hello from b"),
        },
    )

    stage = DaemonParseStage(max_workers=2, max_inflight_bytes=10_000_000)
    try:
        warmed = stage.warm(_config(tmp_path), limit=10, max_payload_bytes=10_000_000)
    finally:
        stage.shutdown()

    assert warmed == 2
    assert len(stage.cache) == 2

    with sqlite3.connect(f"file:{tmp_path / 'source.db'}?mode=ro", uri=True) as conn:
        rows = list(conn.execute("SELECT raw_id, source_path FROM raw_sessions ORDER BY raw_id"))
    raw_ids_by_path = {str(row[1]): str(row[0]) for row in rows}

    for source_path in ("a.jsonl", "b.jsonl"):
        raw_id = raw_ids_by_path[source_path]
        assert stage.cache.contains(raw_id)
        sessions, payload_bytes, _kind = stage.cache.pop(raw_id)  # type: ignore[misc]
        assert len(sessions) == 1
        assert payload_bytes > 0


def test_warm_dispatches_to_worker_threads_not_the_caller_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: the parse must genuinely run on the pool's own thread.

    If the extraction regressed to calling the parser inline on the calling
    thread (defeating the entire point of moving parse off the writer
    hold), the recorded thread name would equal the caller's thread name
    and this assertion would fail.
    """
    _seed_raws(tmp_path, {"a.jsonl": _codex_payload("session-a", "hello")})

    from polylogue.sources import revision_backfill

    observed_thread_names: list[str] = []
    real_worker = revision_backfill.census_parse_worker

    def spying_worker(*args: object, **kwargs: object) -> object:
        observed_thread_names.append(threading.current_thread().name)
        return real_worker(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(revision_backfill, "census_parse_worker", spying_worker)

    stage = DaemonParseStage(max_workers=2, max_inflight_bytes=10_000_000)
    try:
        warmed = stage.warm(_config(tmp_path), limit=10, max_payload_bytes=10_000_000)
    finally:
        stage.shutdown()

    assert warmed == 1
    assert len(observed_thread_names) == 1
    assert observed_thread_names[0] != threading.current_thread().name
    assert observed_thread_names[0].startswith("polylogue-parse-stage")


def test_warm_enforces_inflight_bytes_budget(tmp_path: Path) -> None:
    """A tiny budget admits only what fits; the rest is simply left uncached
    (the writer-held pass will parse it normally -- never an error)."""
    _seed_raws(
        tmp_path,
        {
            "a.jsonl": _codex_payload("session-a", "x" * 500),
            "b.jsonl": _codex_payload("session-b", "y" * 500),
        },
    )

    stage = DaemonParseStage(max_workers=2, max_inflight_bytes=1)
    try:
        warmed = stage.warm(_config(tmp_path), limit=10, max_payload_bytes=10_000_000)
    finally:
        stage.shutdown()

    # Neither payload fits a 1-byte budget.
    assert warmed == 0
    assert len(stage.cache) == 0


def test_warm_skips_raws_already_present_in_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second warm pass over the same still-uncensused backlog must not
    redispatch a raw the cache already holds."""
    _seed_raws(tmp_path, {"a.jsonl": _codex_payload("session-a", "hello")})

    from polylogue.sources import revision_backfill

    dispatch_count = 0
    real_worker = revision_backfill.census_parse_worker

    def counting_worker(*args: object, **kwargs: object) -> object:
        nonlocal dispatch_count
        dispatch_count += 1
        return real_worker(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(revision_backfill, "census_parse_worker", counting_worker)

    stage = DaemonParseStage(max_workers=2, max_inflight_bytes=10_000_000)
    try:
        first = stage.warm(_config(tmp_path), limit=10, max_payload_bytes=10_000_000)
        second = stage.warm(_config(tmp_path), limit=10, max_payload_bytes=10_000_000)
    finally:
        stage.shutdown()

    assert first == 1
    assert second == 0
    assert dispatch_count == 1


def test_warm_returns_zero_when_no_candidates_pending(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)

    stage = DaemonParseStage(max_workers=2, max_inflight_bytes=10_000_000)
    try:
        warmed = stage.warm(_config(tmp_path), limit=10, max_payload_bytes=10_000_000)
    finally:
        stage.shutdown()

    assert warmed == 0
    assert len(stage.cache) == 0


def test_warm_times_out_on_a_hung_worker_and_leaves_it_uncached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CodeRabbit (PR #3168): a worker that never returns (e.g. an
    unresponsive filesystem read) must not block ``warm()`` forever -- since
    ``warm()`` is awaited directly ahead of ``run_sync`` in the periodic
    conveyor loop, an unbounded wait here would stall every subsequent drain
    pass indefinitely. The hung raw is simply left uncached; a real
    writer-held pass would reparse it normally, identical to any other
    prefetch miss."""
    _seed_raws(tmp_path, {"a.jsonl": _codex_payload("session-a", "hello")})

    from polylogue.sources import revision_backfill

    dispatched = threading.Event()

    def hanging_worker(*args: object, **kwargs: object) -> object:
        dispatched.set()
        time.sleep(0.3)  # far longer than the test's tiny warm timeout below
        pytest.fail("hung worker must not be awaited past the warm() timeout")

    monkeypatch.setattr(revision_backfill, "census_parse_worker", hanging_worker)

    stage = DaemonParseStage(max_workers=1, max_inflight_bytes=10_000_000, warm_timeout_seconds=0.02)
    try:
        started = time.monotonic()
        warmed = stage.warm(_config(tmp_path), limit=10, max_payload_bytes=10_000_000)
        elapsed = time.monotonic() - started
    finally:
        stage.shutdown()

    assert dispatched.wait(timeout=1.0)
    assert warmed == 0
    assert len(stage.cache) == 0
    # warm() returned close to its own timeout, not after the hung worker's
    # sleep -- proving the wait is genuinely bounded, not merely reordered.
    assert elapsed < 0.3


def test_max_inflight_bytes_default_is_adaptive_and_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whale-memory budget scales with physical RAM, clamped to [64MiB, 2GiB].

    The old fixed 64 MiB default starved bulk-scale warm on whale corpora
    (measured live: 0.37 raws/s stalled on cache admission vs 56.7 raws/s
    with an adequate budget) — the budget must grow on capable machines
    while keeping the 64 MiB floor semantics on small ones.
    """
    from polylogue.daemon import parse_prefetch as pp

    monkeypatch.delenv("POLYLOGUE_DAEMON_PARSE_STAGE_MAX_INFLIGHT_BYTES", raising=False)

    # 32 GiB machine -> hits the 2 GiB ceiling (32 GiB / 16 = 2 GiB).
    monkeypatch.setattr(pp, "_physical_memory_bytes", lambda: 32 * 1024**3)
    assert pp.daemon_parse_stage_max_inflight_bytes() == 2 * 1024**3

    # 512 MiB machine -> clamped up to the 64 MiB floor.
    monkeypatch.setattr(pp, "_physical_memory_bytes", lambda: 512 * 1024**2)
    assert pp.daemon_parse_stage_max_inflight_bytes() == 64 * 1024**2

    # 8 GiB machine -> proportional (8 GiB / 16 = 512 MiB).
    monkeypatch.setattr(pp, "_physical_memory_bytes", lambda: 8 * 1024**3)
    assert pp.daemon_parse_stage_max_inflight_bytes() == 512 * 1024**2

    # Unknown physical memory -> conservative floor.
    monkeypatch.setattr(pp, "_physical_memory_bytes", lambda: None)
    assert pp.daemon_parse_stage_max_inflight_bytes() == 64 * 1024**2

    # Explicit env override always wins.
    monkeypatch.setenv("POLYLOGUE_DAEMON_PARSE_STAGE_MAX_INFLIGHT_BYTES", "123456789")
    assert pp.daemon_parse_stage_max_inflight_bytes() == 123456789
