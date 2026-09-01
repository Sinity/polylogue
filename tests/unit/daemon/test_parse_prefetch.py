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

import asyncio
import sqlite3
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.config import Config
from polylogue.core.enums import BlockType, Provider
from polylogue.daemon.parse_prefetch import DaemonParseStage, estimate_parsed_tree_bytes
from polylogue.sources.parsers.base_models import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

pytestmark = pytest.mark.uses_real_clock(
    "DaemonParseStage.warm() timeout test measures real elapsed wall-clock against a genuinely hung worker thread to prove the wait is bounded, not merely reordered; frozen_clock cannot substitute for a real ThreadPoolExecutor future's wait timeout."
)


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


def test_warm_raw_ids_enforces_aggregate_payload_budget(tmp_path: Path) -> None:
    """One warm operation never retains every raw merely because each fits."""
    payloads = {
        "a.jsonl": _codex_payload("session-a", "same-sized-a"),
        "b.jsonl": _codex_payload("session-b", "same-sized-b"),
        "c.jsonl": _codex_payload("session-c", "same-sized-c"),
    }
    _seed_raws(tmp_path, payloads)
    with sqlite3.connect(tmp_path / "source.db") as connection:
        raw_ids = [
            str(row[0]) for row in connection.execute("SELECT raw_id FROM raw_sessions ORDER BY source_path").fetchall()
        ]

    stage = DaemonParseStage(max_workers=3, max_inflight_bytes=10_000_000)
    try:
        warmed = stage.warm_raw_ids(
            _config(tmp_path),
            raw_ids=raw_ids,
            max_payload_bytes=max(len(payload) for payload in payloads.values()),
        )
    finally:
        stage.shutdown()

    assert warmed == 1
    assert len(stage.cache) == 1


def test_warm_candidate_scope_excludes_ordinary_queue_members(tmp_path: Path) -> None:
    """Whale warming discovers only the selected authority component.

    This drives the production ``DaemonParseStage.warm`` route with two
    independent raw rows.  A global preview would admit both rows, while the
    candidate-specific scope used by the whale daemon route admits only the
    selected seed and leaves the ordinary-pass row for its own conveyor.
    """
    _seed_raws(
        tmp_path,
        {
            "whale-seed.jsonl": _codex_payload("whale-seed", "warm whale"),
            "ordinary.jsonl": _codex_payload("ordinary", "ordinary queue"),
        },
    )
    with sqlite3.connect(f"file:{tmp_path / 'source.db'}?mode=ro", uri=True) as conn:
        rows = dict(conn.execute("SELECT source_path, raw_id FROM raw_sessions"))
    seed = str(rows["whale-seed.jsonl"])
    ordinary = str(rows["ordinary.jsonl"])

    stage = DaemonParseStage(max_workers=1, max_inflight_bytes=10_000_000)
    try:
        warmed = stage.warm(
            _config(tmp_path),
            limit=10,
            max_payload_bytes=10_000_000,
            raw_artifact_id=seed,
        )
        assert warmed == 1
        assert stage.cache.contains(seed)
        assert not stage.cache.contains(ordinary)
    finally:
        stage.shutdown()


def test_writer_admission_rejects_until_cancelled_parse_worker_drains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled caller cannot make an active parse worker writer-safe."""
    import threading

    _seed_raws(tmp_path, {"blocked.jsonl": _codex_payload("blocked", "drain me")})
    started = threading.Event()
    release = threading.Event()
    real_worker = __import__(
        "polylogue.sources.revision_backfill", fromlist=["census_parse_worker"]
    ).census_parse_worker

    def blocked_worker(*args: object, **kwargs: object) -> object:
        started.set()
        release.wait(timeout=5)
        return real_worker(*args, **kwargs)

    monkeypatch.setattr("polylogue.sources.revision_backfill.census_parse_worker", blocked_worker)
    stage = DaemonParseStage(max_workers=1, max_inflight_bytes=10_000_000)
    thread = threading.Thread(
        target=lambda: stage.warm(_config(tmp_path), limit=1, max_payload_bytes=10_000_000),
        daemon=True,
    )
    thread.start()
    assert started.wait(timeout=2)
    assert stage.writer_admission_ready() is False
    assert stage.wait_until_idle(timeout=0.01) is False
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert stage.writer_admission_ready() is True
    assert stage.wait_until_idle(timeout=0) is True
    stage.shutdown()


def test_shutdown_is_process_bounded_with_wedged_parse_worker(tmp_path: Path) -> None:
    """A wedged prefetch worker cannot extend daemon process shutdown."""
    initialize_active_archive_root(tmp_path)
    started = threading.Event()
    release = threading.Event()

    def wedged_worker() -> None:
        started.set()
        release.wait(timeout=5)

    stage = DaemonParseStage(max_workers=1, max_inflight_bytes=100)
    future = stage._executor.submit(wedged_worker)
    try:
        assert started.wait(timeout=2)
        started_shutdown = time.perf_counter()
        stage.shutdown()
        elapsed = time.perf_counter() - started_shutdown
        # The wedged worker holds its slot for 5s; a shutdown that awaited it
        # would take that long. The ceiling only needs to sit clearly below
        # that signal while tolerating scheduler delay on a loaded host.
        assert elapsed < 2.0
        assert all(thread.daemon for thread in stage._executor._threads)
        assert not future.done()
    finally:
        release.set()
        future.result(timeout=2)


def test_warm_async_cancellation_does_not_join_default_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production async warm route is cancellation-safe around a wedged worker."""
    from polylogue.archive.revision_authority import RawRevisionKind
    from polylogue.sources import census_parse_stage, revision_backfill

    initialize_active_archive_root(tmp_path)
    descriptor = (Provider.CODEX, "hash", "capture.jsonl", RawRevisionKind.FULL, 67)
    monkeypatch.setattr(
        census_parse_stage,
        "raw_materialization_pending_census_raw_ids",
        lambda *_args, **_kwargs: ["raw-67"],
    )
    monkeypatch.setattr(
        census_parse_stage,
        "raw_materialization_readonly_descriptors",
        lambda *_args: {"raw-67": descriptor},
    )
    started = threading.Event()
    release = threading.Event()

    def blocked_worker(raw_id: str, *args: object) -> object:
        started.set()
        # These waits bound a genuinely hung test, not scheduling latency: a
        # saturated full-corpus run (3 pytest processes x 8 xdist workers)
        # can delay executor thread start-up past small single-digit
        # timeouts, which twice failed this test at corpus scope only.
        release.wait(timeout=30)
        return raw_id, [], None

    monkeypatch.setattr(revision_backfill, "census_parse_worker", blocked_worker)
    stage = DaemonParseStage(max_workers=1, max_inflight_bytes=1_000, warm_timeout_seconds=300)

    async def scenario() -> None:
        task = asyncio.create_task(
            stage.warm_async(_config(tmp_path), limit=1, max_payload_bytes=100, raw_artifact_id="raw-67")
        )
        assert await asyncio.to_thread(started.wait, 2)
        started_at = time.perf_counter()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert time.perf_counter() - started_at < 0.5
        assert stage.writer_admission_ready()
        assert not stage.wait_until_idle(timeout=0.01)
        release.set()
        assert stage.wait_until_idle(timeout=5)

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        stage.shutdown()


def test_warm_source_payload_admission_bounds_submitted_full_parses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker submission set respects the source-payload budget, not count alone."""
    from polylogue.archive.revision_authority import RawRevisionKind
    from polylogue.sources import census_parse_stage, revision_backfill

    initialize_active_archive_root(tmp_path)
    raw_ids = ["raw-a", "raw-b", "raw-c"]
    descriptors = dict.fromkeys(raw_ids, (Provider.CODEX, "hash", "capture.jsonl", RawRevisionKind.FULL, 60))
    submitted: list[str] = []

    def fake_worker(raw_id: str, *args: object) -> object:
        submitted.append(raw_id)
        return raw_id, [], None

    monkeypatch.setattr(census_parse_stage, "raw_materialization_readonly_descriptors", lambda *_args: descriptors)
    monkeypatch.setattr(revision_backfill, "census_parse_worker", fake_worker)
    stage = DaemonParseStage(max_workers=3, max_inflight_bytes=100, max_cached_tree_bytes=10_000_000)
    try:
        stage.warm_raw_ids(_config(tmp_path), raw_ids=raw_ids, max_payload_bytes=100)
    finally:
        stage.shutdown()

    assert submitted == ["raw-a"]


def test_completed_result_retains_payload_reservation_until_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed executor result cannot admit a duplicate before consumption."""
    from polylogue.archive.revision_authority import RawRevisionKind
    from polylogue.sources import census_parse_stage, revision_backfill

    initialize_active_archive_root(tmp_path)
    descriptor = (Provider.CODEX, "hash", "capture.jsonl", RawRevisionKind.FULL, 67)
    monkeypatch.setattr(
        census_parse_stage,
        "raw_materialization_readonly_descriptors",
        lambda *_args: {"raw-67": descriptor},
    )
    submitted: list[str] = []
    worker_done = threading.Event()
    result_consumption_started = threading.Event()
    release_result = threading.Event()
    real_result = Future.result
    result_gate_used = threading.Event()

    def completed_worker(raw_id: str, *args: object) -> object:
        submitted.append(raw_id)
        worker_done.set()
        return raw_id, [], None

    def gated_result(self: Future[object], timeout: float | None = None) -> object:
        if not result_gate_used.is_set():
            result_gate_used.set()
            result_consumption_started.set()
            assert release_result.wait(timeout=5)
        return real_result(self, timeout)

    monkeypatch.setattr(revision_backfill, "census_parse_worker", completed_worker)
    monkeypatch.setattr(Future, "result", gated_result)
    stage = DaemonParseStage(max_workers=2, max_inflight_bytes=100, warm_timeout_seconds=1)
    first = threading.Thread(
        target=lambda: stage.warm_raw_ids(_config(tmp_path), raw_ids=["raw-67"], max_payload_bytes=100),
        daemon=True,
    )
    first.start()
    try:
        assert worker_done.wait(timeout=2)
        assert result_consumption_started.wait(timeout=2)
        # The worker has completed, but its parsed result is still resident in
        # the Future. A second warm must not submit another full payload.
        assert stage.warm_raw_ids(_config(tmp_path), raw_ids=["raw-67"], max_payload_bytes=100) == 0
        assert submitted == ["raw-67"]
        release_result.set()
        first.join(timeout=5)
        assert not first.is_alive()
    finally:
        release_result.set()
        first.join(timeout=5)
        stage.shutdown()


def test_submit_failure_cleans_every_admitted_future_and_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mid-batch submit error cannot strand prior futures or byte reservations."""
    from polylogue.archive.revision_authority import RawRevisionKind
    from polylogue.sources import census_parse_stage, revision_backfill

    initialize_active_archive_root(tmp_path)
    descriptors = dict.fromkeys(
        ("raw-a", "raw-b", "raw-c"),
        (Provider.CODEX, "hash", "capture.jsonl", RawRevisionKind.FULL, 60),
    )
    monkeypatch.setattr(census_parse_stage, "raw_materialization_readonly_descriptors", lambda *_args: descriptors)
    started = threading.Event()
    release = threading.Event()

    def blocked_worker(raw_id: str, *args: object) -> object:
        started.set()
        # These waits bound a genuinely hung test, not scheduling latency: a
        # saturated full-corpus run (3 pytest processes x 8 xdist workers)
        # can delay executor thread start-up past small single-digit
        # timeouts, which twice failed this test at corpus scope only.
        release.wait(timeout=30)
        return raw_id, [], None

    monkeypatch.setattr(revision_backfill, "census_parse_worker", blocked_worker)
    stage = DaemonParseStage(max_workers=2, max_inflight_bytes=300)
    real_submit = stage._executor.submit
    submit_count = 0

    def fail_second_submit(function: Callable[..., object], *args: object, **kwargs: object) -> Future[object]:
        nonlocal submit_count
        submit_count += 1
        if submit_count == 2:
            raise RuntimeError("synthetic submit failure")
        return real_submit(function, *args, **kwargs)

    monkeypatch.setattr(stage._executor, "submit", fail_second_submit)
    try:
        with pytest.raises(RuntimeError, match="synthetic submit failure"):
            stage.warm_raw_ids(_config(tmp_path), raw_ids=list(descriptors), max_payload_bytes=300)
        assert started.wait(timeout=30)
        release.set()
        assert stage.wait_until_idle(timeout=30)
        assert stage.writer_admission_ready()
        assert stage._pending_payload_bytes == 0
        assert stage._pending_payload_by_raw_id == {}
    finally:
        release.set()
        stage.shutdown()


def test_consecutive_timeout_retries_share_global_payload_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timed-out raw remains admitted globally until its worker completes."""
    from polylogue.archive.revision_authority import RawRevisionKind
    from polylogue.sources import census_parse_stage, revision_backfill

    initialize_active_archive_root(tmp_path)
    descriptor = (Provider.CODEX, "hash", "capture.jsonl", RawRevisionKind.FULL, 67)
    monkeypatch.setattr(
        census_parse_stage,
        "raw_materialization_readonly_descriptors",
        lambda *_args: {"raw-67": descriptor},
    )
    started = threading.Event()
    release = threading.Event()
    submitted: list[str] = []

    def blocked_worker(raw_id: str, *args: object) -> object:
        submitted.append(raw_id)
        started.set()
        release.wait(timeout=5)
        return raw_id, [], None

    monkeypatch.setattr(revision_backfill, "census_parse_worker", blocked_worker)
    stage = DaemonParseStage(max_workers=1, max_inflight_bytes=100, warm_timeout_seconds=0.01)
    try:
        assert stage.warm_raw_ids(_config(tmp_path), raw_ids=["raw-67"], max_payload_bytes=100) == 0
        assert started.wait(timeout=2)
        assert stage.writer_admission_ready() is False
        # The second call sees the timed-out worker's 67-byte reservation;
        # it cannot enqueue a duplicate despite the cache still being empty.
        assert stage.warm_raw_ids(_config(tmp_path), raw_ids=["raw-67"], max_payload_bytes=100) == 0
        assert submitted == ["raw-67"]
    finally:
        release.set()
        assert stage.wait_until_idle(timeout=5)
        assert stage.writer_admission_ready()
        stage.shutdown()


def test_warm_parses_indexed_raw_with_missing_parser_receipt(tmp_path: Path) -> None:
    """The daemon warmer includes all-raw census debt, not only replay debt.

    The raw is already materialized in ``index.db`` but its parser receipt is
    removed. ``DaemonParseStage.warm`` must still parse it outside the writer
    hold, because ``repair_raw_materialization`` will census that same raw.
    Replacing the preview's parser-census selector with the ordinary replay
    selector makes ``warmed`` zero and leaves the cache empty.
    """
    initialize_active_archive_root(tmp_path)
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="indexed-missing-receipt",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="warm this census debt")],
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id, _session_id = archive.write_raw_and_parsed(
            session,
            payload=_codex_payload("indexed-missing-receipt", "warm this census debt"),
            source_path="indexed-missing-receipt.jsonl",
            acquired_at_ms=1,
        )

    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("DELETE FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,))
        conn.commit()
    with sqlite3.connect(f"file:{tmp_path / 'index.db'}?mode=ro", uri=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)

    stage = DaemonParseStage(max_workers=2, max_inflight_bytes=10_000_000)
    try:
        warmed = stage.warm(_config(tmp_path), limit=10, max_payload_bytes=10_000_000)
    finally:
        stage.shutdown()

    assert warmed == 1
    assert stage.cache.contains(raw_id)
    sessions, _payload_bytes, _kind = stage.cache.pop(raw_id)  # type: ignore[misc]
    assert [parsed.provider_session_id for parsed in sessions] == ["indexed-missing-receipt"]


def test_warm_recovers_append_native_id_not_source_path_stem(tmp_path: Path) -> None:
    """polylogue-6lyh1: the daemon's off-writer-hold warmer must apply the
    same APPEND fallback-identity recovery as the sequential path
    (``parse_retained_raw_sessions``'s ``archive.raw_native_id`` lookup),
    not silently fall back to ``Path(source_path).stem``.

    The raw is written APPEND-kind, with a payload carrying no
    ``session_meta`` record (so the parser genuinely has nothing else to
    fall back on) and a ``source_path`` stem deliberately different from the
    recorded native_id -- so a regression to stem-based fallback is
    observable, not silently masked by the two happening to agree.
    """
    from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    payload = (
        b'{"type":"response_item","payload":{"type":"message","id":"m0","role":"user",'
        b'"content":[{"type":"input_text","text":"hello"}]}}\n'
    )
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path="stem-does-not-match.jsonl",
            acquired_at_ms=1,
            raw_id="raw-append",
            native_id="session-real-native-id",
            revision=RawRevisionEnvelope(
                logical_source_key="codex:session-real-native-id",
                kind=RawRevisionKind.APPEND,
                source_revision="raw-append-revision",
                acquisition_generation=0,
                predecessor_source_revision="raw-append-predecessor",
                append_start_offset=0,
                append_end_offset=len(payload),
                authority=RawRevisionAuthority.QUARANTINED,
            ),
        )

    stage = DaemonParseStage(max_workers=2, max_inflight_bytes=10_000_000)
    try:
        warmed = stage.warm(_config(tmp_path), limit=10, max_payload_bytes=10_000_000)
    finally:
        stage.shutdown()

    assert warmed == 1
    sessions, _payload_bytes, _kind = stage.cache.pop("raw-append")  # type: ignore[misc]
    assert len(sessions) == 1
    assert sessions[0].provider_session_id == "session-real-native-id"


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
        time.sleep(5.0)  # far longer than the test's tiny warm timeout below
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
    # 5s sleep -- proving the wait is genuinely bounded, not merely reordered.
    assert elapsed < 0.02 * 10


def test_max_inflight_bytes_default_is_adaptive_and_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whale-memory budget scales with physical RAM, clamped to [64MiB, 2GiB].

    The old fixed 64 MiB default starved bulk-scale warm on whale corpora
    (measured live: 0.37 raws/s stalled on cache admission vs 56.7 raws/s
    with an adequate budget) — the budget must grow on capable machines
    while keeping the 64 MiB floor semantics on small ones.
    """
    # polylogue-czq2: the adaptive-budget resolution functions (and the
    # ``_physical_memory_bytes`` helper they call) now live in
    # ``polylogue.sources.census_parse_stage`` -- ``polylogue.daemon.parse_prefetch``
    # only re-exports the same function objects, so monkeypatching an
    # attribute there would rebind the re-export's local name, not the
    # global the functions actually read from when they execute.
    from polylogue.sources import census_parse_stage as pp

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


# --- polylogue-xb4i: parsed-tree-byte accounting (estimator + eviction) ----


def _session_with_text(native_id: str, text_len: int) -> ParsedSession:
    """One session, one message, one block, all carrying ``text_len`` chars.

    Deliberately constructed directly (not run through a provider parser) so
    tree size is exactly controlled for size-comparison assertions below.
    """
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=native_id,
        messages=[
            ParsedMessage(
                provider_message_id=f"{native_id}-m0",
                role=Role.USER,
                text="x" * text_len,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="x" * text_len)],
            )
        ],
    )


def _deep_size_bytes(obj: object, seen: set[int] | None = None) -> int:
    """Manual recursive object-graph size walk -- a from-scratch equivalent
    of ``pympler.asizeof.asizeof`` (not a repo dependency) used ONLY here,
    to calibrate/verify ``estimate_parsed_tree_bytes``'s cheap structural
    formula against ground truth. Never used on the production hot path --
    that is exactly the cost ``estimate_parsed_tree_bytes`` exists to avoid."""
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for key, value in obj.items():
            size += _deep_size_bytes(key, seen)
            size += _deep_size_bytes(value, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += _deep_size_bytes(item, seen)
    elif hasattr(obj, "__dict__"):
        size += _deep_size_bytes(obj.__dict__, seen)
    return size


def test_estimate_parsed_tree_bytes_is_monotone_in_content_size() -> None:
    """Production dependency: ``estimate_parsed_tree_bytes`` itself. A
    mutation that stopped reading ``message.text``/``block.text`` lengths
    (e.g. hardcoding a fixed per-session constant) would make these three
    estimates equal instead of strictly increasing, and this would fail."""
    small = estimate_parsed_tree_bytes([_session_with_text("s", 10)])
    medium = estimate_parsed_tree_bytes([_session_with_text("s", 1_000)])
    large = estimate_parsed_tree_bytes([_session_with_text("s", 100_000)])

    assert small < medium < large

    # Also monotone in session COUNT at fixed per-session size.
    one_session = estimate_parsed_tree_bytes([_session_with_text("s", 500)])
    two_sessions = estimate_parsed_tree_bytes([_session_with_text("a", 500), _session_with_text("b", 500)])
    assert two_sessions > one_session


def test_estimate_parsed_tree_bytes_within_3x_of_deep_measurement() -> None:
    """Production dependency: the calibrated constants
    ``_ESTIMATOR_BYTES_PER_CHAR``/``_ESTIMATOR_OBJECT_OVERHEAD_BYTES`` inside
    ``estimate_parsed_tree_bytes``. If either constant were set to a wildly
    wrong value (e.g. 0, or 1000x too small), the estimate would fall
    outside a 3x band of ``_deep_size_bytes``'s independent ground-truth
    measurement on the same synthetic session and this would fail -- this is
    the test that keeps the calibration comment in parse_prefetch.py honest
    against actual measured memory, not just internally self-consistent."""
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="calibration",
        messages=[
            ParsedMessage(
                provider_message_id=f"m{i}",
                role=Role.USER,
                text="x" * 300,
                blocks=[
                    ParsedContentBlock(type=BlockType.TEXT, text="y" * 200),
                    ParsedContentBlock(type=BlockType.TEXT, text="z" * 200),
                ],
            )
            for i in range(50)
        ],
    )

    estimate = estimate_parsed_tree_bytes([session])
    deep = _deep_size_bytes(session)

    assert deep / 3 <= estimate <= deep * 3


def test_warm_never_retains_a_tree_exceeding_the_whole_cache_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production dependency: ``DaemonParseStage.warm_raw_ids``'s per-item
    whale check (compares ``estimate_parsed_tree_bytes(sessions)`` against
    ``self._max_cached_tree_bytes`` BEFORE calling ``self.cache.try_admit``).
    Deleting that check (or its ``continue``) would let a whale raw whose
    payload happens to fit the generous inflight-bytes budget below sail
    into the cache regardless of its parsed-tree size -- exactly the
    2026-07-20 earlyoom failure mode -- and every assertion here would flip."""
    _seed_raws(tmp_path, {"whale.jsonl": _codex_payload("session-whale", "seed")})

    from polylogue.sources import revision_backfill

    def whale_worker(
        raw_id: str,
        provider_value: str,
        blob_hash: str,
        source_path: str,
        is_stream: bool,
        blob_root_str: str,
        source_db_path_str: str,
        kind_token: str,
        native_id: str | None,
    ) -> object:
        return raw_id, [_session_with_text("session-whale", 50_000)], None

    monkeypatch.setattr(revision_backfill, "census_parse_worker", whale_worker)

    whale_tree_bytes = estimate_parsed_tree_bytes([_session_with_text("session-whale", 50_000)])
    tiny_budget = whale_tree_bytes // 2
    assert tiny_budget > 0

    stage = DaemonParseStage(
        max_workers=1,
        max_inflight_bytes=10_000_000,  # generous: payload bytes are NOT the constraint here
        max_cached_tree_bytes=tiny_budget,
    )
    try:
        warmed = stage.warm(_config(tmp_path), limit=10, max_payload_bytes=10_000_000)
    finally:
        stage.shutdown()

    assert warmed == 0
    assert len(stage.cache) == 0
    assert stage.cached_tree_bytes_total == 0


def test_cache_evicts_to_stay_within_tree_bytes_budget_under_pressure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production dependency: ``DaemonParseStage._register_cached_tree_bytes``'s
    eviction loop (and the ledger it maintains, ``_tree_bytes_by_raw_id`` /
    ``_cached_tree_bytes_total``). Two equal-size raws are admitted (each
    individually well under the whole-cache budget, so the whale check never
    fires); together they exceed the budget. A mutation that turned the
    eviction ``while`` loop into a no-op would leave both entries cached with
    ``cached_tree_bytes_total`` above budget, and the assertions below would
    fail."""
    _seed_raws(
        tmp_path,
        {
            "a.jsonl": _codex_payload("session-a", "seed-a"),
            "b.jsonl": _codex_payload("session-b", "seed-b"),
        },
    )

    from polylogue.sources import revision_backfill

    source_path_to_native_id = {"a.jsonl": "session-a", "b.jsonl": "session-b"}

    def equal_size_worker(
        raw_id: str,
        provider_value: str,
        blob_hash: str,
        source_path: str,
        is_stream: bool,
        blob_root_str: str,
        source_db_path_str: str,
        kind_token: str,
        native_id_arg: str | None,
    ) -> object:
        native_id = source_path_to_native_id[source_path]
        return raw_id, [_session_with_text(native_id, 5_000)], None

    monkeypatch.setattr(revision_backfill, "census_parse_worker", equal_size_worker)

    one_tree_bytes = estimate_parsed_tree_bytes([_session_with_text("session-a", 5_000)])
    # Fits exactly one tree, not two.
    budget = int(one_tree_bytes * 1.5)

    stage = DaemonParseStage(
        max_workers=1,
        max_inflight_bytes=10_000_000,
        max_cached_tree_bytes=budget,
    )
    try:
        warmed = stage.warm(_config(tmp_path), limit=10, max_payload_bytes=10_000_000)
    finally:
        stage.shutdown()

    # Both raws were individually admitted at the time they were parsed --
    # eviction is a post-hoc budget correction, not an admission rejection.
    assert warmed == 2
    # But only one survives the tree-bytes budget.
    assert len(stage.cache) == 1
    assert stage.cached_tree_bytes_total <= budget
    assert stage.cached_tree_bytes_total == one_tree_bytes


def test_max_cached_tree_bytes_default_is_adaptive_and_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production dependency: ``daemon_parse_stage_max_cached_tree_bytes``.
    Mirrors ``test_max_inflight_bytes_default_is_adaptive_and_clamped`` for
    the SECOND (tree-bytes, not payload-bytes) budget -- distinct clamp
    range [256 MiB, 4 GiB] and distinct RAM fraction (1/8, not 1/16). A
    mutation that swapped in the inflight-bytes clamp/fraction, or dropped
    the env override, would flip these assertions."""
    # polylogue-czq2: the adaptive-budget resolution functions (and the
    # ``_physical_memory_bytes`` helper they call) now live in
    # ``polylogue.sources.census_parse_stage`` -- ``polylogue.daemon.parse_prefetch``
    # only re-exports the same function objects, so monkeypatching an
    # attribute there would rebind the re-export's local name, not the
    # global the functions actually read from when they execute.
    from polylogue.sources import census_parse_stage as pp

    monkeypatch.delenv("POLYLOGUE_DAEMON_PARSE_STAGE_MAX_CACHED_TREE_BYTES", raising=False)

    # 64 GiB machine -> hits the 4 GiB ceiling (64 GiB / 8 = 8 GiB, clamped down).
    monkeypatch.setattr(pp, "_physical_memory_bytes", lambda: 64 * 1024**3)
    assert pp.daemon_parse_stage_max_cached_tree_bytes() == 4 * 1024**3

    # 1 GiB machine -> clamped up to the 256 MiB floor (1 GiB / 8 = 128 MiB).
    monkeypatch.setattr(pp, "_physical_memory_bytes", lambda: 1 * 1024**3)
    assert pp.daemon_parse_stage_max_cached_tree_bytes() == 256 * 1024**2

    # 16 GiB machine -> proportional (16 GiB / 8 = 2 GiB).
    monkeypatch.setattr(pp, "_physical_memory_bytes", lambda: 16 * 1024**3)
    assert pp.daemon_parse_stage_max_cached_tree_bytes() == 2 * 1024**3

    # Unknown physical memory -> conservative floor.
    monkeypatch.setattr(pp, "_physical_memory_bytes", lambda: None)
    assert pp.daemon_parse_stage_max_cached_tree_bytes() == 256 * 1024**2

    # Explicit env override always wins, even over an adaptive value that
    # would otherwise differ.
    monkeypatch.setenv("POLYLOGUE_DAEMON_PARSE_STAGE_MAX_CACHED_TREE_BYTES", "987654321")
    assert pp.daemon_parse_stage_max_cached_tree_bytes() == 987654321
