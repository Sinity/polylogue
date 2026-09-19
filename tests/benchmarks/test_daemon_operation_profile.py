"""Managed production-route profile for daemon/CLI architecture selection.

The profile intentionally names every workload in the packet.  The route
benchmarks below exercise the installed CLI or direct typed UDS operation;
bulk ingest, derivation catch-up, and inactive-candidate construction remain
declared in the same profile so a future daemon fixture cannot silently omit
their throughput denominators.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from resource import RUSAGE_SELF, getrusage
from time import perf_counter, sleep

import pytest

from polylogue.daemon.execution import MAX_BACKGROUND_STARVATION_S, DaemonBackpressureError
from polylogue.daemon_client import DaemonClient
from tests.benchmarks.cli_profile import INTERACTION_WORKLOADS, PROFILE_METRICS, profile_manifest, record_metrics
from tests.benchmarks.helpers import BenchmarkFixture, benchmark_one_shot
from tests.infra.benchmark_archives import seed_benchmark_archive
from tests.infra.daemon_operations import DaemonOperationStack, running_daemon_operations
from tests.infra.workload_declarations import BenchmarkWorkloadTier

pytest_plugins = ("tests.benchmarks.test_daemon_uds",)

pytestmark = pytest.mark.uses_real_clock(
    "The profile waits for a real AF_UNIX server and measures installed-process wall-clock behavior."
)


def _installed_cli() -> list[str]:
    """Locate the console script this run's interpreter would dispatch.

    The warm-status lane measures the installed CLI, and a hard-coded path
    that misses used to be the only thing standing between this lane and a
    measurement: an unprovisioned worktree has no ``.venv``. Preference order:

    1. this checkout's own ``.venv`` console script — it is the one whose
       ``import polylogue`` is guaranteed to resolve inside this checkout;
    2. the console script beside the running interpreter — right when the
       environment is provisioned elsewhere, but it can belong to a shared
       venv wired to a different checkout, so it is not tried first;
    3. ``python -m polylogue`` — the same product entry point with a little
       extra interpreter startup, so the lane measures rather than refusing.
    """

    candidates = (
        Path(__file__).parents[2] / ".venv" / "bin" / "polylogue",
        Path(sys.executable).parent / "polylogue",
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return [str(candidate)]
    return [sys.executable, "-m", "polylogue"]


def _operation(client: DaemonClient, name: str, payload: dict[str, object] | None = None) -> dict[str, object]:
    result = client.operation(name, payload or {})
    assert isinstance(result, dict)
    assert result.get("protocol") == "polylogue.daemon-operation/v1"
    assert result.get("error") is None
    return result


@pytest.mark.benchmark
def test_bench_daemon_warm_status(benchmark: BenchmarkFixture, bench_daemon_uds_client: DaemonClient) -> None:
    """Installed CLI warm status includes process, UDS, and rendering cost."""

    del bench_daemon_uds_client
    env = {**os.environ, "POLYLOGUE_FORCE_PLAIN": "1"}

    def run() -> subprocess.CompletedProcess[str]:
        started = perf_counter()
        result = subprocess.run(
            [*_installed_cli(), "--plain", "status", "--format", "json"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed_ms = (perf_counter() - started) * 1000
        record_metrics(
            benchmark,
            warm_roundtrip_ms=elapsed_ms,
            bytes=len(result.stdout),
            rows=result.stdout.count("\n"),
        )
        return result

    result = benchmark_one_shot(benchmark, run)
    assert result.returncode == 0, result.stderr


@pytest.mark.benchmark
def test_bench_daemon_static_completion(benchmark: BenchmarkFixture, bench_daemon_uds_client: DaemonClient) -> None:
    def run() -> dict[str, object]:
        return _operation(bench_daemon_uds_client, "completion", {"kind": "field", "incomplete": ""})

    result = benchmark_one_shot(benchmark, run)
    assert isinstance(result["result"], dict)
    record_metrics(
        benchmark,
        static_completion_ms=bench_daemon_uds_client.last_elapsed_ms or 0,
        bytes=len(json.dumps(result, separators=(",", ":")).encode()),
    )


@pytest.mark.benchmark
def test_bench_daemon_live_completion(benchmark: BenchmarkFixture, bench_daemon_uds_client: DaemonClient) -> None:
    def run() -> dict[str, object]:
        return _operation(bench_daemon_uds_client, "completion", {"kind": "terminal-source", "incomplete": ""})

    result = benchmark_one_shot(benchmark, run)
    assert isinstance(result["result"], dict)
    record_metrics(
        benchmark,
        live_completion_ms=bench_daemon_uds_client.last_elapsed_ms or 0,
        bytes=len(json.dumps(result, separators=(",", ":")).encode()),
    )


@pytest.mark.benchmark
def test_bench_daemon_cancellation(benchmark: BenchmarkFixture, bench_daemon_uds_client: DaemonClient) -> None:
    def run() -> dict[str, object]:
        return _operation(bench_daemon_uds_client, "cli.query", {"params": {"limit": 1}})

    result = benchmark_one_shot(benchmark, run)
    assert result["progress"] == {"state": "complete"}
    record_metrics(
        benchmark,
        cancellation_ms=bench_daemon_uds_client.last_elapsed_ms or 0,
        bytes=len(json.dumps(result, separators=(",", ":")).encode()),
    )


@pytest.mark.benchmark
def test_bench_daemon_concurrent_reads(benchmark: BenchmarkFixture, bench_daemon_uds_client: DaemonClient) -> None:
    socket_path = bench_daemon_uds_client.socket_path
    elapsed: list[int] = []

    def run() -> list[dict[str, object]]:
        def one(index: int) -> dict[str, object]:
            # Distinct params per worker: identical requests are served from the
            # daemon read cache, which measures the cache rather than the
            # interference this benchmark names.
            client = DaemonClient(socket_path, timeout_s=2)
            result = _operation(client, "cli.query", {"params": {"limit": 5 + index, "offset": index}})
            elapsed.append(client.last_elapsed_ms or 0)
            return result

        with ThreadPoolExecutor(max_workers=4) as pool:
            return list(pool.map(one, range(4)))

    results = benchmark_one_shot(benchmark, run)
    assert len(results) == 4
    assert all(result["error"] is None for result in results)
    record_metrics(benchmark, concurrent_interference_p95_ms=max(elapsed, default=0))


@pytest.fixture
def bench_mixed_load_stack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[DaemonOperationStack]:
    """A daemon sized for the mixed-load lane rather than for saturation.

    The shared ``bench_daemon_uds_stack`` runs a deliberately tiny kernel (two
    workers, four queue units). That is right for the single-request lanes, but
    under this lane's reads + writes + background units it exhausts admission,
    and the lane then measures a ``compute_backpressure`` refusal instead of
    service under load. The capacity is named here so the contention the test
    asserts is contention for the writer and for dispatch, not for the queue
    reservation of an undersized fixture.

    The archive root is this test's own: the lane writes real tags, and the
    session-scoped root the read-only lanes share must not accumulate them.
    """

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("POLYLOGUE_SCHEMA_VALIDATION", "off")
    monkeypatch.delenv("POLYLOGUE_NO_DAEMON", raising=False)
    monkeypatch.delenv("POLYLOGUE_DAEMON", raising=False)

    def seed(root: Path) -> None:
        seed_benchmark_archive(root / "index.db", BenchmarkWorkloadTier.SMOKE)

    with running_daemon_operations(
        archive_root,
        seed_archive=seed,
        compute_workers=4,
        compute_queue_units=16,
    ) as stack:
        yield stack


@pytest.mark.benchmark
def test_bench_daemon_mixed_load(benchmark: BenchmarkFixture, bench_mixed_load_stack: DaemonOperationStack) -> None:
    """Interactive reads stay served while writes and background units contend.

    Two contention sources run against the one daemon for the whole measured
    window:

    * **Write load** — feeder threads issue real ``mutation.session.tag``
      operations over their own UDS connections, so every one of them takes
      the daemon's single write coordinator and commits to ``user.db`` while
      the reads below are in flight.
    * **Background compute** — feeder threads submit ``bulk-candidate`` units
      into the same bounded kernel the interactive reads are admitted to.

    The measured operation is the interactive side: eight ``cli.query`` reads
    over four connections. The background denominator is completed background
    operations per mixed-load second, and queue delay is the kernel's own
    longest admission-to-dispatch wait.

    Anti-vacuity: this is red if the daemon stops *serving* reads under that
    load rather than merely being constructible. A read that is refused,
    backpressured, times out on its 5s client deadline, or comes back with an
    empty page fails the assertions below (demonstrated by stalling the
    ``cli.query`` read path past that deadline: every read then returns a
    result-less envelope and the test goes red) — as does a run where no write and
    no background unit actually completed during the window, which would mean
    the reads were never contended at all.
    """

    kernel = bench_mixed_load_stack.execution_kernel
    socket_path = bench_mixed_load_stack.client.socket_path
    archive_root = str(bench_mixed_load_stack.archive_root)
    seeded = _operation(bench_mixed_load_stack.client, "cli.query", {"params": {"limit": 5}})
    page = seeded["result"]
    assert isinstance(page, dict)
    items = page["items"]
    assert isinstance(items, list) and items, "mixed-load contention needs a seeded read page"
    session_ids = [str(item["id"]) for item in items if isinstance(item, dict)]
    assert session_ids

    stop = threading.Event()
    background_completed = 0
    peak_queue_units = 0
    peak_queue_bytes = 0
    writes_completed = 0
    write_latency_ms: list[int] = []
    write_failures: list[str] = []
    counters = threading.Lock()

    def background_unit() -> None:
        sleep(0.005)

    def observe_admission() -> None:
        """Retain queue high-water marks while the real mixed load is active."""

        nonlocal peak_queue_bytes, peak_queue_units
        admission = kernel.snapshot()
        with counters:
            peak_queue_units = max(peak_queue_units, admission.queued_units)
            peak_queue_bytes = max(peak_queue_bytes, admission.queued_bytes)

    def keep_background_busy() -> None:
        nonlocal background_completed
        while not stop.is_set():
            try:
                submitted = kernel.submit(background_unit, admission_class="bulk-candidate")
            except DaemonBackpressureError:
                observe_admission()
                sleep(0.005)
                continue
            observe_admission()
            with suppress(Exception):
                submitted.future.result(timeout=5)
                with counters:
                    background_completed += 1

    def keep_writing(worker: int) -> None:
        """Hold the daemon's single writer with real audited tag mutations."""

        nonlocal writes_completed
        client = DaemonClient(socket_path, timeout_s=10)
        round_index = 0
        while not stop.is_set():
            tag = f"bench-mixed-load-{worker}-{round_index}"
            round_index += 1
            try:
                envelope = client.operation(
                    "mutation.session.tag",
                    {"session_ids": session_ids[:1], "tags": [tag]},
                    archive_root=archive_root,
                )
            except Exception as error:  # recorded, then asserted after the window
                write_failures.append(f"{type(error).__name__}: {error}")
                return
            if not isinstance(envelope, dict) or envelope.get("error") is not None:
                write_failures.append(f"write envelope carried an error: {envelope}")
                return
            with counters:
                writes_completed += 1
                write_latency_ms.append(client.last_elapsed_ms or 0)
            # Paced, not a denial-of-service: the lane measures reads served
            # under a steady stream of real writes, not the daemon's behavior
            # when two threads mutate as fast as the socket accepts.
            sleep(0.01)

    elapsed: list[int] = []

    def run() -> list[dict[str, object]]:
        def one(index: int) -> dict[str, object]:
            client = DaemonClient(socket_path, timeout_s=5)
            # Distinct page sizes per read: identical params would be served
            # from the daemon's read cache, and the lane would time the cache
            # rather than reads executed against the archive under write load.
            result = _operation(client, "cli.query", {"params": {"limit": 2 + index % 4}})
            elapsed.append(client.last_elapsed_ms or 0)
            return result

        with ThreadPoolExecutor(max_workers=4) as pool:
            return list(pool.map(one, range(8)))

    feeders = [threading.Thread(target=keep_background_busy, daemon=True) for _ in range(2)]
    feeders += [threading.Thread(target=keep_writing, args=(worker,), daemon=True) for worker in range(2)]
    started = perf_counter()
    for feeder in feeders:
        feeder.start()
    try:
        results = benchmark_one_shot(benchmark, run)
    finally:
        stop.set()
        for feeder in feeders:
            feeder.join(timeout=30)
    duration_s = max(perf_counter() - started, 1e-6)

    # The last write's own control unit can still be settling when its client
    # reply is already back, so drain is a bounded wait rather than an instant.
    drain_deadline = perf_counter() + 5
    snapshot = kernel.snapshot()
    while snapshot.used_units and perf_counter() < drain_deadline:
        sleep(0.05)
        snapshot = kernel.snapshot()
    observe_admission()
    assert not write_failures, write_failures
    assert len(results) == 8
    # Every interactive read completed and was served a page, while the writer
    # and the background classes were busy. A refusal, a deadline, or an
    # accepted-but-empty envelope is a read the daemon did not serve.
    for result in results:
        assert result["error"] is None, result["error"]
        assert result["outcome"] == "completed", result["outcome"]
        served = result["result"]
        assert isinstance(served, dict), result["outcome"]
        rows = served["items"]
        assert isinstance(rows, list) and rows
    # Mixed-load progress: the contention was real on both axes, and no
    # background unit waited past the declared starvation window.
    assert writes_completed > 0
    assert background_completed > 0
    assert snapshot.background_max_wait_s < MAX_BACKGROUND_STARVATION_S
    assert snapshot.used_units == 0, snapshot
    assert peak_queue_units <= snapshot.capacity_units
    assert peak_queue_bytes <= snapshot.capacity_bytes
    record_metrics(
        benchmark,
        concurrent_interference_p95_ms=max(elapsed, default=0),
        writer_hold_ms=max(write_latency_ms, default=0),
        background_operations=background_completed,
        background_throughput=background_completed / duration_s,
        queue_delay_ms=int(snapshot.background_max_wait_s * 1000),
        peak_queue_units=peak_queue_units,
        peak_queue_bytes=peak_queue_bytes,
        peak_rss_kib=getrusage(RUSAGE_SELF).ru_maxrss,
    )


def test_profile_declares_all_packet_workloads() -> None:
    """Anti-vacuity: deleting a required workload makes the profile fail."""

    assert sys.version_info >= (3, 14)
    assert getattr(sys, "_is_gil_enabled", lambda: True)() is False
    assert {item.name for item in INTERACTION_WORKLOADS} == {
        "cold-status",
        "warm-status",
        "find-read",
        "static-completion",
        "live-completion",
        "fuzzy-launch",
        "pagination",
        "cancellation",
        "concurrent-reads",
        "incremental-ingest",
        "derivation-catch-up",
        "inactive-candidate",
    }


def test_profile_declares_mixed_load_queue_high_water_metrics() -> None:
    """Mixed-load output must retain queue depth and queued-byte maxima.

    Anti-vacuity: removing either profile metric makes the benchmark helper
    refuse its measured high-water value, so a drained final snapshot cannot
    masquerade as bounded mixed-load admission.
    """

    expected = {"peak_queue_units", "peak_queue_bytes"}
    manifest_metrics = profile_manifest()["metrics"]
    assert isinstance(manifest_metrics, list)
    assert expected <= set(PROFILE_METRICS)
    assert expected <= set(manifest_metrics)
