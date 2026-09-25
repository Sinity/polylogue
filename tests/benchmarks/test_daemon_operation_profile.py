"""Managed production-route profile for daemon/CLI architecture selection.

The profile intentionally names every workload in the packet.  The route
benchmarks below exercise the installed CLI or direct typed UDS operation;
bulk ingest, derivation catch-up, and inactive-candidate construction remain
declared in the same profile so a future daemon fixture cannot silently omit
their throughput denominators.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from pathlib import Path
from resource import RUSAGE_SELF, getrusage
from time import perf_counter, sleep
from typing import Any

import pytest

from polylogue.daemon.execution import MAX_BACKGROUND_STARVATION_S, DaemonBackpressureError
from polylogue.daemon.socket_path import daemon_socket_path
from polylogue.daemon.write_coordinator import DaemonWriteEvent
from polylogue.daemon_client import DaemonClient
from tests.benchmarks.cli_profile import INTERACTION_WORKLOADS, PROFILE_METRICS, profile_manifest, record_metrics
from tests.benchmarks.helpers import BenchmarkFixture, benchmark_repeated
from tests.infra.benchmark_archives import seed_benchmark_archive
from tests.infra.daemon_operations import DaemonOperationStack, running_daemon_operations
from tests.infra.workload_declarations import BenchmarkWorkloadTier

pytest_plugins = ("tests.benchmarks.test_daemon_uds",)

pytestmark = pytest.mark.uses_real_clock(
    "The profile waits for a real AF_UNIX server and measures installed-process wall-clock behavior."
)


def _installed_cli(*, installed_only: bool = False) -> list[str]:
    """Locate the console script this run's interpreter would dispatch.

    The warm-status lane measures the installed CLI, and a hard-coded path
    that misses used to be the only thing standing between this lane and a
    measurement: an unprovisioned worktree has no ``.venv``. Preference order:

    1. this checkout's own ``.venv`` console script — it is the one whose
       ``import polylogue`` is guaranteed to resolve inside this checkout;
    2. the console script beside the running interpreter — only for callers
       that allow fallback; it can belong to a shared venv wired to another
       checkout;
    3. ``python -m polylogue`` — only for callers that allow fallback, with a
       little extra interpreter startup.
    """

    checkout_root = Path(__file__).resolve().parents[2]
    checkout_candidate = checkout_root / ".venv" / "bin" / "polylogue"
    try:
        checkout_candidate.resolve(strict=True).relative_to(checkout_root)
        checkout_owned = True
    except (OSError, ValueError):
        checkout_owned = False
    if checkout_owned and checkout_candidate.is_file() and os.access(checkout_candidate, os.X_OK):
        return [str(checkout_candidate)]
    if installed_only:
        raise RuntimeError("the benchmark requires this checkout's installed polylogue console script")
    interpreter_candidate = Path(sys.executable).parent / "polylogue"
    if interpreter_candidate.is_file() and os.access(interpreter_candidate, os.X_OK):
        return [str(interpreter_candidate)]
    return [sys.executable, "-m", "polylogue"]


def _operation(client: DaemonClient, name: str, payload: dict[str, object] | None = None) -> dict[str, object]:
    result = client.operation(name, payload or {})
    assert isinstance(result, dict)
    assert result.get("protocol") == "polylogue.daemon-operation/v1"
    assert result.get("error") is None
    return result


def _validate_successful_find_response(returncode: int, stdout: str, stderr: str) -> dict[str, object]:
    """Reject CLI errors, empty results, and non-product JSON before timing is recorded."""
    assert returncode == 0, stderr
    payload = json.loads(stdout)
    assert isinstance(payload, dict), stdout
    outcome = payload.get("outcome")
    assert isinstance(outcome, dict) and outcome.get("state") == "ok", stdout
    assert payload.get("source") == "daemon", stdout
    items = payload.get("items")
    assert isinstance(items, list) and items, stdout
    assert all(isinstance(item, dict) for item in items), stdout
    total = payload.get("total")
    assert type(total) is int and total > 0, stdout
    return payload


def test_installed_cli_only_uses_a_console_script_inside_this_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "checkout"
    module_file = checkout / "tests" / "benchmarks" / "test_daemon_operation_profile.py"
    module_file.parent.mkdir(parents=True)
    monkeypatch.setattr(sys.modules[__name__], "__file__", str(module_file))

    shared_bin = tmp_path / "shared-venv" / "bin"
    shared_bin.mkdir(parents=True)
    shared_python = shared_bin / "python"
    shared_python.write_text("", encoding="utf-8")
    shared_cli = shared_bin / "polylogue"
    shared_cli.write_text("#!/bin/sh\n", encoding="utf-8")
    shared_cli.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(shared_python))

    with pytest.raises(RuntimeError, match="this checkout's installed polylogue"):
        _installed_cli(installed_only=True)

    checkout_cli = checkout / ".venv" / "bin" / "polylogue"
    checkout_cli.parent.mkdir(parents=True)
    checkout_cli.write_text("#!/bin/sh\n", encoding="utf-8")
    checkout_cli.chmod(0o755)
    assert _installed_cli(installed_only=True) == [str(checkout_cli)]


def test_installed_cli_timing_requires_successful_nonempty_product_envelope() -> None:
    good = json.dumps(
        {
            "items": [{"id": "codex-session:seeded"}],
            "outcome": {"state": "ok", "reason": None, "detail": {}},
            "source": "daemon",
            "total": 1,
        }
    )
    assert _validate_successful_find_response(0, good, "")["total"] == 1

    with pytest.raises(AssertionError):
        _validate_successful_find_response(1, good, "CLI failed")
    with pytest.raises(AssertionError):
        _validate_successful_find_response(0, good.replace('"daemon"', '"direct"'), "")
    with pytest.raises(AssertionError):
        _validate_successful_find_response(
            0,
            json.dumps({"error": {"code": "daemon_unavailable"}, "items": [{"id": "x"}]}),
            "",
        )
    with pytest.raises(AssertionError):
        _validate_successful_find_response(
            0,
            json.dumps({"items": [], "outcome": {"state": "empty"}, "total": 0}),
            "",
        )


@pytest.mark.benchmark
def test_bench_daemon_warm_status(benchmark: BenchmarkFixture, bench_daemon_uds_client: DaemonClient) -> None:
    """Installed CLI warm status includes process, UDS, and rendering cost."""

    assert bench_daemon_uds_client.socket_path == daemon_socket_path(os.environ["POLYLOGUE_ARCHIVE_ROOT"]), (
        "installed CLI status must use the production-derived archive socket"
    )
    env = {**os.environ, "POLYLOGUE_FORCE_PLAIN": "1"}

    def run() -> subprocess.CompletedProcess[str]:
        started = perf_counter()
        result = subprocess.run(
            [*_installed_cli(installed_only=True), "--plain", "status", "--format", "json"],
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

    result = benchmark_repeated(benchmark, run)
    # A seeded benchmark archive can legitimately report a typed degraded
    # status (exit 1) while still returning the complete JSON operation
    # envelope. Treat that as an available status response; an empty or
    # malformed payload remains an invalid measurement.
    assert result.returncode in (0, 1), result.stderr
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict) and payload, result.stdout


@pytest.mark.benchmark
def test_bench_daemon_static_completion(benchmark: BenchmarkFixture, bench_daemon_uds_client: DaemonClient) -> None:
    def run() -> dict[str, object]:
        return _operation(bench_daemon_uds_client, "completion", {"kind": "field", "incomplete": ""})

    result = benchmark_repeated(benchmark, run)
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

    result = benchmark_repeated(benchmark, run)
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

    result = benchmark_repeated(benchmark, run)
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

    results = benchmark_repeated(benchmark, run)
    assert len(results) == 4
    assert all(result["error"] is None for result in results)
    record_metrics(benchmark, concurrent_interference_p95_ms=max(elapsed, default=0))


#: Sessions in the SMOKE tier this lane seeds (tests/infra/workload_declarations.py:313).
#: Every offset below stays far inside it, so no read can be handed an empty page.
_MIXED_LOAD_SEEDED_SESSIONS = 1_000

#: Distinct tag names each write worker cycles over one session. The write load
#: has to stay bounded for the lane to take repeated rounds -- see
#: ``keep_writing`` -- and this is the bound.
_MIXED_LOAD_TAG_CYCLE = 8


def mixed_load_read_params(read_index: int) -> dict[str, int]:
    """Return ``cli.query`` params for read ``read_index`` of the mixed-load window.

    Every read in the window must reach the archive query path, and in this
    lane nothing but the params can make it. The daemon read cache keys on
    ``(operation, archive_root, generation, epoch, params fingerprint)``
    (polylogue/storage/search/cache.py:157-165), and the epoch only moves for a
    write that reports ``changed_session_ids``
    (polylogue/archive/write_effects.py:166-167). ``mutation.session.tag``
    commits with ``changed_session_ids=()``
    (polylogue/storage/sqlite/archive_tiers/archive.py:4262), so this lane's
    entire write load leaves the cache epoch untouched.

    That is why the previous ``2 + index % 4`` was not merely inelegant: over
    ``range(8)`` it yields ``2,3,4,5,2,3,4,5``, so four of the eight reads this
    lane reported as archive latency were cache hits. Pairing the limit with a
    monotonic offset makes the fingerprint unique for every read of the whole
    repeated window, not just within one round.
    """
    return {"limit": 2 + read_index % 4, "offset": read_index}


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
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("POLYLOGUE_SCHEMA_VALIDATION", "off")
    monkeypatch.delenv("POLYLOGUE_NO_DAEMON", raising=False)
    monkeypatch.delenv("POLYLOGUE_DAEMON", raising=False)
    monkeypatch.delenv("POLYLOGUE_DAEMON_URL", raising=False)
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    socket_path = daemon_socket_path(archive_root, runtime_dir=str(runtime_dir))

    def seed(root: Path) -> None:
        seed_benchmark_archive(root / "index.db", BenchmarkWorkloadTier.SMOKE)

    with running_daemon_operations(
        archive_root,
        seed_archive=seed,
        compute_workers=4,
        compute_queue_units=16,
        socket_path=socket_path,
    ) as stack:
        yield stack


@pytest.mark.benchmark
def test_bench_daemon_mixed_load(
    benchmark: BenchmarkFixture,
    bench_mixed_load_stack: DaemonOperationStack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interactive reads stay served while writes and background units contend.

    Two contention sources run against the one daemon for the whole measured
    window:

    * **Write load** — feeder threads issue real ``mutation.session.tag``
      operations over their own UDS connections, so every one of them takes
      the daemon's single write coordinator and commits to ``user.db`` while
      the reads below are in flight.
    * **Background compute** — feeder threads submit ``bulk-candidate`` units
      into the same bounded kernel the interactive reads are admitted to.

    The measured operation is the interactive side: eight direct ``cli.query``
    UDS reads over four connections, taken over five rounds, plus one installed
    CLI ``find`` subprocess after those rounds while the same background load
    remains active.
    The CLI launcher uses the production-derived socket path; this stack binds
    there instead of using the helper's random test socket. The background
    denominator is completed bulk operations per mixed-load second, compared
    with the same unit on this same kernel without contention. That ratio is
    recorded for calibration only; the accepted fairness contract is nonzero
    background progress and the declared maximum admission wait.

    The lane takes rounds rather than one shot, which is a change of claim as
    much as of shape: a latency budget describes a distribution, and a
    ``rounds=1`` lane reports its single sample as p50 and p95 alike. Two
    things used to make a second round measure something the first one did
    not, and both are removed above -- the write load now cycles a bounded tag
    set instead of growing one the measured read joins, and the read params
    stay unique across the whole window rather than repeating per round.

    Anti-vacuity: this is red if the daemon stops *serving* reads under that
    load rather than merely being constructible. A read that is refused,
    backpressured, times out on its 5s client deadline, or comes back with an
    empty page fails the assertions below (demonstrated by stalling the
    ``cli.query`` read path past that deadline: every read then returns a
    result-less envelope and the test goes red) — as does a run where no write and
    no background unit actually completed during the window, which would mean
    the reads were never contended at all.
    """

    # Attribute the route in the daemon process itself.  Client wall time is
    # not enough: it conflates admission, snapshot pinning, query compute,
    # response encoding, and the instrumentation/transport tail.  These
    # wrappers are deliberately benchmark-only and preserve the production
    # route unchanged.
    import polylogue.operations.daemon_execution as daemon_execution
    import polylogue.operations.daemon_reads as daemon_reads
    from polylogue.storage.search.cache import current_cache_epoch

    phase_lock = threading.Lock()
    phase_name = "quiet"
    phase_names = (
        "admission_queue",
        "read_frame_acquisition",
        "compute",
        "cache_invalidation",
        "instrumentation_tail",
        "serialization_tail",
    )
    phase_samples: dict[str, dict[str, list[float]]] = {
        "quiet": {name: [] for name in phase_names},
        "writer": {name: [] for name in phase_names},
    }
    pending_compute: dict[str, deque[float]] = {
        "quiet": deque(),
        "writer": deque(),
    }
    frame_started = threading.local()

    real_open_operation_read = daemon_execution.open_operation_read  # type: ignore[attr-defined]

    @contextmanager
    def timed_open_operation_read(*args: Any, **kwargs: Any) -> Iterator[object]:
        # Mirrors open_operation_read's own signature via Any: it takes typed
        # keywords (Path, EmbeddingRecipe, QueryExecutionContext, ...) that a
        # ``**kwargs: object`` forward cannot satisfy.
        started = perf_counter()
        with real_open_operation_read(*args, **kwargs) as snapshot:
            elapsed_ms = (perf_counter() - started) * 1000
            frame_started.elapsed_ms = elapsed_ms
            yield snapshot

    real_query_payload = daemon_reads._query_payload

    def timed_query_payload(*args: Any, **kwargs: Any) -> Any:
        started = perf_counter()
        try:
            return real_query_payload(*args, **kwargs)
        finally:
            elapsed_ms = (perf_counter() - started) * 1000
            with phase_lock:
                current_phase = phase_name
                pending_compute[current_phase].append(elapsed_ms)

    monkeypatch.setattr(daemon_execution, "open_operation_read", timed_open_operation_read)
    monkeypatch.setattr(daemon_reads, "_query_payload", timed_query_payload)

    kernel = bench_mixed_load_stack.execution_kernel
    socket_path = bench_mixed_load_stack.client.socket_path
    archive_root = str(bench_mixed_load_stack.archive_root)
    assert socket_path == daemon_socket_path(archive_root), (
        "installed CLI and direct UDS clients must reach the same production-derived archive socket"
    )
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
    write_failures: list[str] = []
    write_hold_measurements: list[tuple[float, float]] = []
    write_event_lock = threading.Lock()
    counters = threading.Lock()

    def observe_write_event(event: DaemonWriteEvent) -> None:
        if event.phase != "released" or event.hold_seconds is None:
            return
        if event.hold_budget_s is None:
            return
        with write_event_lock:
            write_hold_measurements.append((event.hold_seconds * 1000, event.hold_budget_s * 1000))

    # Capture the coordinator's actual gate hold, not UDS roundtrip time. The
    # stack is benchmark-owned and shuts down with this fixture.
    stack_coordinator = bench_mixed_load_stack.write_coordinator
    previous_observer = stack_coordinator._observer
    stack_coordinator._observer = observe_write_event

    def background_unit() -> None:
        sleep(0.005)

    # Establish an unloaded bulk-operation denominator with the same kernel,
    # unit, and submit path used in the mixed window. This fraction is reported
    # as exploratory; the accepted contract names nonzero progress and a
    # bounded starvation window, not a minimum throughput ratio.
    quiet_completed = 0
    quiet_lock = threading.Lock()
    quiet_stop = threading.Event()
    quiet_failures: list[str] = []

    def count_quiet_completion() -> None:
        nonlocal quiet_completed
        while not quiet_stop.is_set():
            try:
                submitted = kernel.submit(background_unit, admission_class="bulk-candidate")
                submitted.future.result(timeout=5)
                with quiet_lock:
                    quiet_completed += 1
            except DaemonBackpressureError:
                sleep(0.005)
            except Exception as error:
                quiet_failures.append(f"{type(error).__name__}: {error}")
                return

    quiet_feeders = [threading.Thread(target=count_quiet_completion, daemon=True) for _ in range(2)]
    quiet_started = perf_counter()
    for feeder in quiet_feeders:
        feeder.start()
    sleep(1.0)
    quiet_duration_s = max(perf_counter() - quiet_started, 1e-6)
    with quiet_lock:
        quiet_completed_in_window = quiet_completed
    quiet_stop.set()
    for feeder in quiet_feeders:
        feeder.join(timeout=5)
    assert all(not feeder.is_alive() for feeder in quiet_feeders), "quiet bulk feeders did not drain"
    assert not quiet_failures, quiet_failures
    assert quiet_completed_in_window > 0, "quiet bulk baseline produced no completions"
    quiet_background_throughput = quiet_completed_in_window / quiet_duration_s

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
        """Hold the daemon's single writer with real audited tag mutations.

        Each worker cycles a bounded set of tag names over one session, and the
        bound is what makes the repeated rounds legitimate rather than a
        convenience. ``cli.query`` is not blind to tags: ``list_summaries``
        ``LEFT JOIN``s the user-overlay tag relation on every plain listing
        (polylogue/storage/sqlite/archive_tiers/archive.py:6335-6337, over the
        ``user_tier.assertions`` union at :7844), and ``tags`` is an
        unconditional field of every returned row
        (polylogue/operations/daemon_reads.py:407). So a tag backlog IS an
        input to the operation this lane times, and a fresh tag per write would
        have made round N read a session carrying everything rounds 1..N-1 left
        on it -- the lane really could only have taken one honest round.

        Cycling instead of growing holds the tag set at
        ``_MIXED_LOAD_TAG_CYCLE`` per session for the whole window, so every
        round reads the same shape. Re-adding a present tag is idempotent in
        the archive but is still a full ``mutation.session.tag`` operation
        taking the daemon's single write coordinator, the mutation transaction
        and the audit tier -- which is the contention this lane is about. The
        assertion after the window checks the bound actually held.

        Measured, not reasoned: with a fresh tag per write instead of the cycle,
        the later rounds' reads do not merely slow down, they breach the 5s
        client deadline and come back ``outcome == "timed-out"``.

        Add-only rather than add/remove: ``mutate-remove-tag`` never sets
        ``allowed_surfaces`` (polylogue/operations/specs.py:380-388), so it
        defaults to the empty tuple (:69) and
        ``MutationTransaction.prepare_bound`` refuses it on EVERY surface
        (polylogue/operations/mutation_transaction.py:773-774). Removal is not
        available to this lane to undo its own writes with.
        """

        nonlocal writes_completed
        client = DaemonClient(socket_path, timeout_s=10)
        target = [session_ids[worker % len(session_ids)]]
        round_index = 0
        while not stop.is_set():
            params: dict[str, object] = {
                "session_ids": target,
                "tags": [f"bench-mixed-load-{worker}-{round_index % _MIXED_LOAD_TAG_CYCLE}"],
            }
            round_index += 1
            try:
                envelope = client.operation(
                    "mutation.session.tag",
                    params,
                    archive_root=archive_root,
                )
            except Exception as error:  # recorded, then asserted after the window
                write_failures.append(f"{type(error).__name__}: {error}")
                return
            if not isinstance(envelope, dict) or envelope.get("error") is not None:
                error_detail = envelope.get("error") if isinstance(envelope, dict) else envelope
                write_failures.append(f"write envelope carried an error: {error_detail!r} for {params}")
                return
            with counters:
                writes_completed += 1
            # Paced, not a denial-of-service: the lane measures reads served
            # under a steady stream of real writes, not the daemon's behavior
            # when two threads mutate as fast as the socket accepts.
            sleep(0.01)

    elapsed: list[int] = []
    installed_cli_elapsed_ms: list[float] = []
    read_index = itertools.count()

    def percentile(values: list[float], fraction: float) -> float:
        ordered = sorted(values)
        if not ordered:
            return 0.0
        # Nearest-rank percentile, so p95/p99 include the upper-tail sample.
        index = max(0, min(len(ordered) - 1, int((len(ordered) * fraction + 0.999999) - 1)))
        return round(ordered[index], 3)

    def installed_cli_read(offset: int) -> None:
        """Exercise the installed CLI against the canonical daemon socket."""
        env = {**os.environ, "POLYLOGUE_FORCE_PLAIN": "1"}
        started = perf_counter()
        result = subprocess.run(
            [
                *_installed_cli(installed_only=True),
                "--plain",
                "find",
                "--format",
                "json",
                "--limit",
                "5",
                "--offset",
                str(offset),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        elapsed_ms = (perf_counter() - started) * 1000
        _validate_successful_find_response(result.returncode, result.stdout, result.stderr)
        installed_cli_elapsed_ms.append(elapsed_ms)

    def phase_read(params: Mapping[str, object]) -> dict[str, object]:
        before_epoch = current_cache_epoch()
        client = DaemonClient(socket_path, timeout_s=5)
        result = _operation(client, "cli.query", {"params": params})
        client_ms = float(client.last_elapsed_ms or 0)
        frame_started.client_ms = client_ms
        timing = result.get("timing")
        server_ms = float(timing.get("elapsed_ms", client_ms)) if isinstance(timing, dict) else client_ms
        admission_queue_ms = float(timing.get("queue_ms", 0.0)) if isinstance(timing, dict) else 0.0
        frame_ms = float(getattr(frame_started, "elapsed_ms", 0.0))
        with phase_lock:
            current_phase = phase_name
            compute_ms = pending_compute[current_phase].popleft() if pending_compute[current_phase] else 0.0
            phase_samples[current_phase]["admission_queue"].append(max(0.0, admission_queue_ms))
            phase_samples[current_phase]["compute"].append(compute_ms)
            phase_samples[current_phase]["read_frame_acquisition"].append(max(0.0, frame_ms))
            phase_samples[current_phase]["cache_invalidation"].append(
                0.0 if current_cache_epoch() == before_epoch else max(0.0, server_ms - frame_ms - compute_ms)
            )
            phase_samples[current_phase]["instrumentation_tail"].append(
                max(0.0, server_ms - admission_queue_ms - frame_ms - compute_ms)
            )
            phase_samples[current_phase]["serialization_tail"].append(max(0.0, client_ms - server_ms))
        return result

    # Quiet control series: same varying semantic requests, before either
    # feeder starts.  It establishes the archive-path floor for comparison.
    for index in range(8):
        phase_read(mixed_load_read_params(index))

    def run() -> list[dict[str, object]]:
        def one(_slot: int) -> dict[str, object]:
            nonlocal phase_name
            # A monotonic counter, not the in-round index: the counter keeps
            # every read of every round on the archive path (see
            # ``mixed_load_read_params``), which an in-round index cannot do
            # because round 2 would repeat round 1's fingerprints exactly.
            params = mixed_load_read_params(next(read_index))
            assert params["offset"] + params["limit"] < _MIXED_LOAD_SEEDED_SESSIONS
            with phase_lock:
                phase_name = "writer"
            result = phase_read(params)
            elapsed.append(int(getattr(frame_started, "client_ms", 0.0)))
            return result

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(one, range(8)))
        with phase_lock:
            pending_compute["writer"].clear()
        return results

    feeders = [threading.Thread(target=keep_background_busy, daemon=True) for _ in range(2)]
    feeders += [threading.Thread(target=keep_writing, args=(worker,), daemon=True) for worker in range(2)]
    started = perf_counter()
    for feeder in feeders:
        feeder.start()
    try:
        results = benchmark_repeated(benchmark, run)
        # Sample the installed process once after the repeated direct-read
        # timing, while the same writer/bulk feeders are still active.
        installed_cli_read(next(read_index))
    finally:
        mixed_window_s = max(perf_counter() - started, 1e-6)
        with counters:
            background_completed_in_window = background_completed
        stop.set()
        for feeder in feeders:
            feeder.join(timeout=30)
        stack_coordinator._observer = previous_observer

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
    assert background_completed_in_window > 0
    assert snapshot.background_max_wait_s < MAX_BACKGROUND_STARVATION_S
    assert snapshot.used_units == 0, snapshot
    assert peak_queue_units <= snapshot.capacity_units
    assert peak_queue_bytes <= snapshot.capacity_bytes
    # The repeatability precondition, checked rather than asserted in prose:
    # every round must read the same session shape, and `cli.query` rows carry
    # user tags, so the write load must not have grown the tag set it reads.
    # A per-write fresh tag makes this red -- and that is the whole reason this
    # lane could previously take only one round.
    tagged = _operation(bench_mixed_load_stack.client, "cli.query", {"params": {"limit": 5}})
    tagged_page = tagged["result"]
    assert isinstance(tagged_page, dict)
    for row in tagged_page["items"]:
        assert isinstance(row, dict)
        bench_tags = [tag for tag in (row.get("tags") or []) if str(tag).startswith("bench-mixed-load-")]
        assert len(bench_tags) <= _MIXED_LOAD_TAG_CYCLE, (
            f"the write load grew the tag set the measured read joins: {len(bench_tags)} tags on "
            f"{row.get('id')}, bound {_MIXED_LOAD_TAG_CYCLE}; later rounds were not reading round 1's shape"
        )
    # A real percentile over the whole repeated window's reads, not the single
    # worst sample ``max()`` used to report under a p95's name. With five
    # rounds of eight reads there are forty samples to take it from.
    ordered = sorted(elapsed)
    interactive_percentiles = {
        "p50_ms": percentile([float(value) for value in ordered], 0.50),
        "p95_ms": percentile([float(value) for value in ordered], 0.95),
        "p99_ms": percentile([float(value) for value in ordered], 0.99),
        "samples": len(ordered),
    }
    interference_p95 = int(interactive_percentiles["p95_ms"])
    max_writer_hold_ms, writer_hold_budget_ms = max(write_hold_measurements, default=(0.0, 0.0))
    loaded_background_throughput = background_completed_in_window / mixed_window_s
    background_throughput_fraction = (
        loaded_background_throughput / quiet_background_throughput if quiet_background_throughput > 0 else 0.0
    )
    phase_report = {
        phase: {
            metric: {
                "p50_ms": percentile(values, 0.50),
                "p95_ms": percentile(values, 0.95),
                "p99_ms": percentile(values, 0.99),
                "samples": len(values),
            }
            for metric, values in metrics.items()
        }
        for phase, metrics in phase_samples.items()
    }
    dominant_phase = max(
        phase_report["writer"],
        key=lambda metric: float(phase_report["writer"][metric]["p95_ms"]),
    )
    record_metrics(
        benchmark,
        concurrent_interference_p95_ms=interference_p95,
        writer_hold_ms=round(max_writer_hold_ms, 3),
        background_operations=background_completed_in_window,
        background_throughput=loaded_background_throughput,
        queue_delay_ms=int(snapshot.background_max_wait_s * 1000),
        peak_queue_units=peak_queue_units,
        peak_queue_bytes=peak_queue_bytes,
        peak_rss_kib=getrusage(RUSAGE_SELF).ru_maxrss,
        mixed_load_phase_percentiles=phase_report,
        mixed_load_dominant_phase=dominant_phase,
        mixed_load_series_summary={
            "quiet": {
                "queue_delay_ms": 0.0,
                "writer_hold_ms": 0.0,
                "background_throughput": round(quiet_background_throughput, 3),
            },
            "writer": {
                "queue_delay_ms": round(float(snapshot.background_max_wait_s) * 1000, 3),
                "writer_hold_ms": round(max_writer_hold_ms, 3),
                "background_throughput": round(loaded_background_throughput, 3),
            },
        },
        mixed_load_interactive_percentiles=interactive_percentiles,
        mixed_load_installed_cli_samples_ms=[round(value, 3) for value in installed_cli_elapsed_ms],
        mixed_load_quiet_background_throughput=round(quiet_background_throughput, 3),
        mixed_load_background_throughput_fraction=round(background_throughput_fraction, 3),
        mixed_load_writer_hold_budget_ms=round(writer_hold_budget_ms, 3),
    )
    assert installed_cli_elapsed_ms, "the mixed-load window must include an installed CLI read"
    assert len(installed_cli_elapsed_ms) == 1, "one installed CLI sample is required under the mixed-load window"
    assert write_hold_measurements, "the mixed-load writer must produce coordinator hold measurements"
    assert writer_hold_budget_ms > 0
    assert all(hold_ms <= budget_ms for hold_ms, budget_ms in write_hold_measurements)


#: Latency the injection adds to one archive query. Leave enough separation
#: from cache-hit transport jitter on a loaded workstation that the positive
#: cache control does not fail merely because the host is busy.
_ARCHIVE_PATH_INJECTION_MS = 500


def test_read_series_separates_cache_hits_from_archive(
    bench_daemon_uds_client: DaemonClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The concurrent-read lane's series must observe the archive, not its cache.

    polylogue-gd9i1 AC2. ``daemon_mixed_load`` once passed its own anti-vacuity
    injection because identical read params were served from the daemon read
    cache: the lane was timing the cache and reporting it as archive latency.
    Both lanes now issue distinct params per worker, but "distinct params" is a
    claim about the code, not a measurement -- nothing proved the reads reached
    ``_query_payload``.

    This is that proof, and it is two-sided by construction:

    * with a controlled latency injection on the archive query path, EVERY read
      in the uncached series must pay it;
    * repeating one read's exact params must NOT pay it, because the cache
      answers before the query path runs.

    The second half is what makes the first meaningful. A run where the cache
    was simply disabled would satisfy an "injection is observed" assertion on
    its own while telling us nothing about whether the lane's params were
    distinct enough to miss.

    Anti-vacuity: give the uncached series identical params and its assertion
    goes red on reads 2-4; remove the cache short-circuit and the cache-hit
    assertion goes red.
    """
    import polylogue.operations.daemon_reads as daemon_reads

    real_query_payload = daemon_reads._query_payload
    injected: list[float] = []

    def slow_query_payload(*args: Any, **kwargs: Any) -> Any:
        injected.append(perf_counter())
        sleep(_ARCHIVE_PATH_INJECTION_MS / 1000)
        return real_query_payload(*args, **kwargs)

    monkeypatch.setattr(daemon_reads, "_query_payload", slow_query_payload)

    socket_path = bench_daemon_uds_client.socket_path

    def read(params: dict[str, object]) -> int:
        client = DaemonClient(socket_path, timeout_s=10)
        _operation(client, "cli.query", {"params": params})
        return client.last_elapsed_ms or 0

    # Offsets nothing else in this module uses, so every one of these is a
    # guaranteed cache miss rather than a hit left by an earlier lane.
    uncached = [read({"limit": 3, "offset": 900 + index}) for index in range(4)]
    assert min(uncached) >= _ARCHIVE_PATH_INJECTION_MS, (
        f"an uncached read did not reach the archive query path: {uncached} ms, "
        f"injection {_ARCHIVE_PATH_INJECTION_MS} ms"
    )
    assert len(injected) == 4, f"the injection ran {len(injected)} times for four distinct reads"

    repeated_params: dict[str, object] = {"limit": 3, "offset": 950}
    first = read(repeated_params)
    assert first >= _ARCHIVE_PATH_INJECTION_MS, f"the cache-populating read skipped the archive: {first} ms"
    hits = [read(repeated_params) for _ in range(3)]
    assert max(hits) < _ARCHIVE_PATH_INJECTION_MS, (
        f"a repeated read paid the archive-path injection, so nothing was served from the read cache: {hits} ms"
    )
    assert len(injected) == 5, f"a cache hit reached the query path: {len(injected)} injections for 8 reads"


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


def test_mixed_load_series_stays_on_the_archive_path_across_rounds(
    bench_daemon_uds_client: DaemonClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mixed-load lane's params must miss the read cache for a whole window.

    ``test_read_series_separates_cache_hits_from_archive`` proves the
    *concurrent-reads* series reaches the archive. It says nothing about this
    lane, and this lane needed saying: its write load is
    ``mutation.session.tag``, which commits ``changed_session_ids=()``
    (polylogue/storage/sqlite/archive_tiers/archive.py:4262) and so never bumps
    the cache epoch. Params are the only thing keeping a read off the cache
    here, and once the lane takes repeated rounds they have to stay distinct
    across all of them, not just within one.

    The proof is two-sided on purpose, because "every read paid the injection"
    is satisfiable by a run where caching was simply not reachable:

    * the series ``mixed_load_read_params`` actually generates must pay the
      archive-path injection on every read of a multi-round window;
    * the superseded ``2 + index % 4`` must NOT, on exactly the same daemon and
      the same injection -- its repeats are served from the cache.

    The second half is the positive control. If it also paid, the first half
    would be evidence about the injection rather than about the params.

    Anti-vacuity: restore ``2 + index % 4`` as the lane's generator and the
    first assertion goes red; make the superseded formula's offsets unique and
    the second goes red.
    """
    import polylogue.operations.daemon_reads as daemon_reads

    real_query_payload = daemon_reads._query_payload
    injected: list[float] = []

    def slow_query_payload(*args: Any, **kwargs: Any) -> Any:
        injected.append(perf_counter())
        sleep(_ARCHIVE_PATH_INJECTION_MS / 1000)
        return real_query_payload(*args, **kwargs)

    monkeypatch.setattr(daemon_reads, "_query_payload", slow_query_payload)
    socket_path = bench_daemon_uds_client.socket_path

    def read(params: Mapping[str, object]) -> int:
        client = DaemonClient(socket_path, timeout_s=10)
        _operation(client, "cli.query", {"params": dict(params)})
        return client.last_elapsed_ms or 0

    # Two rounds of the lane's eight reads. A base offset no other lane in this
    # module uses keeps the window's first reads from being served a hit some
    # earlier test left in the shared daemon's cache.
    base = 700
    window = 2 * 8
    live = [read(mixed_load_read_params(base + index)) for index in range(window)]
    assert min(live) >= _ARCHIVE_PATH_INJECTION_MS, (
        f"a mixed-load read was served from the cache instead of the archive: {live} ms, "
        f"injection {_ARCHIVE_PATH_INJECTION_MS} ms"
    )
    assert len(injected) == window, f"{len(injected)} archive queries for {window} distinct reads"

    # Positive control: the formula this lane used to carry, on the same daemon.
    # Its params repeat every four reads, so the repeats are cache hits -- which
    # is what made half of the lane's reported archive latency a cache timing.
    superseded = [read({"limit": 2 + index % 4, "offset": base + 500}) for index in range(window)]
    assert max(superseded[4:]) < _ARCHIVE_PATH_INJECTION_MS, (
        "the superseded params reached the archive on every read, so this control proves nothing "
        f"about the cache: {superseded} ms"
    )
    assert len(injected) == window + 4, (
        f"the superseded series ran {len(injected) - window} archive queries for {window} reads; "
        "four distinct params should have produced exactly four"
    )
