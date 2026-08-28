"""Managed scaling profile over the production bounded compute adapter."""

from __future__ import annotations

import json
import logging
import resource
import statistics
import sys
import threading
import time
from asyncio import run as run_async
from contextlib import redirect_stderr
from dataclasses import asdict, dataclass
from io import StringIO

from polylogue.daemon.execution import (
    BoundedComputeAdapter,
    DaemonBackpressureError,
    DaemonOperationCancelled,
    current_cancellation,
)
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteEvent
from polylogue.runtime import available_cpus, require_free_threaded_runtime, runtime_identity


@dataclass(frozen=True, slots=True)
class ProfileResult:
    workload: str
    workers: int
    admission_units: int
    submitted: int
    completed: int
    rejected: int
    throughput_per_second: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    queue_count: int
    queue_bytes: int
    rss_kib: int
    cpu_utilization_percent: float
    writer_hold_ms: float
    cancelled: int
    background_progress: int


def _work(units: int) -> str:
    value = b"polylogue-concurrency-profile"
    for _ in range(units):
        value = __import__("hashlib").sha256(value).digest()
    return value.hex()


def run_profile(*, workers: int, admission_units: int, workload: str, jobs: int) -> ProfileResult:
    adapter = BoundedComputeAdapter(
        max_workers=workers,
        queue_units=max(0, admission_units - workers),
        queue_bytes=jobs * 1024,
        thread_name_prefix=f"profile-{workload}",
    )
    started = time.perf_counter()
    cpu_started = resource.getrusage(resource.RUSAGE_SELF)
    durations: list[float] = []
    futures = []
    rejected = 0
    cancelled = 0
    peak_queue_count = 0
    peak_queue_bytes = 0

    def observe_queue() -> None:
        nonlocal peak_queue_count, peak_queue_bytes
        snapshot = adapter.snapshot()
        peak_queue_count = max(peak_queue_count, snapshot.queued_units)
        peak_queue_bytes = max(peak_queue_bytes, snapshot.queued_bytes)

    try:
        cancellation_started = threading.Event()

        def cancellation_probe() -> None:
            cancellation_started.set()
            while True:
                handle = current_cancellation()
                if handle is not None and handle.cancelled:
                    break
                time.sleep(0.001)
            raise DaemonOperationCancelled("profile cancellation probe")

        cancellation_operation = adapter.submit(cancellation_probe, estimated_bytes=1024)
        observe_queue()
        if not cancellation_started.wait(timeout=2):
            raise RuntimeError("managed profile cancellation probe did not start")
        cancellation_operation.cancellation.cancel()
        futures.append((cancellation_operation, time.perf_counter()))
        for index in range(jobs):
            try:
                submitted_at = time.perf_counter()

                def compute(index: int = index) -> str:
                    return _work(200 + index % 50)

                operation = adapter.submit(compute, estimated_bytes=1024)
                observe_queue()
            except DaemonBackpressureError:
                rejected += 1
                continue
            futures.append((operation, submitted_at))
        for operation, submitted_at in futures:
            try:
                operation.future.result(timeout=30)
            except DaemonOperationCancelled:
                cancelled += 1
            else:
                durations.append((time.perf_counter() - submitted_at) * 1000)
    finally:
        adapter.shutdown(wait=True)
    elapsed = max(time.perf_counter() - started, 1e-9)
    cpu_finished = resource.getrusage(resource.RUSAGE_SELF)
    cpu_seconds = (cpu_finished.ru_utime - cpu_started.ru_utime) + (cpu_finished.ru_stime - cpu_started.ru_stime)
    writer_events: list[DaemonWriteEvent] = []
    previous_log_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with redirect_stderr(StringIO()):
            run_async(
                DaemonWriteCoordinator(observer=writer_events.append).run(
                    "profile.writer", lambda: _completed_operation()
                )
            )
    finally:
        logging.disable(previous_log_level)
    writer_hold_seconds = next(
        (event.hold_seconds for event in reversed(writer_events) if event.hold_seconds is not None),
        0.0,
    )
    quantiles = (
        statistics.quantiles(durations, n=100, method="inclusive")
        if len(durations) >= 2
        else [durations[0] if durations else 0.0] * 99
    )
    return ProfileResult(
        workload=workload,
        workers=workers,
        admission_units=admission_units,
        submitted=len(futures),
        completed=len(durations),
        rejected=rejected,
        throughput_per_second=len(durations) / elapsed,
        p50_ms=statistics.median(durations) if durations else 0.0,
        p95_ms=quantiles[94],
        p99_ms=quantiles[98],
        queue_count=peak_queue_count,
        queue_bytes=peak_queue_bytes,
        rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        cpu_utilization_percent=round(cpu_seconds / elapsed / (available_cpus() or 1) * 100, 2),
        writer_hold_ms=round(writer_hold_seconds * 1000, 3),
        cancelled=cancelled,
        background_progress=len(durations),
    )


async def _completed_operation() -> None:
    return None


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    require_free_threaded_runtime(consumer="managed concurrency profile")
    cpu = available_cpus() or 2
    worker_profiles = (1, max(1, min(4, cpu - 1)), max(1, min(8, cpu - 2)))
    workloads = (
        ("tiny-file", 8),
        ("ordinary", 24),
        ("whale", 64),
        ("mixed-ingest", 32),
        ("derivation", 20),
        ("interactive-read", 12),
    )
    results = [
        asdict(run_profile(workers=workers, admission_units=workers + 2, workload=workload, jobs=jobs))
        for workload, jobs in workloads
        for workers in worker_profiles
    ]
    payload = {
        "schema": "polylogue.managed-concurrency-profile.v1",
        "runtime": runtime_identity().to_dict(),
        "workloads": [name for name, _jobs in workloads],
        "results": results,
        "selected_default": {"workers": worker_profiles[-1], "admission_units": worker_profiles[-1] + 2},
        "rejected_configurations": ["unbounded queue", "worker count above CPU-derived bound"],
    }
    if "--json" in args:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(f"managed concurrency profile: {len(results)} measurements")
        for result in results:
            print(
                f"{result['workload']} workers={result['workers']} "
                f"throughput={result['throughput_per_second']:.1f}/s "
                f"p95={result['p95_ms']:.2f}ms rejected={result['rejected']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
