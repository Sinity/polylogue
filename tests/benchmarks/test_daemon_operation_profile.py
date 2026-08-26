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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import pytest

from polylogue.cli.daemon_client import DaemonClient
from tests.benchmarks.cli_profile import INTERACTION_WORKLOADS, record_metrics
from tests.benchmarks.helpers import BenchmarkFixture, benchmark_one_shot

pytest_plugins = ("tests.benchmarks.test_daemon_uds",)

pytestmark = pytest.mark.uses_real_clock(
    "The profile waits for a real AF_UNIX server and measures installed-process wall-clock behavior."
)


def _installed_cli() -> list[str]:
    executable = Path(__file__).parents[2] / ".venv" / "bin" / "polylogue"
    return [str(executable)] if executable.is_file() else [sys.executable, "-m", "polylogue"]


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
            full_render_ms=elapsed_ms,
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
        def one() -> dict[str, object]:
            client = DaemonClient(socket_path, timeout_s=2)
            result = _operation(client, "cli.query", {"params": {"limit": 5}})
            elapsed.append(client.last_elapsed_ms or 0)
            return result

        with ThreadPoolExecutor(max_workers=4) as pool:
            return list(pool.map(lambda _index: one(), range(4)))

    results = benchmark_one_shot(benchmark, run)
    assert len(results) == 4
    assert all(result["error"] is None for result in results)
    record_metrics(benchmark, concurrent_interference_p95_ms=max(elapsed, default=0))


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
