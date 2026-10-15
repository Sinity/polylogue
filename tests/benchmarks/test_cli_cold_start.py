"""Cold subprocess status against real resident EMPTY and typed absence controls.

Timing samples never enable profiling. EMPTY serving is not healthy materialization.
The separate seeded resident component retains healthy-frame and fresh-cache controls.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from time import perf_counter
from typing import Any

import pytest
from watchfiles import watch

from devtools.isolated_environment import isolated_home_environment
from polylogue.operations.demo_resident import demo_resident
from polylogue.readiness.capability import (
    raw_frontier_integrity_is_proven_healthy,
    status_snapshot_has_fresh_provenance,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.benchmarks.cli_profile import record_metrics
from tests.benchmarks.helpers import benchmark_repeated
from tests.infra.benchmark_process import run_with_peak_rss
from tests.infra.cli_cold_start import EmptyResident
from tests.infra.workload_artifacts import build_seeded_archive
from tests.infra.workload_declarations import BenchmarkWorkloadTier, benchmark_corpus_specs


@pytest.fixture(scope="session")
def bench_cli_cold_start_archive_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An empty synthetic archive with its canonical disposable ops tier."""
    root = tmp_path_factory.mktemp("bench-cli-cold-start") / "archive"
    root.mkdir()
    initialize_archive_database(root / "ops.db", ArchiveTier.OPS)
    return root


@pytest.mark.uses_real_clock
@pytest.mark.benchmark(group="cli-cold-start")
@pytest.mark.parametrize("route", ["status", "agents", "daemon"])
def test_bench_cli_status_unavailable_cold(benchmark: Any, bench_cli_cold_start_archive_root: Path, route: str) -> None:
    root = bench_cli_cold_start_archive_root
    home = root.parent / "isolated-home"
    home.mkdir(exist_ok=True)
    env = isolated_home_environment(os.environ, home=home)
    env["POLYLOGUE_ARCHIVE_ROOT"] = str(root)
    env["POLYLOGUE_FORCE_PLAIN"] = "1"
    env.pop("POLYLOGUE_DAEMON_URL", None)
    env.pop("POLYLOGUE_DAEMON_MODE", None)
    env.pop("PYTHONPROFILEIMPORTTIME", None)
    # The test worker has its own disposable bytecode prefix. Measure the
    # checkout entrypoint's cache instead, without purging or prewarming it.
    cache = Path(__file__).resolve().parents[2] / ".cache" / "pycache"
    cache.mkdir(parents=True, exist_ok=True)
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    env["PYTHONPYCACHEPREFIX"] = str(cache)
    executable = Path(sys.executable).parent / ("polylogued" if route == "daemon" else "polylogue")
    args = [str(executable), *(["agents", "status"] if route == "agents" else ["status"]), "--format", "json"]
    samples: list[float] = []

    def invoke() -> int:
        started = perf_counter()
        result, peak_rss_kib = run_with_peak_rss(args, env=env, cwd=home)
        elapsed_ms = (perf_counter() - started) * 1000
        payload = json.loads(result.stdout)
        assert result.returncode == (0 if route == "agents" else 1), result.stderr.decode(errors="replace")
        if route == "agents":
            assert payload["view"] == "status"
            assert isinstance(payload["peers"], list)
            assert isinstance(payload["projection"], dict)
        else:
            assert payload["ok"] is False
            assert payload["daemon_liveness"] is False
            assert payload["status_snapshot"]["state"] == "unavailable"
            assert payload["status_snapshot"]["reason"] == "daemon_absent"
        samples.append(elapsed_ms)
        record_metrics(
            benchmark,
            cold_start_ms=elapsed_ms,
            peak_rss_kib=peak_rss_kib,
            bytes=len(result.stdout),
            rows=result.stdout.count(b"\n"),
        )
        return result.returncode

    benchmark_repeated(benchmark, invoke, rounds=5)
    benchmark.extra_info.update(
        {
            "route": route,
            "cold_samples_ms": samples,
            "launch": args,
            "cwd": str(home),
            "archive_root": str(root),
            "interpreter": sys.executable,
            "daemon": "absent",
            "rounds": 5,
            "profiling": False,
            "bytecode_writes_disabled": "PYTHONDONTWRITEBYTECODE" in env,
            "bytecode_cache_prefix": env["PYTHONPYCACHEPREFIX"],
            "cache_admission": "existing cache; five sequential launches; no prewarm or purge",
        }
    )


@pytest.fixture(scope="session")
def bench_empty_status_resident() -> Iterator[EmptyResident]:
    from tests.infra.cli_cold_start import empty_status_resident

    artifacts = Path(__file__).resolve().parents[2] / ".cache" / "verify" / "cold-status"
    artifacts.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="resident-", dir=artifacts))
    with empty_status_resident(directory) as resident:
        yield resident


@pytest.mark.uses_real_clock
@pytest.mark.benchmark(group="cli-cold-start")
@pytest.mark.parametrize("route", ["status", "agents", "daemon"])
def test_bench_cli_status_cold(benchmark: Any, bench_empty_status_resident: EmptyResident, route: str) -> None:
    from tests.infra.cli_cold_start import assert_served_empty_status

    resident = bench_empty_status_resident
    env = resident.client_environment(route)
    executable = Path(sys.executable).parent / ("polylogued" if route == "daemon" else "polylogue")
    args = [str(executable), *(["agents", "status"] if route == "agents" else ["status"]), "--format", "json"]
    samples: list[float] = []
    attempts: list[tuple[subprocess.CompletedProcess[bytes], float, int]] = []

    def invoke() -> int:
        started = perf_counter()
        result, peak_rss_kib = run_with_peak_rss(args, env=env, cwd=resident.home)
        elapsed_ms = (perf_counter() - started) * 1000
        attempts.append((result, elapsed_ms, peak_rss_kib))
        return result.returncode

    try:
        benchmark_repeated(benchmark, invoke, rounds=5)
    finally:
        # Preserve every returned response, including when a later launch fails.
        # Fixture I/O and validation must not count as command startup time.
        for attempt, (result, elapsed_ms, peak_rss_kib) in enumerate(attempts, start=1):
            (resident.home.parent / f"{route}-attempt-{attempt}.stdout.json").write_bytes(result.stdout)
            (resident.home.parent / f"{route}-attempt-{attempt}.stderr.log").write_bytes(result.stderr)
            (resident.home.parent / f"{route}-attempt-{attempt}.receipt.json").write_text(
                json.dumps(
                    {
                        "elapsed_ms": elapsed_ms,
                        "peak_rss_kib": peak_rss_kib,
                        "returncode": result.returncode,
                        "profiling": False,
                    }
                )
                + "\n"
            )
    for result, elapsed_ms, peak_rss_kib in attempts:
        payload = json.loads(result.stdout)
        if route == "agents":
            assert result.returncode == 0, result.stderr.decode(errors="replace")
            assert payload["view"] == "status"
            assert isinstance(payload["peers"], list) and isinstance(payload["projection"], dict)
        else:
            assert result.returncode == 1, result.stderr.decode(errors="replace")
            assert_served_empty_status(payload, root=resident.root, daemon_pid=resident.daemon_pid)
        samples.append(elapsed_ms)
        record_metrics(
            benchmark,
            cold_start_ms=elapsed_ms,
            peak_rss_kib=peak_rss_kib,
            bytes=len(result.stdout),
            rows=result.stdout.count(b"\n"),
        )
    benchmark.extra_info.update(
        {
            "route": route,
            "cold_samples_ms": samples,
            "launch": args,
            "cwd": str(resident.home),
            "archive_root": str(resident.root),
            "interpreter": sys.executable,
            "daemon": "owned resident",
            "resident_provenance": resident.provenance,
            "rounds": len(samples),
            "profiling": False,
            "bytecode_cache_prefix": env["PYTHONPYCACHEPREFIX"],
            "cache_admission": "existing owned checkout cache; five sequential cold processes; no prewarm or purge",
            "status_semantics": "served authoritative EMPTY with rc1/ok=false; not healthy materialization",
        }
    )


@pytest.fixture
def bench_cli_resident_environment(
    bench_db_1k: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[dict[str, str], dict[str, object]]]:
    """Borrow the original resident against the canonical smoke artifact."""
    root = bench_db_1k.parent
    artifact = build_seeded_archive(benchmark_corpus_specs(BenchmarkWorkloadTier.SMOKE, seed=42))
    manifest = artifact.manifest
    workload: dict[str, object] = {
        "seed": 42,
        "tier": BenchmarkWorkloadTier.SMOKE.value,
        "artifact_key": manifest.key,
        "archive_id": manifest.archive_id,
        "profile_id": manifest.profile_id,
        "recipe_id": manifest.recipe_id,
        "source_semantics_id": manifest.source_semantics_id,
        "archive_schema_id": manifest.archive_schema_id,
        "row_counts": dict(manifest.resources.row_counts),
    }
    home = root.parent / "cold-command-home"
    home.mkdir(exist_ok=True)
    isolated = isolated_home_environment(os.environ, home=home)
    for key, value in isolated.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    with demo_resident(root) as env:
        env.pop("PYTHONPROFILEIMPORTTIME", None)
        env.pop("PYTHONDONTWRITEBYTECODE", None)
        env["PYTHONPYCACHEPREFIX"] = str(Path(__file__).resolve().parents[2] / ".cache" / "pycache")
        args = [str(Path(sys.executable).parent / "polylogue"), "status", "--format", "json"]
        # The listener proves this child's binds. Only the real status producer
        # can prove archive readiness; watch ticks merely schedule observations.
        with closing(watch(root, yield_on_timeout=True, rust_timeout=1000)) as changes:
            while True:
                result = subprocess.run(args, env=env, cwd=home, capture_output=True)
                payload = json.loads(result.stdout)
                if status_snapshot_has_fresh_provenance(payload):
                    # A fresh non-ok archive is a product boundary,
                    # not a startup wait that another tick can certify.
                    assert result.returncode == 0, payload
                    _require_healthy_status(payload)
                    break
                assert payload.get("daemon_liveness") is True, payload
                next(changes)
        workload["ready_status"] = {
            key: payload[key] for key in ("source", "sessions", "messages", "status_snapshot", "raw_frontier_integrity")
        }
        assert Path(payload["archive_root"]).resolve() == root.resolve()
        assert payload["messages"] == manifest.resources.row_counts["messages"] == 1000
        assert payload["sessions"] == manifest.resources.row_counts["sessions"] > 0
        yield env, workload


def _require_healthy_status(payload: dict[str, Any]) -> None:
    assert payload["source"] == "daemon", payload
    assert payload["ok"] is True, payload
    assert payload["daemon_liveness"] is True, payload
    assert status_snapshot_has_fresh_provenance(payload), payload
    snapshot = payload["status_snapshot"]
    assert isinstance(snapshot["frame"], str) and snapshot["frame"], payload
    assert snapshot["frame"] == snapshot["current_frame"], payload
    assert snapshot["frame_changed"] is False, payload
    assert raw_frontier_integrity_is_proven_healthy(payload.get("raw_frontier_integrity")), payload


@pytest.mark.uses_real_clock
@pytest.mark.timeout(0)
@pytest.mark.benchmark(group="cli-cold-start")
@pytest.mark.parametrize("route", ["status", "agents", "daemon"])
@pytest.mark.parametrize("cache_state", ["existing", "empty"])
def test_bench_cli_resident_cold(
    benchmark: Any,
    bench_cli_resident_environment: tuple[dict[str, str], dict[str, object]],
    tmp_path: Path,
    route: str,
    cache_state: str,
) -> None:
    environment, workload = bench_cli_resident_environment
    env = dict(environment)
    home = Path(env["HOME"])
    executable = Path(sys.executable).parent / ("polylogued" if route == "daemon" else "polylogue")
    args = [str(executable), *(["agents", "status"] if route == "agents" else ["status"]), "--format", "json"]
    samples: list[float] = []
    prefixes: list[str] = []
    outcomes: list[dict[str, object]] = []

    def invoke() -> int:
        if cache_state == "empty":
            # A unique unpopulated prefix for EACH launch, without purging the
            # checkout cache or relabeling the warmed confirmation rounds.
            prefix = tmp_path / "bytecode" / str(len(samples))
            prefix.mkdir(parents=True)
            env["PYTHONPYCACHEPREFIX"] = str(prefix)
        prefixes.append(env["PYTHONPYCACHEPREFIX"])
        started = perf_counter()
        result = subprocess.run(args, env=env, cwd=home, capture_output=True)
        elapsed_ms = (perf_counter() - started) * 1000
        attempt = len(samples) + 1
        (tmp_path / f"{route}-{cache_state}-{attempt}.stdout.json").write_bytes(result.stdout)
        (tmp_path / f"{route}-{cache_state}-{attempt}.stderr.log").write_bytes(result.stderr)
        payload = json.loads(result.stdout)
        assert result.returncode == 0, (payload, result.stderr.decode(errors="replace"))
        if route == "agents":
            assert payload["view"] == "status"
            assert isinstance(payload["peers"], list)
            assert isinstance(payload["projection"], dict)
        else:
            _require_healthy_status(payload)
        outcome: dict[str, object] = {"returncode": result.returncode}
        if route != "agents":
            outcome.update(
                {
                    "source": payload["source"],
                    "snapshot_state": payload["status_snapshot"]["state"],
                    "frame": payload["status_snapshot"]["frame"],
                    "current_frame": payload["status_snapshot"]["current_frame"],
                    "frontier_status": payload["raw_frontier_integrity"]["overall_status"],
                }
            )
        outcomes.append(outcome)
        samples.append(elapsed_ms)
        record_metrics(benchmark, cold_start_ms=elapsed_ms, bytes=len(result.stdout), rows=result.stdout.count(b"\n"))
        return result.returncode

    benchmark_repeated(benchmark, invoke, rounds=5)
    assert len(samples) == len(outcomes) == len(prefixes) == 5
    benchmark.extra_info.update(
        {
            "route": route,
            "cold_samples_ms": samples,
            "sample_outcomes": outcomes,
            "launch": args,
            "cwd": str(home),
            "archive_root": env["POLYLOGUE_ARCHIVE_ROOT"],
            "interpreter": sys.executable,
            "daemon": "original resident; canonical seeded smoke archive",
            "rounds": 5,
            "profiling": False,
            "bytecode_writes_disabled": False,
            "bytecode_cache_prefixes": prefixes,
            "cache_state": cache_state,
            "cache_admission": (
                "new empty prefix before every timed launch"
                if cache_state == "empty"
                else "existing checkout cache; no purge; readiness observed before timing"
            ),
            "daemon_startup_included": False,
            "workload": workload,
            "original_successful_empty_slo": "unmet; separate seeded component",
        }
    )
