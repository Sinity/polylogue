"""Five cold installed-process rounds for three public read commands.

Every route runs against one isolated daemon-absent workspace. Profiling is a
separate invocation; ordinary timing samples do not enable import tracing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import pytest

from devtools.isolated_environment import isolated_home_environment
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.benchmarks.cli_profile import record_metrics
from tests.benchmarks.helpers import benchmark_repeated


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
def test_bench_cli_status_cold(benchmark: Any, bench_cli_cold_start_archive_root: Path, route: str) -> None:
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
        result = subprocess.run(args, env=env, cwd=home, capture_output=True, timeout=30)
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
