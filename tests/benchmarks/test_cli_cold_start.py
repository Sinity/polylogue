"""Cold-CLI status latency benchmark (polylogue-8s70 / polylogue-20d.14 / polylogue-jtwu).

Covers: the ``polylogue status`` cold-subprocess path with no daemon reachable
-- one whole Python process invocation, import tax included -- against a
minimal (empty, ops-tier-only) archive. Status is daemon-served, so this path
ends in the typed unavailable status snapshot (exit 1); this is the
"interactive" SLO tier's ``cli_status_cold`` surface and the cost
polylogue-8s70's cProfile/importtime investigation targets.

Run with:
    pytest tests/benchmarks/test_cli_cold_start.py --benchmark-enable -p no:xdist -v
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import pytest

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.benchmarks.cli_profile import record_metrics
from tests.benchmarks.helpers import BenchmarkFixture


@pytest.fixture(scope="session")
def bench_cli_cold_start_archive_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Empty archive_root with only the ops tier initialized.

    Deliberately empty: this surface benchmarks the fixed cold-subprocess
    import/dispatch cost, not query cost over real data (the ``query``/
    ``reader``/``facets`` surfaces already cover that).
    """
    archive_root = tmp_path_factory.mktemp("bench-cli-cold-start") / "archive"
    archive_root.mkdir()
    initialize_archive_database(archive_root / "ops.db", ArchiveTier.OPS)
    return archive_root


@pytest.mark.benchmark(group="cli-cold-start")
def test_bench_cli_status_cold(
    benchmark: BenchmarkFixture,
    bench_cli_cold_start_archive_root: Path,
) -> None:
    env = dict(os.environ)
    env["POLYLOGUE_ARCHIVE_ROOT"] = str(bench_cli_cold_start_archive_root)
    env["POLYLOGUE_FORCE_PLAIN"] = "1"
    # Import census is an explicit profile mode; it is opt-in so ordinary
    # latency samples remain representative of the installed route.
    if os.environ.get("POLYLOGUE_BENCH_IMPORTTIME"):
        env["PYTHONPROFILEIMPORTTIME"] = "1"

    def _invoke() -> int:
        before_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        started = perf_counter()
        result = subprocess.run(
            [sys.executable, "-m", "polylogue", "status", "--format", "json"],
            env=env,
            capture_output=True,
            timeout=30,
        )
        elapsed_ms = (perf_counter() - started) * 1000
        # No daemon serves this empty root, so status answers with its typed
        # unavailable snapshot and exits 1; its reason (``daemon_absent`` or
        # ``status_read_failed``) depends on what the environment's daemon
        # probe finds. A crash or a non-JSON answer is not this lane's path.
        assert result.returncode == 1, result.stderr.decode(errors="replace")
        payload = json.loads(result.stdout)
        assert payload["ok"] is False, payload
        assert payload["status_snapshot"]["state"] == "unavailable", payload
        imported_modules = result.stderr.count(b"import time:")
        record_metrics(
            benchmark,
            cold_start_ms=elapsed_ms,
            peak_rss_kib=max(0, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss - before_rss),
            imported_modules=imported_modules,
            bytes=len(result.stdout),
            rows=result.stdout.count(b"\n"),
        )
        return result.returncode

    benchmark(_invoke)
