"""Cold-CLI status latency benchmark (polylogue-8s70 / polylogue-20d.14 / polylogue-jtwu).

Covers: the direct-fallback ``polylogue status`` cold-subprocess path -- one
whole Python process invocation, import tax included -- against a minimal
(empty, ops-tier-only) archive. This is the "interactive" SLO tier's
``cli_status_cold`` surface: no daemon reachable, so this measures the same
cost polylogue-8s70's own manual cProfile/importtime investigation targeted.

Run with:
    pytest tests/benchmarks/test_cli_cold_start.py --benchmark-enable -p no:xdist -v
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
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

    def _invoke() -> int:
        result = subprocess.run(
            [sys.executable, "-m", "polylogue", "status", "--format", "json"],
            env=env,
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        return result.returncode

    benchmark(_invoke)
