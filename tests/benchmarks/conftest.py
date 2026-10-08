"""Shared fixtures for provider-native benchmark archive tiers.

The benchmark tiers are semantic workload projections. They use the same
content-addressed real-pipeline artifacts as verification fixtures, then clone
one private writable archive for each benchmark fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.infra.benchmark_archives import seed_benchmark_archive
from tests.infra.workload_declarations import BenchmarkWorkloadTier


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-cost-model-full",
        action="store_true",
        default=False,
        help="Run the full stratified rebuild-cost projection (real rebuild passes, ~minutes).",
    )
    parser.addoption(
        "--run-heavy-benchmarks",
        action="store_true",
        default=False,
        help="Run benchmarks marked heavy_benchmark (genuine scale workloads; minutes to hours).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep scale workloads out of every selection that did not ask for them.

    Selecting a benchmark module runs its ordinary tiers. Its heavy tiers are
    a separate lane: they carry a deadline sized to their declared workload
    and run only under ``--run-heavy-benchmarks``.
    """
    if config.getoption("--run-heavy-benchmarks"):
        return
    skip = pytest.mark.skip(reason="heavy scale workload; run with --run-heavy-benchmarks")
    for item in items:
        if item.get_closest_marker("heavy_benchmark") is not None:
            item.add_marker(skip)


def _benchmark_db(tmp_path_factory: pytest.TempPathFactory, *, tier: BenchmarkWorkloadTier) -> Path:
    tier_root = tmp_path_factory.mktemp(f"bench-{tier.value}")
    # Keep the clone helper's sealed ancestor inside the fixture-owned tier
    # directory. The pytest basetemp root may be a restricted tmpfs mount and
    # is owned by the harness rather than this fixture.
    db_path = tier_root / "archive" / "benchmark.db"
    stats = seed_benchmark_archive(db_path, tier)
    print(f"\nbenchmark {tier.value}: {stats}")
    return db_path


@pytest.fixture(scope="session")
def bench_db_1k(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Smoke projection for fixture lifecycle and small-scale probes."""
    return _benchmark_db(tmp_path_factory, tier=BenchmarkWorkloadTier.SMOKE)


@pytest.fixture(scope="session")
def bench_db_5k(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Representative production-shaped benchmark projection."""
    return _benchmark_db(tmp_path_factory, tier=BenchmarkWorkloadTier.REPRESENTATIVE)


@pytest.fixture(scope="session")
def bench_db_10k(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Archive-scale benchmark projection."""
    return _benchmark_db(tmp_path_factory, tier=BenchmarkWorkloadTier.ARCHIVE_SCALE)


@pytest.fixture(scope="session")
def bench_db_50k(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Explicit stress projection for resource-envelope probes."""
    return _benchmark_db(tmp_path_factory, tier=BenchmarkWorkloadTier.STRESS)
