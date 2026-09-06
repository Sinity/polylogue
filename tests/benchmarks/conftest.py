"""Shared fixtures for provider-native benchmark archive tiers.

The benchmark tiers are semantic workload projections. They use the same
content-addressed real-pipeline artifacts as verification fixtures, then clone
one private writable archive for each benchmark fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.infra.benchmark_archives import seed_benchmark_archive
from tests.infra.workload_artifacts import benchmark_workload_tier


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-cost-model-full",
        action="store_true",
        default=False,
        help="Run the full stratified rebuild-cost projection (real rebuild passes, ~minutes).",
    )


def _benchmark_db(tmp_path_factory: pytest.TempPathFactory, *, target_messages: int) -> Path:
    tier = benchmark_workload_tier(target_messages)
    tier_root = tmp_path_factory.mktemp(f"bench-{tier.value}")
    # Keep the clone helper's sealed ancestor inside the fixture-owned tier
    # directory. The pytest basetemp root may be a restricted tmpfs mount and
    # is owned by the harness rather than this fixture.
    db_path = tier_root / "archive" / "benchmark.db"
    stats = seed_benchmark_archive(db_path, target_messages=target_messages)
    print(f"\nbenchmark {tier.value}: {stats}")
    return db_path


@pytest.fixture(scope="session")
def bench_db_1k(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Smoke projection for fixture lifecycle and small-scale probes."""
    return _benchmark_db(tmp_path_factory, target_messages=1_000)


@pytest.fixture(scope="session")
def bench_db_5k(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Representative production-shaped benchmark projection."""
    return _benchmark_db(tmp_path_factory, target_messages=5_000)


@pytest.fixture(scope="session")
def bench_db_10k(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Archive-scale benchmark projection."""
    return _benchmark_db(tmp_path_factory, target_messages=10_000)


@pytest.fixture(scope="session")
def bench_db_50k(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Explicit stress projection for resource-envelope probes."""
    return _benchmark_db(tmp_path_factory, target_messages=50_000)
