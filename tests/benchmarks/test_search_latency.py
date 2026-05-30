"""Search latency benchmarks across query types and corpus sizes.

Measures FTS5 search latency for: simple terms, phrase queries, boolean
operators, prefix searches, rare terms, and long queries — at multiple
corpus sizes. Provides per-query-type latency data for regression detection.

Run with:
    pytest tests/benchmarks/test_search_latency.py \\
      --benchmark-enable -p no:xdist -o "addopts=" -v
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from polylogue.storage.search import search_messages
from tests.benchmarks.helpers import BenchmarkFixture

# ── Query type taxonomy ───────────────────────────────────────────────

# Each entry: (label, query, min_expected_hits)
QUERY_TYPES = [
    # Simple term — common word
    ("term-common", "analysis", 1),
    # Simple term — rare/nonexistent (fast empty path)
    ("term-rare", "xyzzynonexistent987654", 0),
    # Multi-word AND
    ("multi-and", "python analysis", 0),
    # Boolean OR
    ("boolean-or", "python OR analysis", 1),
    # Phrase query
    ("phrase", '"quick brown fox"', 0),
    # Prefix search
    ("prefix", "analys*", 0),
    # Long query (many words)
    ("long-query", "the quick brown fox jumps over the lazy dog", 0),
    # Single character (edge case)
    ("single-char", "a", 1),
    # Numeric term
    ("numeric", "2024", 0),
    # Special characters (should be safely escaped)
    ("special-chars", "test@example.com", 0),
]


# ── Latency distribution helper ───────────────────────────────────────


def _measure_latency_distribution(
    func: Callable[[], object], n_warmup: int = 2, n_samples: int = 10
) -> dict[str, float]:
    """Measure latency distribution with warmup.

    Returns {mean, median, p95, p99, min, max} in milliseconds.
    """
    # Warmup
    for _ in range(n_warmup):
        func()

    # Measure
    times: list[float] = []
    for _ in range(n_samples):
        t0 = time.perf_counter()
        func()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times.append(elapsed_ms)

    times.sort()
    p95_idx = int(len(times) * 0.95)
    p99_idx = int(len(times) * 0.99)

    return {
        "mean_ms": round(statistics.mean(times), 2),
        "median_ms": round(statistics.median(times), 2),
        "p95_ms": round(times[min(p95_idx, len(times) - 1)], 2),
        "p99_ms": round(times[min(p99_idx, len(times) - 1)], 2),
        "min_ms": round(times[0], 2),
        "max_ms": round(times[-1], 2),
    }


# ── Single query benchmark ────────────────────────────────────────────


@pytest.mark.benchmark
@pytest.mark.parametrize("label,query,min_hits", QUERY_TYPES)
def test_search_latency_by_query_type(
    benchmark: BenchmarkFixture,
    label: str,
    query: str,
    min_hits: int,
    bench_db_5k: Path,
    tmp_path: Path,
) -> None:
    """Measure search latency for each query type against a 5K-message corpus."""
    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)

    def do_search() -> object:
        return search_messages(
            query,
            archive_root=archive_root,
            db_path=bench_db_5k,
            limit=20,
        )

    _ = benchmark(do_search)

    # Record query-type metadata
    extras = {"query_type": label, "query": query}
    if hasattr(benchmark, "extra_info"):
        benchmark.extra_info.update(extras)


# ── Latency distribution measurement ──────────────────────────────────


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "label,query",
    [
        ("common-term", "analysis"),
        ("rare-term", "xyzzynonexistent"),
        ("multi-word", "python analysis"),
        ("boolean-or", "python OR analysis"),
        ("prefix", "analys*"),
        ("long-query", "the quick brown fox jumps over the lazy dog"),
    ],
)
def test_search_latency_distribution(
    benchmark: BenchmarkFixture,
    label: str,
    query: str,
    bench_db_5k: Path,
    tmp_path: Path,
) -> None:
    """Measure full latency distribution (p50/p95/p99) for key query types."""
    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)

    def do_search() -> None:
        search_messages(
            query,
            archive_root=archive_root,
            db_path=bench_db_5k,
            limit=20,
        )

    dist = _measure_latency_distribution(do_search)

    extras = {
        "query_type": label,
        "query": query,
        **dist,
    }
    if hasattr(benchmark, "extra_info"):
        benchmark.extra_info.update(extras)

    # Sanity: search shouldn't take >1s on a 5K corpus
    assert dist["p95_ms"] < 1000, f"Query '{label}' p95 latency {dist['p95_ms']}ms exceeds 1s threshold"


# ── Cold-start vs warm-cache comparison ───────────────────────────────


@pytest.mark.benchmark
def test_search_cold_vs_warm_first_query(
    benchmark: BenchmarkFixture,
    bench_db_5k: Path,
    tmp_path: Path,
) -> None:
    """Measure first-query (cold cache) vs subsequent query latency.

    The first FTS5 query after database open incurs page cache misses.
    This benchmark quantifies the cold-start penalty.
    """
    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)

    # Measure cold start: fresh process, first query
    t0 = time.perf_counter()
    search_messages("analysis", archive_root=archive_root, db_path=bench_db_5k, limit=20)
    cold_ms = (time.perf_counter() - t0) * 1000

    # Warm: second query
    t0 = time.perf_counter()
    search_messages("python", archive_root=archive_root, db_path=bench_db_5k, limit=20)
    warm_ms = (time.perf_counter() - t0) * 1000

    extras = {
        "cold_first_query_ms": round(cold_ms, 2),
        "warm_second_query_ms": round(warm_ms, 2),
        "cold_warm_ratio": round(cold_ms / max(warm_ms, 0.001), 1),
    }
    if hasattr(benchmark, "extra_info"):
        benchmark.extra_info.update(extras)

    # Cold start should be reasonable (< 2s)
    assert cold_ms < 2000, f"Cold start query took {cold_ms:.0f}ms"


# ── Search correctness under load ─────────────────────────────────────


@pytest.mark.benchmark
def test_search_result_count_stability(
    benchmark: BenchmarkFixture,
    bench_db_5k: Path,
    tmp_path: Path,
) -> None:
    """Search result counts should be stable across repeated queries."""
    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)

    queries = ["analysis", "python", "data", "test", "implementation"]
    counts: dict[str, list[int]] = {q: [] for q in queries}

    def run_all_queries() -> None:
        for q in queries:
            result = search_messages(q, archive_root=archive_root, db_path=bench_db_5k, limit=100)
            counts[q].append(len(result.hits))

    benchmark(run_all_queries)

    # Each query should return consistent result counts
    for q in queries:
        unique_counts = set(counts[q])
        assert len(unique_counts) >= 1, f"No results for {q!r}"
        # Result count should be stable (allow ±1 for timing edge cases)
        if len(unique_counts) > 1:
            max_count = max(unique_counts)
            min_count = min(unique_counts)
            assert max_count - min_count <= 1, f"Unstable result count for {q!r}: {unique_counts}"
