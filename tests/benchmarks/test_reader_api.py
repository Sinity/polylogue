"""Reader API benchmark tests.

Covers: daemon HTTP endpoint latency on synthetic data
  - GET /api/sessions       (list summaries)
  - GET /api/sessions/{id}  (session detail)
  - GET /api/facets              (aggregated provider/tag counts)
  - GET /api/status              (archive-level stats)
  - Context image assembly        (future surface — placeholder)
  - Cost rollup                  (future surface — placeholder)

Run with:
    pytest tests/benchmarks/test_reader_api.py --benchmark-enable -p no:xdist -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.benchmarks.helpers import BenchmarkFixture, benchmark_store_call


def _first_conv_id(db_path: Path) -> str:
    """Resolve the first session ID from the seeded benchmark DB."""
    from tests.benchmarks.helpers import open_bench_store

    with open_bench_store(db_path) as store:
        summaries = store.run(store.repository.list_summaries())
        assert summaries, "Seeded benchmark DB has no sessions"
        return str(summaries[0].id)


# ---------------------------------------------------------------------------
# Session listing (GET /api/sessions)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_reader_list_sessions(
    benchmark: BenchmarkFixture,
    bench_db_5k: Path,
) -> None:
    """Benchmark listing session summaries through the filter.

    Matches: GET /api/sessions — returns list-of-summary envelope.
    """
    benchmark_store_call(
        benchmark,
        bench_db_5k,
        lambda store: store.repository.list_summaries(),
    )


@pytest.mark.benchmark
def test_bench_reader_list_with_provider(
    benchmark: BenchmarkFixture,
    bench_db_5k: Path,
) -> None:
    """Benchmark listing sessions filtered by provider."""
    benchmark_store_call(
        benchmark,
        bench_db_5k,
        lambda store: store.repository.list_summaries(origin="claude-code-session"),
    )


# ---------------------------------------------------------------------------
# Session detail (GET /api/sessions/{id})
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_reader_get_session(
    benchmark: BenchmarkFixture,
    bench_db_5k: Path,
) -> None:
    """Benchmark fetching a single session with full messages.

    Matches: GET /api/sessions/{id} — returns session + all messages.
    """
    conv_id = _first_conv_id(bench_db_5k)

    def _get_conv(db_path: Path) -> None:
        from tests.benchmarks.helpers import open_bench_store

        with open_bench_store(db_path) as store:
            conv = store.run(store.repository.get(conv_id))
            assert conv is not None, f"Session {conv_id} not found"
            assert len(conv.messages) > 0

    benchmark(lambda: _get_conv(bench_db_5k))


# ---------------------------------------------------------------------------
# Facets (GET /api/facets)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_reader_facets(
    benchmark: BenchmarkFixture,
    bench_db_5k: Path,
) -> None:
    """Benchmark facet aggregation (provider counts + archive stats).

    Matches: GET /api/facets — archive-level provider and message-type rollups.
    """
    benchmark_store_call(
        benchmark,
        bench_db_5k,
        lambda store: store.repository.aggregate_message_stats(),
    )


# ---------------------------------------------------------------------------
# Status (GET /api/status)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_reader_status(
    benchmark: BenchmarkFixture,
    bench_db_5k: Path,
) -> None:
    """Benchmark archive status snapshot (counts, provider breakdown).

    Matches: GET /api/status — light archive-wide metadata query.
    """
    benchmark_store_call(
        benchmark,
        bench_db_5k,
        lambda store: store.repository.count(),
    )


# ---------------------------------------------------------------------------
# Context image assembly (future reader surface)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_reader_context_image(
    benchmark: BenchmarkFixture,
    bench_db_5k: Path,
) -> None:
    """Benchmark assembling a minimal context image for a session.

    Placeholder for the context-image reader surface.  Uses the same
    get-session path as the detail endpoint today.
    """
    conv_id = _first_conv_id(bench_db_5k)

    def _context_image(db_path: Path) -> None:
        from tests.benchmarks.helpers import open_bench_store

        with open_bench_store(db_path) as store:
            conv = store.run(store.repository.get(conv_id))
            assert conv is not None
            # A real context image would assemble messages + insights + neighbors.
            # For now this measures the core fetch cost.
            _ = (conv.id, len(conv.messages), conv.word_count)

    benchmark(lambda: _context_image(bench_db_5k))


# ---------------------------------------------------------------------------
# Cost rollup (future reader surface)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_reader_cost_rollup(
    benchmark: BenchmarkFixture,
    bench_db_5k: Path,
) -> None:
    """Benchmark computing a cost rollup over sessions.

    Placeholder for the cost-rollup reader surface.  Uses provider-scoped
    stats as the nearest available query today.
    """
    benchmark_store_call(
        benchmark,
        bench_db_5k,
        lambda store: store.repository.get_stats_by("origin"),
    )
