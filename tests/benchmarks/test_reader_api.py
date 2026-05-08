"""Reader API benchmark tests.

Covers: daemon HTTP endpoint latency on synthetic data
  - GET /api/conversations       (list summaries)
  - GET /api/conversations/{id}  (conversation detail)
  - GET /api/facets              (aggregated provider/tag counts)
  - GET /api/status              (archive-level stats)
  - Context pack assembly        (future surface — placeholder)
  - Cost rollup                  (future surface — placeholder)

Run with:
    pytest tests/benchmarks/test_reader_api.py --benchmark-enable -p no:xdist -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.benchmarks.helpers import BenchmarkFixture, benchmark_store_call


def _first_conv_id(db_path: Path) -> str:
    """Resolve the first conversation ID from the seeded benchmark DB."""
    from tests.benchmarks.helpers import open_bench_store

    with open_bench_store(db_path) as store:
        summaries = store.run(store.repository.filter().list_summaries())
        assert summaries, "Seeded benchmark DB has no conversations"
        return str(summaries[0].id)


# ---------------------------------------------------------------------------
# Conversation listing (GET /api/conversations)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_reader_list_conversations(
    benchmark: BenchmarkFixture,
    bench_db_5k: Path,
) -> None:
    """Benchmark listing conversation summaries through the filter.

    Matches: GET /api/conversations — returns list-of-summary envelope.
    """
    benchmark_store_call(
        benchmark,
        bench_db_5k,
        lambda store: store.repository.filter().list_summaries(),
    )


@pytest.mark.benchmark
def test_bench_reader_list_with_provider(
    benchmark: BenchmarkFixture,
    bench_db_5k: Path,
) -> None:
    """Benchmark listing conversations filtered by provider."""
    benchmark_store_call(
        benchmark,
        bench_db_5k,
        lambda store: store.repository.filter().provider("claude-code").list_summaries(),
    )


# ---------------------------------------------------------------------------
# Conversation detail (GET /api/conversations/{id})
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_reader_get_conversation(
    benchmark: BenchmarkFixture,
    bench_db_5k: Path,
) -> None:
    """Benchmark fetching a single conversation with full messages.

    Matches: GET /api/conversations/{id} — returns conversation + all messages.
    """
    conv_id = _first_conv_id(bench_db_5k)

    def _get_conv(db_path: Path) -> None:
        from tests.benchmarks.helpers import open_bench_store

        with open_bench_store(db_path) as store:
            conv = store.run(store.repository.get(conv_id))
            assert conv is not None, f"Conversation {conv_id} not found"
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
# Context pack assembly (future reader surface)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_reader_context_pack(
    benchmark: BenchmarkFixture,
    bench_db_5k: Path,
) -> None:
    """Benchmark assembling a minimal context pack for a session.

    Placeholder for the context-pack reader surface.  Uses the same
    get-conversation path as the detail endpoint today.
    """
    conv_id = _first_conv_id(bench_db_5k)

    def _context_pack(db_path: Path) -> None:
        from tests.benchmarks.helpers import open_bench_store

        with open_bench_store(db_path) as store:
            conv = store.run(store.repository.get(conv_id))
            assert conv is not None
            # A real context pack would assemble messages + insights + neighbors.
            # For now this measures the core fetch cost.
            _ = (conv.id, len(conv.messages), conv.word_count)

    benchmark(lambda: _context_pack(bench_db_5k))


# ---------------------------------------------------------------------------
# Cost rollup (future reader surface)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_reader_cost_rollup(
    benchmark: BenchmarkFixture,
    bench_db_5k: Path,
) -> None:
    """Benchmark computing a cost rollup over conversations.

    Placeholder for the cost-rollup reader surface.  Uses provider-scoped
    stats as the nearest available query today.
    """
    benchmark_store_call(
        benchmark,
        bench_db_5k,
        lambda store: store.repository.get_stats_by("provider"),
    )
