"""Regression benchmark for polylogue-dcz5: prove the free-threaded parse win.

polylogue-dcz5's deploy phase existed to unlock ``ThreadPoolExecutor`` census
parse (``_parse_unique_retained_raws`` in
``polylogue.sources.revision_backfill``, gated by
``parallel_threads_effective()`` in ``polylogue.pipeline.services.process_pool``)
on a genuinely free-threaded (no-GIL) interpreter. The gate that used to keep
this path from ever running in production
(``daemon_parse_stage_split``, a config flag defaulted off) was deleted in
``ef8a4c3d0`` -- ``_maybe_warm_raw_materialization_parse_stage`` now always
attempts the off-writer-hold warm; a GIL build or a warm failure degrades to
the unmodified sequential in-hold parse. So the remaining, previously-unmet
acceptance criterion is a *measurement*: does thread-parallel parse actually
win on the interpreter this process is running, using the exact dispatch
function the daemon calls (not a synthetic ThreadPoolExecutor toy)?

This benchmark calls ``_parse_unique_retained_raws`` itself at
``ingest_workers=1`` (forces the sequential branch) and at
``ingest_workers=N`` (takes the thread-pool branch iff
``parallel_threads_effective()`` is true) against an identical synthetic
Codex raw corpus, and records the wall-clock ratio. It is diagnostic, not a
strict pass/fail gate on a specific ratio -- host core count and interpreter
build vary -- but it DOES assert that parallel dispatch is never slower than
sequential by more than noise, which would indicate the GIL-safety guard
itself regressed (the one correctness property that must never flip).

Run with:
    pytest tests/benchmarks/test_parse_stage_thread_scaling.py --benchmark-enable -p no:xdist -v
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from polylogue.pipeline.services.process_pool import parallel_threads_effective
from polylogue.sources.revision_backfill import _parse_unique_retained_raws
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.revision_backfill_benchmark import build_independent_raw_corpus

# A larger, more CPU-bound population than SMALL_PAYLOAD_SHAPE/LARGE_PAYLOAD_SHAPE:
# many small-to-medium raws so JSON-decode + ParsedSession construction
# (the actual CPU-bound work under test) dominates over fixed per-call
# overhead, without making the benchmark itself slow.
_RAW_COUNT = 240
_AVG_PAYLOAD_BYTES = 80_000


def _time_parse(archive_root: Path, raw_ids: list[str], *, ingest_workers: int) -> float:
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        descriptors = {
            raw_id: (*archive.raw_revision_descriptor(raw_id), archive.raw_native_id(raw_id)) for raw_id in raw_ids
        }
        started = time.perf_counter()
        results = _parse_unique_retained_raws(archive, raw_ids, descriptors=descriptors, ingest_workers=ingest_workers)
        elapsed = time.perf_counter() - started
    for raw_id, outcome in results.items():
        assert not isinstance(outcome, Exception), f"{raw_id} failed to parse: {outcome}"
    return elapsed


@pytest.mark.benchmark
def test_parse_stage_thread_scaling(tmp_path: Path) -> None:
    """Measure sequential vs thread-parallel census parse wall time on this
    process's actual interpreter build, using the real daemon dispatch
    function end to end (not a synthetic executor)."""
    is_free_threaded = parallel_threads_effective()
    workers = min(16, (__import__("os").cpu_count() or 2) - 2) or 1

    seq_root = tmp_path / "sequential"
    raw_ids = build_independent_raw_corpus(seq_root, raw_count=_RAW_COUNT, avg_payload_bytes=_AVG_PAYLOAD_BYTES)
    sequential_seconds = _time_parse(seq_root, raw_ids, ingest_workers=1)

    par_root = tmp_path / "parallel"
    raw_ids_2 = build_independent_raw_corpus(par_root, raw_count=_RAW_COUNT, avg_payload_bytes=_AVG_PAYLOAD_BYTES)
    assert raw_ids_2 == raw_ids, "corpus builder must be deterministic for a fair before/after comparison"
    parallel_seconds = _time_parse(par_root, raw_ids, ingest_workers=workers)

    speedup = sequential_seconds / max(parallel_seconds, 1e-9)
    print(
        f"\nparse-stage thread scaling (interpreter={sys.version.split()[0]}, "
        f"free_threaded={is_free_threaded}, workers={workers}, "
        f"raw_count={_RAW_COUNT}, avg_payload_bytes={_AVG_PAYLOAD_BYTES}): "
        f"sequential={sequential_seconds:.4f}s, parallel={parallel_seconds:.4f}s, speedup={speedup:.2f}x"
    )

    # The one correctness property that must never regress: threaded dispatch
    # must never be dramatically slower than sequential. On a GIL build this
    # ratio hovers near 1.0 (0.93x-0.96x was the polylogue-7mtf control-run
    # finding -- thread overhead with no parallel win, not a regression). On
    # a genuinely free-threaded build it should show a real multi-x win. Both
    # cases pass this floor; only a GIL-mistaken-for-free-threaded dispatch
    # (which would reintroduce the ~5000x writer-latency hazard this whole
    # gate exists to prevent) would plausibly show catastrophic slowdown here.
    assert parallel_seconds < sequential_seconds * 1.5, (
        f"threaded parse dispatch was slower than sequential by more than noise "
        f"(sequential={sequential_seconds:.4f}s, parallel={parallel_seconds:.4f}s) -- "
        "if free_threaded=True this is unexpected and worth investigating before trusting "
        "the daemon's off-writer-hold warm to actually help."
    )
    if is_free_threaded:
        assert speedup > 1.5, (
            f"expected a genuine free-threaded speedup (was {speedup:.2f}x, "
            f"sequential={sequential_seconds:.4f}s, parallel={parallel_seconds:.4f}s) -- "
            "polylogue-7mtf's control run measured 3.9x-9.6x at w=4..16; a near-1x ratio "
            "on a free-threaded interpreter would mean the parse workload is not actually "
            "running in parallel (e.g. an accidental global lock reintroduced upstream)."
        )
