"""Measure actual shared census parsing with one and several admission slots.

This synthetic benchmark uses the same immutable-descriptor route as census.
It remains a measurement selection; normal focused correctness runs do not
run it. Both variants use the selected free-threaded runtime and shared
compute owner, with no alternate GIL or process dispatch.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from polylogue.core.compute import compute_adapter
from polylogue.runtime import require_free_threaded_runtime
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
    """Measure one versus several admission slots on the supported runtime."""
    identity = require_free_threaded_runtime(consumer="census parse measurement")
    workers = compute_adapter().snapshot().by_class("incremental-background").ceiling_slots

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
        f"free_threaded={identity.free_threaded}, workers={workers}, "
        f"raw_count={_RAW_COUNT}, avg_payload_bytes={_AVG_PAYLOAD_BYTES}): "
        f"sequential={sequential_seconds:.4f}s, parallel={parallel_seconds:.4f}s, speedup={speedup:.2f}x"
    )

    assert parallel_seconds < sequential_seconds * 1.5, (
        f"shared census parse slowdown: one={sequential_seconds:.4f}s, several={parallel_seconds:.4f}s"
    )
    if workers > 1:
        assert speedup > 1.5, f"expected a free-threaded parse speedup: {speedup:.2f}x with {workers} shared slots"
