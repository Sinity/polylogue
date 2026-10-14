"""Measure current captured Raw parser admission with complete content identity."""

from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path
from time import perf_counter

import pytest

from devtools.measurement_receipts import emit_receipt
from polylogue.core.compute import BoundedComputeAdapter, compute_adapter
from polylogue.core.enums import ValidationMode
from polylogue.core.sql_settlement import retain_native_sql_lifetimes
from polylogue.pipeline.ids import session_revision_projection
from polylogue.runtime import require_free_threaded_runtime
from polylogue.storage.derived.raw import NeutralRawPreparation, RawObservationDerivation, _parse_captured_neutral
from tests.infra.live_ingest import prepared_live_convergence_owner
from tests.infra.revision_backfill_benchmark import build_independent_raw_corpus

_RAW_COUNT = 240
_PAYLOAD_BYTES = 80_000


async def _time_parse(
    root: Path, raw_ids: list[str], *, sequential: bool
) -> tuple[float, dict[str, tuple[str, ...]], int]:
    # Ordinary law fixtures default to one worker. Borrow the actual shared
    # daemon capacity here, because the measured window is the intention.
    async with prepared_live_convergence_owner(root, compute_adapter=compute_adapter()) as owner:
        adapter = owner._compute_adapter
        slots = adapter.snapshot().by_class("incremental-background").ceiling_slots

        def capture(compute: BoundedComputeAdapter) -> NeutralRawPreparation:
            page = RawObservationDerivation(root, compute_adapter=compute).capture_neutral_raws(raw_ids)
            assert page is not None
            assert set(page.captures) == set(raw_ids)
            return page

        page = await owner.run_convergence_sync("test.parse-scaling.capture", capture, adapter)

        def prepare(raw_id: str) -> tuple[str, tuple[str, ...]]:
            captured = page.captures[raw_id]
            provider = captured.descriptor[0]
            # This is the production closed-file parse body, with the captured
            # Source operands and immutable sidecar/schema context of its owner.
            from polylogue.storage.derived.raw import _neutral_artifact_key

            with retain_native_sql_lifetimes(Path(page.scratch_owner.name)):
                artifact = _parse_captured_neutral(
                    raw_id,
                    captured,
                    _neutral_artifact_key(raw_id, page.operands[raw_id], ValidationMode.ADVISORY),
                    page.contexts[provider],
                    Path(page.scratch_owner.name),
                )
                try:
                    assert artifact.error is None and not artifact.deferred
                    return raw_id, tuple(
                        session_revision_projection(session).session_hash.hex() for session in artifact.iter_sessions()
                    )
                finally:
                    artifact.discard()

        def measure() -> tuple[float, dict[str, tuple[str, ...]], int]:
            started = perf_counter()
            if sequential:
                results = [
                    adapter.submit(
                        partial(prepare, raw_id),
                        admission_class="incremental-background",
                        estimated_bytes=page.captures[raw_id].descriptor[4],
                    ).future.result()
                    for raw_id in raw_ids
                ]
            else:
                results = list(
                    adapter.map(
                        prepare,
                        raw_ids,
                        admission_class="incremental-background",
                        estimated_bytes=lambda raw_id: page.captures[raw_id].descriptor[4],
                    )
                )
            return perf_counter() - started, dict(results), slots

        try:
            # Outside a compute parent: nested maps intentionally run serially
            # and would not measure the background admission window.
            return await asyncio.to_thread(measure)
        finally:
            page.close()


@pytest.mark.benchmark
def test_parse_stage_background_window_preserves_bytes_and_scales(tmp_path: Path) -> None:
    runtime = require_free_threaded_runtime(consumer="parse-stage-thread-scaling")
    sequential_root, parallel_root = tmp_path / "sequential", tmp_path / "parallel"
    sequential_ids = build_independent_raw_corpus(
        sequential_root, raw_count=_RAW_COUNT, avg_payload_bytes=_PAYLOAD_BYTES
    )
    parallel_ids = build_independent_raw_corpus(parallel_root, raw_count=_RAW_COUNT, avg_payload_bytes=_PAYLOAD_BYTES)
    assert sequential_ids == parallel_ids
    sequential_s, sequential_content, sequential_slots = asyncio.run(
        _time_parse(sequential_root, sequential_ids, sequential=True)
    )
    parallel_s, parallel_content, parallel_slots = asyncio.run(
        _time_parse(parallel_root, parallel_ids, sequential=False)
    )
    assert sequential_slots == parallel_slots
    assert parallel_content == sequential_content
    assert set(parallel_content) == set(parallel_ids)
    assert all(len(digests) == 1 for digests in parallel_content.values())
    speedup = sequential_s / parallel_s
    emit_receipt(
        "parse-stage-thread-scaling",
        {
            "runtime": runtime.to_dict(),
            "raw_count": _RAW_COUNT,
            "payload_bytes": _PAYLOAD_BYTES,
            "background_ceiling_slots": parallel_slots,
            "admission_windows": [1, parallel_slots],
            "sequential_s": sequential_s,
            "parallel_s": parallel_s,
            "speedup": speedup,
            "content": parallel_content,
        },
    )
    assert parallel_s < sequential_s * 1.5
    if parallel_slots > 1:
        assert speedup > 1.5, f"actual {parallel_slots}-slot parser speedup was {speedup:.3f}"
