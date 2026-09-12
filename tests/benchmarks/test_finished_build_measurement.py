"""Sealed, non-live finished-build measurement for the reindex route.

The executable arms use ``backfill_historical_revision_evidence``, the normal
retained-raw authority, census, replay, finalization, and close route.  The
input is a deterministic 516-raw Codex slice whose raw identities, blob
digests, and exact byte count are sealed into each receipt.

Raw replay has no SessionShard hand-off.  Shard and process arms are therefore
declared capability refusals, not silently replaced with direct archive-writer
experiments.  The writer-level transport laws remain in
``tests/unit/storage/test_session_shards.py``.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import resource
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import pytest

from polylogue.sources.live.metrics import LiveBatchMetrics
from polylogue.sources.revision_backfill import (
    RevisionBackfillResult,
    RevisionCensusResult,
    backfill_historical_revision_evidence,
    census_historical_revision_evidence,
    split_parse_and_apply_seconds,
)
from polylogue.storage.index_generation import IndexGenerationStore
from tests.infra.archive_templates import (
    bootstrap_archive_root,
    clone_archive_template,
    finalize_archive_template,
)
from tests.infra.reindex_differential import (
    DerivedModelSnapshot,
    assert_derived_model_ready,
    assert_derived_models_equivalent,
    snapshot_derived_model,
)
from tests.infra.revision_backfill_benchmark import build_independent_raw_corpus

_SESSION_COUNT = 516
# ``_codex_raw_payload`` subtracts its JSON envelope before padding.  This
# target consequently seals precisely 210,554,832 bytes (200.800735 MiB).
_RAW_PAYLOAD_TARGET_BYTES = 408_129
_SEALED_INPUT_BYTES = 210_554_832
_SEALED_INPUT_DIGEST = "b970c56fd5478c928e12eb97c92737fe351907e1e1edeafc14c7104489a345ed"
_WORKER_COUNTS = (1, 4, 12)


@dataclass(frozen=True, slots=True)
class _Arm:
    name: str
    uses_owned_inactive_generation: bool
    uses_shard_transport: bool
    worker_mode: str = "thread"
    refusal_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _SealedInput:
    digest: str
    bytes: int
    raw_count: int


@dataclass(frozen=True, slots=True)
class _ArmReceipt:
    arm: str
    worker_mode: str
    worker_count: int
    input_digest: str
    wall_seconds: float
    self_cpu_seconds: float
    child_cpu_seconds: float
    peak_rss_bytes: int
    archive_bytes: int
    stage_timings_s: dict[str, float]
    metrics: dict[str, object]
    snapshot: DerivedModelSnapshot


@dataclass(frozen=True, slots=True)
class _SourceCensusReceipt:
    input_digest: str
    wall_seconds: float
    self_cpu_seconds: float
    peak_rss_bytes: int
    scanned: int
    classified_full: int
    quarantined: int
    logical_key_count: int


_ARMS = (
    _Arm("retained-index-inline", uses_owned_inactive_generation=False, uses_shard_transport=False),
    _Arm("deferred-index-fresh-inline", uses_owned_inactive_generation=True, uses_shard_transport=False),
    _Arm(
        "retained-index-shard",
        uses_owned_inactive_generation=False,
        uses_shard_transport=True,
        refusal_reason="raw replay has no production SessionShard/PreparedSessionShardRows hand-off",
    ),
    _Arm(
        "deferred-index-fresh-shard",
        uses_owned_inactive_generation=True,
        uses_shard_transport=True,
        refusal_reason="raw replay has no production SessionShard/PreparedSessionShardRows hand-off",
    ),
    _Arm(
        "retained-index-inline-process",
        uses_owned_inactive_generation=False,
        uses_shard_transport=False,
        worker_mode="process",
        refusal_reason="raw replay dispatches its production parser through ThreadPoolExecutor only",
    ),
)


def _input_slice(root: Path) -> _SealedInput:
    """Acquire one deterministic raw input through the ordinary source writer."""
    bootstrap_archive_root(root)
    build_independent_raw_corpus(
        root,
        raw_count=_SESSION_COUNT,
        avg_payload_bytes=_RAW_PAYLOAD_TARGET_BYTES,
        authoritative_source=True,
    )
    return _sealed_input(root)


def _sealed_input(root: Path) -> _SealedInput:
    """Read the immutable raw identity/byte manifest without changing it."""
    with sqlite3.connect(root / "source.db") as conn:
        rows = conn.execute("SELECT raw_id, blob_hash FROM raw_sessions ORDER BY raw_id").fetchall()
    digest = hashlib.sha256()
    byte_count = 0
    for raw_id, blob_hash in rows:
        blob_hex = bytes(blob_hash).hex() if isinstance(blob_hash, bytes) else str(blob_hash)
        blob_path = root / "blob" / blob_hex[:2] / blob_hex[2:]
        size = blob_path.stat().st_size
        byte_count += size
        digest.update(f"{raw_id}:{blob_hex}:{size}\n".encode())
    if len(rows) != _SESSION_COUNT or byte_count != _SEALED_INPUT_BYTES:
        raise AssertionError(
            f"sealed input shape drifted: raws={len(rows)} bytes={byte_count} "
            f"expected={_SESSION_COUNT}/{_SEALED_INPUT_BYTES}"
        )
    input_digest = digest.hexdigest()
    if input_digest != _SEALED_INPUT_DIGEST:
        raise AssertionError(f"sealed input digest drifted: {input_digest} != {_SEALED_INPUT_DIGEST}")
    return _SealedInput(digest=input_digest, bytes=byte_count, raw_count=len(rows))


def _arm_root(template: Path, destination: Path, sealed: _SealedInput) -> Path:
    """Clone the same sealed source tree for one isolated production arm."""
    clone_archive_template(template, destination)
    if _sealed_input(destination) != sealed:
        raise AssertionError("cloned finished-build arm does not carry the sealed input")
    return destination


def _prepare_censused_template(root: Path, sealed: _SealedInput) -> _SourceCensusReceipt:
    """Record the production source-admission prerequisite once, before arms.

    An owned inactive generation intentionally refuses an uncensused source.
    This source-only phase records those durable parser receipts and does not
    create an index output. It is preparation, not part of either arm's
    elapsed time.
    """
    before = resource.getrusage(resource.RUSAGE_SELF)
    started = perf_counter()
    result: RevisionCensusResult = census_historical_revision_evidence(root)
    wall_seconds = perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_SELF)
    if result.scanned != sealed.raw_count or result.quarantined:
        raise AssertionError(
            f"sealed source census is incomplete: scanned={result.scanned} quarantined={result.quarantined}"
        )
    return _SourceCensusReceipt(
        input_digest=sealed.digest,
        wall_seconds=wall_seconds,
        self_cpu_seconds=(after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime),
        peak_rss_bytes=int(after.ru_maxrss) * 1024,
        scanned=result.scanned,
        classified_full=result.classified_full,
        quarantined=result.quarantined,
        logical_key_count=len(result.logical_keys),
    )


@pytest.fixture(scope="module")
def _censused_input_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, _SealedInput, _SourceCensusReceipt]:
    """Build and seal one raw-identical, production-censused template."""
    template = tmp_path_factory.mktemp("finished-build-template") / "sealed-input"
    sealed = _input_slice(template)
    census = _prepare_censused_template(template, sealed)
    finalize_archive_template(template)
    return template, sealed, census


def _candidate_root(root: Path, arm: _Arm) -> tuple[Path, tuple[str, str] | None]:
    if not arm.uses_owned_inactive_generation:
        return root, None
    generation = IndexGenerationStore.for_archive_root(root).create(source_snapshot="finished-build-sealed-input")
    return Path(generation.index_path).parent, (generation.generation_id, generation.owner_id)


def _session_ids(index_path: Path) -> tuple[str, ...]:
    with sqlite3.connect(index_path) as conn:
        return tuple(str(row[0]) for row in conn.execute("SELECT session_id FROM sessions ORDER BY session_id"))


def _archive_bytes(index_path: Path) -> int:
    return sum(path.stat().st_size for path in (index_path, index_path.with_suffix(".db-wal")) if path.exists())


def _live_metrics(
    result: RevisionBackfillResult,
    sealed: _SealedInput,
    *,
    archive_bytes: int,
    wall_seconds: float,
    peak_rss_bytes: int,
    worker_count: int,
) -> LiveBatchMetrics:
    """Adapt the production result to the existing live metric vocabulary."""
    parse_seconds, apply_seconds = split_parse_and_apply_seconds(result.stage_timings_s)
    return LiveBatchMetrics(
        queued_file_count=sealed.raw_count,
        needed_file_count=sealed.raw_count,
        skipped_file_count=0,
        succeeded_file_count=result.replayed_logical_sources,
        failed_file_count=result.quarantined,
        source_group_count=sealed.raw_count,
        input_bytes=sealed.bytes,
        source_payload_read_bytes=sealed.bytes,
        cursor_fingerprint_read_bytes=0,
        ingest_worker_count_max=worker_count,
        append_file_count=0,
        full_file_count=sealed.raw_count,
        archive_bytes_before=0,
        archive_bytes_after=archive_bytes,
        archive_write_bytes_delta=archive_bytes,
        parse_time_s=parse_seconds,
        convergence_time_s=apply_seconds,
        total_time_s=wall_seconds,
        ingested_bytes=sealed.bytes if result.quarantined == 0 and result.adoption_deferred == 0 else 0,
        failed_bytes=0,
        refused_bytes_by_reason={},
        ingested_session_count=result.replayed_logical_sources,
        ingested_message_count=0,
        changed_session_count=result.replayed_logical_sources,
        rss_peak_self_mb=peak_rss_bytes / (1024 * 1024),
        stage_timings_s=dict(result.stage_timings_s),
    )


def _run_arm(root: Path, sealed: _SealedInput, arm: _Arm, *, worker_count: int) -> _ArmReceipt:
    if arm.uses_shard_transport or arm.worker_mode != "thread":
        raise RuntimeError(arm.refusal_reason or "unsupported finished-build capability")
    destination, owned_generation = _candidate_root(root, arm)
    before = resource.getrusage(resource.RUSAGE_SELF)
    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = perf_counter()
    result = backfill_historical_revision_evidence(
        destination,
        owned_inactive_generation=owned_generation,
        ingest_workers=worker_count,
    )
    wall_seconds = perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_SELF)
    children_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    index_path = destination / "index.db"
    ids = _session_ids(index_path)
    if len(ids) != sealed.raw_count:
        raise AssertionError(f"{arm.name} lost population: sessions={len(ids)} expected={sealed.raw_count}")
    snapshot = snapshot_derived_model(
        destination,
        index_path,
        session_ids=ids[:3],
        search_queries=("amg1-payload",),
    )
    assert_derived_model_ready(snapshot)
    archive_bytes = _archive_bytes(index_path)
    metrics = _live_metrics(
        result,
        sealed,
        archive_bytes=archive_bytes,
        wall_seconds=wall_seconds,
        peak_rss_bytes=int(after.ru_maxrss) * 1024,
        worker_count=worker_count,
    )
    if metrics.unaccounted_bytes:
        raise AssertionError(f"{arm.name} left unclassified input bytes: {metrics.unaccounted_bytes}")
    if result.quarantined or result.adoption_deferred:
        raise AssertionError(
            f"{arm.name} cannot produce a speed verdict with quarantined={result.quarantined} "
            f"or adoption_deferred={result.adoption_deferred}"
        )
    return _ArmReceipt(
        arm=arm.name,
        worker_mode=arm.worker_mode,
        worker_count=worker_count,
        input_digest=sealed.digest,
        wall_seconds=wall_seconds,
        self_cpu_seconds=(after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime),
        child_cpu_seconds=(children_after.ru_utime + children_after.ru_stime)
        - (children_before.ru_utime + children_before.ru_stime),
        peak_rss_bytes=int(after.ru_maxrss) * 1024,
        archive_bytes=archive_bytes,
        stage_timings_s=dict(result.stage_timings_s),
        metrics=metrics.to_payload(),
        snapshot=snapshot,
    )


def test_finished_build_measurement_declares_capability_boundary() -> None:
    """No direct writer attachment may impersonate a production replay arm."""
    assert _WORKER_COUNTS == (1, 4, 12)
    assert {arm.name for arm in _ARMS} == {
        "retained-index-inline",
        "deferred-index-fresh-inline",
        "retained-index-shard",
        "deferred-index-fresh-shard",
        "retained-index-inline-process",
    }
    refused = [arm for arm in _ARMS if arm.uses_shard_transport or arm.worker_mode != "thread"]
    assert all(arm.refusal_reason for arm in refused)
    route_source = inspect.getsource(backfill_historical_revision_evidence)
    assert "prepare_session_shard" not in route_source
    assert "PreparedSessionShardRows" not in route_source
    assert "ProcessPoolExecutor" not in inspect.getsource(
        __import__("polylogue.sources.revision_backfill", fromlist=["*"])
    )
    sealed = _SealedInput(digest="sealed", bytes=0, raw_count=0)
    for arm in refused:
        with pytest.raises(RuntimeError, match="production|ThreadPoolExecutor"):
            _run_arm(Path("not-opened-for-capability-refusal"), sealed, arm, worker_count=1)


@pytest.mark.benchmark
@pytest.mark.storage_scale
@pytest.mark.timeout(900)
@pytest.mark.parametrize("worker_count", _WORKER_COUNTS)
def test_finished_build_measurement_runs_sealed_inline_arms_at_declared_scale(
    tmp_path: Path,
    worker_count: int,
    _censused_input_template: tuple[Path, _SealedInput, _SourceCensusReceipt],
) -> None:
    """Measure the supported thread counts with fresh, interleaved arm roots.

    Every arm gets a clone of the one sealed source tree.  N=4 reverses the
    retained/fresh order, avoiding a fixed route-order conclusion while still
    retaining an immediately comparable pair at each supported worker count.
    """
    template, sealed, source_census = _censused_input_template
    retained, deferred_fresh = _ARMS[:2]
    ordered_arms = (deferred_fresh, retained) if worker_count == 4 else (retained, deferred_fresh)
    receipts = [
        _run_arm(
            _arm_root(template, tmp_path / f"{arm.name}-n{worker_count}", sealed),
            sealed,
            arm,
            worker_count=worker_count,
        )
        for arm in ordered_arms
    ]
    assert {receipt.input_digest for receipt in receipts} == {sealed.digest}
    by_arm = {receipt.arm: receipt for receipt in receipts}
    assert_derived_models_equivalent(
        by_arm[retained.name].snapshot,
        by_arm[deferred_fresh.name].snapshot,
    )
    assert all(receipt.metrics["unaccounted_bytes"] == 0 for receipt in receipts)
    assert all(receipt.metrics["failed_file_count"] == 0 for receipt in receipts)
    assert all(receipt.metrics["refused_bytes"] == 0 for receipt in receipts)
    # Each count has one interleaved pair. The receipts show the observed
    # ordering, but one pair is insufficient to declare a timing winner.
    verdict = {
        "conclusion": "no-winner",
        "reason": "one interleaved retained/fresh pair at this worker count; repeat before ranking",
    }
    print(
        "finished-build-measurement="
        + json.dumps(
            {
                "receipts": [
                    {
                        **{key: value for key, value in asdict(receipt).items() if key != "snapshot"},
                        "snapshot": "derived-model-equivalent-and-ready",
                    }
                    for receipt in receipts
                ],
                "source_census": asdict(source_census),
                "verdict": verdict,
            },
            sort_keys=True,
        )
    )
