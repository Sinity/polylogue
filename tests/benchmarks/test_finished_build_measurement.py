"""Sealed, non-live finished-build measurement for the reindex route.

The executable arms use ``backfill_historical_revision_evidence``, the normal
retained-raw authority, census, replay, finalization, and close route.  The
input is a deterministic 516-raw Codex slice whose raw identities, blob
digests, and exact byte count are sealed into each receipt.

Frozen inactive-generation replay uses the same sealed ``SessionShard`` writer
handoff as live ingest. The retained-index and process cells remain typed
capability refusals, rather than silently becoming inline runs or direct
archive-writer experiments. The writer-level transport laws remain in
``tests/unit/storage/test_session_shards.py``.
"""

from __future__ import annotations

import inspect
import resource
import sqlite3
import subprocess
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import cast
from unittest.mock import patch

import pytest

from devtools.measurement_receipts import emit_receipt
from polylogue.core.compute import compute_window_length
from polylogue.sources import revision_backfill
from polylogue.sources.live.metrics import LiveBatchMetrics
from polylogue.sources.revision_backfill import (
    PreparedRevisionReplayResult,
    RevisionCensusResult,
    backfill_historical_revision_evidence,
    census_historical_revision_evidence,
    split_parse_and_apply_seconds,
)
from polylogue.storage.index_generation import IndexGenerationStore
from polylogue.storage.sqlite import runtime_indexes
from tests.infra.archive_templates import (
    bootstrap_archive_root,
    finalize_archive_template,
)
from tests.infra.reindex_differential import (
    DerivedModelSnapshot,
    FinishedBuildOutput,
    FinishedBuildRoute,
    FinishedBuildWorkIdentity,
    SealedRawInput,
    assert_finished_builds_equivalent,
    capture_finished_build_output,
    capture_streamed_finished_build_fingerprint,
    clone_sealed_arm,
    finished_build_work_identity,
    seal_raw_input,
)
from tests.infra.revision_backfill_benchmark import (
    LARGE_PARENT_SHARED_PREFIX_SHAPE,
    build_independent_raw_corpus,
    build_large_parent_shared_prefix_raw_corpus,
)
from tests.infra.workload_artifacts import FinishedBuildResourceMeasurement, FinishedBuildResourceProbe

#: The committed-baseline name this arm records under.  ``devtools bench
#: baseline --record`` promotes the emitted receipt to
#: ``tests/benchmarks/baselines/<name>.json``.
_MEASUREMENT_NAME = "finished-build-sealed-516-raw"
_SESSION_COUNT = 516
# ``_codex_raw_payload`` subtracts its JSON envelope before padding.  This
# target consequently seals precisely 210,554,832 bytes (200.800735 MiB).
_RAW_PAYLOAD_TARGET_BYTES = 408_129
_SEALED_INPUT_BYTES = 210_554_832
_SEALED_INPUT_DIGEST = "b970c56fd5478c928e12eb97c92737fe351907e1e1edeafc14c7104489a345ed"
# This is a bounded selected-arm measurement, not a transport or worker-scaling
# matrix. Four is the ordinary ThreadPoolExecutor setting selected for this
# sealed frozen-replay profile; a different width or transport belongs to its
# own measured decision.
_SELECTED_TRANSPORT_WORKER_COUNT = 4
_WAL_SAMPLE_INTERVAL_S = 0.05


@dataclass(frozen=True, slots=True)
class _Arm:
    name: str
    uses_owned_inactive_generation: bool
    uses_shard_transport: bool
    worker_mode: str = "thread"
    refusal_reason: str | None = None
    defer_secondary_indexes: bool | None = None


@dataclass(frozen=True, slots=True)
class _ArmReceipt:
    arm: str
    worker_mode: str
    worker_count: int
    input_digest: str
    work: FinishedBuildWorkIdentity
    route: FinishedBuildRoute
    candidate_git_sha: str
    candidate_checkout_dirty: bool
    index_generation_identity: str
    resources: FinishedBuildResourceMeasurement
    non_decode_backfill_seconds: float
    archive_bytes: int
    index_wal_sampled_peak_bytes: int
    index_wal_final_bytes: int
    index_wal_sample_interval_s: float
    index_wal_sample_count: int
    stage_timings_s: dict[str, float]
    metrics: dict[str, object]
    fresh_build: bool
    deferred_secondary_indexes: bool
    deferred_secondary_index_names: tuple[str, ...]
    derived_table_census: tuple[str, ...]
    schema_object_census: tuple[tuple[str, str], ...]
    schema_identity: str
    canonical_logical_digest: str
    fts_source_rows: int
    fts_indexed_rows: int
    public_index_count: int
    open_convergence_debt_count: int
    offered_raw_count: int
    ingested_raw_count: int
    refused_raw_count: int
    deferred_raw_count: int
    failed_raw_count: int
    shard_lowering_degraded_count: int
    skipped_raw_count: int
    output_session_count: int
    output_message_count: int
    output_block_count: int
    # The completed output is released before rendering.  The digest and
    # censuses are the compact receipt; retaining the full logical projection
    # would turn this finished-build measurement into a memory-scaling test.
    snapshot: DerivedModelSnapshot | None


@dataclass(frozen=True, slots=True)
class _SourceCensusReceipt:
    input_digest: str
    wall_seconds: float
    self_cpu_seconds: float
    peak_rss_self_process_lifetime_bytes: int
    scanned: int
    classified_full: int
    quarantined: int
    logical_key_count: int


def _compact_receipt(receipt: _ArmReceipt) -> _ArmReceipt:
    """Release the completed projection after its terminal checks succeeded."""
    snapshot = receipt.snapshot
    assert snapshot is not None
    return replace(receipt, snapshot=None)


def _finished_output(receipt: _ArmReceipt) -> FinishedBuildOutput:
    snapshot = receipt.snapshot
    if snapshot is None:
        raise AssertionError("finished-build comparison requires retained snapshots")
    return FinishedBuildOutput(
        work=receipt.work,
        route=receipt.route,
        canonical_logical_digest=receipt.canonical_logical_digest,
        schema_object_census=receipt.schema_object_census,
        schema_identity=receipt.schema_identity,
        output_session_count=receipt.output_session_count,
        output_message_count=receipt.output_message_count,
        output_block_count=receipt.output_block_count,
        resources=receipt.resources,
        snapshot=snapshot,
    )


def _receipt_payload(receipt: _ArmReceipt) -> dict[str, object]:
    """Render a compact receipt without recursively copying its projection."""
    if receipt.snapshot is not None:
        raise AssertionError("finished-build receipt must release its compared snapshot before rendering")
    return {
        "arm": receipt.arm,
        "worker_mode": receipt.worker_mode,
        "worker_count": receipt.worker_count,
        "input_digest": receipt.input_digest,
        "work": asdict(receipt.work),
        "route": asdict(receipt.route),
        "candidate_git_sha": receipt.candidate_git_sha,
        "candidate_checkout_dirty": receipt.candidate_checkout_dirty,
        "index_generation_identity": receipt.index_generation_identity,
        "resources": receipt.resources.to_payload(),
        "non_decode_backfill_seconds": receipt.non_decode_backfill_seconds,
        "archive_bytes": receipt.archive_bytes,
        "archive_bytes_scope": "final index.db plus final index.db-wal file sizes; not allocated bytes or write I/O",
        "index_wal_sampled_peak_bytes": receipt.index_wal_sampled_peak_bytes,
        "index_wal_final_bytes": receipt.index_wal_final_bytes,
        "index_wal_sample_interval_s": receipt.index_wal_sample_interval_s,
        "index_wal_sample_count": receipt.index_wal_sample_count,
        "stage_timings_s": receipt.stage_timings_s,
        "stage_timings_scope": "nested and concurrent observations; do not sum entries or treat non-decode time as writer hold",
        "resource_scope": "self process CPU, RSS, and I/O; child CPU is a separate cumulative delta",
        "route_scope": "direct retained-raw backfill and finished-output checks; excludes cold daemon intake and source preparation",
        "metrics": receipt.metrics,
        "metrics_scope": "LiveBatchMetrics compatibility projection; convergence_time_s is non-decode backfill time, and archive_write_bytes_delta is final archive file size, not written bytes",
        "fresh_build": receipt.fresh_build,
        "deferred_secondary_indexes": receipt.deferred_secondary_indexes,
        "deferred_secondary_index_names": receipt.deferred_secondary_index_names,
        "derived_table_census": receipt.derived_table_census,
        "schema_object_census": receipt.schema_object_census,
        "schema_identity": receipt.schema_identity,
        "canonical_logical_digest": receipt.canonical_logical_digest,
        "fts_source_rows": receipt.fts_source_rows,
        "fts_indexed_rows": receipt.fts_indexed_rows,
        "public_index_count": receipt.public_index_count,
        "open_convergence_debt_count": receipt.open_convergence_debt_count,
        "offered_raw_count": receipt.offered_raw_count,
        "ingested_raw_count": receipt.ingested_raw_count,
        "refused_raw_count": receipt.refused_raw_count,
        "deferred_raw_count": receipt.deferred_raw_count,
        "failed_raw_count": receipt.failed_raw_count,
        "shard_lowering_degraded_count": receipt.shard_lowering_degraded_count,
        "skipped_raw_count": receipt.skipped_raw_count,
        "output_session_count": receipt.output_session_count,
        "output_message_count": receipt.output_message_count,
        "output_block_count": receipt.output_block_count,
        "snapshot": "derived-model-equivalent-and-ready",
    }


_SELECTED_ARM = _Arm(
    "deferred-index-fresh-shard",
    uses_owned_inactive_generation=True,
    uses_shard_transport=True,
    defer_secondary_indexes=True,
)
_RETAINED_INDEX_CONTROL = _Arm(
    "retained-index-fresh-shard",
    uses_owned_inactive_generation=True,
    uses_shard_transport=True,
    defer_secondary_indexes=False,
)
_DEFERRED_INDEX_INLINE_COMPARISON = _Arm(
    "deferred-index-fresh-inline",
    uses_owned_inactive_generation=True,
    uses_shard_transport=False,
    defer_secondary_indexes=True,
)
_RETAINED_INDEX_INLINE_COMPARISON = _Arm(
    "retained-index-fresh-inline",
    uses_owned_inactive_generation=True,
    uses_shard_transport=False,
    defer_secondary_indexes=False,
)
# These are declared non-cells, not a benchmark matrix.  The receipt keeps the
# decision boundary auditable without executing a direct-writer arm or a
# process mode the production backfill dispatcher does not own.
_REJECTED_ALTERNATIVES = (
    _Arm(
        "retained-index-inline",
        uses_owned_inactive_generation=False,
        uses_shard_transport=False,
        refusal_reason="not the selected fresh-build transport profile",
    ),
    _Arm(
        "deferred-index-fresh-shard-process",
        uses_owned_inactive_generation=True,
        uses_shard_transport=True,
        worker_mode="process",
        refusal_reason="production raw replay dispatches parsing through ThreadPoolExecutor only",
    ),
)


def _input_slice(root: Path) -> SealedRawInput:
    """Acquire one deterministic raw input through the ordinary source writer."""
    bootstrap_archive_root(root)
    build_independent_raw_corpus(
        root,
        raw_count=_SESSION_COUNT,
        avg_payload_bytes=_RAW_PAYLOAD_TARGET_BYTES,
        authoritative_source=True,
    )
    return _sealed_input(root)


def _sealed_input(root: Path) -> SealedRawInput:
    """Read the shared raw manifest and hold it to this measurement's seal."""
    sealed = seal_raw_input(root)
    if sealed.raw_count != _SESSION_COUNT or sealed.byte_count != _SEALED_INPUT_BYTES:
        raise AssertionError(
            f"sealed input shape drifted: raws={sealed.raw_count} bytes={sealed.byte_count} "
            f"expected={_SESSION_COUNT}/{_SEALED_INPUT_BYTES}"
        )
    if sealed.digest != _SEALED_INPUT_DIGEST:
        raise AssertionError(f"sealed input digest drifted: {sealed.digest} != {_SEALED_INPUT_DIGEST}")
    return sealed


def _prepare_censused_template(root: Path, sealed: SealedRawInput) -> _SourceCensusReceipt:
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
        peak_rss_self_process_lifetime_bytes=int(after.ru_maxrss) * 1024,
        scanned=result.scanned,
        classified_full=result.classified_full,
        quarantined=result.quarantined,
        logical_key_count=len(result.logical_keys),
    )


@pytest.fixture(scope="module")
def _censused_input_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, SealedRawInput, _SourceCensusReceipt]:
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


def _wal_bytes(index_path: Path) -> int:
    try:
        return index_path.with_name(f"{index_path.name}-wal").stat().st_size
    except FileNotFoundError:
        return 0


class _SampledWalSize:
    """Sample one index WAL outside the measured caller at a bounded rate."""

    def __init__(self, index_path: Path, *, interval_s: float = _WAL_SAMPLE_INTERVAL_S) -> None:
        if interval_s <= 0:
            raise ValueError("WAL sample interval must be positive")
        self.index_path = index_path
        self.interval_s = interval_s
        self.peak_bytes = 0
        self.final_bytes = 0
        self.sample_count = 0
        self._stop = threading.Event()
        self._error: OSError | None = None
        self._thread = threading.Thread(target=self._run, name="finished-build-wal-sampler", daemon=True)

    def _sample(self) -> None:
        size = _wal_bytes(self.index_path)
        self.peak_bytes = max(self.peak_bytes, size)
        self.sample_count += 1
        self.final_bytes = size

    def _run(self) -> None:
        try:
            self._sample()
            while not self._stop.wait(self.interval_s):
                self._sample()
        except OSError as exc:
            self._error = exc

    def start(self) -> None:
        self._thread.start()

    def finish(self) -> None:
        self._stop.set()
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("index WAL sampling failed") from self._error
        self._sample()


def _candidate_checkout() -> tuple[str, bool]:
    repository = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(("git", "-C", str(repository), "rev-parse", "HEAD"), text=True).strip()
    dirty = bool(subprocess.check_output(("git", "-C", str(repository), "status", "--porcelain"), text=True))
    return sha, dirty


def _completed_generation_index_path(root: Path, receipt: _ArmReceipt) -> Path:
    generation_id = receipt.index_generation_identity.removeprefix("index-generation:")
    if generation_id == receipt.index_generation_identity:
        raise AssertionError("finished arm has no owned inactive generation")
    return Path(IndexGenerationStore.for_archive_root(root, repair_anchor=False).load(generation_id).index_path)


def _work_identity(
    sealed: SealedRawInput, *, uses_shard_transport: bool, worker_count: int
) -> FinishedBuildWorkIdentity:
    """Bind the sealed source, exact route code, and one selected profile."""
    routes: tuple[Callable[..., object] | type, ...]
    if uses_shard_transport:
        routes = (backfill_historical_revision_evidence, revision_backfill._FrozenReplayShardTransport)
    else:
        routes = (backfill_historical_revision_evidence,)
    return finished_build_work_identity(
        sealed,
        profile=(
            f"finished-build:sealed-{sealed.raw_count}-raw:workers-{worker_count}:owned-inactive-generation:"
            f"{'session-shard' if uses_shard_transport else 'inline-replay'}"
        ),
        routes=routes,
    )


def _live_metrics(
    result: PreparedRevisionReplayResult,
    sealed: SealedRawInput,
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
        input_bytes=sealed.byte_count,
        source_payload_read_bytes=sealed.byte_count,
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
        ingested_bytes=sealed.byte_count if result.quarantined == 0 and result.adoption_deferred == 0 else 0,
        failed_bytes=0,
        refused_bytes_by_reason={},
        ingested_session_count=result.replayed_logical_sources,
        ingested_message_count=0,
        changed_session_count=result.replayed_logical_sources,
        rss_peak_self_mb=peak_rss_bytes / (1024 * 1024),
        stage_timings_s=dict(result.stage_timings_s),
    )


def _run_arm(
    root: Path,
    sealed: SealedRawInput,
    arm: _Arm,
    *,
    worker_count: int,
) -> _ArmReceipt:
    if arm not in (
        _SELECTED_ARM,
        _RETAINED_INDEX_CONTROL,
        _DEFERRED_INDEX_INLINE_COMPARISON,
        _RETAINED_INDEX_INLINE_COMPARISON,
    ):
        raise RuntimeError("finished-build measurement runs only the declared selected arm")
    # Report the existing shared admission window used by production census.
    effective_workers = compute_window_length(sealed.raw_count, worker_count)
    resource_probe = FinishedBuildResourceProbe.start()
    destination, owned_generation = _candidate_root(root, arm)
    index_path = destination / "index.db"
    wal_sampler = _SampledWalSize(index_path)
    wal_sampler.start()
    observed_deferred_indexes: list[tuple[str, ...]] = []
    observed_restored_indexes: list[tuple[str, ...]] = []
    original_defer = runtime_indexes.defer_secondary_indexes_sync
    original_restore = runtime_indexes.restore_deferred_secondary_indexes_sync

    def record_defer(conn: sqlite3.Connection) -> tuple[str, ...]:
        dropped = original_defer(conn)
        remaining = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
        if remaining.intersection(dropped):
            raise AssertionError("deferred reader indexes remained present during the write phase")
        observed_deferred_indexes.append(dropped)
        return dropped

    def record_restore(conn: sqlite3.Connection) -> None:
        original_restore(conn)
        names = tuple(
            str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index' ORDER BY name")
        )
        if observed_deferred_indexes and not set(observed_deferred_indexes[-1]) <= set(names):
            raise AssertionError("the index boundary returned before every deferred index was recreated")
        observed_restored_indexes.append(names)

    try:
        with (
            patch.object(runtime_indexes, "defer_secondary_indexes_sync", record_defer),
            patch.object(runtime_indexes, "restore_deferred_secondary_indexes_sync", record_restore),
        ):
            result = backfill_historical_revision_evidence(
                destination,
                owned_inactive_generation=owned_generation,
                ingest_workers=effective_workers,
                use_session_shards=arm.uses_shard_transport,
                defer_secondary_indexes=arm.defer_secondary_indexes,
            )
        ids = _session_ids(index_path)
        if len(ids) != sealed.raw_count:
            raise AssertionError(f"{arm.name} lost population: sessions={len(ids)} expected={sealed.raw_count}")
        output = capture_finished_build_output(
            destination,
            index_path,
            work=_work_identity(sealed, uses_shard_transport=arm.uses_shard_transport, worker_count=effective_workers),
            route=FinishedBuildRoute.from_production_callable(arm.name, backfill_historical_revision_evidence),
            resource_probe=resource_probe,
            session_ids=ids[:3],
            search_queries=("amg1-payload",),
        )
    finally:
        wal_sampler.finish()
    archive_bytes = _archive_bytes(index_path)
    _parse_seconds, non_decode_backfill_seconds = split_parse_and_apply_seconds(result.stage_timings_s)
    metrics = _live_metrics(
        result,
        sealed,
        archive_bytes=archive_bytes,
        wall_seconds=output.resources.elapsed_seconds,
        peak_rss_bytes=output.resources.peak_rss_self_bytes,
        worker_count=effective_workers,
    )
    if metrics.unaccounted_bytes:
        raise AssertionError(f"{arm.name} left unclassified input bytes: {metrics.unaccounted_bytes}")
    if result.quarantined or result.adoption_deferred:
        raise AssertionError(
            f"{arm.name} cannot produce a speed verdict with quarantined={result.quarantined} "
            f"or adoption_deferred={result.adoption_deferred}"
        )
    population_total = result.replayed_logical_sources + result.adoption_deferred + result.quarantined
    if result.scanned != sealed.raw_count or population_total != sealed.raw_count:
        raise AssertionError(
            f"{arm.name} has unclassified raw population: offered={sealed.raw_count} "
            f"scanned={result.scanned} classified={population_total}"
        )
    if output.output_session_count != result.replayed_logical_sources:
        raise AssertionError(
            f"{arm.name} output population differs from replay receipt: "
            f"sessions={output.output_session_count} replayed={result.replayed_logical_sources}"
        )
    if non_decode_backfill_seconds <= 0:
        raise AssertionError("selected frozen replay reported no non-decode backfill time")
    if len(observed_deferred_indexes) > 1 or len(observed_restored_indexes) > 1:
        raise AssertionError(f"{arm.name} crossed the index boundary more than once")
    deferred_index_names = observed_deferred_indexes[0] if observed_deferred_indexes else ()
    if bool(deferred_index_names) != bool(observed_restored_indexes):
        raise AssertionError(f"{arm.name} did not restore exactly the indexes observed as deferred")
    if arm.defer_secondary_indexes is True and deferred_index_names != runtime_indexes.DEFERRED_SECONDARY_INDEX_NAMES:
        raise AssertionError(f"{arm.name} deferred unexpected indexes: {deferred_index_names!r}")
    candidate_git_sha, candidate_checkout_dirty = _candidate_checkout()
    return _ArmReceipt(
        arm=arm.name,
        worker_mode=arm.worker_mode if effective_workers > 1 else "sequential",
        worker_count=effective_workers,
        input_digest=sealed.digest,
        work=output.work,
        route=output.route,
        candidate_git_sha=candidate_git_sha,
        candidate_checkout_dirty=candidate_checkout_dirty,
        index_generation_identity=("active" if owned_generation is None else f"index-generation:{owned_generation[0]}"),
        resources=output.resources,
        non_decode_backfill_seconds=non_decode_backfill_seconds,
        archive_bytes=archive_bytes,
        index_wal_sampled_peak_bytes=wal_sampler.peak_bytes,
        index_wal_final_bytes=wal_sampler.final_bytes,
        index_wal_sample_interval_s=wal_sampler.interval_s,
        index_wal_sample_count=wal_sampler.sample_count,
        stage_timings_s=dict(result.stage_timings_s),
        metrics=metrics.to_payload(),
        fresh_build=arm.uses_owned_inactive_generation,
        deferred_secondary_indexes=bool(deferred_index_names),
        deferred_secondary_index_names=deferred_index_names,
        derived_table_census=tuple(table for table, _projection in output.snapshot.tables),
        schema_object_census=output.schema_object_census,
        schema_identity=output.schema_identity,
        canonical_logical_digest=output.canonical_logical_digest,
        fts_source_rows=output.snapshot.fts.source_rows,
        fts_indexed_rows=output.snapshot.fts.indexed_rows,
        public_index_count=output.snapshot.fts.public_index_count,
        open_convergence_debt_count=len(output.snapshot.open_debt),
        offered_raw_count=sealed.raw_count,
        ingested_raw_count=result.replayed_logical_sources,
        refused_raw_count=0,
        deferred_raw_count=result.adoption_deferred,
        failed_raw_count=result.quarantined,
        shard_lowering_degraded_count=result.shard_lowering_degraded,
        skipped_raw_count=0,
        output_session_count=output.output_session_count,
        output_message_count=output.output_message_count,
        output_block_count=output.output_block_count,
        snapshot=output.snapshot,
    )


def test_finished_build_measurement_declares_capability_boundary() -> None:
    """The one measurement binds the owned production shard route directly."""
    assert _SELECTED_TRANSPORT_WORKER_COUNT > 0
    assert _SELECTED_ARM.name == "deferred-index-fresh-shard"
    assert _SELECTED_ARM.worker_mode == "thread"
    assert _SELECTED_ARM.uses_owned_inactive_generation
    assert _SELECTED_ARM.uses_shard_transport
    route_source = inspect.getsource(backfill_historical_revision_evidence)
    assert "prepare_session_shard" in inspect.getsource(revision_backfill._FrozenReplayShardTransport)
    assert "attached_session_shard" in route_source
    assert {arm.name for arm in _REJECTED_ALTERNATIVES} == {
        "retained-index-inline",
        "deferred-index-fresh-shard-process",
    }
    assert all(arm.refusal_reason for arm in _REJECTED_ALTERNATIVES)
    sealed = SealedRawInput(digest="sealed", byte_count=1, raw_count=1)
    for arm in _REJECTED_ALTERNATIVES:
        with pytest.raises(RuntimeError, match="declared selected arm"):
            _run_arm(Path("not-opened-for-capability-refusal"), sealed, arm, worker_count=1)


def test_finished_build_measurement_compacts_projection_before_rendering() -> None:
    """A retained projection must not be recursively copied into the receipt."""

    class UncopyableSnapshot:
        def __deepcopy__(self, memo: object) -> object:
            del memo
            raise AssertionError("receipt renderer tried to copy the full logical projection")

    receipt = _ArmReceipt(
        arm="test",
        worker_mode="thread",
        worker_count=1,
        input_digest="sealed",
        work=FinishedBuildWorkIdentity("source", "code", "profile"),
        route=FinishedBuildRoute("test", "polylogue.example.route"),
        candidate_git_sha="a" * 40,
        candidate_checkout_dirty=False,
        index_generation_identity="index-generation:test",
        resources=FinishedBuildResourceMeasurement(0.0, 0.0, 0.0, 0, 0, 0, 0),
        non_decode_backfill_seconds=0.0,
        archive_bytes=0,
        index_wal_sampled_peak_bytes=0,
        index_wal_final_bytes=0,
        index_wal_sample_interval_s=_WAL_SAMPLE_INTERVAL_S,
        index_wal_sample_count=0,
        stage_timings_s={},
        metrics={},
        fresh_build=False,
        deferred_secondary_indexes=False,
        deferred_secondary_index_names=(),
        derived_table_census=(),
        schema_object_census=(),
        schema_identity="schema",
        canonical_logical_digest="digest",
        fts_source_rows=0,
        fts_indexed_rows=0,
        public_index_count=0,
        open_convergence_debt_count=0,
        offered_raw_count=0,
        ingested_raw_count=0,
        refused_raw_count=0,
        deferred_raw_count=0,
        failed_raw_count=0,
        shard_lowering_degraded_count=0,
        skipped_raw_count=0,
        output_session_count=0,
        output_message_count=0,
        output_block_count=0,
        snapshot=cast(DerivedModelSnapshot, UncopyableSnapshot()),
    )

    compact = _compact_receipt(receipt)

    assert compact.snapshot is None
    payload = _receipt_payload(compact)
    assert payload["snapshot"] == "derived-model-equivalent-and-ready"
    assert payload["candidate_git_sha"] == "a" * 40
    assert payload["index_generation_identity"] == "index-generation:test"
    assert "writer_apply_seconds" not in payload
    assert "candidate_identity" not in payload


def test_finished_build_wal_sampler_distinguishes_sampled_peak_from_final(tmp_path: Path) -> None:
    index = tmp_path / "index.db"
    wal = tmp_path / "index.db-wal"
    sampler = _SampledWalSize(index, interval_s=0.01)
    sampler.start()
    try:
        wal.write_bytes(b"x" * 32)
        deadline = perf_counter() + 1.0
        while sampler.peak_bytes != 32 and perf_counter() < deadline:
            threading.Event().wait(0.01)
        assert sampler.peak_bytes == 32
        wal.unlink()
    finally:
        sampler.finish()
    assert sampler.sample_count >= 2
    assert sampler.peak_bytes == 32
    assert sampler.final_bytes == 0


def test_streamed_fingerprint_reads_completed_inactive_generation(tmp_path: Path) -> None:
    template = tmp_path / "sealed-input"
    bootstrap_archive_root(template)
    build_independent_raw_corpus(template, raw_count=1, avg_payload_bytes=2_000, authoritative_source=True)
    census = census_historical_revision_evidence(template)
    assert census.scanned == 1 and census.quarantined == 0
    sealed = seal_raw_input(template)
    finalize_archive_template(template)
    archive_root = clone_sealed_arm(template, tmp_path / "arm", sealed)
    receipt = _run_arm(archive_root, sealed, _SELECTED_ARM, worker_count=1)
    index_path = _completed_generation_index_path(archive_root, receipt)
    assert index_path != archive_root / "index.db"
    streamed = capture_streamed_finished_build_fingerprint(
        index_path.parent,
        index_path,
        scratch_root=tmp_path,
        session_ids=_session_ids(index_path)[:3],
        search_queries=("amg1-payload",),
        include_threads=True,
    )
    assert streamed.canonical_logical_digest == receipt.canonical_logical_digest


def test_index_deferral_comparison_repeats_interleaved_finished_builds(tmp_path: Path) -> None:
    """Compare retained and deferred indexes on independent and shared-prefix input.

    The input combines eight independent raws with a bounded parent and eight
    children that each repeat its 24-message prefix. The four runs alternate
    index policy so each is repeated on both sides of the other. Each run times
    the complete backfill and finished-output checks, including boundary index
    restoration, FTS and derived finalization, schema identity, and commit.
    """
    template = tmp_path / "sealed-input"
    bootstrap_archive_root(template)
    build_independent_raw_corpus(
        template,
        raw_count=8,
        avg_payload_bytes=2_000,
        authoritative_source=True,
    )
    build_large_parent_shared_prefix_raw_corpus(template)
    census = census_historical_revision_evidence(template)
    expected_raw_count = 8 + LARGE_PARENT_SHARED_PREFIX_SHAPE["child_count"] + 1
    assert census.scanned == expected_raw_count
    assert census.quarantined == 0
    sealed = seal_raw_input(template)
    finalize_archive_template(template)
    with sqlite3.connect(template / "index.db") as conn:
        expected_schema_objects = tuple(
            (str(kind), str(name))
            for kind, name in conn.execute(
                """
                SELECT type, name
                FROM sqlite_master
                WHERE name NOT LIKE 'sqlite_%'
                ORDER BY type, name
                """
            )
        )
        expected_schema_identity_row = conn.execute(
            "SELECT identity FROM schema_identity WHERE tier = 'index'"
        ).fetchone()
    assert expected_schema_identity_row is not None
    expected_schema_identity = str(expected_schema_identity_row[0])

    order = (
        _DEFERRED_INDEX_INLINE_COMPARISON,
        _RETAINED_INDEX_INLINE_COMPARISON,
        _RETAINED_INDEX_INLINE_COMPARISON,
        _DEFERRED_INDEX_INLINE_COMPARISON,
    )
    receipts = [
        _run_arm(
            clone_sealed_arm(template, tmp_path / f"comparison-{index}-{arm.name}", sealed),
            sealed,
            arm,
            worker_count=4,
        )
        for index, arm in enumerate(order)
    ]
    reference = receipts[0]
    for receipt in receipts[1:]:
        assert_finished_builds_equivalent(_finished_output(reference), _finished_output(receipt))

    streamed_archive = tmp_path / f"comparison-0-{order[0].name}"
    streamed_index_path = _completed_generation_index_path(streamed_archive, reference)
    streamed_candidate_root = streamed_index_path.parent
    streamed_session_ids = _session_ids(streamed_index_path)[:3]
    streamed = capture_streamed_finished_build_fingerprint(
        streamed_candidate_root,
        streamed_index_path,
        scratch_root=tmp_path,
        session_ids=streamed_session_ids,
        search_queries=("amg1-payload",),
        include_threads=True,
    )
    assert streamed.canonical_logical_digest == reference.canonical_logical_digest
    assert streamed.schema_object_census == reference.schema_object_census
    assert streamed.schema_identity == reference.schema_identity
    assert streamed.output_session_count == reference.output_session_count
    assert streamed.output_message_count == reference.output_message_count
    assert streamed.output_block_count == reference.output_block_count

    assert [receipt.deferred_secondary_indexes for receipt in receipts] == [True, False, False, True]
    assert [receipt.deferred_secondary_index_names for receipt in receipts] == [
        runtime_indexes.DEFERRED_SECONDARY_INDEX_NAMES,
        (),
        (),
        runtime_indexes.DEFERRED_SECONDARY_INDEX_NAMES,
    ]
    assert all(receipt.fresh_build for receipt in receipts)
    assert all(receipt.offered_raw_count == receipt.ingested_raw_count == sealed.raw_count for receipt in receipts)
    assert all(
        receipt.refused_raw_count
        == receipt.deferred_raw_count
        == receipt.failed_raw_count
        == receipt.shard_lowering_degraded_count
        == receipt.skipped_raw_count
        == 0
        for receipt in receipts
    )
    elapsed = [receipt.resources.elapsed_seconds for receipt in receipts]
    assert all(seconds > 0 for seconds in elapsed)
    assert all(receipt.schema_object_census == reference.schema_object_census for receipt in receipts)
    assert all(receipt.schema_identity == reference.schema_identity for receipt in receipts)
    assert all(receipt.canonical_logical_digest == reference.canonical_logical_digest for receipt in receipts)
    assert all(receipt.schema_object_census == expected_schema_objects for receipt in receipts)
    assert all(receipt.schema_identity == expected_schema_identity for receipt in receipts)
    assert sealed.raw_count == expected_raw_count
    assert reference.output_session_count == expected_raw_count
    assert reference.output_message_count == (
        8 + LARGE_PARENT_SHARED_PREFIX_SHAPE["parent_message_count"] + LARGE_PARENT_SHARED_PREFIX_SHAPE["child_count"]
    )
    with sqlite3.connect(streamed_index_path) as conn:
        prefix_links = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM session_links
                WHERE inheritance = 'prefix-sharing'
                  AND resolved_dst_session_id IS NOT NULL
                  AND branch_point_message_id IS NOT NULL
                """
            ).fetchone()[0]
        )
    assert prefix_links == LARGE_PARENT_SHARED_PREFIX_SHAPE["child_count"]
    emitted = emit_receipt(
        "finished-build-index-deferral-comparison",
        {
            "input": {"digest": sealed.digest, "bytes": sealed.byte_count, "raw_count": sealed.raw_count},
            "run_order": [receipt.arm for receipt in receipts],
            "arms": [_receipt_payload(_compact_receipt(receipt)) for receipt in receipts],
            "verdict": {
                "conclusion": "equivalent-finished-output",
                "reason": "interleaved retained/deferred index controls produced identical completed logical and schema digests; no speed ranking is claimed at this small scale",
            },
        },
    )
    print(f"finished-build-index-deferral comparison receipt: {emitted}")


@pytest.mark.benchmark
@pytest.mark.storage_scale
@pytest.mark.timeout(900)
def test_finished_build_measurement_runs_sealed_production_arms_at_declared_scale(
    tmp_path: Path,
    _censused_input_template: tuple[Path, SealedRawInput, _SourceCensusReceipt],
) -> None:
    """Measure one completed production route over the sealed 516-raw input.

    This does not rank transports or widths. The selected arm is the owned
    inactive-generation, sealed-shard route; terminal FTS, debt, schema, and
    public reads finish before the resource probe closes.
    """
    worker_count = _SELECTED_TRANSPORT_WORKER_COUNT
    template, sealed, source_census = _censused_input_template
    receipt = _compact_receipt(
        _run_arm(
            clone_sealed_arm(template, tmp_path / f"{_SELECTED_ARM.name}-n{worker_count}", sealed),
            sealed,
            _SELECTED_ARM,
            worker_count=worker_count,
        )
    )
    assert receipt.input_digest == sealed.digest
    assert receipt.metrics["unaccounted_bytes"] == 0
    assert receipt.metrics["failed_file_count"] == 0
    assert receipt.metrics["refused_bytes"] == 0
    assert receipt.offered_raw_count == receipt.ingested_raw_count == sealed.raw_count
    assert (
        receipt.refused_raw_count
        == receipt.deferred_raw_count
        == receipt.failed_raw_count
        == receipt.skipped_raw_count
        == 0
    )
    assert receipt.fts_source_rows == receipt.fts_indexed_rows
    assert receipt.open_convergence_debt_count == 0
    assert receipt.output_session_count == sealed.raw_count
    assert receipt.output_message_count and receipt.output_block_count
    assert receipt.fresh_build
    assert receipt.deferred_secondary_indexes
    assert receipt.resources.elapsed_seconds > 0
    assert receipt.resources.storage_bytes > 0
    # The receipt goes to the measurement-receipt route, not to stdout: a
    # printed receipt is evidence only for whoever was watching the run
    # (polylogue-cjyfw).  ``devtools bench baseline --record <path> --reason
    # ...`` promotes this observation to the committed baseline.
    emitted = emit_receipt(
        _MEASUREMENT_NAME,
        {
            "receipt": _receipt_payload(receipt),
            "source_census": asdict(source_census),
            "rejected_alternatives": [
                {"arm": arm.name, "reason": arm.refusal_reason} for arm in _REJECTED_ALTERNATIVES
            ],
            "verdict": {
                "conclusion": "single-arm-observation",
                "reason": "one selected production profile; no transport or width ranking claimed",
            },
        },
    )
    print(f"finished-build-measurement receipt: {emitted}")
