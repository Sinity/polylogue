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

import hashlib
import inspect
import resource
import sqlite3
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import cast

import pytest

from devtools.measurement_receipts import emit_receipt
from polylogue.sources import revision_backfill
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
    FinishedBuildRoute,
    FinishedBuildWorkIdentity,
    capture_finished_build_output,
)
from tests.infra.revision_backfill_benchmark import build_independent_raw_corpus
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
    work: FinishedBuildWorkIdentity
    route: FinishedBuildRoute
    candidate_identity: str
    resources: FinishedBuildResourceMeasurement
    writer_apply_seconds: float
    archive_bytes: int
    stage_timings_s: dict[str, float]
    metrics: dict[str, object]
    fresh_build: bool
    deferred_secondary_indexes: bool
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
    peak_rss_bytes: int
    scanned: int
    classified_full: int
    quarantined: int
    logical_key_count: int


def _compact_receipt(receipt: _ArmReceipt) -> _ArmReceipt:
    """Release the completed projection after its terminal checks succeeded."""
    snapshot = receipt.snapshot
    assert snapshot is not None
    return replace(receipt, snapshot=None)


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
        "candidate_identity": receipt.candidate_identity,
        "resources": receipt.resources.to_payload(),
        # The frozen-replay route holds the single SQLite writer internally;
        # its writer-side stage ledger is the only truthful hold denominator.
        # It is not a DaemonWriteCoordinator lease and is labelled accordingly.
        "writer_apply_seconds": receipt.writer_apply_seconds,
        "archive_bytes": receipt.archive_bytes,
        "stage_timings_s": receipt.stage_timings_s,
        "metrics": receipt.metrics,
        "fresh_build": receipt.fresh_build,
        "deferred_secondary_indexes": receipt.deferred_secondary_indexes,
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
        "deferred-index-fresh-inline",
        uses_owned_inactive_generation=True,
        uses_shard_transport=False,
        refusal_reason="does not exercise the selected sealed-shard transport",
    ),
    _Arm(
        "retained-index-shard",
        uses_owned_inactive_generation=False,
        uses_shard_transport=True,
        refusal_reason="sealed shards require the owned inactive-generation route",
    ),
    _Arm(
        "deferred-index-fresh-shard-process",
        uses_owned_inactive_generation=True,
        uses_shard_transport=True,
        worker_mode="process",
        refusal_reason="production raw replay dispatches parsing through ThreadPoolExecutor only",
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


def _work_identity(sealed: _SealedInput) -> FinishedBuildWorkIdentity:
    """Bind the sealed source, exact route code, and one selected profile."""
    route_source = inspect.getsource(backfill_historical_revision_evidence)
    shard_source = inspect.getsource(revision_backfill._FrozenReplayShardTransport)
    code_digest = hashlib.sha256((route_source + shard_source).encode()).hexdigest()
    return FinishedBuildWorkIdentity(
        source_identity=f"sha256:{sealed.digest}",
        code_identity=f"sha256:{code_digest}",
        profile_identity=("finished-build:sealed-516-raw:thread-4:owned-inactive-generation:session-shard"),
    )


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


def _run_arm(
    root: Path,
    sealed: _SealedInput,
    arm: _Arm,
    *,
    worker_count: int,
) -> _ArmReceipt:
    if arm != _SELECTED_ARM:
        raise RuntimeError("finished-build measurement runs only the declared selected arm")
    destination, owned_generation = _candidate_root(root, arm)
    resource_probe = FinishedBuildResourceProbe.start()
    result = backfill_historical_revision_evidence(
        destination,
        owned_inactive_generation=owned_generation,
        ingest_workers=worker_count,
        use_session_shards=arm.uses_shard_transport,
    )
    index_path = destination / "index.db"
    ids = _session_ids(index_path)
    if len(ids) != sealed.raw_count:
        raise AssertionError(f"{arm.name} lost population: sessions={len(ids)} expected={sealed.raw_count}")
    output = capture_finished_build_output(
        destination,
        index_path,
        work=_work_identity(sealed),
        route=FinishedBuildRoute.from_production_callable(arm.name, backfill_historical_revision_evidence),
        resource_probe=resource_probe,
        session_ids=ids[:3],
        search_queries=("amg1-payload",),
    )
    archive_bytes = _archive_bytes(index_path)
    _parse_seconds, writer_apply_seconds = split_parse_and_apply_seconds(result.stage_timings_s)
    metrics = _live_metrics(
        result,
        sealed,
        archive_bytes=archive_bytes,
        wall_seconds=output.resources.elapsed_seconds,
        peak_rss_bytes=output.resources.peak_rss_self_bytes,
        worker_count=worker_count,
    )
    if metrics.unaccounted_bytes:
        raise AssertionError(f"{arm.name} left unclassified input bytes: {metrics.unaccounted_bytes}")
    if result.quarantined or result.adoption_deferred:
        raise AssertionError(
            f"{arm.name} cannot produce a speed verdict with quarantined={result.quarantined} "
            f"or adoption_deferred={result.adoption_deferred}"
        )
    population_total = result.replayed_logical_sources + result.adoption_deferred + result.quarantined
    if population_total != sealed.raw_count:
        raise AssertionError(
            f"{arm.name} has unclassified raw population: offered={sealed.raw_count} classified={population_total}"
        )
    if output.output_session_count != result.replayed_logical_sources:
        raise AssertionError(
            f"{arm.name} output population differs from replay receipt: "
            f"sessions={output.output_session_count} replayed={result.replayed_logical_sources}"
        )
    if writer_apply_seconds <= 0:
        raise AssertionError("selected frozen replay reported no serialized writer-apply time")
    return _ArmReceipt(
        arm=arm.name,
        worker_mode=arm.worker_mode,
        worker_count=worker_count,
        input_digest=sealed.digest,
        work=output.work,
        route=output.route,
        candidate_identity=("active" if owned_generation is None else f"index-generation:{owned_generation[0]}"),
        resources=output.resources,
        writer_apply_seconds=writer_apply_seconds,
        archive_bytes=archive_bytes,
        stage_timings_s=dict(result.stage_timings_s),
        metrics=metrics.to_payload(),
        fresh_build=arm.uses_owned_inactive_generation,
        deferred_secondary_indexes=arm.uses_owned_inactive_generation,
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
    assert "ProcessPoolExecutor" not in inspect.getsource(
        __import__("polylogue.sources.revision_backfill", fromlist=["*"])
    )
    assert {arm.name for arm in _REJECTED_ALTERNATIVES} == {
        "retained-index-inline",
        "deferred-index-fresh-inline",
        "retained-index-shard",
        "deferred-index-fresh-shard-process",
    }
    assert all(arm.refusal_reason for arm in _REJECTED_ALTERNATIVES)
    sealed = _SealedInput(digest="sealed", bytes=0, raw_count=0)
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
        candidate_identity="candidate",
        resources=FinishedBuildResourceMeasurement(0.0, 0.0, 0.0, 0, 0, 0, 0),
        writer_apply_seconds=0.0,
        archive_bytes=0,
        stage_timings_s={},
        metrics={},
        fresh_build=False,
        deferred_secondary_indexes=False,
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
        skipped_raw_count=0,
        output_session_count=0,
        output_message_count=0,
        output_block_count=0,
        snapshot=cast(DerivedModelSnapshot, UncopyableSnapshot()),
    )

    compact = _compact_receipt(receipt)

    assert compact.snapshot is None
    assert _receipt_payload(compact)["snapshot"] == "derived-model-equivalent-and-ready"


@pytest.mark.benchmark
@pytest.mark.storage_scale
@pytest.mark.timeout(900)
def test_finished_build_measurement_runs_sealed_production_arms_at_declared_scale(
    tmp_path: Path,
    _censused_input_template: tuple[Path, _SealedInput, _SourceCensusReceipt],
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
            _arm_root(template, tmp_path / f"{_SELECTED_ARM.name}-n{worker_count}", sealed),
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
