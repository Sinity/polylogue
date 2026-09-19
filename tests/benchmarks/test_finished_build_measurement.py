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
import json
import resource
import sqlite3
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

import pytest

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
# This is a transport comparison, not a worker-scaling experiment.  Four is
# the one ordinary ThreadPoolExecutor setting selected for this sealed arm;
# changing it belongs to a separate parser-scaling measurement.
_SELECTED_TRANSPORT_WORKER_COUNT = 4
_INTERLEAVED_REPETITIONS = 2


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
    repetition: int
    input_digest: str
    wall_seconds: float
    self_cpu_seconds: float
    child_cpu_seconds: float
    peak_rss_bytes: int
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
    # A completed arm is compared to the retained reference immediately.  The
    # receipt that survives to the final renderer deliberately releases this
    # potentially large, full logical projection: the digest and censuses are
    # the compact receipt, while retaining every projection would turn a
    # finished-build measurement into a memory-scaling benchmark of its own.
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


@dataclass(frozen=True, slots=True)
class _CapabilityReceipt:
    """One declared arm that the current ordinary route cannot execute.

    Keeping these alongside completed-arm receipts prevents a result renderer
    from presenting an inline-only timing as a shard/process comparison.
    """

    arm: str
    worker_mode: str
    worker_count: int
    status: Literal["unsupported"]
    reason: str


def _compare_and_compact_receipt(
    reference: DerivedModelSnapshot | None, receipt: _ArmReceipt
) -> tuple[DerivedModelSnapshot, _ArmReceipt]:
    """Compare one full projection, then release it from the retained receipt.

    A finished-build arm's projection can contain the complete synthetic
    payload through several ordinary index projections.  Keeping all of those
    snapshots solely to print the compact receipt makes peak memory depend on
    the number of controls, rather than the production route being measured.
    """
    snapshot = receipt.snapshot
    assert snapshot is not None
    if reference is not None:
        assert_derived_models_equivalent(reference, snapshot)
    return (snapshot if reference is None else reference), replace(receipt, snapshot=None)


def _receipt_payload(receipt: _ArmReceipt) -> dict[str, object]:
    """Render a compact receipt without recursively copying its projection."""
    if receipt.snapshot is not None:
        raise AssertionError("finished-build receipt must release its compared snapshot before rendering")
    return {
        "arm": receipt.arm,
        "worker_mode": receipt.worker_mode,
        "worker_count": receipt.worker_count,
        "repetition": receipt.repetition,
        "input_digest": receipt.input_digest,
        "wall_seconds": receipt.wall_seconds,
        "self_cpu_seconds": receipt.self_cpu_seconds,
        "child_cpu_seconds": receipt.child_cpu_seconds,
        "peak_rss_bytes": receipt.peak_rss_bytes,
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


_ARMS = (
    _Arm("retained-index-inline", uses_owned_inactive_generation=False, uses_shard_transport=False),
    _Arm("deferred-index-fresh-inline", uses_owned_inactive_generation=True, uses_shard_transport=False),
    _Arm(
        "retained-index-shard",
        uses_owned_inactive_generation=False,
        uses_shard_transport=True,
        refusal_reason="sealed shard replay is an owned inactive-generation production route",
    ),
    _Arm(
        "deferred-index-fresh-shard",
        uses_owned_inactive_generation=True,
        uses_shard_transport=True,
    ),
    _Arm(
        "retained-index-inline-process",
        uses_owned_inactive_generation=False,
        uses_shard_transport=False,
        worker_mode="process",
        refusal_reason="raw replay dispatches its production parser through ThreadPoolExecutor only",
    ),
    _Arm(
        "retained-index-shard-process",
        uses_owned_inactive_generation=False,
        uses_shard_transport=True,
        worker_mode="process",
        refusal_reason="raw replay dispatches its production parser through ThreadPoolExecutor only",
    ),
    _Arm(
        "deferred-index-fresh-inline-process",
        uses_owned_inactive_generation=True,
        uses_shard_transport=False,
        worker_mode="process",
        refusal_reason="raw replay dispatches its production parser through ThreadPoolExecutor only",
    ),
    _Arm(
        "deferred-index-fresh-shard-process",
        uses_owned_inactive_generation=True,
        uses_shard_transport=True,
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


def _finished_schema_census(index_path: Path) -> tuple[tuple[tuple[str, str], ...], str]:
    """Read the declared finished schema shape after the route has closed."""
    with sqlite3.connect(index_path) as conn:
        objects = tuple(
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
        row = conn.execute("SELECT identity FROM schema_identity WHERE tier = 'index'").fetchone()
    if row is None or not str(row[0]):
        raise AssertionError("finished build has no index schema identity")
    return objects, str(row[0])


def _canonical_logical_digest(snapshot: DerivedModelSnapshot) -> str:
    """Hash the sorted differential projection rather than a database image."""

    def default(value: object) -> str:
        if isinstance(value, bytes):
            return value.hex()
        raise TypeError(f"cannot canonically encode {type(value).__name__}")

    payload = json.dumps(asdict(snapshot), default=default, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


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


def _run_arm(
    root: Path,
    sealed: _SealedInput,
    arm: _Arm,
    *,
    worker_count: int,
    repetition: int = 1,
) -> _ArmReceipt:
    if arm.worker_mode != "thread" or (arm.uses_shard_transport and not arm.uses_owned_inactive_generation):
        raise RuntimeError(arm.refusal_reason or "unsupported finished-build capability")
    destination, owned_generation = _candidate_root(root, arm)
    before = resource.getrusage(resource.RUSAGE_SELF)
    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = perf_counter()
    result = backfill_historical_revision_evidence(
        destination,
        owned_inactive_generation=owned_generation,
        ingest_workers=worker_count,
        use_session_shards=arm.uses_shard_transport,
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
    schema_object_census, schema_identity = _finished_schema_census(index_path)
    canonical_logical_digest = _canonical_logical_digest(snapshot)
    with sqlite3.connect(index_path) as conn:
        output_session_count, output_message_count, output_block_count = (
            int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("sessions", "messages", "blocks")
        )
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
    population_total = result.replayed_logical_sources + result.adoption_deferred + result.quarantined
    if population_total != sealed.raw_count:
        raise AssertionError(
            f"{arm.name} has unclassified raw population: offered={sealed.raw_count} classified={population_total}"
        )
    if output_session_count != result.replayed_logical_sources:
        raise AssertionError(
            f"{arm.name} output population differs from replay receipt: "
            f"sessions={output_session_count} replayed={result.replayed_logical_sources}"
        )
    return _ArmReceipt(
        arm=arm.name,
        worker_mode=arm.worker_mode,
        worker_count=worker_count,
        repetition=repetition,
        input_digest=sealed.digest,
        wall_seconds=wall_seconds,
        self_cpu_seconds=(after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime),
        child_cpu_seconds=(children_after.ru_utime + children_after.ru_stime)
        - (children_before.ru_utime + children_before.ru_stime),
        peak_rss_bytes=int(after.ru_maxrss) * 1024,
        archive_bytes=archive_bytes,
        stage_timings_s=dict(result.stage_timings_s),
        metrics=metrics.to_payload(),
        fresh_build=arm.uses_owned_inactive_generation,
        deferred_secondary_indexes=arm.uses_owned_inactive_generation,
        derived_table_census=tuple(table for table, _projection in snapshot.tables),
        schema_object_census=schema_object_census,
        schema_identity=schema_identity,
        canonical_logical_digest=canonical_logical_digest,
        fts_source_rows=snapshot.fts.source_rows,
        fts_indexed_rows=snapshot.fts.indexed_rows,
        public_index_count=snapshot.fts.public_index_count,
        open_convergence_debt_count=len(snapshot.open_debt),
        offered_raw_count=sealed.raw_count,
        ingested_raw_count=result.replayed_logical_sources,
        refused_raw_count=0,
        deferred_raw_count=result.adoption_deferred,
        failed_raw_count=result.quarantined,
        skipped_raw_count=0,
        output_session_count=output_session_count,
        output_message_count=output_message_count,
        output_block_count=output_block_count,
        snapshot=snapshot,
    )


def _capability_receipts(*, worker_count: int) -> tuple[_CapabilityReceipt, ...]:
    """Expose every omitted matrix cell as a refusal, never a timing sample."""
    return tuple(
        _CapabilityReceipt(
            arm=arm.name,
            worker_mode=arm.worker_mode,
            worker_count=worker_count,
            status="unsupported",
            reason=arm.refusal_reason or "unsupported finished-build capability",
        )
        for arm in _ARMS
        if arm.worker_mode != "thread" or (arm.uses_shard_transport and not arm.uses_owned_inactive_generation)
    )


def test_finished_build_measurement_declares_capability_boundary() -> None:
    """No direct writer attachment may impersonate a production replay arm."""
    assert _SELECTED_TRANSPORT_WORKER_COUNT > 0
    assert _INTERLEAVED_REPETITIONS == 2
    assert {arm.name for arm in _ARMS} == {
        "retained-index-inline",
        "deferred-index-fresh-inline",
        "retained-index-shard",
        "deferred-index-fresh-shard",
        "retained-index-inline-process",
        "retained-index-shard-process",
        "deferred-index-fresh-inline-process",
        "deferred-index-fresh-shard-process",
    }
    refused = [
        arm
        for arm in _ARMS
        if arm.worker_mode != "thread" or (arm.uses_shard_transport and not arm.uses_owned_inactive_generation)
    ]
    assert all(arm.refusal_reason for arm in refused)
    route_source = inspect.getsource(backfill_historical_revision_evidence)
    assert "prepare_session_shard" in inspect.getsource(revision_backfill._FrozenReplayShardTransport)
    assert "attached_session_shard" in route_source
    assert "ProcessPoolExecutor" not in inspect.getsource(
        __import__("polylogue.sources.revision_backfill", fromlist=["*"])
    )
    sealed = _SealedInput(digest="sealed", bytes=0, raw_count=0)
    for arm in refused:
        with pytest.raises(RuntimeError, match="production|ThreadPoolExecutor"):
            _run_arm(Path("not-opened-for-capability-refusal"), sealed, arm, worker_count=1)
    refusal_receipts = _capability_receipts(worker_count=1)
    assert {receipt.arm for receipt in refusal_receipts} == {arm.name for arm in refused}
    assert all(receipt.reason for receipt in refusal_receipts)


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
        repetition=1,
        input_digest="sealed",
        wall_seconds=0.0,
        self_cpu_seconds=0.0,
        child_cpu_seconds=0.0,
        peak_rss_bytes=0,
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

    reference, compact = _compare_and_compact_receipt(None, receipt)

    assert reference is receipt.snapshot
    assert compact.snapshot is None
    assert _receipt_payload(compact)["snapshot"] == "derived-model-equivalent-and-ready"


@pytest.mark.benchmark
@pytest.mark.storage_scale
@pytest.mark.timeout(900)
def test_finished_build_measurement_runs_sealed_production_arms_at_declared_scale(
    tmp_path: Path,
    _censused_input_template: tuple[Path, _SealedInput, _SourceCensusReceipt],
) -> None:
    """Measure one selected transport choice with fresh, interleaved arm roots.

    Every completed arm gets a clone of the one sealed source tree and the two
    repetitions reverse their order, so a route cannot inherit one fixed cache
    or order position. Process and retained-index shard cells remain explicit
    production-policy refusals, not unmeasured matrix cells.
    """
    worker_count = _SELECTED_TRANSPORT_WORKER_COUNT
    template, sealed, source_census = _censused_input_template
    arms_by_name = {arm.name: arm for arm in _ARMS}
    retained = arms_by_name["retained-index-inline"]
    deferred_fresh = arms_by_name["deferred-index-fresh-inline"]
    deferred_fresh_shard = arms_by_name["deferred-index-fresh-shard"]
    supported_arms = (retained, deferred_fresh, deferred_fresh_shard)
    ordered_first = (deferred_fresh_shard, deferred_fresh, retained)
    receipts: list[_ArmReceipt] = []
    reference_snapshot: DerivedModelSnapshot | None = None
    for repetition in range(1, _INTERLEAVED_REPETITIONS + 1):
        ordered_arms = ordered_first if repetition % 2 else tuple(reversed(ordered_first))
        for arm in ordered_arms:
            receipt = _run_arm(
                _arm_root(template, tmp_path / f"{arm.name}-n{worker_count}-r{repetition}", sealed),
                sealed,
                arm,
                worker_count=worker_count,
                repetition=repetition,
            )
            reference_snapshot, compact_receipt = _compare_and_compact_receipt(reference_snapshot, receipt)
            receipts.append(compact_receipt)
    assert {receipt.input_digest for receipt in receipts} == {sealed.digest}
    assert len(receipts) == len(supported_arms) * _INTERLEAVED_REPETITIONS
    assert {receipt.arm for receipt in receipts} == {arm.name for arm in supported_arms}
    assert len({receipt.canonical_logical_digest for receipt in receipts}) == 1
    assert len({receipt.derived_table_census for receipt in receipts}) == 1
    assert len({receipt.schema_object_census for receipt in receipts}) == 1
    assert len({receipt.schema_identity for receipt in receipts}) == 1
    assert all(receipt.metrics["unaccounted_bytes"] == 0 for receipt in receipts)
    assert all(receipt.metrics["failed_file_count"] == 0 for receipt in receipts)
    assert all(receipt.metrics["refused_bytes"] == 0 for receipt in receipts)
    assert all(receipt.offered_raw_count == receipt.ingested_raw_count for receipt in receipts)
    assert all(
        receipt.refused_raw_count
        == receipt.deferred_raw_count
        == receipt.failed_raw_count
        == receipt.skipped_raw_count
        == 0
        for receipt in receipts
    )
    assert all(receipt.fts_source_rows == receipt.fts_indexed_rows for receipt in receipts)
    assert all(receipt.open_convergence_debt_count == 0 for receipt in receipts)
    assert all(receipt.output_session_count == sealed.raw_count for receipt in receipts)
    assert all(receipt.output_message_count and receipt.output_block_count for receipt in receipts)
    assert {receipt.fresh_build for receipt in receipts if "fresh" in receipt.arm} == {True}
    assert {receipt.deferred_secondary_indexes for receipt in receipts if "fresh" in receipt.arm} == {True}
    # Two interleaved controls are useful evidence, but there is no declared
    # statistical decision rule for a route ranking. Keep that limit explicit.
    capability_receipts = _capability_receipts(worker_count=worker_count)
    verdict = {
        "conclusion": "no-winner",
        "reason": (
            "two interleaved retained/fresh-inline/fresh-shard controls have no pre-registered "
            "ranking rule; process mode remains unsupported by production policy"
        ),
    }
    print(
        "finished-build-measurement="
        + json.dumps(
            {
                "receipts": [
                    {
                        **_receipt_payload(receipt),
                    }
                    for receipt in receipts
                ],
                "source_census": asdict(source_census),
                "capability_refusals": [asdict(receipt) for receipt in capability_receipts],
                "verdict": verdict,
            },
            sort_keys=True,
        )
    )
