"""Equivalent finished builds across two production backfill arms.

The sealed 516-raw measurement in
``tests/benchmarks/test_finished_build_measurement.py`` runs ONE selected
production profile and declares the others non-cells. It therefore never
compares two arms, and the finished-build comparator
(``assert_finished_builds_equivalent``) had no consumer at all.

This module is the comparison, at a synthetic scale that runs without live
data or a dedicated measurement host: one sealed raw population, cloned into
two isolated arms, each completed through
``backfill_historical_revision_evidence`` --

* baseline: the retained active-index replay, the shape
  ``RawObservationPublisher.publish`` drives in production;
* replacement: the owned inactive generation with the sealed session-shard
  transport, the shape a fresh cold build drives.

Equivalence is over the completed logical output (every comparable index
table, the public insight reads, FTS readiness, open debt and the canonical
digest), not over a database page image, and each arm keeps its own
production callable identity and resource receipt. Scale is deliberately not
a claim here: this fixes *what* the arms must agree on, so a host-window
measurement run only has to add the scale.

Anti-vacuity: ``test_finished_build_comparison_rejects_a_diverged_or_indebted_arm``
executes both mutations a hollow comparator would survive -- one unresolved
convergence-debt row and one deleted ``blocks`` row.

The replacement arm's measured elapsed time was once dominated by
``spill_prefetch.decode_concurrent`` rather than by its own work: the AUTO
pipeline-decode prefetcher blocked for a full 30 s SQLite busy timeout on its
first reparse inside the owned-generation bulk-build route (0.10 s with
``pipeline_decode=False``, same output). polylogue-cz17d fixed that at the
source -- a prefetch worker no longer opens a handle to an EXCLUSIVE-locked
owned generation. Both arms still run the production default and this module
still encodes no timing tolerance, so a regression re-appears as elapsed time
in the receipt rather than being hidden by a bound.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from devtools.measurement_receipts import emit_receipt
from polylogue.sources import revision_backfill
from polylogue.sources.revision_backfill import (
    backfill_historical_revision_evidence,
    census_historical_revision_evidence,
    split_parse_and_apply_seconds,
)
from polylogue.storage.index_generation import IndexGenerationStore
from tests.infra.archive_templates import bootstrap_archive_root, finalize_archive_template
from tests.infra.reindex_differential import (
    FinishedBuildOutput,
    FinishedBuildRoute,
    SealedRawInput,
    assert_finished_builds_equivalent,
    capture_finished_build_output,
    clone_sealed_arm,
    finished_build_work_identity,
    seal_raw_input,
)
from tests.infra.revision_backfill_benchmark import build_independent_raw_corpus
from tests.infra.workload_artifacts import FinishedBuildResourceProbe

# Small enough to run in an ordinary selection, large enough that the replay
# phase does real per-cohort work in both arms. The scale is not the claim.
_RAW_COUNT = 8
_RAW_PAYLOAD_BYTES = 2_000
_WORK_PROFILE = "finished-build-equivalence:synthetic-8-raw:codex"
#: The measurement-receipt name this comparison emits under. It is an
#: observation, not a committed baseline: promoting one is the deliberate
#: ``devtools bench baseline --record`` step.
_MEASUREMENT_NAME = "finished-build-equivalence-synthetic"
# Counts have their own ``stage_counts`` key space, so every non-total entry
# in ``stage_timings_s`` is a duration and can be compared directly.
#: Kernel/clock slack between the route's own ledger and the probe interval.
_LEDGER_TOLERANCE_S = 0.5


@dataclass(frozen=True, slots=True)
class _Arm:
    """One production route shape, named by what production drives it."""

    name: str
    owned_inactive_generation: bool
    use_session_shards: bool
    ingest_workers: int
    production_driver: str


_BASELINE_ARM = _Arm(
    name="retained-active-index",
    owned_inactive_generation=False,
    use_session_shards=False,
    ingest_workers=1,
    production_driver="polylogue.storage.derived.raw.RawObservationPublisher.publish",
)
_REPLACEMENT_ARM = _Arm(
    name="fresh-generation-sealed-shard",
    owned_inactive_generation=True,
    use_session_shards=True,
    ingest_workers=2,
    production_driver="polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.open_cold_build_generation",
)


@dataclass(frozen=True, slots=True)
class _ArmRun:
    arm: _Arm
    output: FinishedBuildOutput
    archive_root: Path
    index_path: Path
    session_ids: tuple[str, ...]
    search_queries: tuple[str, ...]
    replayed_logical_sources: int
    adoption_deferred: int
    quarantined: int
    stage_timings_s: dict[str, float]
    stage_counts: dict[str, int]
    writer_apply_seconds: float

    @property
    def route_total_seconds(self) -> float:
        """The route's own whole-operation figure, as it reported it."""
        total = self.stage_timings_s.get("total")
        if total is None:
            raise AssertionError(f"{self.arm.name} reported no total stage time to attribute")
        return float(total)

    @property
    def dominant_stage(self) -> tuple[str, float]:
        """Name the phase that actually cost the most in this arm."""
        if not self.stage_timings_s:
            raise AssertionError(f"{self.arm.name} recorded no stage ledger to attribute its elapsed time to")
        total = self.route_total_seconds
        durations = {stage: seconds for stage, seconds in self.stage_timings_s.items() if stage != "total"}
        impossible = {stage: seconds for stage, seconds in durations.items() if seconds > total + _LEDGER_TOLERANCE_S}
        if impossible:
            raise AssertionError(
                f"{self.arm.name} stage ledger holds non-duration keys exceeding total={total:.3f}s: {impossible}"
            )
        if not durations:
            raise AssertionError(f"{self.arm.name} recorded only a total with no phase to attribute it to")
        return max(durations.items(), key=lambda item: item[1])


def test_replay_prefetch_counts_are_separate_from_stage_durations(tmp_path: Path) -> None:
    with revision_backfill._ParsedSessionSpill(tmp_path, max_cached_payload_bytes=None) as spill:
        prefetcher = revision_backfill._ReplaySpillPrefetcher(
            spill,
            archive_root=tmp_path,
            max_buffered_tree_bytes=1,
        )
        prefetcher.hits = 2
        prefetcher.reparse_hits = 1
        prefetcher.consumed = 1
        prefetcher.decode_seconds = 0.25

        timings = prefetcher.close()
        counts = prefetcher.counts()

    assert timings == {"spill_prefetch.decode_concurrent": 0.25}
    assert counts == {
        "spill_prefetch.hits": 2,
        "spill_prefetch.reparse_hits": 1,
        "spill_prefetch.consumed": 1,
    }


def _run_arm(template: Path, destination: Path, sealed: SealedRawInput, arm: _Arm) -> _ArmRun:
    """Complete one production arm over an isolated clone of the sealed input."""
    archive_root = clone_sealed_arm(template, destination, sealed)
    route_root = archive_root
    if arm.owned_inactive_generation:
        generation = IndexGenerationStore.for_archive_root(archive_root).create(
            source_snapshot="finished-build-equivalence"
        )
        owned = (generation.generation_id, generation.owner_id)
        route_root = Path(generation.index_path).parent
    else:
        owned = None

    probe = FinishedBuildResourceProbe.start()
    result = backfill_historical_revision_evidence(
        route_root,
        owned_inactive_generation=owned,
        ingest_workers=arm.ingest_workers,
        use_session_shards=arm.use_session_shards,
    )
    index_path = route_root / "index.db"
    with sqlite3.connect(f"file:{index_path}?mode=ro", uri=True) as conn:
        session_ids = tuple(str(row[0]) for row in conn.execute("SELECT session_id FROM sessions ORDER BY session_id"))
    read_session_ids = session_ids[:3]
    search_queries = ("amg1-payload",)
    output = capture_finished_build_output(
        route_root,
        index_path,
        work=finished_build_work_identity(
            sealed,
            profile=_WORK_PROFILE,
            routes=(backfill_historical_revision_evidence, revision_backfill._FrozenReplayShardTransport),
        ),
        route=FinishedBuildRoute.from_production_callable(arm.name, backfill_historical_revision_evidence),
        resource_probe=probe,
        session_ids=read_session_ids,
        search_queries=search_queries,
    )
    _parse_seconds, writer_apply_seconds = split_parse_and_apply_seconds(result.stage_timings_s)
    return _ArmRun(
        arm=arm,
        output=output,
        archive_root=route_root,
        index_path=index_path,
        session_ids=read_session_ids,
        search_queries=search_queries,
        replayed_logical_sources=result.replayed_logical_sources,
        adoption_deferred=result.adoption_deferred,
        quarantined=result.quarantined,
        stage_timings_s=dict(result.stage_timings_s),
        stage_counts=dict(result.stage_counts),
        writer_apply_seconds=writer_apply_seconds,
    )


@pytest.fixture(scope="module")
def _sealed_template(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, SealedRawInput]:
    """Build, census and seal one raw population both arms then clone."""
    template = tmp_path_factory.mktemp("finished-build-equivalence") / "sealed-input"
    bootstrap_archive_root(template)
    build_independent_raw_corpus(
        template,
        raw_count=_RAW_COUNT,
        avg_payload_bytes=_RAW_PAYLOAD_BYTES,
        authoritative_source=True,
    )
    census = census_historical_revision_evidence(template)
    assert census.scanned == _RAW_COUNT
    assert census.quarantined == 0
    sealed = seal_raw_input(template)
    assert sealed.raw_count == _RAW_COUNT
    finalize_archive_template(template)
    return template, sealed


@pytest.fixture(scope="module")
def _arms(
    _sealed_template: tuple[Path, SealedRawInput],
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[_ArmRun, _ArmRun]:
    """Complete both production arms once over the same sealed input."""
    template, sealed = _sealed_template
    root = tmp_path_factory.mktemp("finished-build-arms")
    return (
        _run_arm(template, root / _BASELINE_ARM.name, sealed, _BASELINE_ARM),
        _run_arm(template, root / _REPLACEMENT_ARM.name, sealed, _REPLACEMENT_ARM),
    )


def test_both_production_arms_finish_the_same_work(
    _sealed_template: tuple[Path, SealedRawInput],
    _arms: tuple[_ArmRun, _ArmRun],
) -> None:
    """The replacement route's completed output equals the baseline's."""
    _template, sealed = _sealed_template
    baseline, replacement = _arms

    # The comparison is only meaningful if the two arms really ran different
    # production shapes over one identical input.
    assert baseline.arm.production_driver != replacement.arm.production_driver
    assert baseline.index_path != replacement.index_path
    assert baseline.arm.owned_inactive_generation is False
    assert replacement.arm.owned_inactive_generation is True
    assert baseline.output.work == replacement.output.work
    assert baseline.output.route.variant != replacement.output.route.variant
    assert (
        baseline.output.route.callable_identity
        == replacement.output.route.callable_identity
        == "polylogue.sources.revision_backfill.backfill_historical_revision_evidence"
    )

    assert_finished_builds_equivalent(baseline.output, replacement.output)

    for run in (baseline, replacement):
        assert run.output.output_session_count == sealed.raw_count
        assert run.output.output_message_count and run.output.output_block_count


def test_neither_arm_hides_residual_or_refused_work(_arms: tuple[_ArmRun, _ArmRun]) -> None:
    """A partially finished or partly refused arm is not a comparable build."""
    for run in _arms:
        classified = run.replayed_logical_sources + run.adoption_deferred + run.quarantined
        assert classified == _RAW_COUNT, f"{run.arm.name} left raws unclassified"
        assert run.adoption_deferred == 0
        assert run.quarantined == 0
        # ``capture_finished_build_output`` already refuses a stale or
        # indebted generation; assert the receipt states the same facts so a
        # future relaxation there cannot pass silently here.
        assert run.output.snapshot.fts.source_rows == run.output.snapshot.fts.indexed_rows
        assert run.output.snapshot.fts.source_rows > 0
        assert run.output.snapshot.open_debt == ()
        assert run.output.snapshot.fts.public_searches
        assert all(hits for _query, hits in run.output.snapshot.fts.public_searches)


def test_each_arm_attributes_its_elapsed_time_to_a_named_phase(
    _sealed_template: tuple[Path, SealedRawInput],
    _arms: tuple[_ArmRun, _ArmRun],
) -> None:
    """Record whole-operation cost per arm and name its expensive phase.

    Timing is descriptive: this asserts that each arm carries a complete,
    attributable receipt, not that either arm is faster. A ranking needs the
    dedicated measurement host, and this deliberately does not manufacture
    one from an 8-raw fixture.
    """
    _template, sealed = _sealed_template
    receipts = []
    for run in _arms:
        resources = run.output.resources
        assert resources.elapsed_seconds > 0
        assert resources.storage_bytes > 0
        assert run.writer_apply_seconds > 0, f"{run.arm.name} reported no serialized writer-apply time"
        stage, stage_seconds = run.dominant_stage
        assert set(run.stage_counts).isdisjoint(run.stage_timings_s)
        # The route cannot have spent more time than the probe measured around
        # it. This is what stops a receipt from timing one inner phase and
        # presenting it as the finished operation.
        assert run.route_total_seconds <= resources.elapsed_seconds + _LEDGER_TOLERANCE_S, (
            f"{run.arm.name} route ledger total={run.route_total_seconds:.3f}s "
            f"exceeds measured elapsed={resources.elapsed_seconds:.3f}s"
        )
        receipts.append(
            {
                "arm": run.arm.name,
                "production_driver": run.arm.production_driver,
                "ingest_workers": run.arm.ingest_workers,
                "input_digest": sealed.digest,
                "input_bytes": sealed.byte_count,
                "input_raw_count": sealed.raw_count,
                "resources": resources.to_payload(),
                "writer_apply_seconds": run.writer_apply_seconds,
                "dominant_stage": stage,
                "dominant_stage_seconds": stage_seconds,
                "route_total_seconds": run.route_total_seconds,
                "outside_route_seconds": resources.elapsed_seconds - run.route_total_seconds,
                "stage_timings_s": run.stage_timings_s,
                "canonical_logical_digest": run.output.canonical_logical_digest,
                "output_session_count": run.output.output_session_count,
            }
        )
    emitted = emit_receipt(
        _MEASUREMENT_NAME,
        {
            "input": {
                "digest": sealed.digest,
                "bytes": sealed.byte_count,
                "raw_count": sealed.raw_count,
            },
            "arms": receipts,
            "verdict": {
                "conclusion": "equivalent-finished-output",
                "reason": (
                    "both production arms reached one canonical logical digest; "
                    "no transport or width ranking is claimed at this scale"
                ),
            },
        },
    )
    print(f"finished-build-equivalence receipt: {emitted}")


def test_finished_build_comparison_rejects_a_diverged_or_indebted_arm(tmp_path: Path) -> None:
    """Anti-vacuity: mutate a completed arm and watch the comparison fail.

    Two mutations are executed against a real completed archive, because a
    comparator that reads neither the derived rows nor the debt state would
    keep passing the test above after either of them:

    1. one unresolved ``convergence_debt`` row -- an arm that still owes
       derivation work must not be capturable as a finished build;
    2. one deleted ``blocks`` row -- a lost logical row must name its table.
    """
    template = tmp_path / "sealed-input"
    bootstrap_archive_root(template)
    build_independent_raw_corpus(template, raw_count=2, avg_payload_bytes=1_000, authoritative_source=True)
    census_historical_revision_evidence(template)
    sealed = seal_raw_input(template)
    finalize_archive_template(template)

    baseline = _run_arm(template, tmp_path / "baseline", sealed, _BASELINE_ARM)
    mutated = _run_arm(template, tmp_path / "mutated", sealed, _BASELINE_ARM)
    assert_finished_builds_equivalent(baseline.output, mutated.output)

    def recapture() -> FinishedBuildOutput:
        return capture_finished_build_output(
            mutated.archive_root,
            mutated.index_path,
            work=mutated.output.work,
            route=mutated.output.route,
            resource_probe=FinishedBuildResourceProbe.start(),
            session_ids=mutated.session_ids,
            search_queries=mutated.search_queries,
        )

    with sqlite3.connect(mutated.archive_root / "ops.db") as conn:
        conn.execute(
            """
            INSERT INTO convergence_debt (
                debt_id, stage, target_type, target_id, status, priority,
                attempts, last_error, created_at_ms, updated_at_ms
            ) VALUES ('equivalence-probe', 'insights', 'session', 'probe', 'failed', 0, 1, 'probe', 1, 1)
            """
        )
    with pytest.raises(AssertionError, match="convergence debt remains"):
        recapture()

    with sqlite3.connect(mutated.archive_root / "ops.db") as conn:
        conn.execute("DELETE FROM convergence_debt WHERE debt_id = 'equivalence-probe'")
    recaptured = recapture()
    assert_finished_builds_equivalent(baseline.output, recaptured)

    with sqlite3.connect(mutated.index_path) as conn:
        conn.execute("DELETE FROM blocks WHERE block_id = (SELECT block_id FROM blocks ORDER BY block_id LIMIT 1)")
    with pytest.raises(AssertionError, match="derived table blocks differs"):
        assert_finished_builds_equivalent(baseline.output, recapture())
