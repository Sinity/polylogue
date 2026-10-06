"""Equivalent finished builds across two configurations of the production replay route.

The sealed 516-raw measurement in
``tests/benchmarks/test_finished_build_measurement.py`` runs ONE selected
production profile and declares the others non-cells. It therefore never
compares two arms, and the finished-build comparator
(``assert_finished_builds_equivalent``) had no consumer at all.

This module is the comparison, at a synthetic scale that runs without live
data or a dedicated measurement host: one sealed raw population, cloned into
two isolated arms, each completed through the canonical retained replay
route (``RawObservationDerivation`` compute and publish on the daemon raw
owner's admitted worker) --

* baseline: retained replay into the active Index;
* replacement: retained replay into an owned inactive Index generation.

Both arms call the same replay entry point. They do not execute live intake
or cold-start orchestration; equivalence of those drivers is not established
by this fixture.

Equivalence is over the completed logical output (every comparable index
table, the public insight reads, FTS readiness, open debt and the canonical
digest), not over a database page image, and each arm keeps its own
production callable identity and resource receipt. Scale is deliberately not
a claim here: this fixes *what* the arms must agree on, so a host-window
measurement run only has to add the scale.

Anti-vacuity: ``test_finished_build_comparison_rejects_a_diverged_or_indebted_arm``
executes both mutations a hollow comparator would survive -- one unresolved
convergence-debt row and one deleted ``blocks`` row.

Both arms use the production retained-parse route and encode no timing
tolerance. Runtime measurements remain separate from output equivalence.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from devtools.measurement_receipts import emit_receipt
from polylogue.operations.raw_observation_derivation import make_raw_observation_derivation
from polylogue.sources import revision_backfill
from polylogue.sources.live import WatchSource
from polylogue.sources.live.cold_build import (
    ColdBuildGeneration,
    clear_cold_build_generation,
    register_cold_build_generation,
)
from polylogue.storage.derived.raw import RawObservationDerivation
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.write_lease import write_lease
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
from tests.infra.retained_replay import replay_retained_components
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
    """One destination configuration of the canonical retained replay route."""

    name: str
    owned_inactive_generation: bool


_BASELINE_ARM = _Arm(name="retained-active-index", owned_inactive_generation=False)
_REPLACEMENT_ARM = _Arm(name="fresh-owned-generation", owned_inactive_generation=True)
#: The production callable both arms execute, and the code the work identity binds.
_REPLAY_ROUTE = RawObservationDerivation.publish
_ROUTE_CODE = (make_raw_observation_derivation, RawObservationDerivation)


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


def test_replay_enrichment_counts_are_request_local() -> None:
    @revision_backfill._capture_replay_enrichment_degradations
    def replay_probe() -> revision_backfill.PreparedRevisionReplayResult:
        revision_backfill._count_enrichment_degradation("probe")
        return revision_backfill.PreparedRevisionReplayResult(0, 0, 0, 0, 0)

    first = replay_probe()
    second = replay_probe()

    assert first.stage_counts == {"replay_enrichment_degraded.probe": 1}
    assert second.stage_counts == {"replay_enrichment_degraded.probe": 1}


def _merged_publication_ledger(receipts: tuple[object, ...]) -> tuple[dict[str, float], dict[str, int]]:
    timings: dict[str, float] = {}
    counts: dict[str, int] = {}
    for receipt in receipts:
        if not isinstance(receipt, revision_backfill.PreparedRevisionReplayResult):
            continue
        for stage, seconds in receipt.stage_timings_s.items():
            timings[stage] = timings.get(stage, 0.0) + seconds
        for stage, count in receipt.stage_counts.items():
            counts[stage] = counts.get(stage, 0) + count
    return timings, counts


def _run_arm(template: Path, destination: Path, sealed: SealedRawInput, arm: _Arm) -> _ArmRun:
    """Complete one replay configuration over an isolated clone of the sealed input."""
    archive_root = clone_sealed_arm(template, destination, sealed)
    cold_build: ColdBuildGeneration | None = None
    if arm.owned_inactive_generation:
        # The owned generation is the registered cold-build destination, as the
        # daemon's fresh build engages it; the retained owner writes only there.
        with write_lease("test.finished-build.generation", archive_root=archive_root):
            cold_build = ColdBuildGeneration.begin(
                archive_root,
                reason="finished-build-equivalence",
                observed=ColdBuildGeneration.observe_source_baseline(
                    (WatchSource("fixture", archive_root / "absent"),)
                ),
                owner_id="finished-build-equivalence",
            )
        register_cold_build_generation(cold_build)
    generation = None if cold_build is None else cold_build.generation
    index_path = archive_root / "index.db" if generation is None else Path(generation.index_path)

    # Open the writer destination before preparation records its file identity,
    # as the production ingest route's retained destination step does.
    with write_lease("test.finished-build.destination", archive_root=archive_root):
        with (
            ArchiveStore.open_existing(archive_root, read_only=False)
            if cold_build is None
            else cold_build.open_writer()
        ):
            pass

    probe = FinishedBuildResourceProbe.start()
    try:
        run = replay_retained_components(archive_root, owned_generation=generation)
        if cold_build is not None:
            # A finished fresh build publishes its candidate; public reads then
            # serve the replacement generation.
            cold_build.promote()
    finally:
        if cold_build is not None:
            clear_cold_build_generation()
    stage_timings_s, stage_counts = _merged_publication_ledger(run.receipts)
    with sqlite3.connect(f"file:{index_path}?mode=ro", uri=True) as conn:
        session_ids = tuple(str(row[0]) for row in conn.execute("SELECT session_id FROM sessions ORDER BY session_id"))
    read_session_ids = session_ids[:3]
    search_queries = ("amg1-payload",)
    output = capture_finished_build_output(
        archive_root,
        index_path,
        work=finished_build_work_identity(sealed, profile=_WORK_PROFILE, routes=_ROUTE_CODE),
        route=FinishedBuildRoute.from_production_callable(arm.name, _REPLAY_ROUTE),
        resource_probe=probe,
        session_ids=read_session_ids,
        search_queries=search_queries,
    )
    writer_apply_seconds = sum(
        seconds for stage, seconds in stage_timings_s.items() if stage.endswith(".index_parsed_write")
    )
    return _ArmRun(
        arm=arm,
        output=output,
        archive_root=archive_root,
        index_path=index_path,
        session_ids=read_session_ids,
        search_queries=search_queries,
        replayed_logical_sources=run.replayed_logical_sources,
        adoption_deferred=run.adoption_deferred,
        quarantined=run.quarantined,
        stage_timings_s=stage_timings_s,
        stage_counts=stage_counts,
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
    sealed = seal_raw_input(template)
    assert sealed.raw_count == _RAW_COUNT
    finalize_archive_template(template)
    return template, sealed


@pytest.fixture(scope="module")
def _arms(
    _sealed_template: tuple[Path, SealedRawInput],
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[_ArmRun, _ArmRun]:
    """Complete both replay configurations once over the same sealed input."""
    template, sealed = _sealed_template
    root = tmp_path_factory.mktemp("finished-build-arms")
    return (
        _run_arm(template, root / _BASELINE_ARM.name, sealed, _BASELINE_ARM),
        _run_arm(template, root / _REPLACEMENT_ARM.name, sealed, _REPLACEMENT_ARM),
    )


def test_both_replay_configurations_finish_the_same_work(
    _sealed_template: tuple[Path, SealedRawInput],
    _arms: tuple[_ArmRun, _ArmRun],
) -> None:
    """The replacement route's completed output equals the baseline's."""
    _template, sealed = _sealed_template
    baseline, replacement = _arms

    # Configuration differences are real; different driver entry points are
    # not. The callable identity below must name the route actually executed.
    assert baseline.index_path != replacement.index_path
    assert baseline.arm.owned_inactive_generation is False
    assert replacement.arm.owned_inactive_generation is True
    assert baseline.output.work == replacement.output.work
    assert baseline.output.route.variant != replacement.output.route.variant
    assert (
        baseline.output.route.callable_identity
        == replacement.output.route.callable_identity
        == "polylogue.storage.derived.raw.RawObservationDerivation.publish"
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
                "replay_callable": run.output.route.callable_identity,
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
                    "both replay configurations reached one canonical logical digest; "
                    "no transport or width ranking is claimed at this scale"
                ),
            },
        },
    )
    # The receipt must attribute the callable we actually executed, never
    # claim that an uncalled live-intake/cold-start driver was exercised.
    observed_arms = json.loads(emitted.read_text(encoding="utf-8"))["measurement"]["arms"]
    assert [arm["replay_callable"] for arm in observed_arms] == [run.output.route.callable_identity for run in _arms]


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
            ) VALUES ('equivalence-probe', 'derived', 'session_id', 'probe', 'failed', 0, 1, 'probe', 1, 1)
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


def test_route_code_digest_moves_when_a_transitive_dependency_changes(tmp_path: Path) -> None:
    """Editing a module the route imports changes the work's code identity.

    Anti-vacuity: hashing only ``inspect.getsource(route)`` leaves the digest
    unchanged when ``polylogue/dep.py`` changes, so arms built from different
    code would compare as the same work.
    """
    import importlib.util

    from tests.infra.reindex_differential import route_code_digest

    package = tmp_path / "polylogue"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    dependency = package / "dep.py"
    dependency.write_text("RULE = 1\n", encoding="utf-8")
    route_file = package / "route.py"
    route_file.write_text(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from polylogue.dep import RULE\n\n\n"
        "def route() -> None:\n"
        "    return None\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("finished_build_route_probe", route_file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    before = route_code_digest((module.route,), checkout=tmp_path)
    dependency.write_text("RULE = 2\n", encoding="utf-8")
    after = route_code_digest((module.route,), checkout=tmp_path)

    assert before != after
