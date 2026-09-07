"""The query laws hold on every surface, and the census measures what it claims.

One adversarial corpus is built once through the production write route, then
every declared law runs across the Python facade, the root CLI ``find`` verb,
the daemon's ``/api/query-units`` route and the MCP ``query`` tool. Four
production mutations prove the laws can fail, and a serialized workload census
over a reflink copy of the same archive produces the resource receipts.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from tests.infra.query_census import (
    ArchiveSnapshot,
    CensusConcurrencyError,
    CensusObservation,
    CensusSnapshotError,
    reflink_archive_snapshot,
    run_workload_census,
)
from tests.infra.query_contract import (
    CENSUS_FAMILIES,
    QUERY_LAW_EXEMPTIONS,
    QUERY_LAWS,
    REQUIRED_PATHOLOGIES,
    SURFACE_NAMES,
    UNIT_PROBE_BY_UNIT,
    UNIT_PROBES,
    law_applies,
)
from tests.infra.query_corpus import QueryCorpus, build_query_corpus_sync
from tests.infra.query_differential import (
    QUERY_LAW_MUTANTS,
    LawMutant,
    LawRun,
    evaluate_query_laws,
    surface_bench,
)

pytestmark = pytest.mark.timeout(600)


@pytest.fixture(scope="module")
def query_law_corpus(tmp_path_factory: pytest.TempPathFactory) -> QueryCorpus:
    return build_query_corpus_sync(tmp_path_factory.mktemp("query-law-archive"))


def _evaluate(corpus: QueryCorpus, *, probes: Sequence[object] = UNIT_PROBES) -> LawRun:
    async def _run() -> LawRun:
        async with surface_bench(corpus.archive_root) as bench:
            return await evaluate_query_laws(bench, probes=probes)  # type: ignore[arg-type]

    return asyncio.run(_run())


@pytest.fixture(scope="module")
def query_law_run(query_law_corpus: QueryCorpus) -> LawRun:
    return _evaluate(query_law_corpus)


def test_query_law_corpus_carries_every_required_pathology(query_law_corpus: QueryCorpus) -> None:
    """The fixture is adversarial by declaration, not by good intentions.

    Anti-vacuity: dropping the duplicate, missing or late tool result, the
    lineage fan-out, the growth session or the large payload from the corpus
    builder fails its own completeness check before a law ever runs.
    """

    for pathology in REQUIRED_PATHOLOGIES:
        session_id = query_law_corpus.pathology_session(pathology)
        assert session_id, f"{pathology} names no session"
        assert f"claude-code-session:{session_id}" in query_law_corpus.session_ids


def test_query_law_every_declared_law_holds_on_every_surface(query_law_run: LawRun) -> None:
    """No law is violated at any declared unit/surface coordinate.

    Anti-vacuity: the mutation tests below break one production route each
    and require exactly the law that names the mutation to turn red here.
    """

    violations = [f"{outcome.coordinate}: {outcome.detail}" for outcome in query_law_run.violations]
    assert not violations, "query laws violated:\n" + "\n".join(violations)


def test_query_law_run_covers_every_declared_law(query_law_run: LawRun) -> None:
    """Every declared law was actually evaluated, not merely declared."""

    covered = query_law_run.covered_law_ids
    missing = sorted(law.law_id for law in QUERY_LAWS if law.law_id not in covered)
    assert not missing, f"declared laws that never ran: {missing}"


def test_query_law_run_covers_every_unit_that_is_not_exempt(query_law_run: LawRun) -> None:
    """A unit is either evaluated for a per-unit law or carries an exemption."""

    per_unit_laws = {outcome.law_id for outcome in query_law_run.outcomes if outcome.unit is not None}
    gaps: list[str] = []
    for law_id in sorted(per_unit_laws):
        evaluated = query_law_run.covered_units(law_id)
        for probe in UNIT_PROBES:
            if probe.unit in evaluated:
                continue
            if not law_applies(law_id, unit=probe.unit):
                continue
            gaps.append(f"{law_id}[{probe.unit}]")
    assert not gaps, f"units silently skipped by a law in force: {gaps}"


def test_query_law_exemptions_are_recorded_as_outcomes(query_law_run: LawRun) -> None:
    """Every declared exemption shows up in the run with its stated reason."""

    exempt = {(outcome.law_id, outcome.surface) for outcome in query_law_run.outcomes if outcome.status == "exempt"}
    for exemption in QUERY_LAW_EXEMPTIONS:
        if exemption.surface is None:
            continue
        assert (exemption.law_id, exemption.surface) in exempt, (
            f"exemption {exemption.law_id}/{exemption.surface} was declared but never recorded"
        )
    for outcome in query_law_run.outcomes:
        if outcome.status == "exempt":
            assert outcome.exemption.strip(), f"{outcome.coordinate} is exempt without a reason"


@pytest.mark.parametrize("mutant", QUERY_LAW_MUTANTS, ids=lambda mutant: mutant.mutant_id)
def test_query_law_mutation_fails_the_production_harness(mutant: LawMutant, query_law_corpus: QueryCorpus) -> None:
    """A broken pushdown, continuation, public type or ref route turns its law red.

    Each mutation patches one production seam -- structural predicate
    lowering, continuation advance, MCP serialization, row payload refs --
    and the run must report exactly the laws whose ``anti_vacuity`` sentence
    names that defect.
    """

    probes = (UNIT_PROBE_BY_UNIT["message"], UNIT_PROBE_BY_UNIT["action"])
    with mutant.apply():
        run = _evaluate(query_law_corpus, probes=probes)
    violated = {outcome.law_id for outcome in run.violations}
    missing = sorted(set(mutant.expected_violations) - violated)
    assert not missing, f"{mutant.mutant_id} did not turn {missing} red; violated={sorted(violated)}"


def test_query_law_unmutated_run_is_green_for_the_same_probes(query_law_corpus: QueryCorpus) -> None:
    """The mutation tests compare against a green baseline, not a red one."""

    probes = (UNIT_PROBE_BY_UNIT["message"], UNIT_PROBE_BY_UNIT["action"])
    run = _evaluate(query_law_corpus, probes=probes)
    assert not run.violations, [outcome.coordinate for outcome in run.violations]


# ---------------------------------------------------------------------------
# Workload census
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def census_snapshot(
    query_law_corpus: QueryCorpus, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[ArchiveSnapshot]:
    destination = tmp_path_factory.mktemp("query-law-census") / "copy"
    snapshot = reflink_archive_snapshot(query_law_corpus.archive_root, destination)
    yield snapshot
    shutil.rmtree(snapshot.root, ignore_errors=True)


@pytest.fixture(scope="module")
def census(census_snapshot: ArchiveSnapshot) -> tuple[CensusObservation, ...]:
    return asyncio.run(run_workload_census(census_snapshot))


def test_query_law_census_covers_every_declared_family(census: tuple[CensusObservation, ...]) -> None:
    assert tuple(observation.family.family_id for observation in census) == tuple(
        family.family_id for family in CENSUS_FAMILIES
    )


@pytest.mark.parametrize("family_id", [family.family_id for family in CENSUS_FAMILIES])
def test_query_law_census_family_matches_the_cheapest_correct_primitive(
    census: tuple[CensusObservation, ...], family_id: str
) -> None:
    """The routed answer equals the cheapest primitive's, for a larger plan.

    Anti-vacuity: the broken-pushdown mutation makes the routed identity set
    diverge from the primitive's, which fails this comparison.
    """

    observation = next(item for item in census if item.family.family_id == family_id)
    assert observation.identity_match, observation.identity_detail
    assert observation.rows_emitted == observation.rows_visited_cheapest
    assert observation.cheaper_primitive, (
        f"routed plan weight {observation.routed_plan_weight} is not above the primitive's "
        f"{observation.cheapest_plan_weight}, so the comparison proves nothing"
    )


@pytest.mark.parametrize("family_id", [family.family_id for family in CENSUS_FAMILIES])
def test_query_law_census_classifies_every_scan_and_materialization(
    census: tuple[CensusObservation, ...], family_id: str
) -> None:
    """Every full scan and temp B-tree is linked to a declared owner.

    Anti-vacuity: an allowance that matches nothing is reported as stale, so
    a declaration cannot survive the plan it was written for.
    """

    observation = next(item for item in census if item.family.family_id == family_id)
    unclassified = sorted({finding.detail for finding in observation.unclassified_scans})
    assert not unclassified, f"{family_id}: unclassified plan steps {unclassified}"
    assert not observation.stale_allowances, (
        f"{family_id}: declared allowances that matched nothing {observation.stale_allowances}"
    )


@pytest.mark.parametrize("family_id", [family.family_id for family in CENSUS_FAMILIES])
def test_query_law_census_predicate_is_pushed_below_the_ranked_window(
    census: tuple[CensusObservation, ...], family_id: str
) -> None:
    """The selective restriction reaches SQL before the first ranked window.

    This is the 2026-07-15 plan shape stated as an invariant: a predicate
    applied only after a global window filters an archive-wide materialization.
    """

    observation = next(item for item in census if item.family.family_id == family_id)
    assert observation.pushdown_held, observation.pushdown_detail


@pytest.mark.parametrize("family_id", [family.family_id for family in CENSUS_FAMILIES])
def test_query_law_census_receipt_carries_exact_resource_evidence(
    census: tuple[CensusObservation, ...], family_id: str
) -> None:
    """Each family emits one WorkloadReceipt whose budgets were evaluated.

    Anti-vacuity: a phase that reported no measurement produces a
    ``measurement-unavailable`` verdict rather than a pass, so an absent
    number cannot read as a green budget.
    """

    from polylogue.scenarios.workload import BudgetSemantics, BudgetVerdict

    observation = next(item for item in census if item.family.family_id == family_id)
    receipt = observation.receipt
    assert receipt.receipt_id
    phase = receipt.phases[0]
    assert phase.name == "query"
    assert phase.wall_ms is not None and phase.cpu_ms is not None
    assert phase.peak_rss_bytes is not None
    assert phase.response_bytes == observation.response_bytes
    assert phase.sqlite_vm_steps is not None
    assert phase.cleanup_complete is True
    gates = [result for result in receipt.budget_results if result.semantics is BudgetSemantics.REGRESSION_GATE]
    assert gates, "the receipt evaluated no regression gate"
    for result in gates:
        assert result.verdict is BudgetVerdict.PASS, (
            f"{family_id} exceeded {result.measure.value}: {result.observed} > {result.maximum}"
        )
    unavailable = [
        result for result in receipt.budget_results if result.verdict is BudgetVerdict.MEASUREMENT_UNAVAILABLE
    ]
    for result in unavailable:
        assert result.observed is None


def test_query_law_census_records_response_bytes_for_every_surface(query_law_corpus: QueryCorpus) -> None:
    """Response size is measured per surface, not assumed equal across them."""

    async def _run() -> dict[str, int]:
        async with surface_bench(query_law_corpus.archive_root) as bench:
            sizes: dict[str, int] = {}
            for name in SURFACE_NAMES:
                page = await bench.surface(name).page(
                    UNIT_PROBE_BY_UNIT["action"].scoped_expression, unit="action", limit=3
                )
                sizes[name] = page.response_bytes
            return sizes

    sizes = asyncio.run(_run())
    assert set(sizes) == set(SURFACE_NAMES)
    assert all(size > 0 for size in sizes.values()), sizes


def test_query_law_census_refuses_a_second_parallel_walk(census_snapshot: ArchiveSnapshot) -> None:
    """One census at a time: parallel EQP walks over a big archive are the hazard.

    Anti-vacuity: removing the module lock lets the second walk start, and
    this expectation of a refusal fails.
    """

    from tests.infra import query_census

    assert query_census._CENSUS_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(CensusConcurrencyError):
            asyncio.run(run_workload_census(census_snapshot, families=()))
    finally:
        query_census._CENSUS_LOCK.release()

    # The lock is released, so the same call now succeeds: the refusal is
    # about concurrency, not about a permanently wedged census.
    assert asyncio.run(run_workload_census(census_snapshot, families=())) == ()


def test_query_law_census_refuses_a_partial_or_self_targeted_snapshot(
    query_law_corpus: QueryCorpus, tmp_path: Path
) -> None:
    """A census copy is verified before it is read.

    Anti-vacuity: a destination missing a tier, or one that is the archive
    itself, is refused loudly instead of censused as if it were complete.
    """

    with pytest.raises(CensusSnapshotError):
        reflink_archive_snapshot(query_law_corpus.archive_root, query_law_corpus.archive_root)

    partial = tmp_path / "partial-source"
    partial.mkdir()
    (partial / "index.db").write_bytes(b"")
    with pytest.raises(CensusSnapshotError):
        reflink_archive_snapshot(partial, tmp_path / "partial-copy")


def test_query_law_census_leaves_durable_tiers_byte_identical(census_snapshot: ArchiveSnapshot) -> None:
    """The census never mutates durable evidence, even on its own copy."""

    from tests.infra.query_census import assert_snapshot_unmutated

    assert_snapshot_unmutated(census_snapshot)
    assert census_snapshot.root != census_snapshot.source
