"""Laws the rebuild replay schedule owes, over generated lineage graphs.

``_lineage_aware_replay_schedule`` reorders a rebuild's logical keys so a
parent's cohort replays before its children's. Ordering is the only thing it
may change: the same keys must be replayed, each typed by why it sits where
it does, and the archive must reach the same state whichever order it was
visited in. The fixed examples in ``tests/unit/sources/test_revision_backfill.py``
cover one parent with five children; these cover the shapes that family never
reaches -- cycles, self-parent claims, absent parents, aliased spellings,
contested cohort claims and reordered source items.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.core.enums import Provider
from polylogue.sources import revision_backfill
from polylogue.sources.revision_backfill import (
    ReplaySchedule,
    ReplayTopologyState,
    _lineage_aware_replay_schedule,
    _ParsedSessionSpill,
    backfill_historical_revision_evidence,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.replay_lineage import (
    CODEX_ORIGIN,
    LineageGraph,
    alias_one_raw_logical_key,
    assert_schedule_is_faithful,
    codex_lineage_payload,
    generate_fork_forest,
    generate_lineage_graph,
    index_content_manifest,
    seed_lineage_graph,
    write_lineage_graph,
)

#: Pathological-family seeds, chosen so the sweep covers every
#: ``ReplayTopologyState``: 0 has self-parents and absent parents, 1 and 2 add
#: multi-level descendants, 4 closes a two-key cycle, 13 a four-key one, 28
#: mixes a cycle with self-parents.
LAW_SEEDS = (0, 1, 2, 4, 13, 28)

#: The subset the replay laws sweep: every state, both cycle sizes, at a
#: third of the rebuild cost -- each seed replays the same archive three
#: times.
REPLAY_LAW_SEEDS = (0, 1, 4, 13)

#: Fork-forest seeds for the full-output law.
FOREST_SEEDS = (0, 1, 2)

NODE_COUNT = 6

#: Each of these tests builds several six-session archives and replays some
#: of them end to end -- seconds on a quiet host, minutes when the corpus run
#: and a dozen sibling workers are on the same disk. The default 120 s bound
#: is a hang guard, not a budget for that.
ARCHIVE_HEAVY_TIMEOUT_S = 600


def _census(root: Path) -> None:
    """Run the census stage a rebuild runs before it schedules anything."""
    with (
        ArchiveStore.open_existing(root, read_only=False) as archive,
        _ParsedSessionSpill(root, max_cached_payload_bytes=None) as spill,
    ):
        revision_backfill._census_historical_revision_evidence(
            archive,
            spill,
            selected_raw_ids=None,
            max_payload_bytes=None,
        )
        archive.commit()


def _schedule_of(root: Path) -> tuple[ReplaySchedule, frozenset[str]]:
    """Compute a censused archive's replay schedule the way replay does."""
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        _expanded, logical_keys = archive.expand_raw_membership_selection(None)
        with _ParsedSessionSpill(root, max_cached_payload_bytes=None) as spill:
            return _lineage_aware_replay_schedule(set(logical_keys), archive, spill, root), frozenset(logical_keys)


def _censused_schedule(root: Path) -> tuple[ReplaySchedule, frozenset[str]]:
    """Census ``root`` the way a rebuild does, then compute its schedule."""
    _census(root)
    return _schedule_of(root)


def _seeded_schedule(root: Path, graph: LineageGraph) -> tuple[ReplaySchedule, frozenset[str]]:
    """Seed, census and schedule ``graph`` through ONE archive open.

    Opening an archive retains a descriptor per durable tier past the
    context manager, so a test that builds many archives must not open each
    of them three times over.
    """
    initialize_active_archive_root(root)
    with (
        ArchiveStore.open_existing(root, read_only=False) as archive,
        _ParsedSessionSpill(root, max_cached_payload_bytes=None) as spill,
    ):
        write_lineage_graph(archive, graph)
        revision_backfill._census_historical_revision_evidence(
            archive,
            spill,
            selected_raw_ids=None,
            max_payload_bytes=None,
        )
        archive.commit()
        _expanded, logical_keys = archive.expand_raw_membership_selection(None)
        schedule = _lineage_aware_replay_schedule(set(logical_keys), archive, spill, root)
    return schedule, frozenset(logical_keys)


ScheduleFn = Callable[[set[str], ArchiveStore, _ParsedSessionSpill, Path], ReplaySchedule]


def _forced_schedule(order: list[str]) -> ScheduleFn:
    """A stand-in scheduler pinning ``order``, asserting nothing about topology.

    Used to replay one archive under an order the production scheduler would
    never choose, so the two runs can be compared.
    """

    def _schedule(
        logical_keys: set[str],
        archive: ArchiveStore,
        spill: _ParsedSessionSpill,
        archive_root: Path,
    ) -> ReplaySchedule:
        assert set(order) == set(logical_keys), "a forced order must cover the rebuild exactly"
        return ReplaySchedule(
            order=tuple(order),
            topology=dict.fromkeys(order, ReplayTopologyState.ROOT),
            parent_of=dict.fromkeys(order, None),
        )

    return _schedule


def _replay_under(
    root: Path,
    graph: LineageGraph,
    order: list[str] | None,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, list[tuple[object, ...]]], tuple[int, int, int]]:
    """Seed ``graph`` at ``root`` and replay it under ``order`` (``None`` = the
    production schedule). Returns the index manifest plus the replay outcome."""
    seed_lineage_graph(root, graph)
    with monkeypatch.context() as patched:
        if order is not None:
            patched.setattr(revision_backfill, "_lineage_aware_replay_schedule", _forced_schedule(order))
        result = backfill_historical_revision_evidence(root)
    return (
        index_content_manifest(root),
        (result.replayed_logical_sources, result.quarantined, result.adoption_deferred),
    )


def _schedule_variants(logical_keys: frozenset[str], seed: int) -> dict[str, list[str] | None]:
    """The three schedules the laws compare: lexicographic (the pre-fix
    order), a seeded shuffle, and the production lineage-aware order."""
    randomized = sorted(logical_keys)
    random.Random(seed * 104729).shuffle(randomized)
    return {"lexicographic": sorted(logical_keys), "randomized": randomized, "topological": None}


@pytest.mark.timeout(ARCHIVE_HEAVY_TIMEOUT_S)
@settings(max_examples=6, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    seed=st.integers(min_value=0, max_value=512),
    node_count=st.integers(min_value=2, max_value=7),
)
def test_generated_graph_schedule_is_faithful_and_typed(
    tmp_path_factory: pytest.TempPathFactory, seed: int, node_count: int
) -> None:
    """Every generated shape schedules every key exactly once, typed by the
    rule that placed it, and the states agree with an oracle read off the
    graph rather than off the scheduler.

    Anti-vacuity: a scheduler that dropped a cycle member, invented a parent
    or duplicated a key fails ``assert_schedule_is_faithful``; one that typed
    an absent parent as a plain root fails the oracle comparison. Both
    mutations are exercised explicitly below.
    """
    graph = generate_lineage_graph(seed, node_count=node_count)
    schedule, logical_keys = _seeded_schedule(tmp_path_factory.mktemp("generated-graph"), graph)

    assert logical_keys == graph.logical_keys
    assert_schedule_is_faithful(schedule, logical_keys)
    assert dict(schedule.topology) == graph.expected_topology()


@pytest.mark.parametrize("seed", LAW_SEEDS)
@pytest.mark.timeout(ARCHIVE_HEAVY_TIMEOUT_S)
def test_schedule_does_not_depend_on_source_item_order(tmp_path: Path, seed: int) -> None:
    """Acquiring the same sessions in a different order must not move any key
    in the schedule: the order is a function of the lineage graph alone."""
    graph = generate_lineage_graph(seed, node_count=NODE_COUNT)
    reordered = LineageGraph(nodes=graph.nodes, write_order=tuple(reversed(graph.write_order)))

    forward_schedule, forward_keys = _seeded_schedule(tmp_path / "forward", graph)
    reversed_schedule, reversed_keys = _seeded_schedule(tmp_path / "reversed", reordered)

    assert forward_keys == reversed_keys
    assert list(forward_schedule.order) == list(reversed_schedule.order)
    assert dict(forward_schedule.topology) == dict(reversed_schedule.topology)


@pytest.mark.parametrize("seed", REPLAY_LAW_SEEDS)
@pytest.mark.timeout(ARCHIVE_HEAVY_TIMEOUT_S)
def test_replay_membership_is_equal_under_every_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    """Whatever the shape -- cycles and dangling claims included -- the SET of
    sessions a rebuild produces and its replay outcome are the same under the
    lexicographic, randomized and lineage-aware schedules. Scheduling changes
    when work happens, never whether it happens.

    Anti-vacuity: a schedule that omits one key replays one session fewer;
    ``test_dropping_a_cycle_member_changes_what_gets_replayed`` runs that
    mutation through this same route and shows the assertion fire.
    """
    graph = generate_lineage_graph(seed, node_count=NODE_COUNT)
    variants = _schedule_variants(graph.logical_keys, seed)
    assert variants["lexicographic"] != variants["randomized"], "the forced orders must actually differ"

    observed: dict[str, tuple[frozenset[object], tuple[int, int, int]]] = {}
    for label, order in variants.items():
        manifest, outcome = _replay_under(tmp_path / label, graph, order, monkeypatch)
        observed[label] = (frozenset(row[0] for row in manifest["sessions"]), outcome)

    assert observed["lexicographic"] == observed["topological"]
    assert observed["randomized"] == observed["topological"]
    assert observed["topological"][0] == frozenset(graph.logical_keys)


@pytest.mark.parametrize("seed", FOREST_SEEDS)
@pytest.mark.timeout(ARCHIVE_HEAVY_TIMEOUT_S)
def test_fork_forest_output_is_equal_under_every_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    """On a fork forest -- every parent present, no claim self-referential or
    cyclic -- the three schedules must reach byte-identical index content, not
    merely the same session set. This is the family a rebuild is expected to
    converge on: the deferred-tail normalization repairs a child replayed
    early, so order becomes pure wall-clock.

    Deeper chains and cycles do NOT hold this law today; see
    ``test_replay_membership_is_equal_under_every_schedule`` for what does.
    """
    graph = generate_fork_forest(seed, node_count=NODE_COUNT)
    variants = _schedule_variants(graph.logical_keys, seed)

    manifests: dict[str, tuple[dict[str, list[tuple[object, ...]]], tuple[int, int, int]]] = {}
    for label, order in variants.items():
        manifests[label] = _replay_under(tmp_path / label, graph, order, monkeypatch)

    assert manifests["lexicographic"] == manifests["topological"]
    assert manifests["randomized"] == manifests["topological"]


def test_aliased_spelling_is_typed_and_replays_beside_its_identity(tmp_path: Path) -> None:
    """A legacy row spelling one identity in provider-wire form must be typed
    an alias of the public-origin spelling and scheduled right after it, not
    treated as an unrelated root at its own lexicographic position -- which
    for this fixture is last."""
    root = tmp_path / "aliased"
    initialize_active_archive_root(root)
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for native_id in ("aaa", "zzz"):
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=codex_lineage_payload(native_id, [f"{native_id}-0"]),
                source_path=f"{native_id}.jsonl",
                acquired_at_ms=1,
            )
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=codex_lineage_payload("aaa", ["aaa-0", "aaa-1"]),
            source_path="aaa-legacy.jsonl",
            acquired_at_ms=2,
        )
    _census(root)
    alias_spelling = alias_one_raw_logical_key(root, source_path="aaa-legacy.jsonl")

    schedule, logical_keys = _schedule_of(root)

    canonical_spelling = f"{CODEX_ORIGIN}:aaa"
    assert {alias_spelling, canonical_spelling} <= logical_keys
    assert_schedule_is_faithful(schedule, logical_keys)
    assert schedule.topology[alias_spelling] is ReplayTopologyState.ALIAS
    assert schedule.parent_of[alias_spelling] == canonical_spelling
    order = list(schedule.order)
    assert order.index(alias_spelling) == order.index(canonical_spelling) + 1
    assert order != sorted(logical_keys)


def test_contested_cohort_claim_resolves_to_the_newest_acquisition(tmp_path: Path) -> None:
    """When a key's raws claim different parents, the newest acquisition wins
    and the same claim wins on every run -- the schedule may not depend on
    which row SQLite happens to return first."""
    root = tmp_path / "contested"
    initialize_active_archive_root(root)
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for native_id in ("early-parent", "late-parent"):
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=codex_lineage_payload(native_id, [f"{native_id}-only"]),
                source_path=f"{native_id}.jsonl",
                acquired_at_ms=1,
            )
        for acquired_at_ms, claimed in ((2, "early-parent"), (3, "late-parent")):
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=codex_lineage_payload(
                    "contested",
                    [f"{claimed}-only", "contested-tail"],
                    forked_from_id=claimed,
                ),
                source_path=f"contested-{claimed}.jsonl",
                acquired_at_ms=acquired_at_ms,
            )

    schedule, logical_keys = _censused_schedule(root)
    repeated, _keys = _censused_schedule(root)

    contested = f"{CODEX_ORIGIN}:contested"
    assert_schedule_is_faithful(schedule, logical_keys)
    assert schedule.topology[contested] is ReplayTopologyState.DESCENDANT
    assert schedule.parent_of[contested] == f"{CODEX_ORIGIN}:late-parent"
    assert list(repeated.order) == list(schedule.order)


def _mutated(
    schedule: ReplaySchedule,
    *,
    order: tuple[str, ...] | None = None,
    parent_of: dict[str, str | None] | None = None,
) -> ReplaySchedule:
    """A copy of ``schedule`` with one field corrupted, for mutation tests."""
    return ReplaySchedule(
        order=schedule.order if order is None else order,
        topology=dict(schedule.topology),
        parent_of=dict(schedule.parent_of) if parent_of is None else parent_of,
    )


@pytest.fixture
def cyclic_schedule(tmp_path: Path) -> tuple[ReplaySchedule, frozenset[str]]:
    """A real schedule over seed 13's four-key lineage cycle."""
    graph = generate_lineage_graph(13, node_count=NODE_COUNT)
    schedule, logical_keys = _seeded_schedule(tmp_path / "cyclic", graph)
    assert any(state is ReplayTopologyState.CYCLE for state in schedule.topology.values())
    return schedule, logical_keys


def test_dropping_a_cycle_member_fails_the_schedule_laws(
    cyclic_schedule: tuple[ReplaySchedule, frozenset[str]],
) -> None:
    """Mutation: silently skipping one member of a strongly connected
    component must be caught, not absorbed as "the cycle was unorderable"."""
    schedule, logical_keys = cyclic_schedule
    dropped = next(key for key, state in schedule.topology.items() if state is ReplayTopologyState.CYCLE)
    mutant = _mutated(schedule, order=tuple(key for key in schedule.order if key != dropped))

    with pytest.raises(AssertionError, match="membership differs"):
        assert_schedule_is_faithful(mutant, logical_keys)


def test_inventing_a_parent_fails_the_schedule_laws(
    cyclic_schedule: tuple[ReplaySchedule, frozenset[str]],
) -> None:
    """Mutation: resolving a parent that is not in the rebuild at all."""
    schedule, logical_keys = cyclic_schedule
    parent_of = dict(schedule.parent_of)
    parent_of[schedule.order[0]] = f"{CODEX_ORIGIN}:never-ingested"
    mutant = _mutated(schedule, parent_of=parent_of)

    with pytest.raises(AssertionError, match="invented parent"):
        assert_schedule_is_faithful(mutant, logical_keys)


def test_changing_membership_fails_the_schedule_laws(
    cyclic_schedule: tuple[ReplaySchedule, frozenset[str]],
) -> None:
    """Mutation: replaying a key the rebuild never selected, and replaying one
    key twice."""
    schedule, logical_keys = cyclic_schedule

    fabricated = _mutated(schedule, order=(*schedule.order, f"{CODEX_ORIGIN}:fabricated"))
    with pytest.raises(AssertionError, match="membership differs"):
        assert_schedule_is_faithful(fabricated, logical_keys)

    duplicated = _mutated(schedule, order=(*schedule.order, schedule.order[0]))
    with pytest.raises(AssertionError, match="repeats a key"):
        assert_schedule_is_faithful(duplicated, logical_keys)


@pytest.mark.timeout(ARCHIVE_HEAVY_TIMEOUT_S)
def test_dropping_a_cycle_member_changes_what_gets_replayed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The membership law is not vacuous: forcing a schedule that omits one
    cycle member through the production rebuild route leaves that session out
    of the index."""
    graph = generate_lineage_graph(13, node_count=NODE_COUNT)
    complete_manifest, complete_outcome = _replay_under(tmp_path / "complete", graph, None, monkeypatch)

    keys = sorted(graph.logical_keys)
    dropped = keys[0]
    partial_root = tmp_path / "partial"
    seed_lineage_graph(partial_root, graph)
    with monkeypatch.context() as patched:
        patched.setattr(
            revision_backfill,
            "_lineage_aware_replay_schedule",
            lambda logical_keys, archive, spill, archive_root: ReplaySchedule(
                order=tuple(key for key in sorted(logical_keys) if key != dropped),
                topology={},
                parent_of={},
            ),
        )
        partial_result = backfill_historical_revision_evidence(partial_root)

    partial_sessions = {row[0] for row in index_content_manifest(partial_root)["sessions"]}
    complete_sessions = {row[0] for row in complete_manifest["sessions"]}
    assert dropped in complete_sessions
    assert dropped not in partial_sessions
    assert partial_result.replayed_logical_sources < complete_outcome[0]
