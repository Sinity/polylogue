"""Laws the replay schedule owes, over generated lineage graphs.

``_lineage_aware_replay_schedule`` orders a retained component's byte-typed
rebuild keys so a parent's cohort replays before its children's. Ordering is
the only thing it may change: the same keys must be scheduled, each typed by
why it sits where it does. The schedule laws call it directly over a real
Source tier: its representative raws come from the production Source query
and its parent claims from the production retained parse.

The replay laws drive the canonical retained owner
(``tests.infra.retained_replay``) over the same graph acquired in different
orders, and require the same index outcome whichever order was used. The
fixed examples in ``tests/unit/sources/test_revision_backfill.py`` cover one
parent with five children; these cover the shapes that family never reaches
-- cycles, self-parents, absent parents, aliased spellings, contested claims
and reordered source items.
"""

from __future__ import annotations

import random
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.core.enums import Provider
from polylogue.sources.parsers.base_models import ParsedSession
from polylogue.sources.revision_backfill import (
    ReplaySchedule,
    ReplayTopologyState,
    _lineage_aware_replay_schedule,
    _PreparedReplayInputs,
    parse_retained_raw_sessions,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead
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
)
from tests.infra.retained_jsonl import prepared_source_fixture
from tests.infra.retained_replay import RetainedReplayRun, replay_retained_components

#: Pathological-family seeds, chosen so the sweep covers every
#: ``ReplayTopologyState``: 0 has self-parents and absent parents, 1 and 2 add
#: multi-level descendants, 4 closes a two-key cycle, 13 a four-key one, 28
#: mixes a cycle with self-parents.
LAW_SEEDS = (0, 1, 2, 4, 13, 28)

#: The subset the replay laws sweep: every state, both cycle sizes, at a
#: third of the rebuild cost -- each seed replays the same graph three times.
REPLAY_LAW_SEEDS = (0, 1, 4, 13)

#: Fork-forest seeds for the full-output law.
FOREST_SEEDS = (0, 1, 2)

NODE_COUNT = 6

#: Each replay law builds several six-session archives and replays each end to
#: end -- seconds on a quiet host, minutes when sibling workers share the disk.
#: The default 120 s bound is a hang guard, not a budget for that.
ARCHIVE_HEAVY_TIMEOUT_S = 600


class _RetainedParsedInputs(_PreparedReplayInputs):
    """Parent claims read through the production retained parse of each raw."""

    def __init__(self, source_read: PreparedSessionSourceRead) -> None:
        super().__init__({})
        self._source_read = source_read

    def for_raw(self, raw_id: str) -> tuple[Sequence[ParsedSession], int]:
        return parse_retained_raw_sessions(self._source_read, raw_id), 0


def _assign_rebuild_keys(root: Path) -> frozenset[str]:
    """Give each acquired raw the byte-typed rebuild key of its own session.

    The canonical route schedules byte-typed rebuild keys (``raw_sessions.
    logical_source_key``); acquisition leaves it unset until a revision
    chain records it. The key is the raw's own parsed session identity, so a
    raw re-acquiring a session shares that session's key. Written through an
    independent Source connection, the way legacy rows are planted.
    """
    with sqlite3.connect(f"file:{root / 'source.db'}?mode=ro", uri=True) as source:
        raw_ids = [str(row[0]) for row in source.execute("SELECT raw_id FROM raw_sessions")]
    with prepared_source_fixture(root) as source_read:
        keys = {
            raw_id: f"{CODEX_ORIGIN}:{parse_retained_raw_sessions(source_read, raw_id)[0].provider_session_id}"
            for raw_id in raw_ids
        }
    with sqlite3.connect(root / "source.db") as source:
        source.executemany(
            "UPDATE raw_sessions SET logical_source_key = ? WHERE raw_id = ?",
            [(key, raw_id) for raw_id, key in keys.items()],
        )
    return frozenset(keys.values())


def _rebuild_keys(root: Path) -> frozenset[str]:
    with sqlite3.connect(f"file:{root / 'source.db'}?mode=ro", uri=True) as source:
        return frozenset(
            str(row[0])
            for row in source.execute(
                "SELECT DISTINCT logical_source_key FROM raw_sessions WHERE logical_source_key IS NOT NULL"
            )
        )


def _schedule_of(root: Path) -> tuple[ReplaySchedule, frozenset[str]]:
    """Schedule ``root``'s rebuild keys with the production scheduler."""
    logical_keys = _rebuild_keys(root)
    with prepared_source_fixture(root) as source_read:
        schedule = _lineage_aware_replay_schedule(set(logical_keys), source_read, _RetainedParsedInputs(source_read))
    return schedule, logical_keys


def _seeded_schedule(root: Path, graph: LineageGraph) -> tuple[ReplaySchedule, frozenset[str]]:
    """Acquire ``graph``, assign its rebuild keys, then schedule them."""
    seed_lineage_graph(root, graph)
    _assign_rebuild_keys(root)
    return _schedule_of(root)


def _acquired_in(graph: LineageGraph, key_order: Sequence[str]) -> LineageGraph:
    """The same graph, acquired in ``key_order``."""
    index_of = {f"{CODEX_ORIGIN}:{node.native_id}": index for index, node in enumerate(graph.nodes)}
    return LineageGraph(nodes=graph.nodes, write_order=tuple(index_of[key] for key in key_order))


def _replay_acquired_in(
    root: Path, graph: LineageGraph, key_order: Sequence[str]
) -> tuple[dict[str, list[tuple[object, ...]]], RetainedReplayRun]:
    """Acquire ``graph`` in ``key_order`` at ``root`` and replay it canonically."""
    seed_lineage_graph(root, _acquired_in(graph, key_order))
    run = replay_retained_components(root)
    return index_content_manifest(root), run


def _order_variants(graph: LineageGraph, seed: int, tmp_path: Path) -> dict[str, list[str]]:
    """Lexicographic, a seeded shuffle, and the production lineage order."""
    keys = sorted(graph.logical_keys)
    randomized = list(keys)
    random.Random(seed * 104729).shuffle(randomized)
    lineage, _keys = _seeded_schedule(tmp_path / "lineage-order", graph)
    return {"lexicographic": keys, "randomized": randomized, "lineage": list(lineage.order)}


def _outcome(run: RetainedReplayRun) -> tuple[int, int, int]:
    return (run.replayed_logical_sources, run.quarantined, run.adoption_deferred)


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
def test_replay_membership_is_equal_under_every_acquisition_order(tmp_path: Path, seed: int) -> None:
    """Whatever the shape -- cycles and dangling claims included -- the SET of
    sessions canonical replay produces and its outcome are the same whether
    the raws were acquired in lexicographic, randomized or lineage order.
    Order changes when work happens, never whether it happens.

    Anti-vacuity: ``test_leaving_a_member_out_changes_what_gets_replayed``
    replays one session fewer through this same route and shows the set
    assertion fire.
    """
    graph = generate_lineage_graph(seed, node_count=NODE_COUNT)
    variants = _order_variants(graph, seed, tmp_path)
    assert variants["lexicographic"] != variants["randomized"], "the orders must actually differ"

    observed: dict[str, tuple[frozenset[object], tuple[int, int, int]]] = {}
    for label, order in variants.items():
        manifest, run = _replay_acquired_in(tmp_path / label, graph, order)
        observed[label] = (frozenset(row[0] for row in manifest["sessions"]), _outcome(run))

    assert observed["lexicographic"] == observed["lineage"]
    assert observed["randomized"] == observed["lineage"]
    assert observed["lineage"][0] == frozenset(graph.logical_keys)


@pytest.mark.parametrize("seed", FOREST_SEEDS)
@pytest.mark.timeout(ARCHIVE_HEAVY_TIMEOUT_S)
def test_fork_forest_output_is_equal_under_every_acquisition_order(tmp_path: Path, seed: int) -> None:
    """On a fork forest -- every parent present, no claim self-referential or
    cyclic -- the three acquisition orders must reach byte-identical index
    content, not merely the same session set. The deferred-tail normalization
    repairs a child replayed before its parent, so order is pure wall-clock.

    Deeper chains and cycles do NOT hold this law today; see
    ``test_replay_membership_is_equal_under_every_acquisition_order`` for what
    does.
    """
    graph = generate_fork_forest(seed, node_count=NODE_COUNT)
    variants = _order_variants(graph, seed, tmp_path)

    observed = {label: _replay_acquired_in(tmp_path / label, graph, order) for label, order in variants.items()}
    manifests = {label: (manifest, _outcome(run)) for label, (manifest, run) in observed.items()}

    assert manifests["lexicographic"] == manifests["lineage"]
    assert manifests["randomized"] == manifests["lineage"]


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
                canonical_source_path=f"{native_id}.jsonl",
                acquired_at_ms=1,
            )
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=codex_lineage_payload("aaa", ["aaa-0", "aaa-1"]),
            source_path="aaa-legacy.jsonl",
            canonical_source_path="aaa-legacy.jsonl",
            acquired_at_ms=2,
        )
    _assign_rebuild_keys(root)
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
                canonical_source_path=f"{native_id}.jsonl",
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
                canonical_source_path=f"contested-{claimed}.jsonl",
                acquired_at_ms=acquired_at_ms,
            )
    _assign_rebuild_keys(root)

    schedule, logical_keys = _schedule_of(root)
    repeated, _keys = _schedule_of(root)

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
def test_leaving_a_member_out_changes_what_gets_replayed(tmp_path: Path) -> None:
    """The membership law is not vacuous: replaying a cycle graph without one
    of its raws through the canonical route leaves that session out of the
    index, and replays one logical source fewer."""
    graph = generate_lineage_graph(13, node_count=NODE_COUNT)
    complete_manifest, complete_run = _replay_acquired_in(tmp_path / "complete", graph, sorted(graph.logical_keys))

    partial_root = tmp_path / "partial"
    seed_lineage_graph(partial_root, graph)
    with sqlite3.connect(f"file:{partial_root / 'source.db'}?mode=ro", uri=True) as source:
        raws = [(str(row[0]), str(row[1])) for row in source.execute("SELECT raw_id, source_path FROM raw_sessions")]
    dropped_raw, dropped_path = min(raws, key=lambda raw: raw[1])
    dropped_native_id = dropped_path.removesuffix(".jsonl")
    partial_run = replay_retained_components(
        partial_root, selected_raw_ids=[raw_id for raw_id, _path in raws if raw_id != dropped_raw]
    )

    def native_ids(manifest: dict[str, list[tuple[object, ...]]]) -> set[str]:
        return {str(session_id).rpartition(":")[2] for session_id in (row[0] for row in manifest["sessions"])}

    assert dropped_native_id in native_ids(complete_manifest)
    assert dropped_native_id not in native_ids(index_content_manifest(partial_root))
    assert partial_run.replayed_logical_sources < complete_run.replayed_logical_sources
