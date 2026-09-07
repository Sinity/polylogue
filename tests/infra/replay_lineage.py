"""Generated lineage graphs and schedule oracles for replay-order laws.

A rebuild's replay schedule (``_lineage_aware_replay_schedule``) is allowed to
choose any visiting order; it is not allowed to change which logical keys are
replayed or what they produce. These builders make the graph shapes that
claim otherwise -- forests, cycles, self-parents, absent parents, aliased
spellings, contested cohort claims -- cheap to generate and to check against
an oracle derived from the graph rather than from the scheduler.
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.sources.revision_backfill import ReplaySchedule, ReplayTopologyState
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

#: Public origin prefix every generated key carries; Codex is the smallest
#: real parser that expresses a parent claim (``forked_from_id``).
CODEX_ORIGIN = "codex-session"

#: Provider-wire prefix for the same identity -- the alias spelling a legacy
#: row can still carry in ``raw_sessions.logical_source_key``.
CODEX_PROVIDER = Provider.CODEX.value

#: Native id of a parent claim that is deliberately never ingested.
ABSENT_PARENT = "absent-parent"


def codex_lineage_payload(
    session_id: str,
    message_texts: list[str],
    *,
    forked_from_id: str | None = None,
) -> bytes:
    """Build a codex JSONL raw with one message per ``message_texts`` entry.

    Mirrors how a real Codex resume payload looks: a ``session_meta`` record
    carrying ``forked_from_id`` when this session is a resume/fork, followed
    by ``response_item`` message records. Passing the SAME leading
    ``message_texts`` for a parent and one of its children (plus extra tail
    entries on the child) reproduces the on-disk shape #2467's deferred-tail
    extraction exists for: the child's JSONL physically re-contains the
    parent's entire prefix.
    """
    meta_payload: dict[str, object] = {"id": session_id, "timestamp": "2026-06-01T00:00:00Z"}
    if forked_from_id is not None:
        meta_payload["forked_from_id"] = forked_from_id
    lines = [json.dumps({"type": "session_meta", "payload": meta_payload}, separators=(",", ":"))]
    for position, text in enumerate(message_texts):
        lines.append(
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "id": f"m{position}",
                        "role": "user" if position % 2 == 0 else "assistant",
                        "content": [{"type": "input_text", "text": text}],
                    },
                },
                separators=(",", ":"),
            )
        )
    return ("\n".join(lines) + "\n").encode()


@dataclass(frozen=True, slots=True)
class LineageNode:
    """One generated session: its native id and the parent it claims."""

    native_id: str
    #: Native id this session claims as its parent. ``None`` is a root; a
    #: value absent from the graph is an external/missing parent; the node's
    #: own id is a self-parent claim.
    parent_native_id: str | None
    tail_length: int


@dataclass(frozen=True, slots=True)
class LineageGraph:
    """A generated lineage shape plus the order its raws were acquired in."""

    nodes: tuple[LineageNode, ...]
    #: Indices into ``nodes``, giving the order raws are written in. Replay
    #: membership and output must not depend on it.
    write_order: tuple[int, ...]

    @property
    def logical_keys(self) -> frozenset[str]:
        return frozenset(f"{CODEX_ORIGIN}:{node.native_id}" for node in self.nodes)

    def node(self, native_id: str) -> LineageNode:
        return next(node for node in self.nodes if node.native_id == native_id)

    def resolved_parent_key(self, key: str) -> str | None:
        """The parent edge the scheduler must resolve INSIDE this graph."""
        node = self.node(key.removeprefix(f"{CODEX_ORIGIN}:"))
        claimed = node.parent_native_id
        if claimed is None or claimed == node.native_id:
            return None
        parent_key = f"{CODEX_ORIGIN}:{claimed}"
        return parent_key if parent_key in self.logical_keys else None

    def expected_topology(self) -> dict[str, ReplayTopologyState]:
        """The state each key must carry, derived from the graph alone."""
        states: dict[str, ReplayTopologyState] = {}
        for node in self.nodes:
            key = f"{CODEX_ORIGIN}:{node.native_id}"
            claimed = node.parent_native_id
            if claimed is None:
                states[key] = ReplayTopologyState.ROOT
            elif claimed == node.native_id:
                states[key] = ReplayTopologyState.SELF_PARENT
            elif f"{CODEX_ORIGIN}:{claimed}" not in self.logical_keys:
                states[key] = ReplayTopologyState.UNRESOLVED_PARENT
            else:
                states[key] = ReplayTopologyState.DESCENDANT
        for key in self.cycle_keys():
            states[key] = ReplayTopologyState.CYCLE
        return states

    def cycle_keys(self) -> frozenset[str]:
        """Keys sitting ON a parent cycle, walked from the graph directly."""
        members: set[str] = set()
        for key in self.logical_keys:
            walk: list[str] = []
            cursor: str | None = key
            while cursor is not None and cursor not in walk:
                walk.append(cursor)
                cursor = self.resolved_parent_key(cursor)
            if cursor is not None and cursor == key:
                members.update(walk)
        return frozenset(members)

    def message_texts(self, native_id: str, _walked: tuple[str, ...] = ()) -> list[str]:
        """This session's messages, physically re-containing its parent's.

        A claim that closes a cycle contributes no prefix -- there is no
        well-founded ancestor text to replay -- which keeps generation total
        without inventing content.
        """
        node = self.node(native_id)
        parent_key = self.resolved_parent_key(f"{CODEX_ORIGIN}:{native_id}")
        prefix: list[str] = []
        if parent_key is not None:
            parent_id = parent_key.removeprefix(f"{CODEX_ORIGIN}:")
            if parent_id not in _walked:
                prefix = self.message_texts(parent_id, (*_walked, native_id))
        return [*prefix, *(f"{native_id}-tail-{index}" for index in range(node.tail_length))]


def generate_lineage_graph(seed: int, *, node_count: int) -> LineageGraph:
    """Draw one lineage shape: forests, deep chains, cycles, self-parents and
    absent parents all arise from the same claim draw, so a seed sweep covers
    the shapes fixed examples do not reach.

    Native ids are ``s00``..``sNN`` and a parent may be any index, so
    lexicographic order and lineage order disagree on most seeds.
    """
    rng = random.Random(seed)
    native_ids = [f"s{index:02d}" for index in range(node_count)]
    nodes: list[LineageNode] = []
    for index, native_id in enumerate(native_ids):
        claim = rng.choice(("none", "node", "node", "self", "absent"))
        if claim == "none":
            parent = None
        elif claim == "self":
            parent = native_id
        elif claim == "absent":
            parent = f"{ABSENT_PARENT}-{index}"
        else:
            parent = rng.choice(native_ids)
        nodes.append(LineageNode(native_id=native_id, parent_native_id=parent, tail_length=rng.randint(1, 3)))
    write_order = list(range(node_count))
    rng.shuffle(write_order)
    return LineageGraph(nodes=tuple(nodes), write_order=tuple(write_order))


def generate_fork_forest(seed: int, *, node_count: int) -> LineageGraph:
    """Draw a fork forest: several roots, each with zero or more children that
    physically re-contain their root's prefix, and no claim left dangling.

    This is the family a rebuild is expected to converge on identically
    whatever order it visits: every parent is present, no claim is
    self-referential, and no chain is deeper than one fork. Siblings give the
    duplicate-parent-claim case; ``write_order`` gives the reordered-source
    case.
    """
    rng = random.Random(0xF0 + seed)
    native_ids = [f"s{index:02d}" for index in range(node_count)]
    root_count = rng.randint(1, max(1, node_count - 1))
    roots = native_ids[:root_count]
    nodes = [
        LineageNode(native_id=native_id, parent_native_id=None, tail_length=rng.randint(1, 3)) for native_id in roots
    ]
    for native_id in native_ids[root_count:]:
        nodes.append(
            LineageNode(
                native_id=native_id,
                parent_native_id=rng.choice(roots),
                tail_length=rng.randint(1, 3),
            )
        )
    write_order = list(range(node_count))
    rng.shuffle(write_order)
    return LineageGraph(nodes=tuple(nodes), write_order=tuple(write_order))


def write_lineage_graph(archive: ArchiveStore, graph: LineageGraph) -> None:
    """Write ``graph``'s raws into an already-open archive, in its own
    acquisition order."""
    for acquisition, index in enumerate(graph.write_order):
        node = graph.nodes[index]
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=codex_lineage_payload(
                node.native_id,
                graph.message_texts(node.native_id),
                forked_from_id=node.parent_native_id,
            ),
            source_path=f"{node.native_id}.jsonl",
            acquired_at_ms=1 + acquisition,
        )


def seed_lineage_graph(root: Path, graph: LineageGraph) -> None:
    """Materialize ``graph`` as codex raws in a fresh archive at ``root``."""
    initialize_active_archive_root(root)
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        write_lineage_graph(archive, graph)


def alias_one_raw_logical_key(root: Path, *, source_path: str) -> str:
    """Rewrite the raw acquired from ``source_path`` to the provider-wire
    spelling of its logical key, and return that spelling.

    Legacy rows predate the public-origin normalization, so one identity can
    still be present under two spellings at once; the scheduler must
    recognize the second as an alias of the first rather than as an
    unrelated root at its own lexicographic position.
    """
    with sqlite3.connect(root / "source.db") as conn:
        row = conn.execute(
            "SELECT raw_id, logical_source_key FROM raw_sessions WHERE source_path = ?",
            (source_path,),
        ).fetchone()
        assert row is not None, f"no raw was acquired from {source_path}"
        raw_id, logical_key = row
        aliased = f"{CODEX_PROVIDER}:{str(logical_key).partition(':')[2]}"
        conn.execute(
            "UPDATE raw_sessions SET logical_source_key = ? WHERE raw_id = ?",
            (aliased, raw_id),
        )
    return aliased


def index_content_manifest(root: Path) -> dict[str, list[tuple[object, ...]]]:
    """Full ordered row dump of the content tables the replay writes --
    the same equivalence currency as PR #3469's MANIFESTS IDENTICAL proof."""
    order_column = {
        "sessions": "session_id",
        "messages": "message_id",
        "blocks": "block_id",
        "session_links": "src_session_id, dst_origin, dst_native_id, link_type",
    }
    with sqlite3.connect(root / "index.db") as conn:
        return {
            table: conn.execute(f"SELECT * FROM {table} ORDER BY {order}").fetchall()
            for table, order in order_column.items()
        }


def assert_schedule_is_faithful(schedule: ReplaySchedule, logical_keys: frozenset[str]) -> None:
    """Every law a schedule owes its caller, independent of graph shape.

    Anti-vacuity: dropping a member of a strongly connected component,
    inventing a parent outside the rebuild, or changing membership each
    break one of these assertions -- see the mutation tests in
    ``tests/property/test_replay_schedule_topology.py``.
    """
    order = list(schedule.order)
    assert len(order) == len(set(order)), f"schedule repeats a key: {order}"
    assert set(order) == set(logical_keys), (
        f"schedule membership differs from the rebuild's: missing "
        f"{sorted(set(logical_keys) - set(order))}, fabricated {sorted(set(order) - set(logical_keys))}"
    )
    assert set(schedule.topology) == set(logical_keys), "every key carries exactly one topology state"
    assert set(schedule.parent_of) == set(logical_keys), "every key carries a resolved parent entry"

    position = {key: index for index, key in enumerate(order)}
    edged = {ReplayTopologyState.DESCENDANT, ReplayTopologyState.ALIAS, ReplayTopologyState.CYCLE}
    for key, parent in schedule.parent_of.items():
        state = schedule.topology[key]
        if parent is None:
            assert state not in {ReplayTopologyState.DESCENDANT, ReplayTopologyState.ALIAS}, (
                f"{key} is typed {state} but resolved no parent"
            )
            continue
        assert parent in logical_keys, f"{key} resolved an invented parent {parent!r}"
        assert state in edged, f"{key} resolved parent {parent!r} but is typed {state}"
        if state is not ReplayTopologyState.CYCLE:
            assert position[key] > position[parent], f"{key} replays before its parent {parent}"

    cycle_keys = {key for key, state in schedule.topology.items() if state is ReplayTopologyState.CYCLE}
    for key in cycle_keys:
        parent = schedule.parent_of[key]
        assert parent is not None and parent in cycle_keys, (
            f"{key} is typed a cycle member but its parent {parent!r} is not on the cycle"
        )
    # One member per cycle -- the component's entry point -- necessarily
    # precedes its own parent; every other cycle member still follows one.
    entry_points = [key for key in cycle_keys if position[key] < position[schedule.parent_of[key] or key]]
    assert len(entry_points) == len(_cycle_components(cycle_keys, schedule.parent_of)), (
        "each cycle is entered exactly once"
    )


def _cycle_components(cycle_keys: set[str], parent_of: Mapping[str, str | None]) -> list[frozenset[str]]:
    """Partition cycle members into their individual cycles."""
    parents = {key: value for key, value in parent_of.items() if key in cycle_keys and value is not None}
    components: list[frozenset[str]] = []
    seen: set[str] = set()
    for key in sorted(cycle_keys):
        if key in seen:
            continue
        walk: list[str] = []
        cursor = key
        while cursor not in walk:
            walk.append(cursor)
            cursor = parents[cursor]
        seen.update(walk)
        components.append(frozenset(walk))
    return components
