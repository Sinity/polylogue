"""Session topology read model tests (#866).

Synthetic corpora with known forks/sidechains/subagents/continuations and
unresolved native parents, asserting that the derived topology matches
expectations across roots/ancestors/descendants/siblings + cycle and
unresolved-edge surfaces.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.core.enums import Origin
from polylogue.insights.topology import (
    SessionTopology,
    TopologyEdgeKind,
)
from polylogue.storage.insights.topology import derive_session_topology_sync
from polylogue.storage.runtime import SessionRecord
from polylogue.types import ContentHash, SessionId
from tests.infra.storage_records import SessionBuilder, db_setup


# Archive session ids derive from the builder's native id (``ext-<conv_id>``)
# and the claude-code origin. Parent references carry the parent's native id
# (``ext-<parent>``) so the archive link resolver can match
# ``dst_native_id`` against ``sessions.native_id``.
def _native(token: str) -> str:
    return f"claude-code-session:ext-{token}"


def _seed_lineage(db_path: Path) -> None:
    """Seed: root → continuation child → fork grandchild, plus subagent + sidechain off the root."""

    SessionBuilder(db_path, "root").provider("claude-code").title("Root session").add_message(
        role="user", text="kickoff"
    ).save()

    SessionBuilder(db_path, "continuation").provider("claude-code").title("Resume").parent_session(
        "ext-root"
    ).branch_type("continuation").add_message(role="user", text="continue").save()

    SessionBuilder(db_path, "fork").provider("claude-code").title("Fork from continuation").parent_session(
        "ext-continuation"
    ).branch_type("fork").add_message(role="user", text="fork").save()

    SessionBuilder(db_path, "subagent").provider("claude-code").title("Subagent off root").parent_session(
        "ext-root"
    ).branch_type("subagent").add_message(role="assistant", text="agent").save()

    SessionBuilder(db_path, "sidechain").provider("claude-code").title("Sidechain off root").parent_session(
        "ext-root"
    ).branch_type("sidechain").add_message(role="user", text="side").save()


@pytest.mark.asyncio
async def test_topology_rooted_subtree_from_any_node(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    _seed_lineage(db_path)

    polylogue = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        # Requesting topology from a deep descendant must still anchor at the root.
        topo = await polylogue.get_session_topology(_native("fork"))
    finally:
        await polylogue.close()

    assert isinstance(topo, SessionTopology)
    assert str(topo.root_id) == _native("root")
    assert str(topo.target_id) == _native("fork")
    assert not topo.cycle_detected

    node_ids = {str(node.session_id) for node in topo.nodes}
    assert node_ids == {_native(t) for t in ("root", "continuation", "fork", "subagent", "sidechain")}

    by_id = {str(node.session_id): node for node in topo.nodes}
    assert by_id[_native("root")].is_root is True
    assert by_id[_native("root")].depth == 0
    assert by_id[_native("continuation")].depth == 1
    assert by_id[_native("fork")].depth == 2  # via continuation
    assert by_id[_native("subagent")].depth == 1
    assert by_id[_native("sidechain")].depth == 1

    edge_by_child = {str(edge.child_id): edge for edge in topo.edges if edge.resolved}
    assert edge_by_child[_native("continuation")].kind is TopologyEdgeKind.CONTINUATION
    assert edge_by_child[_native("fork")].kind is TopologyEdgeKind.FORK
    assert edge_by_child[_native("subagent")].kind is TopologyEdgeKind.SUBAGENT
    assert edge_by_child[_native("sidechain")].kind is TopologyEdgeKind.SIDECHAIN


@pytest.mark.asyncio
async def test_topology_ancestors_descendants_siblings(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    _seed_lineage(db_path)

    polylogue = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        topo = await polylogue.get_session_topology(_native("root"))
    finally:
        await polylogue.close()
    assert topo is not None

    assert [str(cid) for cid in topo.ancestors(_native("fork"))] == [_native("root"), _native("continuation")]
    assert [str(cid) for cid in topo.ancestors(_native("root"))] == []

    assert {str(cid) for cid in topo.descendants(_native("root"))} == {
        _native("continuation"),
        _native("fork"),
        _native("subagent"),
        _native("sidechain"),
    }
    assert [str(cid) for cid in topo.descendants(_native("continuation"))] == [_native("fork")]

    sibling_ids = {str(cid) for cid in topo.siblings(_native("subagent"))}
    assert sibling_ids == {_native("continuation"), _native("sidechain")}


@pytest.mark.asyncio
async def test_topology_missing_session_returns_none(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    polylogue = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        topo = await polylogue.get_session_topology("never-ingested")
    finally:
        await polylogue.close()
    assert topo is None


@pytest.mark.asyncio
async def test_topology_unresolved_native_parent_via_session_links(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)

    # An orphan: it asserts a native parent pointer whose parent
    # session has not been ingested, so the archive link resolver leaves
    # the session_links row unresolved (dst_session_id IS NULL). The topology
    # must surface this as an unresolved edge so late repair (#866 AC) has
    # something to reconcile.
    SessionBuilder(db_path, "orphan").provider("claude-code").title("Orphan child").parent_session(
        "missing-parent-uuid"
    ).add_message(role="user", text="orphan").save()

    polylogue = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        topo = await polylogue.get_session_topology(_native("orphan"))
    finally:
        await polylogue.close()
    assert topo is not None
    # The orphan is its own topology root in the resolved subtree.
    assert str(topo.root_id) == _native("orphan")

    unresolved = topo.unresolved_edges()
    assert len(unresolved) == 1
    edge = unresolved[0]
    assert edge.resolved is False
    assert edge.parent_id is None
    assert edge.parent_native_id == "missing-parent-uuid"
    assert edge.kind is TopologyEdgeKind.UNRESOLVED_NATIVE


def test_topology_sync_derivation_cycle_detected() -> None:
    """Synthetic cycle: A → B → A must be reported, not infinite-loop."""

    a = SessionRecord(
        session_id=SessionId("A"),
        origin=Origin.from_string("unknown-export"),
        native_id="ext-A",
        content_hash=ContentHash("hash-a"),
        parent_session_id=SessionId("B"),
    )
    b = SessionRecord(
        session_id=SessionId("B"),
        origin=Origin.from_string("unknown-export"),
        native_id="ext-B",
        content_hash=ContentHash("hash-b"),
        parent_session_id=SessionId("A"),
    )
    by_id = {"A": a, "B": b}

    topo = derive_session_topology_sync(
        "A",
        fetch=by_id.get,
        fetch_children=lambda _cid: [],
    )
    assert topo is not None
    assert topo.cycle_detected is True
    # The cycle terminates the ancestry walk at the first repeated node.
    assert str(topo.root_id) in {"A", "B"}


def test_topology_sync_derivation_resolves_full_tree() -> None:
    """Sync derivation walks ancestors then descendants without DB."""

    root = SessionRecord(
        session_id=SessionId("root"),
        origin=Origin.from_string("codex-session"),
        native_id="ext-root",
        content_hash=ContentHash("h-root"),
    )
    cont = SessionRecord(
        session_id=SessionId("cont"),
        origin=Origin.from_string("codex-session"),
        native_id="ext-cont",
        content_hash=ContentHash("h-cont"),
        parent_session_id=SessionId("root"),
        branch_type=None,
    )
    children: dict[str, list[SessionRecord]] = {"root": [cont], "cont": []}
    by_id = {"root": root, "cont": cont}

    topo = derive_session_topology_sync(
        "cont",
        fetch=by_id.get,
        fetch_children=lambda cid: children.get(cid, []),
    )
    assert topo is not None
    assert not topo.cycle_detected
    assert str(topo.root_id) == "root"
    ids = {str(node.session_id) for node in topo.nodes}
    assert ids == {"root", "cont"}
    # cont has no branch_type → edge kind falls back to UNKNOWN.
    cont_edge = next(edge for edge in topo.edges if str(edge.child_id) == "cont")
    assert cont_edge.kind is TopologyEdgeKind.UNKNOWN
    assert cont_edge.resolved is True
