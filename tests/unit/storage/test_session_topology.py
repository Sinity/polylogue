"""Session topology read model tests (#866).

Synthetic corpora with known forks/sidechains/subagents/continuations and
unresolved native parents, asserting that the derived topology matches
expectations across roots/ancestors/descendants/siblings + cycle and
unresolved-edge surfaces.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.insights.topology import (
    SessionTopology,
    TopologyEdgeKind,
)
from polylogue.storage.insights.topology import derive_session_topology_sync
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.runtime import ConversationRecord
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.types import ContentHash, ConversationId
from tests.infra.storage_records import ConversationBuilder, db_setup


def _seed_lineage(db_path: Path) -> None:
    """Seed: root → continuation child → fork grandchild, plus subagent + sidechain off the root."""

    ConversationBuilder(db_path, "root").provider("claude-code").title("Root session").add_message(
        role="user", text="kickoff"
    ).save()

    ConversationBuilder(db_path, "continuation").provider("claude-code").title("Resume").parent_conversation(
        "root"
    ).branch_type("continuation").add_message(role="user", text="continue").save()

    ConversationBuilder(db_path, "fork").provider("claude-code").title("Fork from continuation").parent_conversation(
        "continuation"
    ).branch_type("fork").add_message(role="user", text="fork").save()

    ConversationBuilder(db_path, "subagent").provider("claude-code").title("Subagent off root").parent_conversation(
        "root"
    ).branch_type("subagent").add_message(role="assistant", text="agent").save()

    ConversationBuilder(db_path, "sidechain").provider("claude-code").title("Sidechain off root").parent_conversation(
        "root"
    ).branch_type("sidechain").add_message(role="user", text="side").save()


@pytest.mark.asyncio
async def test_topology_rooted_subtree_from_any_node(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    _seed_lineage(db_path)

    async with ConversationRepository(backend=SQLiteBackend(db_path=db_path)) as repo:
        # Requesting topology from a deep descendant must still anchor at the root.
        topo = await repo.get_session_topology("fork")

    assert isinstance(topo, SessionTopology)
    assert str(topo.root_id) == "root"
    assert str(topo.target_id) == "fork"
    assert not topo.cycle_detected

    node_ids = {str(node.conversation_id) for node in topo.nodes}
    assert node_ids == {"root", "continuation", "fork", "subagent", "sidechain"}

    by_id = {str(node.conversation_id): node for node in topo.nodes}
    assert by_id["root"].is_root is True
    assert by_id["root"].depth == 0
    assert by_id["continuation"].depth == 1
    assert by_id["fork"].depth == 2  # via continuation
    assert by_id["subagent"].depth == 1
    assert by_id["sidechain"].depth == 1

    edge_by_child = {str(edge.child_id): edge for edge in topo.edges if edge.resolved}
    assert edge_by_child["continuation"].kind is TopologyEdgeKind.CONTINUATION
    assert edge_by_child["fork"].kind is TopologyEdgeKind.FORK
    assert edge_by_child["subagent"].kind is TopologyEdgeKind.SUBAGENT
    assert edge_by_child["sidechain"].kind is TopologyEdgeKind.SIDECHAIN


@pytest.mark.asyncio
async def test_topology_ancestors_descendants_siblings(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    _seed_lineage(db_path)

    async with ConversationRepository(backend=SQLiteBackend(db_path=db_path)) as repo:
        topo = await repo.get_session_topology("root")
    assert topo is not None

    assert [str(cid) for cid in topo.ancestors("fork")] == ["root", "continuation"]
    assert [str(cid) for cid in topo.ancestors("root")] == []

    assert {str(cid) for cid in topo.descendants("root")} == {
        "continuation",
        "fork",
        "subagent",
        "sidechain",
    }
    assert [str(cid) for cid in topo.descendants("continuation")] == ["fork"]

    sibling_ids = {str(cid) for cid in topo.siblings("subagent")}
    assert sibling_ids == {"continuation", "sidechain"}


@pytest.mark.asyncio
async def test_topology_missing_conversation_returns_none(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    async with ConversationRepository(backend=SQLiteBackend(db_path=db_path)) as repo:
        topo = await repo.get_session_topology("never-ingested")
    assert topo is None


@pytest.mark.asyncio
async def test_topology_unresolved_native_parent_via_provider_meta(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)

    # An orphan: no resolved parent row, but provider_meta carries the
    # native pointer. The topology must surface this as an unresolved edge
    # so late repair (#866 AC) has something to reconcile.
    ConversationBuilder(db_path, "orphan").provider("claude-code").title("Orphan child").provider_meta(
        {"parent_session_id": "missing-parent-uuid"}
    ).add_message(role="user", text="orphan").save()

    async with ConversationRepository(backend=SQLiteBackend(db_path=db_path)) as repo:
        topo = await repo.get_session_topology("orphan")
    assert topo is not None
    # The orphan is its own topology root in the resolved subtree.
    assert str(topo.root_id) == "orphan"

    unresolved = topo.unresolved_edges()
    assert len(unresolved) == 1
    edge = unresolved[0]
    assert edge.resolved is False
    assert edge.parent_id is None
    assert edge.parent_native_id == "missing-parent-uuid"
    assert edge.kind is TopologyEdgeKind.UNRESOLVED_NATIVE


def test_topology_sync_derivation_cycle_detected() -> None:
    """Synthetic cycle: A → B → A must be reported, not infinite-loop."""

    a = ConversationRecord(
        conversation_id=ConversationId("A"),
        provider_name="test",
        provider_conversation_id="ext-A",
        content_hash=ContentHash("hash-a"),
        parent_conversation_id=ConversationId("B"),
    )
    b = ConversationRecord(
        conversation_id=ConversationId("B"),
        provider_name="test",
        provider_conversation_id="ext-B",
        content_hash=ContentHash("hash-b"),
        parent_conversation_id=ConversationId("A"),
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

    root = ConversationRecord(
        conversation_id=ConversationId("root"),
        provider_name="codex",
        provider_conversation_id="ext-root",
        content_hash=ContentHash("h-root"),
    )
    cont = ConversationRecord(
        conversation_id=ConversationId("cont"),
        provider_name="codex",
        provider_conversation_id="ext-cont",
        content_hash=ContentHash("h-cont"),
        parent_conversation_id=ConversationId("root"),
        branch_type=None,
    )
    children: dict[str, list[ConversationRecord]] = {"root": [cont], "cont": []}
    by_id = {"root": root, "cont": cont}

    topo = derive_session_topology_sync(
        "cont",
        fetch=by_id.get,
        fetch_children=lambda cid: children.get(cid, []),
    )
    assert topo is not None
    assert not topo.cycle_detected
    assert str(topo.root_id) == "root"
    ids = {str(node.conversation_id) for node in topo.nodes}
    assert ids == {"root", "cont"}
    # cont has no branch_type → edge kind falls back to UNKNOWN.
    cont_edge = next(edge for edge in topo.edges if str(edge.child_id) == "cont")
    assert cont_edge.kind is TopologyEdgeKind.UNKNOWN
    assert cont_edge.resolved is True
