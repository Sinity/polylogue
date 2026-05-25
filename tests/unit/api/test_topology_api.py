"""Typed topology read-model API contract tests (#1261 / #866 slice D).

These tests pin the four typed lineage methods on the
:class:`~polylogue.api.Polylogue` facade — ``get_session_topology``,
``get_ancestors``, ``get_descendants``, ``get_siblings``, and
``get_thread`` — against synthetic lineages seeded through
``ConversationBuilder``. They are the public-API equivalent of the
storage-level coverage in
``tests/unit/storage/test_session_topology.py`` and exist so future
refactors of the substrate cannot silently regress the surface contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.insights.topology import ConversationRef, LogicalSession, SessionTopology
from tests.infra.storage_records import ConversationBuilder, db_setup


def _seed_lineage(db_path: Path) -> None:
    """Seed: root → continuation → fork, with subagent + sidechain off root."""

    ConversationBuilder(db_path, "root").provider("claude-code").title("Root").add_message(
        role="user", text="kickoff"
    ).save()
    ConversationBuilder(db_path, "continuation").provider("claude-code").title("Continuation").parent_conversation(
        "root"
    ).branch_type("continuation").add_message(role="user", text="continue").save()
    ConversationBuilder(db_path, "fork").provider("claude-code").title("Fork").parent_conversation(
        "continuation"
    ).branch_type("fork").add_message(role="user", text="fork").save()
    ConversationBuilder(db_path, "subagent").provider("claude-code").title("Subagent").parent_conversation(
        "root"
    ).branch_type("subagent").add_message(role="assistant", text="agent").save()
    ConversationBuilder(db_path, "sidechain").provider("claude-code").title("Sidechain").parent_conversation(
        "root"
    ).branch_type("sidechain").add_message(role="user", text="side").save()


@pytest.mark.asyncio
async def test_get_session_topology_returns_typed_envelope(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    _seed_lineage(db_path)

    polylogue = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        topology = await polylogue.get_session_topology("fork")
    finally:
        await polylogue.close()

    assert isinstance(topology, SessionTopology)
    assert str(topology.root_id) == "root"
    assert str(topology.target_id) == "fork"
    assert not topology.cycle_detected
    assert {str(n.conversation_id) for n in topology.nodes} == {
        "root",
        "continuation",
        "fork",
        "subagent",
        "sidechain",
    }


@pytest.mark.asyncio
async def test_get_ancestors_returns_root_to_parent_refs(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    _seed_lineage(db_path)

    polylogue = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        ancestors = await polylogue.get_ancestors("fork")
        root_ancestors = await polylogue.get_ancestors("root")
    finally:
        await polylogue.close()

    assert all(isinstance(ref, ConversationRef) for ref in ancestors)
    assert [str(ref.conversation_id) for ref in ancestors] == ["root", "continuation"]
    # Each ref carries provider/title context so callers do not re-fetch.
    assert ancestors[0].source_name == "claude-code"
    assert ancestors[0].title == "Root"
    assert ancestors[0].depth == 0
    # Root has no ancestors.
    assert root_ancestors == []


@pytest.mark.asyncio
async def test_get_descendants_bfs_order(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    _seed_lineage(db_path)

    polylogue = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        descendants_of_root = await polylogue.get_descendants("root")
        descendants_of_continuation = await polylogue.get_descendants("continuation")
        leaf_descendants = await polylogue.get_descendants("fork")
    finally:
        await polylogue.close()

    assert {str(ref.conversation_id) for ref in descendants_of_root} == {
        "continuation",
        "fork",
        "subagent",
        "sidechain",
    }
    assert [str(ref.conversation_id) for ref in descendants_of_continuation] == ["fork"]
    assert leaf_descendants == []


@pytest.mark.asyncio
async def test_get_siblings_excludes_self(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    _seed_lineage(db_path)

    polylogue = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        siblings = await polylogue.get_siblings("subagent")
        root_siblings = await polylogue.get_siblings("root")
    finally:
        await polylogue.close()

    sibling_ids = {str(ref.conversation_id) for ref in siblings}
    assert sibling_ids == {"continuation", "sidechain"}
    # Root has no resolved parent, hence no siblings.
    assert root_siblings == []


@pytest.mark.asyncio
async def test_get_thread_orders_ancestors_self_descendants(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    _seed_lineage(db_path)

    polylogue = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        thread = await polylogue.get_thread("continuation")
    finally:
        await polylogue.close()

    ids = [str(ref.conversation_id) for ref in thread]
    # Ancestors (root) first, then self (continuation), then descendants (fork).
    assert ids == ["root", "continuation", "fork"]


@pytest.mark.asyncio
async def test_get_logical_session_returns_compact_read_pull_view(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    _seed_lineage(db_path)

    polylogue = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        logical = await polylogue.get_logical_session("continuation")
    finally:
        await polylogue.close()

    assert isinstance(logical, LogicalSession)
    assert str(logical.conversation_id) == "continuation"
    assert str(logical.root_id) == "root"
    assert [str(ref.conversation_id) for ref in logical.thread] == ["root", "continuation", "fork"]
    assert {str(ref.conversation_id) for ref in logical.siblings} == {"subagent", "sidechain"}
    assert [str(ref.conversation_id) for ref in logical.descendants] == ["fork"]
    assert logical.cycle_detected is False


@pytest.mark.asyncio
async def test_topology_api_unknown_conversation_returns_empty(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    _seed_lineage(db_path)

    polylogue = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        topology = await polylogue.get_session_topology("never-ingested")
        ancestors = await polylogue.get_ancestors("never-ingested")
        descendants = await polylogue.get_descendants("never-ingested")
        siblings = await polylogue.get_siblings("never-ingested")
        thread = await polylogue.get_thread("never-ingested")
        logical = await polylogue.get_logical_session("never-ingested")
    finally:
        await polylogue.close()

    assert topology is None
    assert ancestors == []
    assert descendants == []
    assert siblings == []
    assert thread == []
    assert logical is None


@pytest.mark.asyncio
async def test_topology_api_root_only_conversation(workspace_env: dict[str, Path]) -> None:
    """A conversation with no parent and no children projects an empty graph."""

    db_path = db_setup(workspace_env)
    ConversationBuilder(db_path, "lonely").provider("claude-code").title("Lonely").add_message(
        role="user", text="solo"
    ).save()

    polylogue = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        topology = await polylogue.get_session_topology("lonely")
        ancestors = await polylogue.get_ancestors("lonely")
        descendants = await polylogue.get_descendants("lonely")
        siblings = await polylogue.get_siblings("lonely")
        thread = await polylogue.get_thread("lonely")
    finally:
        await polylogue.close()

    assert topology is not None
    assert str(topology.root_id) == "lonely"
    assert ancestors == []
    assert descendants == []
    assert siblings == []
    # Thread on a lonely node is just that node.
    assert [str(ref.conversation_id) for ref in thread] == ["lonely"]
