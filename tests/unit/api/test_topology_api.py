"""Typed topology read-model API contract tests (#1261 / #866 slice D).

These tests pin the four typed lineage methods on the
:class:`~polylogue.api.Polylogue` facade — ``get_session_topology``,
``get_ancestors``, ``get_descendants``, ``get_siblings``, and
``get_thread`` — against synthetic lineages seeded through
``SessionBuilder``. They are the public-API equivalent of the
storage-level coverage in
``tests/unit/storage/test_session_topology.py`` and exist so future
refactors of the substrate cannot silently regress the surface contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.analysis.topology import LogicalSession, SessionRef, SessionTopology
from polylogue.api import Polylogue
from tests.infra.frozen_clock import FrozenClock
from tests.infra.storage_records import SessionBuilder, db_setup


# Archive session ids derive from the builder's provider_session_id
# (``ext-<conv_id>``) and the claude-code origin. Parent references are the
# parent's provider-native id (``ext-<parent>``) so the archive link
# resolver can match ``dst_session_native_id`` against ``sessions.native_id``.
def _native(token: str) -> str:
    return f"claude-code-session:ext-{token}"


def _seed_lineage(db_path: Path) -> None:
    """Seed: root → continuation → fork, with subagent + sidechain off root."""

    SessionBuilder(db_path, "root").provider("claude-code").title("Root").add_message(
        role="user", text="kickoff"
    ).save()
    SessionBuilder(db_path, "continuation").provider("claude-code").title("Continuation").parent_session(
        "ext-root"
    ).branch_type("continuation").add_message(role="user", text="continue").save()
    SessionBuilder(db_path, "fork").provider("claude-code").title("Fork").parent_session(
        "ext-continuation"
    ).branch_type("fork").add_message(role="user", text="fork").save()
    SessionBuilder(db_path, "subagent").provider("claude-code").title("Subagent").parent_session(
        "ext-root"
    ).branch_type("subagent").add_message(role="assistant", text="agent").save()
    SessionBuilder(db_path, "sidechain").provider("claude-code").title("Sidechain").parent_session(
        "ext-root"
    ).branch_type("sidechain").add_message(role="user", text="side").save()


@pytest.mark.asyncio
async def test_get_session_topology_returns_typed_envelope(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    _seed_lineage(db_path)

    polylogue = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        topology = await polylogue.get_session_topology(_native("fork"))
    finally:
        await polylogue.close()

    assert isinstance(topology, SessionTopology)
    assert str(topology.root_id) == _native("root")
    assert str(topology.target_id) == _native("fork")
    assert not topology.cycle_detected
    assert {str(n.session_id) for n in topology.nodes} == {
        _native("root"),
        _native("continuation"),
        _native("fork"),
        _native("subagent"),
        _native("sidechain"),
    }


@pytest.mark.asyncio
async def test_get_ancestors_returns_root_to_parent_refs(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    _seed_lineage(db_path)

    polylogue = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        ancestors = await polylogue.get_ancestors(_native("fork"))
        root_ancestors = await polylogue.get_ancestors(_native("root"))
    finally:
        await polylogue.close()

    assert all(isinstance(ref, SessionRef) for ref in ancestors)
    assert [str(ref.session_id) for ref in ancestors] == [_native("root"), _native("continuation")]
    # Each ref carries provider/title context so callers do not re-fetch.
    assert ancestors[0].origin == "claude-code-session"
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
        descendants_of_root = await polylogue.get_descendants(_native("root"))
        descendants_of_continuation = await polylogue.get_descendants(_native("continuation"))
        leaf_descendants = await polylogue.get_descendants(_native("fork"))
    finally:
        await polylogue.close()

    assert {str(ref.session_id) for ref in descendants_of_root} == {
        _native("continuation"),
        _native("fork"),
        _native("subagent"),
        _native("sidechain"),
    }
    assert [str(ref.session_id) for ref in descendants_of_continuation] == [_native("fork")]
    assert leaf_descendants == []


@pytest.mark.asyncio
async def test_get_siblings_excludes_self(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    _seed_lineage(db_path)

    polylogue = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        siblings = await polylogue.get_siblings(_native("subagent"))
        root_siblings = await polylogue.get_siblings(_native("root"))
    finally:
        await polylogue.close()

    sibling_ids = {str(ref.session_id) for ref in siblings}
    assert sibling_ids == {_native("continuation"), _native("sidechain")}
    # Root has no resolved parent, hence no siblings.
    assert root_siblings == []


@pytest.mark.asyncio
async def test_get_thread_orders_ancestors_self_descendants(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    _seed_lineage(db_path)

    polylogue = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        thread = await polylogue.get_thread(_native("continuation"))
    finally:
        await polylogue.close()

    ids = [str(ref.session_id) for ref in thread]
    # Ancestors (root) first, then self (continuation), then descendants (fork).
    assert ids == [_native("root"), _native("continuation"), _native("fork")]


@pytest.mark.asyncio
async def test_get_logical_session_returns_compact_read_pull_view(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    _seed_lineage(db_path)

    polylogue = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        logical = await polylogue.get_logical_session(_native("continuation"))
    finally:
        await polylogue.close()

    assert isinstance(logical, LogicalSession)
    assert str(logical.session_id) == _native("continuation")
    assert str(logical.root_id) == _native("root")
    assert [str(ref.session_id) for ref in logical.thread] == [
        _native("root"),
        _native("continuation"),
        _native("fork"),
    ]
    assert {str(ref.session_id) for ref in logical.siblings} == {_native("subagent"), _native("sidechain")}
    assert [str(ref.session_id) for ref in logical.descendants] == [_native("fork")]
    assert logical.cycle_detected is False


@pytest.mark.asyncio
async def test_topology_api_unknown_session_returns_empty(workspace_env: dict[str, Path]) -> None:
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
async def test_topology_api_root_only_session(workspace_env: dict[str, Path]) -> None:
    """A session with no parent and no children projects an empty graph."""

    db_path = db_setup(workspace_env)
    SessionBuilder(db_path, "lonely").provider("claude-code").title("Lonely").add_message(
        role="user", text="solo"
    ).save()

    polylogue = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        topology = await polylogue.get_session_topology(_native("lonely"))
        ancestors = await polylogue.get_ancestors(_native("lonely"))
        descendants = await polylogue.get_descendants(_native("lonely"))
        siblings = await polylogue.get_siblings(_native("lonely"))
        thread = await polylogue.get_thread(_native("lonely"))
    finally:
        await polylogue.close()

    assert topology is not None
    assert str(topology.root_id) == _native("lonely")
    assert ancestors == []
    assert descendants == []
    assert siblings == []
    # Thread on a lonely node is just that node.
    assert [str(ref.session_id) for ref in thread] == [_native("lonely")]


@pytest.mark.asyncio
async def test_topology_api_pages_a_stable_bounded_bfs_envelope(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    _seed_lineage(db_path)
    polylogue = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        first = await polylogue.get_session_topology(_native("root"), node_limit=2)
        second = await polylogue.get_session_topology(_native("root"), node_offset=2, node_limit=2)
    finally:
        await polylogue.close()
    assert first is not None and second is not None
    assert [str(node.session_id) for node in first.nodes] == [_native("root"), _native("continuation")]
    assert first.nodes_complete is False
    assert first.continuation == "node-offset:2"
    assert [str(node.session_id) for node in second.nodes] == [_native("sidechain"), _native("subagent")]
    assert {str(node.session_id) for node in first.nodes}.isdisjoint(str(node.session_id) for node in second.nodes)


@pytest.mark.asyncio
async def test_topology_read_pins_parent_replacement_to_one_snapshot(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """F290/F291: a committed reparent between BFS reads cannot invent two parents."""
    import aiosqlite

    from polylogue.storage.sqlite.connection_profile import open_connection
    from polylogue.storage.sqlite.queries import session_links as links_q

    db_path = db_setup(workspace_env)
    for token in ("old-parent", "new-parent"):
        SessionBuilder(db_path, token).provider("claude-code").add_message(role="user", text=token).save()
    SessionBuilder(db_path, "moving-child").provider("claude-code").parent_session("ext-old-parent").branch_type(
        "continuation"
    ).add_message(role="user", text="child").save()
    original = links_q.list_session_links_to_session
    replaced = False

    async def reparent_after_inbound_read(
        conn: aiosqlite.Connection, session_id: str, *, limit: int | None
    ) -> list[dict[str, object]]:
        nonlocal replaced
        rows = await original(conn, session_id, limit=limit)
        if session_id == _native("old-parent") and not replaced:
            assert any(row["src_session_id"] == _native("moving-child") for row in rows)
            # Mutate a real second SQLite connection after the root's old
            # inbound edge was read, before BFS revisits the child's edge.
            with open_connection(db_path) as writer:
                changed = writer.execute(
                    """UPDATE session_links SET dst_native_id = ?, resolved_dst_session_id = ?
                       WHERE src_session_id = ? AND dst_native_id = ?""",
                    ("ext-new-parent", _native("new-parent"), _native("moving-child"), "ext-old-parent"),
                )
                assert changed.rowcount == 1
                writer.commit()
            replaced = True
        return rows

    monkeypatch.setattr(links_q, "list_session_links_to_session", reparent_after_inbound_read)
    polylogue = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        first = await polylogue.get_session_topology(_native("moving-child"))
        second = await polylogue.get_session_topology(_native("moving-child"))
    finally:
        await polylogue.close()

    assert replaced
    assert first is not None and second is not None
    assert str(first.root_id) == _native("old-parent")
    assert str(second.root_id) == _native("new-parent")
    assert not first.conflicting_parent_detected
    assert not second.conflicting_parent_detected
    assert len(first.edges) == len(second.edges) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("borrowed_transaction", [False, True])
@pytest.mark.parametrize("fail_read", [False, True])
async def test_topology_snapshot_releases_only_its_own_transaction(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    borrowed_transaction: bool,
    fail_read: bool,
) -> None:
    """The production snapshot owner must clean up failures without ending caller transactions."""
    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager

    import aiosqlite

    from polylogue.storage.sqlite.queries import sessions as sessions_q
    from polylogue.storage.sqlite.query_store import SQLiteQueryStore

    db_path = db_setup(workspace_env)
    SessionBuilder(db_path, "present").provider("claude-code").add_message(role="user", text="present").save()
    if fail_read:

        async def fail(conn: aiosqlite.Connection, session_id: str) -> None:
            raise RuntimeError("synthetic read interruption")

        monkeypatch.setattr(sessions_q, "get_session", fail)

    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        if borrowed_transaction:
            await conn.execute("BEGIN")

        @asynccontextmanager
        async def connection() -> AsyncIterator[aiosqlite.Connection]:
            yield conn

        queries = SQLiteQueryStore(connection_factory=connection)
        if fail_read:
            with pytest.raises(RuntimeError):
                await queries.get_session_topology("absent")
        else:
            assert await queries.get_session_topology("absent") is None
        assert conn.in_transaction is borrowed_transaction
        if borrowed_transaction:
            await conn.rollback()


@pytest.mark.asyncio
async def test_exhaustive_helpers_cover_a_chain_beyond_the_default_page(
    workspace_env: dict[str, Path], frozen_clock: FrozenClock
) -> None:
    from tests.infra.storage_records import seed_topology_chain

    db_path = db_setup(workspace_env)
    ids = seed_topology_chain(db_path, 205)
    async with Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path) as api:
        page = await api.get_session_topology(ids[-1])
        ancestors = await api.get_ancestors(ids[-1])
        descendants = await api.get_descendants(ids[0])
        thread = await api.get_thread(ids[-1])
        logical = await api.get_logical_session(ids[-1])
        siblings = await api.get_siblings(ids[-1])

    assert page is not None and not page.nodes_complete and page.continuation is not None
    assert len(page.nodes) == 200
    assert [str(ref.session_id) for ref in ancestors] == list(ids[:-1])
    assert [str(ref.session_id) for ref in descendants] == list(ids[1:])
    assert [str(ref.session_id) for ref in thread] == list(ids)
    assert logical is not None and [str(ref.session_id) for ref in logical.thread] == list(ids)
    assert siblings == []


@pytest.mark.asyncio
async def test_exhaustive_siblings_cover_the_complete_edge_relation(
    workspace_env: dict[str, Path], frozen_clock: FrozenClock
) -> None:
    from tests.infra.storage_records import seed_topology_star

    db_path = db_setup(workspace_env)
    ids = seed_topology_star(db_path, 503)
    async with Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path) as api:
        siblings = await api.get_siblings(ids[-1])
        graph = await api.get_session_topology(ids[0], node_limit=None, edge_limit=None)

    assert graph is not None and graph.nodes_complete and graph.edges_complete and graph.continuation is None
    assert len(graph.nodes) == 503 and len(graph.edges) == 502
    assert {str(ref.session_id) for ref in siblings} == set(ids[1:-1])
