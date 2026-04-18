"""Property tests for conversation tree operations.

Tests that get_root, get_children, get_parent, list ordering, and
shortest_unique_prefix resolve correctly for Hypothesis-generated
acyclic conversation graphs.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from hypothesis import HealthCheck, given, settings

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository
from tests.infra.strategies.storage import (
    ConversationSpec,
    conversation_graph_strategy,
    expected_sorted_ids,
    expected_tree_ids,
    root_index,
    seed_conversation_graph,
    shortest_unique_prefix,
)


@given(conversation_graph_strategy(min_conversations=2, max_conversations=6))
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_get_root_returns_node_without_parent(tmp_path: Path, specs: tuple[ConversationSpec, ...]) -> None:
    """get_root resolves to a node whose parent is None."""
    db_path = tmp_path / f"tree-{uuid.uuid4().hex}.db"
    backend = SQLiteBackend(db_path=db_path)
    seed_conversation_graph(db_path, specs)

    repo = ConversationRepository(backend=backend)
    # Pick a random non-root node if possible
    for _i, spec in enumerate(specs):
        root = await repo.get_root(spec.conversation_id)
        parent = await repo.get_parent(root.id)
        assert parent is None, f"Root {root.id} unexpectedly has parent"
    await backend.close()


@given(conversation_graph_strategy(min_conversations=2, max_conversations=6))
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_get_root_matches_expected(tmp_path: Path, specs: tuple[ConversationSpec, ...]) -> None:
    """get_root for each node matches the strategy's expected root."""
    db_path = tmp_path / f"tree-{uuid.uuid4().hex}.db"
    backend = SQLiteBackend(db_path=db_path)
    seed_conversation_graph(db_path, specs)

    repo = ConversationRepository(backend=backend)
    for i, spec in enumerate(specs):
        root = await repo.get_root(spec.conversation_id)
        expected_root_id = specs[root_index(specs, i)].conversation_id
        assert root.id == expected_root_id, (
            f"Node {spec.conversation_id}: expected root {expected_root_id}, got {root.id}"
        )
    await backend.close()


@given(conversation_graph_strategy(min_conversations=2, max_conversations=6))
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_tree_ids_match_expected(tmp_path: Path, specs: tuple[ConversationSpec, ...]) -> None:
    """For each node, the full tree reachable from its root matches expected_tree_ids."""
    db_path = tmp_path / f"tree-{uuid.uuid4().hex}.db"
    backend = SQLiteBackend(db_path=db_path)
    seed_conversation_graph(db_path, specs)

    repo = ConversationRepository(backend=backend)
    for i, spec in enumerate(specs):
        expected = expected_tree_ids(specs, i)
        root = await repo.get_root(spec.conversation_id)

        # Collect all IDs in tree via BFS from root
        tree_ids = {root.id}
        queue = [root.id]
        while queue:
            current = queue.pop(0)
            children = await repo.get_children(current)
            for child in children:
                tree_ids.add(child.id)
                queue.append(child.id)

        assert tree_ids == expected, f"Tree from {spec.conversation_id}: expected {expected}, got {tree_ids}"
    await backend.close()


@given(conversation_graph_strategy(min_conversations=2, max_conversations=6))
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_sorted_ids_match_expected(tmp_path: Path, specs: tuple[ConversationSpec, ...]) -> None:
    """list() returns conversations in expected sort order."""
    db_path = tmp_path / f"tree-{uuid.uuid4().hex}.db"
    backend = SQLiteBackend(db_path=db_path)
    seed_conversation_graph(db_path, specs)

    repo = ConversationRepository(backend=backend)
    convos = await repo.list(limit=len(specs) + 10)
    actual_ids = [c.id for c in convos]
    expected = expected_sorted_ids(specs)
    assert actual_ids == expected, f"Sort mismatch: {actual_ids} != {expected}"
    await backend.close()


@given(conversation_graph_strategy(min_conversations=2, max_conversations=6))
@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_shortest_unique_prefix_resolves_unambiguously(
    tmp_path: Path,
    specs: tuple[ConversationSpec, ...],
) -> None:
    """shortest_unique_prefix for each ID matches exactly one conversation."""
    db_path = tmp_path / f"tree-{uuid.uuid4().hex}.db"
    backend = SQLiteBackend(db_path=db_path)
    seed_conversation_graph(db_path, specs)

    all_ids = tuple(s.conversation_id for s in specs)
    for spec in specs:
        prefix = shortest_unique_prefix(all_ids, spec.conversation_id)
        matches = [cid for cid in all_ids if cid.startswith(prefix)]
        assert len(matches) == 1, (
            f"Prefix '{prefix}' for '{spec.conversation_id}' matched {len(matches)} IDs: {matches}"
        )
        assert matches[0] == spec.conversation_id
    await backend.close()
