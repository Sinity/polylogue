"""Property tests for session tree operations.

Tests that root resolution, child enumeration, parent lookup, list ordering,
and shortest_unique_prefix resolve correctly for Hypothesis-generated acyclic
session graphs over the archive.

Archive session identity differs from the legacy ``session_id``: a seeded
``conv-N-slug`` becomes ``<origin>:ext-conv-N-slug``. Parent edges resolve only
when the child references the parent's ``provider_session_id``
(``ext-<id>``), so this suite seeds the graph locally with that reference shape
rather than through ``seed_session_graph`` (which targets suites that do not
assert resolved multi-node trees).
"""

from __future__ import annotations

import dataclasses
import uuid
from pathlib import Path

from hypothesis import HealthCheck, given, settings

from polylogue.api import Polylogue
from polylogue.archive.session.domain_models import Session
from tests.infra.archive_scenarios import native_session_id_for
from tests.infra.storage_records import SessionBuilder
from tests.infra.strategies.storage import (
    SessionSpec,
    expected_sorted_ids,
    root_index,
    session_graph_strategy,
    shortest_unique_prefix,
)

# Native parent resolution is origin-scoped: a child references its parent by
# ``provider_session_id`` (``ext-<id>``) and the edge resolves only within
# the same origin. The strategy assigns each spec an independent provider, which
# the provider-agnostic ``expected_*`` helpers treat as one logical graph. To
# keep that single-graph assumption true under archive semantics, every spec in a
# generated graph is pinned to one provider before seeding.
_GRAPH_PROVIDER = "claude-ai"


def _uniform_provider(specs: tuple[SessionSpec, ...]) -> tuple[SessionSpec, ...]:
    return tuple(dataclasses.replace(spec, provider=_GRAPH_PROVIDER) for spec in specs)


def _native_id(spec: SessionSpec) -> str:
    return native_session_id_for(spec.provider, spec.session_id)


def _seed_archive_graph(db_path: Path, specs: tuple[SessionSpec, ...]) -> None:
    """Seed the generated graph with native-resolvable parent references.

    The child references the parent's ``provider_session_id``
    (``ext-<session_id>``) so the ArchiveStore write resolves the
    cross-session parent edge.
    """
    for spec in specs:
        builder = (
            SessionBuilder(db_path, spec.session_id)
            .provider(spec.provider)
            .title(spec.title)
            .created_at(spec.created_at)
            .updated_at(spec.updated_at)
        )
        if spec.parent_index is not None:
            builder.parent_session(f"ext-{specs[spec.parent_index].session_id}")
        for message_index, message in enumerate(spec.messages):
            builder.add_message(
                f"{spec.session_id}-m{message_index}",
                role=message.role,
                text=message.text,
                has_tool_use=int(message.has_tool_use),
                has_thinking=int(message.has_thinking),
            )
        builder.save()


async def _root_of(plg: Polylogue, session_id: str) -> Session:
    tree = await plg.get_session_tree(session_id)
    roots = [session for session in tree if session.is_root]
    assert len(roots) == 1, f"expected exactly one root in tree of {session_id}, got {[c.id for c in roots]}"
    return roots[0]


@given(session_graph_strategy(min_sessions=2, max_sessions=6))
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_get_root_returns_node_without_parent(tmp_path: Path, specs: tuple[SessionSpec, ...]) -> None:
    """The resolved root of each node is itself parentless."""
    archive_root = tmp_path / f"tree-{uuid.uuid4().hex}"
    archive_root.mkdir()
    db_path = archive_root / "index.db"
    specs = _uniform_provider(specs)
    _seed_archive_graph(db_path, specs)

    async with Polylogue(archive_root=db_path.parent, db_path=db_path) as plg:
        for spec in specs:
            root = await _root_of(plg, _native_id(spec))
            assert root.parent_id is None, f"Root {root.id} unexpectedly has parent {root.parent_id}"


@given(session_graph_strategy(min_sessions=2, max_sessions=6))
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_get_root_matches_expected(tmp_path: Path, specs: tuple[SessionSpec, ...]) -> None:
    """The resolved root for each node matches the strategy's expected root."""
    archive_root = tmp_path / f"tree-{uuid.uuid4().hex}"
    archive_root.mkdir()
    db_path = archive_root / "index.db"
    specs = _uniform_provider(specs)
    _seed_archive_graph(db_path, specs)

    async with Polylogue(archive_root=db_path.parent, db_path=db_path) as plg:
        for i, spec in enumerate(specs):
            root = await _root_of(plg, _native_id(spec))
            expected_root_id = _native_id(specs[root_index(specs, i)])
            assert root.id == expected_root_id, (
                f"Node {spec.session_id}: expected root {expected_root_id}, got {root.id}"
            )


@given(session_graph_strategy(min_sessions=2, max_sessions=6))
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_tree_ids_match_expected(tmp_path: Path, specs: tuple[SessionSpec, ...]) -> None:
    """The full tree reachable from each node matches expected_tree_ids."""
    archive_root = tmp_path / f"tree-{uuid.uuid4().hex}"
    archive_root.mkdir()
    db_path = archive_root / "index.db"
    specs = _uniform_provider(specs)
    _seed_archive_graph(db_path, specs)

    async with Polylogue(archive_root=db_path.parent, db_path=db_path) as plg:
        for i, spec in enumerate(specs):
            expected = {_native_id(specs[k]) for k in range(len(specs)) if k in _expected_indices(specs, i)}
            tree = await plg.get_session_tree(_native_id(spec))
            tree_ids = {session.id for session in tree}
            assert tree_ids == expected, f"Tree from {spec.session_id}: expected {expected}, got {tree_ids}"


def _expected_indices(specs: tuple[SessionSpec, ...], index: int) -> set[int]:
    expected_root = root_index(specs, index)
    return {position for position in range(len(specs)) if root_index(specs, position) == expected_root}


@given(session_graph_strategy(min_sessions=2, max_sessions=6))
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_sorted_ids_match_expected(tmp_path: Path, specs: tuple[SessionSpec, ...]) -> None:
    """list_sessions returns sessions in expected sort order."""
    archive_root = tmp_path / f"tree-{uuid.uuid4().hex}"
    archive_root.mkdir()
    db_path = archive_root / "index.db"
    specs = _uniform_provider(specs)
    _seed_archive_graph(db_path, specs)

    async with Polylogue(archive_root=db_path.parent, db_path=db_path) as plg:
        convos = await plg.list_sessions(limit=len(specs) + 10)
        actual_ids = [c.id for c in convos]
        expected = [native_session_id_for(_provider_for(specs, cid), cid) for cid in expected_sorted_ids(specs)]
        assert actual_ids == expected, f"Sort mismatch: {actual_ids} != {expected}"


def _provider_for(specs: tuple[SessionSpec, ...], session_id: str) -> str:
    for spec in specs:
        if spec.session_id == session_id:
            return spec.provider
    raise AssertionError(f"unknown session_id {session_id!r}")


@given(session_graph_strategy(min_sessions=2, max_sessions=6))
@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_shortest_unique_prefix_resolves_unambiguously(
    tmp_path: Path,
    specs: tuple[SessionSpec, ...],
) -> None:
    """shortest_unique_prefix for each ID matches exactly one session."""
    archive_root = tmp_path / f"tree-{uuid.uuid4().hex}"
    archive_root.mkdir()
    db_path = archive_root / "index.db"
    specs = _uniform_provider(specs)
    _seed_archive_graph(db_path, specs)

    all_ids = tuple(s.session_id for s in specs)
    for spec in specs:
        prefix = shortest_unique_prefix(all_ids, spec.session_id)
        matches = [cid for cid in all_ids if cid.startswith(prefix)]
        assert len(matches) == 1, f"Prefix '{prefix}' for '{spec.session_id}' matched {len(matches)} IDs: {matches}"
        assert matches[0] == spec.session_id
