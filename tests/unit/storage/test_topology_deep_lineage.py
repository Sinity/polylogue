"""A deep but acyclic lineage must classify without recursing per edge.

The writer admits a parent chain up to ``_CYCLE_WALK_BUDGET`` (1024, see
``storage/sqlite/archive_tiers/write.py``). The topology classifier's cycle
detection walked that chain with a recursive depth-first search, so a lineage
the writer accepted raised ``RecursionError`` when any topology read classified
it -- and because the classifier runs over every link in scope, an unrelated
read of a neighbouring session failed with it.
"""

from __future__ import annotations

import pytest

from polylogue.storage.derived.topology.derivation import (
    TopologyNodeInput,
    compose_session_topology,
)

_DEPTH = 1100


def _linear_lineage(depth: int) -> tuple[list[TopologyNodeInput], list[dict[str, object]]]:
    nodes = [TopologyNodeInput(session_id=f"codex-session:s{i}", origin="codex-session") for i in range(depth)]
    links: list[dict[str, object]] = [
        {
            "src_session_id": f"codex-session:s{i + 1}",
            "dst_origin": "codex-session",
            "dst_native_id": f"s{i}",
            "link_type": "continuation",
            "resolved_dst_session_id": f"codex-session:s{i}",
            "status": "accepted",
            "inheritance": None,
            "branch_point_message_id": None,
            "parent_tool_use_block_id": None,
            "method": None,
            "confidence": 1.0,
            "observed_at_ms": 1000,
            "resolved_at_ms": 1000,
            "evidence_json": "[]",
        }
        for i in range(depth - 1)
    ]
    return nodes, links


def test_deep_acyclic_lineage_composes_without_recursion_error() -> None:
    """Anti-vacuity: with the recursive ``visit`` restored this raises
    ``RecursionError: maximum recursion depth exceeded`` before returning --
    executed against the pre-change file, not asserted. ``_DEPTH`` sits above
    the writer's own 1024-step budget, so the input is one the writer admits.
    """
    assert _DEPTH > 1024
    nodes, links = _linear_lineage(_DEPTH)

    topology = compose_session_topology("codex-session:s0", nodes, links)

    assert topology is not None
    assert len(topology.nodes) == _DEPTH
    assert [edge for edge in topology.edges if edge.composability_reason == "cycle"] == []


def test_a_real_cycle_is_still_quarantined() -> None:
    """The opposite direction: a classifier that never reports a cycle is refuted."""
    nodes = [TopologyNodeInput(session_id=f"codex-session:{name}", origin="codex-session") for name in ("a", "b", "c")]
    links: list[dict[str, object]] = []
    for src, dst in (("a", "b"), ("b", "c"), ("c", "a")):
        links.append(
            {
                "src_session_id": f"codex-session:{src}",
                "dst_origin": "codex-session",
                "dst_native_id": dst,
                "link_type": "continuation",
                "resolved_dst_session_id": f"codex-session:{dst}",
                "status": "accepted",
                "inheritance": None,
                "branch_point_message_id": None,
                "parent_tool_use_block_id": None,
                "method": None,
                "confidence": 1.0,
                "observed_at_ms": 1000,
                "resolved_at_ms": 1000,
                "evidence_json": "[]",
            }
        )

    topology = compose_session_topology("codex-session:a", nodes, links)

    # The composed scope collapses to the target alone once its only inbound
    # edge is quarantined, so one edge is returned, not all three.
    assert topology is not None
    assert topology.edges
    assert all(edge.composability_reason == "cycle" for edge in topology.edges)
    assert not any(edge.composable for edge in topology.edges)


@pytest.mark.parametrize("depth", [2, 16, 1024])
def test_shallow_lineages_are_unchanged(depth: int) -> None:
    nodes, links = _linear_lineage(depth)
    topology = compose_session_topology("codex-session:s0", nodes, links)
    assert topology is not None
    assert len(topology.nodes) == depth
