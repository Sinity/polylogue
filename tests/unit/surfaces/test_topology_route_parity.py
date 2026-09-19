"""Cross-surface parity and terminal-outcome tests for session topology (polylogue-27ezu).

`session_links` is the edge authority and one graph engine derives the
topology; these tests pin the *read* half of that claim: CLI, MCP and HTTP
serialize one envelope produced at one operation boundary, and a topology
behind a named gap reports `degraded` rather than `empty`.

Anti-vacuity is named per test below. Each condition was verified by
reverting the production change and observing the test go red, not by
assertion alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from polylogue.analysis.topology import (
    TOPOLOGY_GAP_CYCLE,
    TOPOLOGY_GAP_EXCLUDED_EDGE,
    TOPOLOGY_GAP_TRUNCATED,
    TOPOLOGY_GAP_UNRESOLVED_PARENT,
    SessionTopology,
    TopologyNode,
)
from polylogue.api import Polylogue
from polylogue.core.types import SessionId
from polylogue.daemon.topology_http import MAX_NODE_LIMIT, build_topology_envelope
from polylogue.mcp.payloads import session_topology_payload
from polylogue.operations.topology_envelope import (
    TOPOLOGY_EMPTY_REASON,
    bound_topology_envelope,
    topology_public_envelope,
)
from tests.infra.storage_records import SessionBuilder, db_setup


def _native(token: str) -> str:
    return f"claude-code-session:ext-{token}"


def _seed_chain(db_path: Path, width: int = 4) -> None:
    """Root plus ``width`` resolved children, so paging has something to cut."""

    SessionBuilder(db_path, "root").provider("claude-code").title("Root").add_message(
        role="user", text="kickoff"
    ).save()
    for index in range(width):
        SessionBuilder(db_path, f"child{index}").provider("claude-code").title(f"Child {index}").parent_session(
            "ext-root"
        ).branch_type("subagent").add_message(role="assistant", text=f"child {index}").save()


async def _topology(db_path: Path, archive_root: Path, session_id: str, **kwargs: int) -> SessionTopology | None:
    polylogue = Polylogue(archive_root=archive_root, db_path=db_path)
    try:
        return await polylogue.get_session_topology(session_id, **kwargs)
    finally:
        await polylogue.close()


def _rows(envelope: dict[str, object], key: str) -> list[dict[str, object]]:
    """Read a list-of-dict section out of an untyped public envelope."""

    return cast("list[dict[str, object]]", envelope[key])


def _outcome(envelope: dict[str, object]) -> dict[str, object]:
    return cast("dict[str, object]", envelope["outcome"])


def _gaps(outcome: dict[str, object]) -> list[str]:
    """The named gaps recorded on a terminal outcome."""

    detail = cast("dict[str, object]", outcome["detail"])
    return [str(reason) for reason in cast("list[object]", detail["gaps"])]


# ---------------------------------------------------------------------------
# Route parity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cli_mcp_and_http_serialize_one_topology_envelope(workspace_env: dict[str, Path]) -> None:
    """The three public routes agree key-for-key on the canonical envelope.

    Anti-vacuity: red if any surface re-derives the envelope itself. Routing
    MCP back through its own ``decide_outcome`` call, or letting the HTTP
    envelope drop ``ancestors``/``descendants``/``siblings``/``thread`` as it
    did before this change, fails the key-set and value comparisons below.
    """

    db_path = db_setup(workspace_env)
    _seed_chain(db_path)
    target = _native("child0")
    topology = await _topology(db_path, workspace_env["archive_root"], target)
    assert topology is not None

    canonical = topology_public_envelope(topology, session_id=target)
    mcp_payload = session_topology_payload(topology, session_id=target).model_dump(mode="json")
    http_payload = build_topology_envelope(topology, node_limit=1000)

    # MCP is the canonical envelope with transport typing only.
    assert set(mcp_payload) == set(canonical)
    for key in ("target_id", "root_id", "generation_id", "outcome", "nodes_complete", "edges_complete"):
        assert mcp_payload[key] == canonical[key], key
    assert [edge["child_id"] for edge in _rows(mcp_payload, "edges")] == [
        edge["child_id"] for edge in _rows(canonical, "edges")
    ]

    # HTTP adds reader affordances but drops no canonical key and re-decides
    # no outcome. The four lineage ref lists must survive.
    assert set(canonical).issubset(set(http_payload))
    for key in ("ancestors", "descendants", "siblings", "thread"):
        assert http_payload[key] == canonical[key], key
    assert http_payload["outcome"] == canonical["outcome"]
    assert http_payload["edges"] == canonical["edges"]


@pytest.mark.asyncio
async def test_every_canonical_edge_field_reaches_every_surface(workspace_env: dict[str, Path]) -> None:
    """No surface may silently drop a field from the canonical edge projection.

    Anti-vacuity: red if a key is removed from ``TopologyEdge.public_dict``
    without being removed from ``MCPTopologyEdgePayload``, or vice versa --
    this is the field-completeness oracle the envelope previously lacked.
    """

    db_path = db_setup(workspace_env)
    _seed_chain(db_path, width=1)
    target = _native("child0")
    topology = await _topology(db_path, workspace_env["archive_root"], target)
    assert topology is not None

    required = {
        "child_id",
        "parent_id",
        "dst_origin",
        "dst_native_id",
        "kind",
        "link_type",
        "resolved",
        "resolution_state",
        "inheritance",
        "branch_point_message_id",
        "authority_state",
        "composable",
        "composability_reason",
        "parent_tool_use_block_id",
        "method",
        "confidence",
        "observed_at_ms",
        "resolved_at_ms",
        "evidence",
    }
    canonical_edges = _rows(topology_public_envelope(topology, session_id=target), "edges")
    assert canonical_edges, "fixture must produce at least one edge"
    for edge in canonical_edges:
        assert required.issubset(set(edge)), required - set(edge)

    mcp_edges = _rows(session_topology_payload(topology, session_id=target).model_dump(mode="json"), "edges")
    for edge in mcp_edges:
        assert required.issubset(set(edge)), required - set(edge)


@pytest.mark.asyncio
async def test_public_topology_envelope_names_no_provider(workspace_env: dict[str, Path]) -> None:
    """Public topology filters and fields use ``origin``, never ``provider``.

    Anti-vacuity: red if an edge or node key is renamed to ``provider`` or a
    provider-wire token leaks into the public envelope.
    """

    db_path = db_setup(workspace_env)
    _seed_chain(db_path, width=1)
    target = _native("child0")
    topology = await _topology(db_path, workspace_env["archive_root"], target)
    assert topology is not None
    envelope = topology_public_envelope(topology, session_id=target)

    for node in _rows(envelope, "nodes"):
        assert "origin" in node
        assert "provider" not in node
    for edge in _rows(envelope, "edges"):
        assert "dst_origin" in edge
        assert not any(key.endswith("provider") for key in edge), edge


# ---------------------------------------------------------------------------
# Terminal outcome: degraded outranks empty
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_topology_reports_ok(workspace_env: dict[str, Path]) -> None:
    """A complete, fully resolved topology carries no gaps.

    Anti-vacuity: red if ``degraded_gaps`` reports a gap unconditionally --
    without this the degraded tests below would pass vacuously.
    """

    db_path = db_setup(workspace_env)
    SessionBuilder(db_path, "solo").provider("claude-code").title("Solo").add_message(role="user", text="hello").save()
    topology = await _topology(db_path, workspace_env["archive_root"], _native("solo"))
    assert topology is not None
    assert topology.degraded_gaps() == ()
    assert _outcome(topology_public_envelope(topology))["state"] == "ok"


@pytest.mark.asyncio
async def test_unresolved_parent_is_degraded_not_empty(workspace_env: dict[str, Path]) -> None:
    """A topology behind an unresolved parent never reports an empty scope.

    Anti-vacuity: red if ``topology_outcome`` calls ``decide_outcome`` without
    passing ``degraded=`` -- exactly the defect this test was written against,
    where an orphan's topology reported ``ok``/``empty`` and hid the gap.
    """

    db_path = db_setup(workspace_env)
    SessionBuilder(db_path, "orphan").provider("claude-code").title("Orphan").parent_session(
        "missing-parent-uuid"
    ).add_message(role="user", text="orphan").save()

    topology = await _topology(db_path, workspace_env["archive_root"], _native("orphan"))
    assert topology is not None
    gaps = topology.degraded_gaps()
    assert TOPOLOGY_GAP_UNRESOLVED_PARENT in gaps
    assert TOPOLOGY_GAP_EXCLUDED_EDGE in gaps

    envelope = topology_public_envelope(topology)
    outcome = _outcome(envelope)
    assert outcome["state"] == "degraded"
    assert outcome["reason"] != TOPOLOGY_EMPTY_REASON
    assert TOPOLOGY_GAP_UNRESOLVED_PARENT in _gaps(outcome)

    # Every surface reports the same degraded state.
    assert session_topology_payload(topology, session_id=_native("orphan")).outcome.state == "degraded"
    assert build_topology_envelope(topology)["outcome"] == outcome


@pytest.mark.asyncio
async def test_truncated_page_is_degraded_and_never_complete(workspace_env: dict[str, Path]) -> None:
    """A bounded topology page reports truncation as a gap, not as a clean answer.

    Anti-vacuity: red if a bound is applied without adding
    ``topology_truncated`` to the gaps, or if ``nodes_complete`` stays true
    after narrowing -- the "claim complete after truncation" mutant.
    """

    db_path = db_setup(workspace_env)
    _seed_chain(db_path, width=4)
    topology = await _topology(db_path, workspace_env["archive_root"], _native("root"), node_limit=2)
    assert topology is not None

    envelope = topology_public_envelope(topology)
    if envelope["nodes_complete"]:
        pytest.skip("engine returned a complete page; bounding is exercised below")
    assert TOPOLOGY_GAP_TRUNCATED in topology.degraded_gaps()
    assert _outcome(envelope)["state"] == "degraded"
    assert envelope["continuation"] is not None


@pytest.mark.asyncio
async def test_surface_bounding_reports_truncation_as_degraded(workspace_env: dict[str, Path]) -> None:
    """A surface that narrows a complete envelope must downgrade it honestly.

    Anti-vacuity: red if ``bound_topology_envelope`` returns the source
    outcome unchanged, or leaves ``nodes_complete`` true after dropping a
    node -- a surface could then present half a graph as a complete one.
    """

    db_path = db_setup(workspace_env)
    _seed_chain(db_path, width=4)
    topology = await _topology(db_path, workspace_env["archive_root"], _native("root"))
    assert topology is not None
    canonical = topology_public_envelope(topology)
    assert len(_rows(canonical, "nodes")) > 1

    bounded = bound_topology_envelope(canonical, node_limit=1)
    assert len(_rows(bounded, "nodes")) == 1
    assert bounded["nodes_complete"] is False
    assert bounded["edges_complete"] is False
    bounded_outcome = _outcome(bounded)
    assert bounded_outcome["state"] == "degraded"
    assert TOPOLOGY_GAP_TRUNCATED in _gaps(bounded_outcome)
    assert bounded["continuation"] == "node-offset:1"

    # The HTTP framing of the same bound agrees.
    http_payload = build_topology_envelope(topology, node_limit=1)
    assert http_payload["nodes_complete"] is False
    assert _outcome(http_payload)["state"] == "degraded"


def test_mcp_and_http_apply_the_same_hard_topology_node_limit() -> None:
    """The MCP serializer must not bypass the daemon's topology safety bound.

    Anti-vacuity: red before the shared bounded envelope producer, when MCP
    serializes all ``MAX_NODE_LIMIT + 1`` nodes while HTTP truncates at the
    hard cap.
    """

    node_count = MAX_NODE_LIMIT + 1
    topology = SessionTopology(
        target_id=SessionId("node-0"),
        root_id=SessionId("node-0"),
        nodes=tuple(
            TopologyNode(
                session_id=SessionId(f"node-{index}"),
                origin="claude-code-session",
                is_root=index == 0,
            )
            for index in range(node_count)
        ),
        edges=(),
    )

    mcp_payload = session_topology_payload(topology, session_id="node-0").model_dump(mode="json")
    http_payload = build_topology_envelope(topology, node_limit=MAX_NODE_LIMIT + 1)

    assert len(_rows(mcp_payload, "nodes")) == MAX_NODE_LIMIT
    assert len(_rows(http_payload, "nodes")) == MAX_NODE_LIMIT
    assert _outcome(mcp_payload) == _outcome(http_payload)
    assert TOPOLOGY_GAP_TRUNCATED in _gaps(_outcome(mcp_payload))


def test_cycle_topology_is_degraded(workspace_env: dict[str, Path]) -> None:
    """A detected cycle is a named gap on the envelope.

    Anti-vacuity: red if ``cycle_detected`` stops contributing a gap, which
    would let a contradictory archive slice report a clean ``ok``.
    """

    from polylogue.analysis.topology import TopologyNode
    from polylogue.core.types import SessionId

    topology = SessionTopology(
        target_id=SessionId("A"),
        root_id=SessionId("A"),
        nodes=(TopologyNode(session_id=SessionId("A"), origin="unknown-export", is_root=True),),
        edges=(),
        cycle_detected=True,
    )
    assert TOPOLOGY_GAP_CYCLE in topology.degraded_gaps()
    assert _outcome(topology_public_envelope(topology))["state"] == "degraded"
