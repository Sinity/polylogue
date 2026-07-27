"""Source-to-graph incident materialization: pure graph-builder behavior.

Anti-vacuity: the fixture session below is compiled through the real,
production ``compile_session_run_projection`` (the same function
``polylogue.operations.incident_evidence_materialization`` calls against a
live archive) -- not a hand-built ``ProjectedRun``/``ObservedEvent`` double.
Removing the ``invoked``/``represented_by``/``produced``/``mentioned`` edge
construction, or the commit/PR/issue mention filter, makes the assertions
below fail against this same real projection.
"""

from __future__ import annotations

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import Role
from polylogue.archive.session.domain_models import Session
from polylogue.core.enums import Origin
from polylogue.core.refs import ObjectRef
from polylogue.core.types import SessionId
from polylogue.insights.incident_evidence_materialization import (
    incident_corpus_snapshot_ref,
    materialize_incident_evidence_graph,
    summarize_incident_graph,
)
from polylogue.insights.transforms import compile_session_run_projection
from polylogue.insights.work_evidence import WorkEvidenceGraph


def _incident_session() -> Session:
    return Session(
        id=SessionId("codex-session:incident-demo"),
        origin=Origin.CODEX_SESSION,
        title="Ship the incident-graph slice",
        git_branch="feature/incident-demo",
        working_directories=("/realm/project/polylogue",),
        messages=MessageCollection(
            messages=[
                Message(id="m1", role=Role.USER, text="Goal: ship the incident graph slice"),
                Message(
                    id="m2",
                    role=Role.ASSISTANT,
                    text="Ran verify and opened the tracking PR.",
                    blocks=[
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "Bash",
                            "tool_input": {"command": "devtools verify --quick"},
                        },
                        {
                            "type": "tool_result",
                            "tool_id": "tool-1",
                            "text": "ok\nhttps://github.com/Sinity/polylogue/pull/4242",
                            "tool_result_exit_code": 0,
                        },
                    ],
                ),
                Message(
                    id="m3",
                    role=Role.ASSISTANT,
                    text="Dispatching a subagent to investigate the archive.",
                    blocks=[
                        {
                            "type": "tool_use",
                            "id": "tool-2",
                            "name": "Task",
                            "tool_input": {
                                "subagent_type": "Explore",
                                "taskId": "task-incident-1",
                                "child_session_id": "codex-session:incident-child-1",
                                "prompt": "Investigate the incident evidence gap.",
                            },
                        },
                        {
                            "type": "tool_result",
                            "tool_id": "tool-2",
                            "text": "Subagent done: confirmed the gap and filed a follow-up.",
                        },
                    ],
                ),
            ]
        ),
    )


def _graph_for_incident_session() -> tuple[Session, WorkEvidenceGraph]:
    session = _incident_session()
    projection = compile_session_run_projection(session)
    corpus_snapshot_ref = incident_corpus_snapshot_ref(
        session_ids=(str(session.id),), runs=projection.runs, events=projection.events
    )
    graph = materialize_incident_evidence_graph(
        graph_id="incident:demo",
        corpus_snapshot_ref=corpus_snapshot_ref,
        runs=projection.runs,
        events=projection.events,
    )
    return session, graph


def test_materialized_graph_has_one_run_node_per_real_run() -> None:
    session, graph = _graph_for_incident_session()
    run_nodes = {node.ref.object_id for node in graph.nodes if node.kind == "run"}
    assert run_nodes == {str(session.id), f"{session.id}:subagent:0:tool-2"}


def test_materialized_graph_links_subagent_run_to_parent_via_invoked_edge() -> None:
    session, graph = _graph_for_incident_session()
    invoked = [(edge.source_ref.object_id, edge.target_ref.object_id) for edge in graph.edges if edge.kind == "invoked"]
    assert (str(session.id), f"{session.id}:subagent:0:tool-2") in invoked


def test_materialized_graph_builds_one_segment_per_run_represented_by_edge() -> None:
    session, graph = _graph_for_incident_session()
    segment_nodes = {node.ref.object_id for node in graph.nodes if node.kind == "session-segment"}
    assert segment_nodes == {
        f"{session.id}:segment",
        f"{session.id}:subagent:0:tool-2:segment",
    }
    represented_by = {
        (edge.source_ref.object_id, edge.target_ref.object_id) for edge in graph.edges if edge.kind == "represented_by"
    }
    assert (str(session.id), f"{session.id}:segment") in represented_by
    assert (f"{session.id}:subagent:0:tool-2", f"{session.id}:subagent:0:tool-2:segment") in represented_by


def test_materialized_graph_derives_claim_from_subagent_self_report() -> None:
    session, graph = _graph_for_incident_session()
    claims = [node for node in graph.nodes if node.kind == "claim"]
    assert len(claims) == 1
    claim = claims[0]
    assert claim.claim_text is not None
    assert "confirmed the gap and filed a follow-up" in claim.claim_text
    produced = [
        (edge.source_ref.object_id, edge.target_ref.object_id) for edge in graph.edges if edge.kind == "produced"
    ]
    assert (f"{session.id}:subagent:0:tool-2", claim.ref.object_id) in produced


def test_materialized_graph_derives_unresolved_effect_from_mentioned_pull_request() -> None:
    session, graph = _graph_for_incident_session()
    effects = [node for node in graph.nodes if node.kind == "effect"]
    assert len(effects) == 1
    effect = effects[0]
    assert effect.ref == ObjectRef(kind="github-pr", object_id="#4242")
    assert effect.association_state == "unresolved"
    assert effect.authority == "inferred"
    mentioned = [
        (edge.source_ref.object_id, edge.target_ref.object_id) for edge in graph.edges if edge.kind == "mentioned"
    ]
    assert (f"{session.id}:segment", "#4242") in mentioned


def test_summarize_incident_graph_reports_counts() -> None:
    session, graph = _graph_for_incident_session()
    summary = summarize_incident_graph(graph, session_ids=(str(session.id),))
    assert summary.session_count == 1
    assert summary.run_count == 2
    assert summary.session_segment_count == 2
    assert summary.claim_count == 1
    assert summary.mentioned_effect_count == 1
    assert summary.edge_count == len(graph.edges)


def test_incident_corpus_snapshot_ref_is_content_addressed_and_stable() -> None:
    session, _ = _graph_for_incident_session()
    projection = compile_session_run_projection(session)
    first = incident_corpus_snapshot_ref(session_ids=(str(session.id),), runs=projection.runs, events=projection.events)
    second = incident_corpus_snapshot_ref(
        session_ids=(str(session.id),), runs=projection.runs, events=projection.events
    )
    assert first == second
    assert first.kind == "context-snapshot"
    assert first.object_id.startswith("incident-evidence-v1:")


def test_materialize_incident_evidence_graph_is_empty_for_no_evidence() -> None:
    graph = materialize_incident_evidence_graph(
        graph_id="incident:empty",
        corpus_snapshot_ref=ObjectRef(kind="context-snapshot", object_id="empty"),
        runs=(),
        events=(),
    )
    assert graph.nodes == ()
    assert graph.edges == ()
