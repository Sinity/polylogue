"""Source-to-graph incident materialization from the archive's own evidence.

An operator investigating a real incident ("agent X did Y around time Z in
repo R") starts from the archive's own record of what actually ran, not an
issue tracker or a separately generated report.

This module is that missing half. It is deliberately *not* a second copy of
``insights.claude_workflow_materializer`` (which reads raw Claude Workflow
orchestration artifacts -- ``workflow_run_snapshot``/``journal``/sidecar/
adopt-manifest -- straight from ``source.db``). Instead it consumes the
already-real, already-evidence-backed :class:`~polylogue.insights.run_projection.ProjectedRun`
and :class:`~polylogue.insights.run_projection.ObservedEvent` rows every
session origin produces (``storage/insights/session/run_projection_rows.py``
computes them from ``sessions``/``blocks`` on every read -- see
``polylogue.storage.repository.insight.run_projection_reads``), and adapts
them into the same generic :class:`~polylogue.insights.work_evidence.WorkEvidenceGraph`
shape ``work_effects``/``work_reconciliation`` already know how to reconcile.
This makes it provider-neutral by construction: Claude Code, Codex, or any
other origin that already flows through ``build_run_projection`` gets the
same incident graph for free, not a Workflow-only one.

Node/edge shape, grounded entirely in real per-session evidence:

- One ``run`` node per :class:`ProjectedRun` (:func:`node_from_projected_run`),
  linked ``invoked`` from a parent run to a subagent run when the run's own
  ``parent_run_ref`` says so -- never inferred from time/name proximity.
- One ``session-segment`` node per run summarizing that run's own
  :class:`ObservedEvent` evidence (:func:`session_segment_from_observed_events`),
  ``represented_by``-linked from the run.
- One ``effect`` node per commit/GitHub-PR/GitHub-issue :class:`ObjectRef` an
  in-session tool call *mentioned* (``ObservedEvent.object_refs``),
  ``mentioned``-linked from the segment that cited it. These are explicitly
  *unresolved, inferred* -- a session mentioning "PR #42" is not the same as
  independently observing PR #42's GitHub state; that corroboration is
  ``insights.work_effects``'s job, layered on top of this graph exactly like
  it already layers onto the Claude Workflow and Beads-ledger graphs.
- One ``claim`` node per subagent's own self-reported final result
  (``ObservedEvent`` of kind ``subagent_finished``, whose ``summary`` is the
  subagent's own ``final_report_preview``) -- a first-class, evidence-backed
  claim about outcome, ``produced``-linked from the run that authored it.
  This is the real analogue of ``build_repository_claim_graph``'s Beads-ledger
  claims, except sourced from the archive's own transcript content, which is
  exactly what an incident investigation starting from "what did the archive
  record" needs and what the ledger-only path cannot supply.

Each ``run`` node now also carries a real ``actor_ref``/``execution_context_ref``
pair (polylogue-h6r, :mod:`polylogue.insights.actor_context`): ``harness`` and
``provider_origin`` are provider-reported facts about how the run executed,
and ``agent_ref`` (when a runtime structurally reports an explicit subagent
persona) is real, provider-observed actor identity -- none of it is
fabricated from bare ``cwd``/``git_branch`` strings, which is exactly the
unproven-authority shortcut OriginSpec (polylogue-2qx) exists to prevent.
Per-session model/instructions evidence lives on ``ParsedSession``, one layer
below ``ProjectedRun``, and is not threaded through this projection today;
:func:`~polylogue.insights.actor_context.actor_ref_from_session` /
:func:`~polylogue.insights.actor_context.execution_context_ref_from_session`
cover that evidence directly for callers with a ``ParsedSession`` in hand.
Everything polylogue-7aw's configuration-artifact ingestion would add
(tools/MCP profile, permissions, runtime build, sampling parameters) is
declared an explicit unknown field on the execution context, never fabricated
or silently omitted.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from polylogue.core.refs import ObjectRef, ObjectRefKind
from polylogue.insights.actor_context import actor_ref_from_run, execution_context_ref_from_run
from polylogue.insights.run_projection import ObservedEvent, ProjectedRun
from polylogue.insights.work_evidence import (
    WorkEvidenceEdge,
    WorkEvidenceEdgeKind,
    WorkEvidenceGraph,
    WorkEvidenceNode,
    node_from_projected_run,
    session_segment_from_observed_events,
)

#: Object-ref kinds an in-session tool call can mention that this module
#: promotes to an independently-reconcilable ``effect`` node. Deliberately a
#: subset of ``ObservedEvent.object_refs`` -- ``file``/``tool-call``/``agent``
#: refs describe the run itself, not an external repository effect a claim
#: could be reconciled against.
_MENTIONABLE_EFFECT_KINDS: Final[frozenset[ObjectRefKind]] = frozenset({"commit", "github-pr", "github-issue"})


def _segment_ref(run_ref: ObjectRef) -> ObjectRef:
    return ObjectRef(kind="work-session-segment", object_id=f"{run_ref.object_id}:segment")


def _claim_ref(event: ObservedEvent) -> ObjectRef:
    return ObjectRef(kind="work-claim", object_id=f"claim:{event.event_ref.object_id}")


def materialize_incident_evidence_graph(
    *,
    graph_id: str,
    corpus_snapshot_ref: ObjectRef,
    runs: Sequence[ProjectedRun],
    events: Sequence[ObservedEvent],
) -> WorkEvidenceGraph:
    """Adapt real per-session run/event evidence into one work-evidence graph.

    Pure: every node/edge is grounded in the ``runs``/``events`` arguments the
    caller already loaded (via the repository's real
    ``query_runs``/``query_observed_events`` routes, or a fixture in tests).
    Nothing here re-derives evidence or reads storage -- see
    ``polylogue.operations.incident_evidence_materialization`` for the
    production caller that resolves an incident's session set and loads
    these.
    """

    nodes: dict[str, WorkEvidenceNode] = {}
    edges: list[tuple[WorkEvidenceEdgeKind, ObjectRef, ObjectRef]] = []

    run_by_ref: dict[str, ProjectedRun] = {}
    for run in runs:
        node = node_from_projected_run(
            run,
            corpus_snapshot_ref=corpus_snapshot_ref,
            actor_ref=actor_ref_from_run(run),
            execution_context_ref=execution_context_ref_from_run(run),
        )
        nodes[node.ref.format()] = node
        run_by_ref[run.run_ref.format()] = run

    for run in runs:
        if run.parent_run_ref is not None and run.parent_run_ref.format() in nodes:
            edges.append(("invoked", run.parent_run_ref, run.run_ref))

    events_by_run: dict[str, list[ObservedEvent]] = defaultdict(list)
    for event in events:
        events_by_run[event.run_ref.format()].append(event)

    for run_ref_str, run_events in events_by_run.items():
        if run_ref_str not in run_by_ref:
            continue
        run = run_by_ref[run_ref_str]
        segment_ref = _segment_ref(run.run_ref)
        segment = session_segment_from_observed_events(
            ref=segment_ref,
            label=f"{run.title or run.run_ref.object_id} transcript segment",
            corpus_snapshot_ref=corpus_snapshot_ref,
            events=tuple(run_events),
        )
        nodes[segment.ref.format()] = segment
        edges.append(("represented_by", run.run_ref, segment_ref))

        for event in run_events:
            for object_ref in event.object_refs:
                if object_ref.kind not in _MENTIONABLE_EFFECT_KINDS:
                    continue
                if object_ref.format() not in nodes:
                    nodes[object_ref.format()] = WorkEvidenceNode(
                        ref=object_ref,
                        kind="effect",
                        label=f"{object_ref.kind} {object_ref.object_id} mentioned in {run.run_ref.object_id}",
                        evidence_refs=event.evidence_refs,
                        corpus_snapshot_ref=corpus_snapshot_ref,
                        authority="inferred",
                        confidence=0.5,
                        association_state="unresolved",
                    )
                edges.append(("mentioned", segment_ref, object_ref))

            if event.kind == "subagent_finished":
                claim_ref = _claim_ref(event)
                nodes[claim_ref.format()] = WorkEvidenceNode(
                    ref=claim_ref,
                    kind="claim",
                    label=f"{run.run_ref.object_id} self-reported result",
                    claim_text=event.summary,
                    evidence_refs=event.evidence_refs,
                    corpus_snapshot_ref=corpus_snapshot_ref,
                    authority="provider",
                    confidence=1.0,
                )
                edges.append(("produced", run.run_ref, claim_ref))

    built_edges: dict[str, WorkEvidenceEdge] = {}
    for kind, source_ref, target_ref in edges:
        source_node = nodes[source_ref.format()]
        target_node = nodes[target_ref.format()]
        edge_ref = ObjectRef(kind="work-edge", object_id=f"{kind}:{source_ref.format()}->{target_ref.format()}")
        if edge_ref.format() in built_edges:
            continue
        built_edges[edge_ref.format()] = WorkEvidenceEdge(
            ref=edge_ref,
            kind=kind,
            source_ref=source_ref,
            target_ref=target_ref,
            evidence_refs=target_node.evidence_refs or source_node.evidence_refs,
            corpus_snapshot_ref=corpus_snapshot_ref,
            authority="provider",
            confidence=1.0,
        )

    return WorkEvidenceGraph(
        graph_id=graph_id,
        corpus_snapshot_ref=corpus_snapshot_ref,
        nodes=tuple(nodes.values()),
        edges=tuple(built_edges.values()),
    )


@dataclass(frozen=True, slots=True)
class IncidentMaterializationSummary:
    """Quantified result of one incident-evidence materialization pass."""

    graph_id: str
    session_count: int
    run_count: int
    session_segment_count: int
    claim_count: int
    mentioned_effect_count: int
    edge_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "graph_id": self.graph_id,
            "session_count": self.session_count,
            "run_count": self.run_count,
            "session_segment_count": self.session_segment_count,
            "claim_count": self.claim_count,
            "mentioned_effect_count": self.mentioned_effect_count,
            "edge_count": self.edge_count,
        }


def summarize_incident_graph(graph: WorkEvidenceGraph, *, session_ids: Sequence[str]) -> IncidentMaterializationSummary:
    """Quantify a materialized incident graph for CLI/report output."""

    return IncidentMaterializationSummary(
        graph_id=graph.graph_id,
        session_count=len(set(session_ids)),
        run_count=sum(node.kind == "run" for node in graph.nodes),
        session_segment_count=sum(node.kind == "session-segment" for node in graph.nodes),
        claim_count=sum(node.kind == "claim" for node in graph.nodes),
        mentioned_effect_count=sum(node.kind == "effect" for node in graph.nodes),
        edge_count=len(graph.edges),
    )


def incident_corpus_snapshot_ref(
    *,
    session_ids: Sequence[str],
    runs: Sequence[ProjectedRun],
    events: Sequence[ObservedEvent],
) -> ObjectRef:
    """Content-address one incident materialization's exact evidence inputs."""

    payload: Mapping[str, object] = {
        "session_ids": sorted(set(session_ids)),
        "run_refs": sorted(run.run_ref.format() for run in runs),
        "event_refs": sorted(event.event_ref.format() for event in events),
    }
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return ObjectRef(kind="context-snapshot", object_id=f"incident-evidence-v1:{digest}")


__all__ = [
    "IncidentMaterializationSummary",
    "incident_corpus_snapshot_ref",
    "materialize_incident_evidence_graph",
    "summarize_incident_graph",
]
