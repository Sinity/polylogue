"""Production read path: materialize an incident's work-evidence graph from the archive.

``insights.incident_evidence_materialization`` is pure -- it adapts whatever
:class:`~polylogue.insights.run_projection.ProjectedRun`/:class:`ObservedEvent`
rows a caller hands it. This module is the one production caller that
resolves *which* sessions an incident investigation means (an explicit
session-id set, or a :class:`~polylogue.archive.filter.filters.SessionFilter`
selection over repo/time/keyword clues -- the same sparse-clue shape the
owning bead's incident describes: "repo, approximate time, roughly N agents
worked on concerns"), loads each session's real content and compiles its full
run projection, and -- when ``apply=True`` -- persists the resulting graph
through the same ``replace_work_evidence_graph`` route the Claude Workflow
materializer and work-effect reconciliation already use.

Loading uses :func:`~polylogue.insights.transforms.compile_session_run_projection`
against each full :class:`~polylogue.archive.session.domain_models.Session`
(the same function ``storage.insights.session.rebuild`` already uses for its
materialization-ledger stamp), not the thinner ``query_runs``/
``query_observed_events`` read-model routes. Those routes are deliberately
CTE/source-derived-only after polylogue-dab/itvd and only ever emit
``session_started``/``tool_finished`` events with bare ``tool-call`` object
refs (see ``storage/sqlite/run_projection_relations.py``) -- no subagent
self-reported results, no mentioned commit/PR/issue refs. Compiling from the
full ``Session`` instead is what actually surfaces those real per-session
signals the incident graph's claim/effect nodes need.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from polylogue.core.errors import PolylogueError
from polylogue.insights.incident_evidence_materialization import (
    IncidentMaterializationSummary,
    incident_corpus_snapshot_ref,
    materialize_incident_evidence_graph,
    summarize_incident_graph,
)
from polylogue.insights.run_projection import ObservedEvent, ProjectedRun
from polylogue.insights.transforms import compile_session_run_projection
from polylogue.insights.work_evidence import WorkEvidenceGraph
from polylogue.storage.query_models import SessionRecordQuery
from polylogue.storage.repository import SessionRepository


class NoIncidentSessionsFoundError(PolylogueError):
    """Raised when an incident's session-selection criteria matched nothing.

    A silent empty graph would read as "the archive proves nothing happened"
    -- exactly the false negative this program exists to eliminate. Callers
    get an explicit, actionable error instead.
    """


@dataclass(frozen=True, slots=True)
class IncidentEvidenceMaterializationResult:
    graph: WorkEvidenceGraph
    summary: IncidentMaterializationSummary
    applied: bool


async def _session_links_for(repository: SessionRepository, session_id: str) -> list[dict[str, object]]:
    """Resolve one session's lineage links, including unlinked children.

    Mirrors ``Polylogue._session_digest``'s own link resolution exactly:
    ``session_links`` rows plus a synthesized ``resolved`` link per child
    session whose ``parent_session_id`` already points here even when no
    explicit ``session_links`` row was recorded for it.
    """

    links: list[dict[str, object]] = list(await repository.queries.list_session_links_for_session(session_id))
    children = await repository.queries.list_sessions(SessionRecordQuery(parent_id=session_id))
    links.extend(
        {
            "dst_origin": child.origin.value,
            "dst_native_id": child.native_id,
            "resolved_dst_session_id": str(child.session_id),
            "status": "resolved",
            "link_type": (child.branch_type.value if child.branch_type is not None else "child"),
        }
        for child in children
    )
    return links


async def materialize_incident_work_evidence(
    repository: SessionRepository,
    *,
    session_ids: Sequence[str],
    graph_id: str,
    apply: bool = False,
) -> IncidentEvidenceMaterializationResult:
    """Build one incident's work-evidence graph from real per-session evidence.

    ``session_ids`` is the caller-resolved incident session set (see
    ``polylogue.cli.commands.materialize_incident_evidence`` for the
    ``SessionFilter``-backed CLI resolution). Each session is loaded in full
    and compiled through the real, already-production
    ``compile_session_run_projection`` path, so the resulting graph reflects
    each session's actual runs, tool calls, subagent reports, and structured
    outcome events -- not a hand-built fixture or an external ledger.
    """

    deduped_ids = tuple(dict.fromkeys(session_ids))
    if not deduped_ids:
        raise NoIncidentSessionsFoundError("materialize_incident_work_evidence requires at least one session id")

    sessions = await repository.get_many(list(deduped_ids))
    if not sessions:
        raise NoIncidentSessionsFoundError(
            f"no sessions found for the {len(deduped_ids)} selected id(s): {', '.join(sorted(deduped_ids))}"
        )

    all_runs: list[ProjectedRun] = []
    all_events: list[ObservedEvent] = []
    for session in sessions:
        session_id = str(session.id)
        session_links = await _session_links_for(repository, session_id)
        run_projection = compile_session_run_projection(session, session_links=session_links)
        all_runs.extend(run_projection.runs)
        all_events.extend(run_projection.events)

    resolved_session_ids = tuple(str(session.id) for session in sessions)
    corpus_snapshot_ref = incident_corpus_snapshot_ref(
        session_ids=resolved_session_ids, runs=all_runs, events=all_events
    )
    graph = materialize_incident_evidence_graph(
        graph_id=graph_id,
        corpus_snapshot_ref=corpus_snapshot_ref,
        runs=all_runs,
        events=all_events,
    )
    if apply:
        await repository.replace_work_evidence_graph(graph)

    summary = summarize_incident_graph(graph, session_ids=resolved_session_ids)
    return IncidentEvidenceMaterializationResult(graph=graph, summary=summary, applied=apply)


__all__ = [
    "IncidentEvidenceMaterializationResult",
    "NoIncidentSessionsFoundError",
    "materialize_incident_work_evidence",
]
