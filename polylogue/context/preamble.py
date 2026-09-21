"""Context preamble composition for the ``read --view context`` surface."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, cast

from polylogue.analysis.lineage_graph import CompactLineageGraph, LineageEdgeRole, LineageNodeRole
from polylogue.context.scheduler import ContextAssembly, ContextItem, record_context_ledger, schedule_context
from polylogue.core.assertions import derive_assertion_context_trust
from polylogue.core.errors import DatabaseError
from polylogue.core.refs import ExecutionContextRef
from polylogue.logging import WARNING, emit, get_logger
from polylogue.storage.sqlite.connection_profile import open_connection
from polylogue.surfaces.compaction import estimate_tokens
from polylogue.surfaces.payloads import (
    AssertionClaimPayload,
    ContextPreamble,
    ContextPreambleAssertionGuidance,
    ContextPreambleGuidance,
    ContextPreambleLineage,
    ContextPreambleOverlapBasis,
    ContextPreambleProjectState,
    ContextPreambleQuotedEvidence,
    ContextPreambleSession,
    ContextTrustClass,
)

if TYPE_CHECKING:
    from polylogue.cli.shared.types import AppEnv

logger = get_logger(__name__)


def _observation_value(value: object) -> str | int | float | bool | None:
    if isinstance(value, (str, int, float, bool)):
        return value if not isinstance(value, str) or value else value
    return None


def _preamble_execution_context(
    *,
    session_id: str | None,
    boundary: str,
    repo_path: str | None,
    cwd: str | None,
    related_limit: int,
    session: object | None,
) -> ExecutionContextRef:
    """Capture the boundary inputs that actually shaped this preamble."""

    fields: dict[str, object] = {"boundary": boundary, "related_limit": related_limit}
    for name, value in (
        ("session_id", session_id),
        ("repo_path", repo_path),
        ("cwd", cwd),
        ("origin", getattr(session, "origin", None) if session is not None else None),
        ("model", getattr(session, "model", None) if session is not None else None),
        ("permission_mode", getattr(session, "permission_mode", None) if session is not None else None),
    ):
        observed = _observation_value(value)
        if observed is not None:
            fields[name] = observed
    unknown_fields = tuple(name for name in ("model", "permission_mode", "runtime") if name not in fields)
    return ExecutionContextRef.from_observation(fields, unknown_fields=unknown_fields)


def _record_preamble_ledger(polylogue: object, assembly: ContextAssembly) -> None:
    """Best-effort persistence for disposable scheduler receipts."""

    config = getattr(polylogue, "config", None)
    archive_root = getattr(config, "archive_root", None)
    if not isinstance(archive_root, (str, Path)):
        return
    ops_db = Path(archive_root) / "ops.db"
    try:
        if not ops_db.exists():
            from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
            from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

            initialize_archive_database(ops_db, ArchiveTier.OPS)
        conn = open_connection(ops_db)
        try:
            record_context_ledger(conn, assembly, observed_at_ms=int(datetime.now(timezone.utc).timestamp() * 1000))
        finally:
            conn.close()
    except (OSError, TypeError, ValueError, sqlite3.Error, DatabaseError):
        # DatabaseError covers a tier this runtime cannot use, including one at
        # a version it has moved past. A disposable receipt never fails the
        # preamble it is a receipt for.
        logger.debug("context preamble: scheduler receipt could not be persisted", exc_info=True)


def _git_project_state(cwd: str | None) -> tuple[ContextPreambleProjectState | None, str | None]:
    """Read branch + recent commits from a local git checkout, best-effort.

    Never raises: a missing/non-git ``cwd`` must not break SessionStart
    context injection (or CLI context composition run outside a repo).

    Returns ``(state, failure)``. ``failure`` is the same
    ``"ExceptionType: message"`` string the other preamble sections record, so
    a git read that *broke* is distinguishable from a cwd that simply is not a
    checkout. A bare swallow here made this the one section where a failed read
    and a clean non-repo build produced byte-identical output.
    """
    import subprocess

    try:
        branch: str | None = None
        commits: list[str] = []
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd or ".",
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
        result2 = subprocess.run(
            ["git", "log", "--oneline", "-5"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd or ".",
        )
        if result2.returncode == 0:
            commits = [line.strip() for line in result2.stdout.strip().split("\n") if line]
        if branch or commits:
            return ContextPreambleProjectState(branch=branch, recent_commits=commits), None
    except Exception as exc:
        # A non-repo cwd is an ordinary answer (returncode != 0 above, no
        # exception). Reaching here means the read itself failed -- git absent,
        # the cwd gone, the 5s timeout blown -- which is a recorded gap, not an
        # empty project section.
        emit(
            "context.preamble.project_state_failed",
            level=WARNING,
            outcome="degraded",
            reason="git_project_state_unreadable",
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        return None, f"{type(exc).__name__}: {exc}"
    return None, None


# --- budget-aware preamble segmentation -------------------------------------
#
# The preamble used to cross the scheduler as ONE item. Under a finite budget
# that reduced to a binary include/exclude, and the exclude branch dropped
# ``recent_related_sessions`` and ``guidance`` wholesale -- the continuation
# material itself -- while keeping the cheap framing, so the result still read
# as a well-formed preamble. The context-image compiler already solves this by
# offering the scheduler one candidate per segment and surfacing each
# budget-dropped segment as a named omission; this follows that scheme rather
# than adding a second one.
#
# ``rank`` is the protection order: the scheduler admits higher ranks first, so
# a lower rank is what sheds first. The order is a claim about what a
# continuation cannot be reconstructed without:
#
#   guidance (50)                 operator/assertion directives scoped to this
#                                 session. Nothing else in the archive restates
#                                 them, and acting without them is acting
#                                 against an instruction the operator gave.
#   recent_related_sessions (40)  the resume candidates -- the reason a
#                                 continuation preamble exists at all.
#   session_lineage (30)          parent/sibling ids. Load-bearing for
#                                 orientation, but re-derivable by one query.
#   project_state (20)            branch + last five commits. The agent can
#                                 read this itself with one command.
#   source_tool_calls (10)        provenance breadcrumb for which tool composed
#                                 the preamble. Useful in a receipt, inert for
#                                 doing the work.
#
# Every sheddable segment that can lose bulk without losing meaning carries a
# reducer, so the scheduler's single degrade attempt can keep a smaller true
# version instead of dropping the segment outright. Reducers never truncate
# assertion prose: a half-quoted operator instruction is worse than a recorded
# omission.
_SEGMENT_RANKS: dict[str, int] = {
    "guidance": 50,
    "recent_related_sessions": 40,
    "session_lineage": 30,
    "project_state": 20,
    "source_tool_calls": 10,
}

# Recorded in ``component_failures`` under this prefix so a budget shed is
# never mistaken for a lookup failure, while still travelling in the declared
# "this section is not complete" channel every consumer already reads.
_BUDGET_FAILURE_PREFIX = "context_budget:"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True, slots=True)
class _PreambleSegment:
    """One independently budgeted section of the preamble."""

    name: str
    full: object
    reduced: object | None
    reduced_detail: str

    @property
    def rank(self) -> int:
        return _SEGMENT_RANKS[self.name]


def _reduced_related(related: list[ContextPreambleSession]) -> ContextPreambleSession | None:
    """Keep the top resume candidate without its two bulkiest fields."""

    if not related:
        return None
    top = related[0]
    return top.model_copy(update={"summary": None, "overlap_basis": None})


def _preamble_segments(
    *,
    source_tool_calls: dict[str, str],
    lineage: ContextPreambleLineage | None,
    related: list[ContextPreambleSession],
    project: ContextPreambleProjectState | None,
    guidance: ContextPreambleGuidance | None,
) -> tuple[_PreambleSegment, ...]:
    """Build the sheddable segments, skipping the ones with nothing in them.

    An absent section produces no candidate at all: "nothing relevant existed"
    must not enter the ledger looking like "the budget took it".
    """

    segments: list[_PreambleSegment] = []
    if guidance is not None and guidance.assertions:
        # Reduce by keeping whole assertions, never by clipping their prose.
        reduced_guidance = (
            guidance.model_copy(update={"assertions": guidance.assertions[:1]})
            if len(guidance.assertions) > 1
            else None
        )
        segments.append(
            _PreambleSegment(
                name="guidance",
                full=guidance,
                reduced=reduced_guidance,
                reduced_detail=f"kept 1 of {len(guidance.assertions)} assertions whole",
            )
        )
    if related:
        reduced_related = _reduced_related(related)
        segments.append(
            _PreambleSegment(
                name="recent_related_sessions",
                full=related,
                reduced=[reduced_related] if reduced_related is not None else None,
                reduced_detail=f"kept the top 1 of {len(related)} candidates without summary/overlap_basis",
            )
        )
    if lineage is not None:
        reduced_lineage = (
            lineage.model_copy(update={"sibling_session_ids": []}) if lineage.sibling_session_ids else None
        )
        segments.append(
            _PreambleSegment(
                name="session_lineage",
                full=lineage,
                reduced=reduced_lineage,
                reduced_detail=f"dropped {len(lineage.sibling_session_ids)} sibling session ids",
            )
        )
    if project is not None:
        reduced_project = project.model_copy(update={"recent_commits": []}) if project.recent_commits else None
        segments.append(
            _PreambleSegment(
                name="project_state",
                full=project,
                reduced=reduced_project,
                reduced_detail=f"dropped {len(project.recent_commits)} recent commits",
            )
        )
    if source_tool_calls:
        segments.append(
            _PreambleSegment(name="source_tool_calls", full=source_tool_calls, reduced=None, reduced_detail="")
        )
    return tuple(segments)


def _segment_payload(value: object) -> object:
    """JSON-ready projection of a segment value, for cost and identity."""

    if isinstance(value, list):
        return [_segment_payload(entry) for entry in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json", exclude_none=True)
    return value


def _candidate_overlap_basis(candidate: object) -> ContextPreambleOverlapBasis | None:
    basis = getattr(candidate, "overlap_basis", None)
    model_dump = getattr(basis, "model_dump", None)
    if not callable(model_dump):
        return None
    raw = model_dump(mode="json")
    if not isinstance(raw, dict):
        return None
    return ContextPreambleOverlapBasis.model_validate(raw)


async def build_context_preamble_payload(
    polylogue: object,
    *,
    session_id: str | None,
    related_limit: int = 5,
    repo_path: str | None = None,
    cwd: str | None = None,
    recent_files: tuple[str, ...] = (),
    source_tool_calls: dict[str, str] | None = None,
    require_session: bool = True,
    boundary: str = "session_start",
    token_budget: int | None = None,
) -> ContextPreamble | None:
    """Build the shared typed context preamble payload for one seed session.

    CLI, MCP, API, and daemon read-view routes all use this builder so the
    context view does not fork into separate browser/MCP/CLI payload shapes.
    """

    conv = await polylogue.get_session(session_id) if session_id else None  # type: ignore[attr-defined]
    if conv is None and require_session:
        return None

    # Each optional section degrades gracefully (the preamble must never crash
    # a SessionStart hook), but every failure is recorded in
    # ``component_failures`` so consumers can distinguish "nothing relevant"
    # from "lookup failed" — silent context loss is invisible by construction.
    component_failures: dict[str, str] = {}

    lineage: ContextPreambleLineage | None = None
    if session_id:
        try:
            # The compact relation (polylogue-4ts.9) carries the seed-relative
            # roles this section needs and hydrates no transcript.
            graph = await polylogue.compact_lineage(  # type: ignore[attr-defined]
                session_id, node_limit=None, edge_limit=None, include_accounting=False
            )
            if graph:
                lineage = _preamble_lineage(graph)
        except Exception as exc:
            component_failures["session_lineage"] = f"{type(exc).__name__}: {exc}"
            logger.warning("context preamble: session lineage lookup failed for %s: %s", session_id, exc)

    related: list[ContextPreambleSession] = []
    try:
        repo = repo_path or (getattr(conv, "git_repository_url", None) if conv is not None else None) or "."
        candidates = await polylogue.find_resume_candidates(  # type: ignore[attr-defined]
            repo_path=str(repo),
            cwd=cwd,
            recent_files=recent_files,
            limit=max(1, related_limit),
        )
        for c in candidates:
            cid = getattr(c, "logical_session_id", None) or getattr(c, "session_id", "") or "?"
            related.append(
                ContextPreambleSession(
                    session_id=str(cid),
                    title=getattr(c, "title", None),
                    date=getattr(c, "date", None),
                    terminal_state=getattr(c, "terminal_state", None),
                    objective_posture=getattr(c, "objective_posture", None),
                    summary=getattr(c, "summary", None),
                    origin=getattr(c, "origin", None),
                    overlap_basis=_candidate_overlap_basis(c),
                )
            )
    except Exception as exc:
        component_failures["recent_related_sessions"] = f"{type(exc).__name__}: {exc}"
        logger.warning("context preamble: resume-candidate lookup failed: %s", exc)

    project: ContextPreambleProjectState | None = None
    git_repo = getattr(conv, "git_repository_url", None) if conv is not None else None
    git_branch = getattr(conv, "git_branch", None) if conv is not None else None
    local_git_state, git_failure = _git_project_state(cwd)
    if git_failure is not None:
        component_failures["project_state"] = git_failure
    if git_repo or git_branch or local_git_state is not None:
        project = ContextPreambleProjectState(
            repo=str(git_repo) if git_repo else None,
            branch=(local_git_state.branch if local_git_state and local_git_state.branch else None)
            or (str(git_branch) if git_branch else None),
            recent_commits=list(local_git_state.recent_commits) if local_git_state else [],
        )

    assertion_guidance: list[ContextPreambleAssertionGuidance] = []
    if session_id:
        try:
            claims = await polylogue.list_assertion_claim_payloads(  # type: ignore[attr-defined]
                target_ref=f"session:{session_id}",
                statuses=("active",),
                context_inject=True,
                limit=20,
            )
            assertion_guidance = [_assertion_guidance_from_claim(claim) for claim in claims]
        except Exception as exc:
            component_failures["assertion_guidance"] = f"{type(exc).__name__}: {exc}"
            logger.warning("context preamble: assertion guidance lookup failed for %s: %s", session_id, exc)

    guidance = ContextPreambleGuidance(assertions=assertion_guidance) if assertion_guidance else None

    # Each section crosses the admission boundary as its own candidate, ranked
    # by how little of it a continuation can be reconstructed without. Under an
    # unbounded budget every segment is admitted and the output is byte-for-byte
    # what it always was; under a finite one the scheduler sheds from the
    # cheapest-to-replace end and every shed is recorded below.
    execution_context = _preamble_execution_context(
        session_id=session_id,
        boundary=boundary,
        repo_path=repo_path,
        cwd=cwd,
        related_limit=related_limit,
        session=conv,
    )
    segments = _preamble_segments(
        source_tool_calls=source_tool_calls or {},
        lineage=lineage,
        related=related,
        project=project,
        guidance=guidance,
    )
    source_name = f"context-{boundary}"
    ref_prefix = f"context-preamble:{boundary}:{session_id or 'anonymous'}"

    # Candidates are emitted in a deterministic order that is deliberately NOT
    # the protection order: the scheduler's tie-break is construction order, so
    # ordering them by name keeps ``ordinal_score`` the only thing that decides
    # what survives a squeeze. The protection order is then a declaration a
    # reviewer can read, not an accident of how this function was written.
    by_ref = {f"{ref_prefix}:{segment.name}": segment for segment in sorted(segments, key=lambda s: s.name)}
    full_content = {ref: _canonical(_segment_payload(seg.full)) for ref, seg in by_ref.items()}
    reduced_content = {
        ref: _canonical(_segment_payload(seg.reduced)) for ref, seg in by_ref.items() if seg.reduced is not None
    }

    def _degrade(item: ContextItem) -> ContextItem | None:
        content = reduced_content.get(item.ref)
        if content is None:
            return None
        return replace(item, content=content, token_cost=estimate_tokens(content))

    class _PreambleSource:
        name = source_name

        def candidates(self, *, moment: str, target_session: str | None) -> tuple[ContextItem, ...]:
            del moment
            return tuple(
                ContextItem(
                    ref=ref,
                    content=full_content[ref],
                    token_cost=estimate_tokens(full_content[ref]),
                    ordinal_score=segment.rank,
                    source=self.name,
                    trust_class="quoted",
                    material_class="evidence",
                    target_session=target_session,
                    degrade=_degrade if ref in reduced_content else None,
                )
                for ref, segment in by_ref.items()
            )

    # An unbounded caller gets exactly the compiled cost, which admits
    # everything; the unsheddable frame (version, injection time, and this
    # very receipt) is deliberately outside the budget, since shedding the
    # receipt would be the one omission nothing could record.
    default_budget = max(sum(estimate_tokens(content) for content in full_content.values()), 1)
    assembly = schedule_context(
        (_PreambleSource(),),
        moment=boundary,
        target_session=session_id,
        execution_context=execution_context,
        token_budget=token_budget if token_budget is not None else default_budget,
    )
    _record_preamble_ledger(polylogue, assembly)

    admitted_content = {item.ref: item.content for item in assembly.quoted_evidence}
    values: dict[str, object] = {}
    for ref, segment in by_ref.items():
        content = admitted_content.get(ref)
        if content is None:
            continue
        values[segment.name] = segment.reduced if content == reduced_content.get(ref) else segment.full

    for row in assembly.ledger:
        shed_segment = by_ref.get(row.item_ref)
        if shed_segment is None:
            continue
        key = f"{_BUDGET_FAILURE_PREFIX}{shed_segment.name}"
        if row.decision == "dropped" and row.disclosure_verdict == "budget":
            # ``row.token_cost`` is the cost of whatever last crossed the
            # boundary, which for a degraded-then-dropped candidate is the
            # reduced form. Report the section's own cost, and say when even
            # the reduced form did not fit.
            cost = estimate_tokens(full_content[row.item_ref])
            reduced = reduced_content.get(row.item_ref)
            tail = f", and {estimate_tokens(reduced)} even reduced" if reduced is not None else ""
            component_failures[key] = (
                f"omitted by the context token budget: the section costs {cost} tokens{tail}, "
                f"with {row.budget_before} remaining"
            )
        elif row.decision == "dropped":
            component_failures[key] = f"rejected at the context admission boundary: {row.authority_reason}"
        elif row.decision == "degraded":
            component_failures[key] = f"reduced by the context token budget: {shed_segment.reduced_detail}"

    return ContextPreamble(
        preamble_version="1.0",
        injected_at=datetime.now(timezone.utc).isoformat(),
        source_tool_calls=cast("dict[str, str]", values.get("source_tool_calls", {})),
        session_lineage=cast("ContextPreambleLineage | None", values.get("session_lineage")),
        recent_related_sessions=cast("list[ContextPreambleSession]", values.get("recent_related_sessions", [])),
        open_issues=[],
        project_state=cast("ContextPreambleProjectState | None", values.get("project_state")),
        guidance=cast("ContextPreambleGuidance | None", values.get("guidance")),
        component_failures=component_failures,
    )


# Assertion rows have no authenticated ContextSource registration yet (37t.11),
# so their arbitrary prose cannot enter this preamble as an operator directive.
_ASSERTION_GUIDANCE_SOURCE_AUTHORITY: ContextTrustClass = "quoted"


def _assertion_guidance_from_claim(claim: AssertionClaimPayload) -> ContextPreambleAssertionGuidance:
    """Render assertion prose according to its provenance-derived authority."""

    trust_class = derive_assertion_context_trust(
        author_kind=getattr(claim, "author_kind", None),
        author_ref=getattr(claim, "author_ref", None),
        status=getattr(claim, "status", None),
        context_policy=getattr(claim, "context_policy", None),
        source_authority=_ASSERTION_GUIDANCE_SOURCE_AUTHORITY,
    )
    text = getattr(claim, "body_text", None) or "(empty assertion)"
    if trust_class == "operator":
        return ContextPreambleAssertionGuidance(
            kind=claim.kind.value,
            trust_class=trust_class,
            operator_instruction=text,
            target_ref=claim.target_ref,
            scope_ref=claim.scope_ref,
            evidence_refs=list(claim.evidence_refs),
        )
    return ContextPreambleAssertionGuidance(
        kind=claim.kind.value,
        trust_class=trust_class,
        quoted_evidence=ContextPreambleQuotedEvidence(text=text),
        target_ref=claim.target_ref,
        scope_ref=claim.scope_ref,
        evidence_refs=list(claim.evidence_refs),
    )


def compose_context_preamble(env: AppEnv, *, session_id: str, related_limit: int = 5) -> str:
    """Compose a context preamble JSON document for a seed session (#1494)."""
    from polylogue.api.sync.bridge import run_coroutine_sync

    preamble = run_coroutine_sync(
        build_context_preamble_payload(
            env.polylogue,
            session_id=session_id,
            related_limit=related_limit,
            source_tool_calls={"compose_context_preamble": "polylogue-cli"},
        )
    )
    if preamble is None:
        env.ui.error(f"Session not found: {session_id}")
        raise SystemExit(1)
    return json.dumps(preamble.model_dump(mode="json", exclude_none=True), indent=2, default=str)


def _preamble_lineage(graph: CompactLineageGraph) -> ContextPreambleLineage:
    """Project the compact lineage graph into the preamble's lineage section."""

    nodes = graph.nodes
    edges = graph.edges
    parent_id = next(
        (
            str(edge.parent_id)
            for edge in edges
            if edge.role is LineageEdgeRole.SEED_PARENT and edge.parent_id is not None
        ),
        None,
    )
    return ContextPreambleLineage(
        logical_session_root=str(graph.root_id),
        parent_session_id=parent_id,
        sibling_session_ids=[str(node.session_id) for node in nodes if node.role is LineageNodeRole.SIBLING],
        continuation_chain_depth=sum(1 for node in nodes if node.role is LineageNodeRole.ANCESTOR),
    )
