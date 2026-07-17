"""Context preamble composition for the ``read --view context`` surface."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from polylogue.core.assertions import derive_assertion_context_trust
from polylogue.logging import get_logger
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
            topology = await polylogue.get_session_topology(session_id)  # type: ignore[attr-defined]
            if topology:
                lineage = ContextPreambleLineage(
                    logical_session_root=getattr(topology, "logical_session_id", None),
                    parent_session_id=getattr(topology, "parent_session_id", None),
                )
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
    if git_repo or git_branch:
        project = ContextPreambleProjectState(
            repo=str(git_repo) if git_repo else None,
            branch=str(git_branch) if git_branch else None,
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
    return ContextPreamble(
        preamble_version="1.0",
        injected_at=datetime.now(timezone.utc).isoformat(),
        source_tool_calls=source_tool_calls or {},
        session_lineage=lineage,
        recent_related_sessions=related,
        open_issues=[],
        project_state=project,
        guidance=guidance,
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
