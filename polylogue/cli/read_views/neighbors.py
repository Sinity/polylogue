"""Neighbor discovery read-view handler."""

from __future__ import annotations

from typing import cast

from polylogue.archive.session.neighbor_candidates import SessionNeighborCandidate
from polylogue.cli.read_views.base import ReadViewInvocation, ReadViewNeighborOptions, deliver_content
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv


def _neighbor_score_label(score: float) -> str:
    return f"{score:.2f}".rstrip("0").rstrip(".")


def _neighbor_candidate_heading(candidate: SessionNeighborCandidate) -> str:
    summary = candidate.summary
    date = f" {summary.display_date.isoformat()}" if summary.display_date else ""
    return (
        f"{candidate.rank}. {candidate.session_id} "
        f"[{summary.origin.value}] {summary.display_title}{date} "
        f"(score {_neighbor_score_label(candidate.score)})"
    )


def _render_neighbors_plain(candidates: list[SessionNeighborCandidate]) -> str:
    if not candidates:
        return "No neighboring candidates found.\n"
    lines = [f"Neighbor candidates ({len(candidates)}):"]
    for candidate in candidates:
        lines.append(_neighbor_candidate_heading(candidate))
        for reason in candidate.reasons:
            evidence = f" ({reason.evidence})" if reason.evidence else ""
            lines.append(f"   - {reason.kind}: {reason.detail}{evidence}")
    return "\n".join(lines) + "\n"


def run_read_neighbors(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render explainable neighbor/near-duplicate candidates for a seed session."""

    from polylogue.api.sync.bridge import run_coroutine_sync
    from polylogue.archive.session.neighbor_candidates import NeighborDiscoveryError
    from polylogue.cli.shared.helper_support import fail
    from polylogue.cli.shared.machine_errors import emit_success
    from polylogue.core.enums import Origin
    from polylogue.core.sources import provider_from_origin
    from polylogue.surfaces.payloads import SessionNeighborCandidatePayload, model_json_document

    query_seed = " ".join(request.query_terms).strip() or None
    if not invocation.session_id and not query_seed:
        fail("read", "read --view neighbors requires a seed (use --id, id:prefix, --latest, or a query).")
    options = cast(ReadViewNeighborOptions, invocation.options or ReadViewNeighborOptions())

    origin = request.params.get("origin")
    provider = provider_from_origin(Origin(str(origin))).value if origin else None

    try:
        candidates = run_coroutine_sync(
            env.polylogue.neighbor_candidates(
                session_id=invocation.session_id,
                query=query_seed,
                provider=provider,
                limit=max(1, options.limit if options.limit is not None else 10),
                window_hours=max(1, options.window_hours),
            )
        )
    except NeighborDiscoveryError as exc:
        fail("read", str(exc))

    if invocation.output_format == "json":
        emit_success(
            {
                "neighbors": [
                    model_json_document(
                        SessionNeighborCandidatePayload.from_candidate(candidate),
                        exclude_none=True,
                    )
                    for candidate in candidates
                ]
            }
        )
        return

    deliver_content(
        env, _render_neighbors_plain(candidates), destination=invocation.destination, out_path=invocation.out_path
    )


__all__ = ["run_read_neighbors"]
