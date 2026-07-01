"""Neighbor discovery read-view handler."""

from __future__ import annotations

from typing import cast

from polylogue.archive.session.neighbor_candidates import SessionNeighborCandidate
from polylogue.cli.read_view_registry import NEIGHBOR_READ_VIEW_OPTION_NAMES
from polylogue.cli.read_views.base import (
    ReadViewInvocation,
    ReadViewNeighborOptions,
    ReadViewOptionValues,
    deliver_content,
)
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv


def build_neighbor_options(values: ReadViewOptionValues) -> ReadViewNeighborOptions:
    """Build options owned by the neighbor-discovery read view."""

    return ReadViewNeighborOptions(
        limit=cast(int | None, values.get("limit")),
        window_hours=cast(int, values.get("window_hours", 24)),
    )


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

    projection = invocation.projection_spec.projection if invocation.projection_spec is not None else None
    limit = (
        projection.neighbor_limit if projection is not None and projection.neighbor_limit is not None else options.limit
    )
    window_hours = (
        projection.neighbor_window_hours
        if projection is not None and projection.neighbor_window_hours is not None
        else options.window_hours
    )
    try:
        candidates = run_coroutine_sync(
            env.polylogue.neighbor_candidates(
                session_id=invocation.session_id,
                query=query_seed,
                provider=provider,
                limit=max(1, limit if limit is not None else 10),
                window_hours=max(1, window_hours),
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


__all__ = ["NEIGHBOR_READ_VIEW_OPTION_NAMES", "build_neighbor_options", "run_read_neighbors"]
