"""Neighbor discovery read-view handler."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from polylogue.cli.operation_kernel import OperationKernelError, OperationRequest
from polylogue.cli.read_dispatch import daemon_route_disabled, dispatch_read
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


def _neighbor_candidate_heading(candidate: Mapping[str, object]) -> str:
    summary = candidate.get("session")
    if not isinstance(summary, Mapping):
        raise ValueError("neighbor result omitted its session summary")
    stamp = summary.get("updated_at") or summary.get("created_at")
    date = f" {str(stamp)[:10]}" if stamp else ""
    score = candidate.get("score")
    if isinstance(score, bool) or not isinstance(score, int | float):
        raise ValueError("neighbor result omitted its numeric score")
    return (
        f"{candidate['rank']}. {summary['id']} "
        f"[{summary['origin']}] {summary['title']}{date} "
        f"(score {_neighbor_score_label(float(score))})"
    )


def _render_neighbors_plain(candidates: list[Mapping[str, object]]) -> str:
    if not candidates:
        return "No neighboring candidates found.\n"
    lines = [f"Neighbor candidates ({len(candidates)}):"]
    for candidate in candidates:
        lines.append(_neighbor_candidate_heading(candidate))
        reasons = candidate.get("reasons")
        if not isinstance(reasons, list):
            raise ValueError("neighbor result omitted its reasons")
        for reason in reasons:
            if not isinstance(reason, Mapping):
                continue
            evidence = f" ({reason['evidence']})" if reason.get("evidence") else ""
            lines.append(f"   - {reason['kind']}: {reason['detail']}{evidence}")
    return "\n".join(lines) + "\n"


def run_read_neighbors(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render explainable neighbor/near-duplicate candidates for a seed session."""

    from polylogue.cli.shared.helper_support import fail
    from polylogue.cli.shared.machine_errors import emit_success

    query_seed = " ".join(request.query_terms).strip() or None
    if not invocation.session_id and not query_seed:
        fail("read", "read --view neighbors requires a seed (use --id, id:prefix, --latest, or a query).")
    options = cast(ReadViewNeighborOptions, invocation.options or ReadViewNeighborOptions())

    origin = request.params.get("origin")

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
        result, _ = dispatch_read(
            env.config,
            OperationRequest(
                "read.neighbors",
                {
                    "session_id": invocation.session_id,
                    "query": query_seed,
                    "origin": str(origin) if origin is not None else None,
                    "limit": max(1, limit if limit is not None else 10),
                    "window_hours": max(1, window_hours),
                },
            ),
            daemon_disabled=daemon_route_disabled(flag=bool(request.params.get("no_daemon"))),
        )
    except OperationKernelError as exc:
        from polylogue.cli.render.outcome import exit_for_read_failure

        exit_for_read_failure(exc)
    payload = result.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("read.neighbors returned no payload")
    raw_candidates = payload.get("neighbors")
    if not isinstance(raw_candidates, list):
        raise ValueError("read.neighbors returned no candidates")
    candidates = [candidate for candidate in raw_candidates if isinstance(candidate, Mapping)]

    if invocation.output_format == "json":
        emit_success({"neighbors": candidates})
        return

    deliver_content(
        env, _render_neighbors_plain(candidates), destination=invocation.destination, out_path=invocation.out_path
    )


__all__ = ["NEIGHBOR_READ_VIEW_OPTION_NAMES", "build_neighbor_options", "run_read_neighbors"]
