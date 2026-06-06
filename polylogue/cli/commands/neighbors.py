"""Neighboring-session discovery command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from polylogue.archive.session.neighbor_candidates import SessionNeighborCandidate

from polylogue.cli.shared.helper_support import fail
from polylogue.cli.shared.machine_errors import emit_success
from polylogue.cli.shared.types import AppEnv
from polylogue.core.enums import Origin
from polylogue.core.sources import provider_from_origin
from polylogue.surfaces.payloads import SessionNeighborCandidatePayload, model_json_document


def _score_label(score: float) -> str:
    return f"{score:.2f}".rstrip("0").rstrip(".")


def _candidate_heading(candidate: SessionNeighborCandidate) -> str:
    summary = candidate.summary
    date = f" {summary.display_date.isoformat()}" if summary.display_date else ""
    return (
        f"{candidate.rank}. {candidate.session_id} "
        f"[{summary.origin.value}] {summary.display_title}{date} "
        f"(score {_score_label(candidate.score)})"
    )


def _render_plain(candidates: list[SessionNeighborCandidate]) -> None:
    if not candidates:
        click.echo("No neighboring candidates found.")
        return

    click.echo(f"Neighbor candidates ({len(candidates)}):")
    for candidate in candidates:
        click.echo(_candidate_heading(candidate))
        for reason in candidate.reasons:
            evidence = f" ({reason.evidence})" if reason.evidence else ""
            click.echo(f"   - {reason.kind}: {reason.detail}{evidence}")


@click.command("neighbors")
@click.option("--id", "session_id", default=None, help="Known session id or prefix")
@click.option("--query", "-q", default=None, help="Fuzzy query to seed candidate discovery")
@click.option("--origin", "-o", default=None, help="Limit candidate scope to one origin")
@click.option("--limit", "-l", "-n", type=int, default=10, show_default=True, help="Maximum candidates")
@click.option(
    "--window-hours",
    type=int,
    default=24,
    show_default=True,
    help="Neighboring time window around --id",
)
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None, help="Output format")
@click.pass_obj
def neighbors_command(
    env: AppEnv,
    session_id: str | None,
    query: str | None,
    origin: str | None,
    limit: int,
    window_hours: int,
    output_format: str | None,
) -> None:
    """Show explainable neighboring or near-duplicate candidates.

    When neither --id nor --query is given, falls back to resolving a
    session via root filters (``--latest``, ``--origin``, etc.) — #1642.
    """
    import click as _click

    from polylogue.api.sync.bridge import run_coroutine_sync
    from polylogue.cli.shared.insight_command_contracts import find_root_params
    from polylogue.cli.shared.latest_resolver import resolve_session_id_from_root_params

    if not session_id and not (query and query.strip()):
        ctx_obj = _click.get_current_context()
        resolved = resolve_session_id_from_root_params(dict(find_root_params(ctx_obj)))
        if resolved:
            session_id = resolved
        else:
            fail("neighbors", "provide --id, --query, or a root filter like --latest")

    try:
        provider = provider_from_origin(Origin(origin)).value if origin is not None else None
        candidates = run_coroutine_sync(
            env.polylogue.neighbor_candidates(
                session_id=session_id,
                query=query,
                provider=provider,
                limit=max(1, limit),
                window_hours=max(1, window_hours),
            )
        )
    except Exception as exc:
        from polylogue.archive.session.neighbor_candidates import NeighborDiscoveryError

        if not isinstance(exc, NeighborDiscoveryError):
            raise
        fail("neighbors", str(exc))

    if output_format == "json":
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

    _render_plain(candidates)


__all__ = ["neighbors_command"]
