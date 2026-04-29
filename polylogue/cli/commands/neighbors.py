"""Neighboring-conversation discovery command."""

from __future__ import annotations

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.cli.shared.helper_support import fail
from polylogue.cli.shared.machine_errors import emit_success
from polylogue.cli.shared.types import AppEnv
from polylogue.lib.conversation.neighbor_candidates import ConversationNeighborCandidate, NeighborDiscoveryError
from polylogue.surfaces.payloads import ConversationNeighborCandidatePayload, model_json_document


def _score_label(score: float) -> str:
    return f"{score:.2f}".rstrip("0").rstrip(".")


def _candidate_heading(candidate: ConversationNeighborCandidate) -> str:
    summary = candidate.summary
    date = f" {summary.display_date.isoformat()}" if summary.display_date else ""
    return (
        f"{candidate.rank}. {candidate.conversation_id} "
        f"[{summary.provider}] {summary.display_title}{date} "
        f"(score {_score_label(candidate.score)})"
    )


def _render_plain(candidates: list[ConversationNeighborCandidate]) -> None:
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
@click.option("--id", "conversation_id", default=None, help="Known conversation id or prefix")
@click.option("--query", "-q", default=None, help="Fuzzy query to seed candidate discovery")
@click.option("--provider", "-p", default=None, help="Limit candidate scope to one provider")
@click.option("--limit", "-n", type=int, default=10, show_default=True, help="Maximum candidates")
@click.option(
    "--window-hours",
    type=int,
    default=24,
    show_default=True,
    help="Neighboring time window around --id",
)
@click.option("--json", "json_mode", is_flag=True, help="Output as JSON")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None, help="Output format")
@click.pass_obj
def neighbors_command(
    env: AppEnv,
    conversation_id: str | None,
    query: str | None,
    provider: str | None,
    limit: int,
    window_hours: int,
    json_mode: bool,
    output_format: str | None,
) -> None:
    """Show explainable neighboring or near-duplicate candidates."""
    if not conversation_id and not (query and query.strip()):
        fail("neighbors", "provide --id or --query")

    try:
        candidates = run_coroutine_sync(
            env.operations.neighbor_candidates(
                conversation_id=conversation_id,
                query=query,
                provider=provider,
                limit=max(1, limit),
                window_hours=max(1, window_hours),
            )
        )
    except NeighborDiscoveryError as exc:
        fail("neighbors", str(exc))

    if json_mode or output_format == "json":
        emit_success(
            {
                "neighbors": [
                    model_json_document(
                        ConversationNeighborCandidatePayload.from_candidate(candidate),
                        exclude_none=True,
                    )
                    for candidate in candidates
                ]
            }
        )
        return

    _render_plain(candidates)


__all__ = ["neighbors_command"]
