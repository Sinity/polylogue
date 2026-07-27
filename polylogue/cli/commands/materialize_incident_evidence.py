"""Materialize a work-evidence graph from the archive's own incident evidence."""

from __future__ import annotations

import json
from datetime import datetime

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.archive.query.spec import QuerySpecError, parse_query_date
from polylogue.cli.shared.types import AppEnv


def _parse_time_bound(field: str, value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return parse_query_date(field, value)
    except QuerySpecError as exc:
        raise click.UsageError(str(exc)) from exc


def _render_plain(payload: dict[str, object]) -> None:
    click.echo(f"Incident work-evidence graph: {payload['graph_id']}")
    click.echo(f"Mode:                         {'apply' if payload['applied'] else 'dry-run'}")
    click.echo(f"Sessions selected:            {payload['session_count']}")
    click.echo(f"Runs:                         {payload['run_count']}")
    click.echo(f"Session segments:             {payload['session_segment_count']}")
    click.echo(f"Self-reported claims:         {payload['claim_count']}")
    click.echo(f"Mentioned (unresolved) effects: {payload['mentioned_effect_count']}")
    click.echo(f"Edges:                        {payload['edge_count']}")
    if not payload["applied"]:
        click.echo("(dry run -- pass --yes to persist the materialized graph)")


@click.command("materialize-incident-evidence")
@click.option(
    "--session-id",
    "session_ids",
    multiple=True,
    help="Explicit session id to include (repeatable). Combine with --repo/--since/--until to widen selection.",
)
@click.option("--repo", "repo_names", multiple=True, help="Restrict selection to session(s) whose repo matches.")
@click.option("--since", default=None, help="Lower session time bound (ISO or relative date).")
@click.option("--until", default=None, help="Upper session time bound (ISO or relative date).")
@click.option("--contains", "contains_text", default=None, help="Free-text clue the session content must contain.")
@click.option(
    "--selection-limit",
    type=int,
    default=200,
    show_default=True,
    help="Maximum sessions the --repo/--since/--until/--contains selection may match.",
)
@click.option("--graph-id", required=True, help="Work-evidence graph id to write, e.g. incident:2026-07-15-workflow.")
@click.option(
    "--yes",
    "apply",
    is_flag=True,
    help="Persist the materialized graph. Without this flag the command is a dry run.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
@click.pass_obj
def materialize_incident_evidence_command(
    env: AppEnv,
    session_ids: tuple[str, ...],
    repo_names: tuple[str, ...],
    since: str | None,
    until: str | None,
    contains_text: str | None,
    selection_limit: int,
    graph_id: str,
    apply: bool,
    output_format: str,
) -> None:
    """Build a work-evidence graph directly from real session/run/action evidence.

    Unlike ``reconcile-work-effects`` (which reconciles an *existing* graph
    against independently observed git/GitHub/Beads effects), this command
    builds the graph itself -- from the archive's own record of what actually
    ran: one node per session run, one per-run transcript segment, one claim
    per subagent's own self-reported final result, and one unresolved
    "mentioned" effect node per commit/PR/issue an in-session tool call cited.
    Select sessions explicitly with one or more --session-id, or widen the
    selection with --repo/--since/--until/--contains (the same sparse clues a
    real incident investigation starts from). Without --yes this only reports
    what would be built.
    """
    from polylogue.archive.filter.filters import SessionFilter
    from polylogue.operations.incident_evidence_materialization import (
        IncidentEvidenceMaterializationResult,
        NoIncidentSessionsFoundError,
        materialize_incident_work_evidence,
    )
    from polylogue.paths import archive_root

    since_bound = _parse_time_bound("since", since)
    until_bound = _parse_time_bound("until", until)

    resolved_session_ids = list(session_ids)
    if repo_names or since_bound is not None or until_bound is not None or contains_text is not None:

        async def _select() -> list[str]:
            session_filter = SessionFilter(archive_root=archive_root())
            if repo_names:
                session_filter = session_filter.repo(*repo_names)
            if since_bound is not None:
                session_filter = session_filter.since(since_bound)
            if until_bound is not None:
                session_filter = session_filter.until(until_bound)
            if contains_text is not None:
                session_filter = session_filter.contains(contains_text)
            session_filter = session_filter.limit(selection_limit)
            summaries = await session_filter.list_summaries()
            return [str(summary.id) for summary in summaries]

        resolved_session_ids.extend(run_coroutine_sync(_select()))

    async def _run() -> IncidentEvidenceMaterializationResult:
        return await materialize_incident_work_evidence(
            env.repository,
            session_ids=resolved_session_ids,
            graph_id=graph_id,
            apply=apply,
        )

    try:
        result = run_coroutine_sync(_run())
    except NoIncidentSessionsFoundError as exc:
        raise click.UsageError(str(exc)) from exc

    payload: dict[str, object] = {
        "mode": "materialize_incident_evidence",
        **result.summary.as_dict(),
        "applied": result.applied,
    }
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    _render_plain(payload)


__all__ = ["materialize_incident_evidence_command"]
