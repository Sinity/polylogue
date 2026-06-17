"""Inspect structured query-completion metadata."""

from __future__ import annotations

import click

from polylogue.archive.query.completions import (
    QUERY_COMPLETION_KINDS,
    QueryCompletionCandidate,
    QueryCompletionError,
    query_completion_candidates,
    query_completion_payload,
)
from polylogue.cli.shared.machine_errors import emit_success

CompletionKind = str


def _render_plain(candidates: list[QueryCompletionCandidate]) -> str:
    if not candidates:
        return "No candidates."
    lines: list[str] = []
    for candidate in candidates:
        danger = " DANGER" if candidate.danger else ""
        stale = " stale" if candidate.stale else ""
        suffix = f"{danger}{stale}".strip()
        suffix = f" [{suffix}]" if suffix else ""
        lines.append(f"{candidate.insert:<24} {candidate.kind:<24} {candidate.source}{suffix}")
        if candidate.description:
            lines.append(f"    {candidate.description}")
    return "\n".join(lines)


@click.command("query-completions")
@click.option("--kind", type=click.Choice(QUERY_COMPLETION_KINDS), required=True, help="Candidate kind to inspect.")
@click.option("--incomplete", default="", show_default=True, help="Current incomplete token.")
@click.option("--unit", help="Structural unit for structural-field completion.")
@click.option("--field", help="Field name for count/date operator completion.")
@click.option(
    "--format", "-f", "output_format", type=click.Choice(["plain", "json"]), default="plain", show_default=True
)
def query_completions_command(
    kind: CompletionKind,
    incomplete: str,
    unit: str | None,
    field: str | None,
    output_format: str,
) -> None:
    """List structured query-completion candidates."""

    try:
        candidates = query_completion_candidates(kind, incomplete=incomplete, unit=unit, field=field)
        payload = query_completion_payload(kind, incomplete=incomplete, unit=unit, field=field)
    except QueryCompletionError as exc:
        raise click.UsageError(str(exc)) from exc
    if output_format == "json":
        emit_success({"query_completions": payload})
        return
    click.echo(_render_plain(candidates))


__all__ = ["query_completions_command"]
