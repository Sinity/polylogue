"""Inspect structured query-completion metadata."""

from __future__ import annotations

from collections.abc import Callable

import click

from polylogue.cli.shared.machine_errors import emit_success
from polylogue.cli.shell_completion_values import (
    QueryCompletionCandidate,
    query_action_candidates,
    query_count_operator_candidates,
    query_date_operator_candidates,
    query_field_candidates,
    query_structural_field_candidates,
    query_structural_unit_candidates,
)

CompletionKind = str
CandidateProvider = Callable[[], list[QueryCompletionCandidate]]

_COMPLETION_KINDS = (
    "field",
    "structural-unit",
    "structural-field",
    "count-operator",
    "date-operator",
    "action",
)


def _candidate_provider(
    kind: CompletionKind, incomplete: str, unit: str | None, field: str | None
) -> CandidateProvider:
    if kind == "field":
        return lambda: query_field_candidates(incomplete)
    if kind == "structural-unit":
        return lambda: query_structural_unit_candidates(incomplete)
    if kind == "structural-field":
        if unit is None:
            raise click.UsageError("--unit is required for --kind structural-field")
        return lambda: query_structural_field_candidates(unit, incomplete)
    if kind == "count-operator":
        if field is None:
            raise click.UsageError("--field is required for --kind count-operator")
        return lambda: query_count_operator_candidates(field, incomplete)
    if kind == "date-operator":
        if field is None:
            raise click.UsageError("--field is required for --kind date-operator")
        return lambda: query_date_operator_candidates(field, incomplete)
    if kind == "action":
        return lambda: query_action_candidates(incomplete)
    raise click.UsageError(f"Unsupported completion kind: {kind}")


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
@click.option("--kind", type=click.Choice(_COMPLETION_KINDS), required=True, help="Candidate kind to inspect.")
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

    candidates = _candidate_provider(kind, incomplete, unit, field)()
    payload = {
        "kind": kind,
        "incomplete": incomplete,
        "unit": unit,
        "field": field,
        "candidates": [candidate.to_payload() for candidate in candidates],
    }
    if output_format == "json":
        emit_success({"query_completions": payload})
        return
    click.echo(_render_plain(candidates))


__all__ = ["query_completions_command"]
