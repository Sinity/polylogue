"""Reconcile a work-evidence graph's claims against observed git/Beads effects."""

from __future__ import annotations

import json
from pathlib import Path

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.archive.query.spec import QuerySpecError, parse_query_date


def _parse_time_bound(field: str, value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = parse_query_date(field, value)
    except QuerySpecError as exc:
        raise click.UsageError(str(exc)) from exc
    assert parsed is not None
    return int(parsed.timestamp() * 1000)


def _render_plain(summary_dict: dict[str, object]) -> None:
    click.echo(f"Work-evidence graph: {summary_dict['graph_id']}")
    click.echo(f"Mode:                {'apply' if summary_dict['applied'] else 'dry-run'}")
    click.echo(
        "Claims:              "
        f"{summary_dict['claims_total']} total, "
        f"{summary_dict['claims_evaluated']} evaluated, "
        f"{summary_dict['claims_unevaluated']} unevaluated"
    )
    effect_counts = summary_dict.get("effect_count_by_authority")
    if isinstance(effect_counts, dict) and effect_counts:
        click.echo("Effects:             " + ", ".join(f"{k}={v}" for k, v in sorted(effect_counts.items())))
    judgment_counts = summary_dict.get("judgment_count_by_evaluation")
    if isinstance(judgment_counts, dict) and judgment_counts:
        click.echo("Judgments:           " + ", ".join(f"{k}={v}" for k, v in sorted(judgment_counts.items())))
    failures = summary_dict.get("adapter_failures")
    if isinstance(failures, tuple | list):
        for failure in failures:
            if isinstance(failure, dict):
                click.echo(f"  adapter unavailable: {failure.get('authority')}: {failure.get('reason')}")
    if not summary_dict["applied"]:
        click.echo("(dry run -- pass --yes to persist the reconciled graph)")


@click.command("reconcile-work-effects")
@click.option(
    "--graph-id",
    required=True,
    help="Work-evidence graph id to reconcile, e.g. claude-workflow:<run-id>.",
)
@click.option(
    "--repo",
    "repo_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Local git checkout to read commit effects from (read-only).",
)
@click.option(
    "--beads-jsonl",
    "beads_jsonl",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Beads interaction ledger (.beads/interactions.jsonl) to read issue-state effects from.",
)
@click.option("--since", default=None, help="Lower effect time bound (ISO or relative date).")
@click.option("--until", default=None, help="Upper effect time bound (ISO or relative date).")
@click.option(
    "--yes",
    "apply",
    is_flag=True,
    help="Persist the reconciled graph. Without this flag the command is a dry run.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def reconcile_work_effects_command(
    graph_id: str,
    repo_path: Path,
    beads_jsonl: Path | None,
    since: str | None,
    until: str | None,
    apply: bool,
    output_format: str,
) -> None:
    """Attach observed git/Beads repository effects to a stored work-evidence graph.

    Reads commit history from --repo and, if given, issue-state changes from
    --beads-jsonl. Claims are only linked to an effect through an explicit
    shared work-item id in their text -- never through time or file
    proximity. Without --yes this only reports what would change.
    """
    from polylogue.insights.work_effects import BeadsIssueEffectAdapter, GitCommitEffectAdapter, RepositoryEffectAdapter
    from polylogue.operations.work_effect_reconciliation import (
        WorkEffectReconciliationSummary,
        WorkEvidenceGraphNotFoundError,
        reconcile_graph_repository_effects,
    )
    from polylogue.paths import active_index_db_path
    from polylogue.storage.repository import SessionRepository

    since_ms = _parse_time_bound("since", since)
    until_ms = _parse_time_bound("until", until)

    adapters: list[RepositoryEffectAdapter] = [GitCommitEffectAdapter(repo_path=repo_path)]
    if beads_jsonl is not None:
        adapters.append(BeadsIssueEffectAdapter(jsonl_path=beads_jsonl))

    async def _run() -> WorkEffectReconciliationSummary:
        async with SessionRepository(db_path=active_index_db_path()) as repository:
            return await reconcile_graph_repository_effects(
                repository,
                graph_id=graph_id,
                adapters=adapters,
                since_ms=since_ms,
                until_ms=until_ms,
                apply=apply,
            )

    try:
        summary = run_coroutine_sync(_run())
    except WorkEvidenceGraphNotFoundError as exc:
        raise click.UsageError(str(exc)) from exc

    payload = {"mode": "reconcile_work_effects", "mutates": apply, **summary.to_dict()}
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    _render_plain(payload)


__all__ = ["reconcile_work_effects_command"]
