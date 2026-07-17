"""Raw-identity repair commands: missing cursors, quarantined/duplicate/mismatched raws."""

from __future__ import annotations

import json

import click

from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.paths import archive_root, render_root


@click.command("raw-authority-frontier")
@click.option("--apply-plan", "plan_ids", multiple=True, help="Exact immutable plan id; repeatable.")
@click.option("--preview-census", default=None, help="Completed dry-run census authorizing --apply-plan.")
@click.option("--yes", "confirmed", is_flag=True, help="Confirm the selected break-glass application.")
@click.option(
    "--output-format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
)
@click.pass_obj
def raw_authority_frontier_command(
    env: AppEnv,
    plan_ids: tuple[str, ...],
    preview_census: str | None,
    confirmed: bool,
    output_format: str,
) -> None:
    """Inspect the complete frontier or apply exact plans as break-glass work."""
    del env
    from polylogue.storage.raw_reconciler import (
        apply_raw_authority_frontier,
        inspect_raw_authority_frontier,
    )

    root = archive_root()
    config = Config(archive_root=root, render_root=render_root(), sources=[])
    try:
        if plan_ids:
            if not confirmed:
                raise click.ClickException("refusing raw-authority application without --yes")
            if preview_census is None:
                raise click.ClickException("--apply-plan requires --preview-census")
            payload = apply_raw_authority_frontier(
                config,
                preview_census_id=preview_census,
                selected_plan_ids=plan_ids,
            ).to_dict()
        else:
            if preview_census is not None or confirmed:
                raise click.ClickException("apply options require at least one --apply-plan")
            payload = inspect_raw_authority_frontier(config).to_dict()
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        if isinstance(exc, click.ClickException):
            raise
        raise click.ClickException(str(exc)) from exc
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    if plan_ids:
        click.echo(
            f"Applied {payload['executed_plan_count']}/{payload['selected_plan_count']} plan(s); "
            f"retryable={payload['retryable_plan_count']} census={payload['census_id']}"
        )
        return
    click.echo(
        f"Frontier {payload['census_id']}: accepted={payload['accepted_head_count']} "
        f"plans={payload['plan_count']} executable={payload['executable_plan_count']}"
    )
    click.echo(f"States: {json.dumps(payload['state_counts'], sort_keys=True)}")
    click.echo(f"Details: {payload['query_handle']}")


@click.command("raw-authority-census")
@click.argument("query_handle")
@click.option("--limit", type=click.IntRange(1, 500), default=100, show_default=True)
@click.option("--offset", type=click.IntRange(min=0), default=None)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
)
@click.pass_obj
def raw_authority_census_command(
    env: AppEnv,
    query_handle: str,
    limit: int,
    offset: int | None,
    output_format: str,
) -> None:
    """Read a bounded page from a durable raw-authority census ledger."""
    del env
    from polylogue.storage.raw_authority import read_raw_authority_census

    try:
        payload = read_raw_authority_census(archive_root(), query_handle, limit=limit, offset=offset)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    census = payload["census"]
    if not isinstance(census, dict):
        raise click.ClickException("invalid raw authority census payload")
    click.echo(
        f"Census {census['census_id']}: plans={census['plan_count']} "
        f"executable={census['executable_plan_count']} residual={census['residual_plan_count']} "
        f"fixed_point={str(census['fixed_point']).lower()}"
    )
    for item in payload["plans"] if isinstance(payload["plans"], list) else []:
        if isinstance(item, dict):
            plan = item.get("plan")
            plan_id = plan.get("plan_id") if isinstance(plan, dict) else "unknown"
            click.echo(f"  {item.get('ordinal')} {item.get('outcome_status')} {plan_id}")
    next_handle = payload.get("next_query_handle")
    if next_handle is not None:
        click.echo(f"Next: {next_handle}")


@click.command("raw-authority-detail")
@click.argument("query_handle")
@click.option("--chunk-chars", type=click.IntRange(256, 65_536), default=16_384, show_default=True)
@click.option("--offset", type=click.IntRange(min=0), default=None)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
)
@click.pass_obj
def raw_authority_detail_command(
    env: AppEnv,
    query_handle: str,
    chunk_chars: int,
    offset: int | None,
    output_format: str,
) -> None:
    """Read one bounded chunk of a complete census or plan document."""
    del env
    from polylogue.storage.raw_authority import read_raw_authority_detail

    try:
        payload = read_raw_authority_detail(
            archive_root(),
            query_handle,
            chunk_chars=chunk_chars,
            offset=offset,
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    click.echo(str(payload["chunk"]), nl=False)
    next_handle = payload.get("next_query_handle")
    if next_handle is not None:
        click.echo(f"\nNext: {next_handle}")


@click.command("raw-authority-blocker-resolve")
@click.option("--blocker-id", required=True, help="Exact unresolved durable blocker identifier.")
@click.option("--reason", required=True, help="Operator rationale recorded in the immutable resolution receipt.")
@click.option(
    "--assertion-id",
    default=None,
    help="Accepted judgment assertion required by a conflicting-authority blocker.",
)
@click.option(
    "--judgment-disposition",
    type=click.Choice(["retain_canonical_authority"]),
    default=None,
    help="Typed authority choice required when resolving a conflicting frontier.",
)
@click.option("--yes", "confirmed", is_flag=True, help="Confirm resolving this blocker against current evidence.")
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
)
@click.pass_obj
def raw_authority_blocker_resolve_command(
    env: AppEnv,
    blocker_id: str,
    reason: str,
    assertion_id: str | None,
    judgment_disposition: str | None,
    confirmed: bool,
    output_format: str,
) -> None:
    """Resolve one stale-plan blocker after replanning current evidence."""
    del env
    if not confirmed:
        raise click.ClickException("refusing to resolve a durable blocker without --yes")
    from polylogue.storage.raw_authority import resolve_raw_authority_blocker

    try:
        receipt = resolve_raw_authority_blocker(
            archive_root(),
            blocker_id,
            resolution=reason,
            assertion_id=assertion_id,
            judgment_disposition=judgment_disposition,
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if output_format == "json":
        click.echo(json.dumps(receipt, indent=2, sort_keys=True))
        return
    click.echo(f"Resolved {blocker_id}")
    current_plan = receipt.get("current_plan")
    if isinstance(current_plan, dict):
        click.echo(f"Current plan: {current_plan.get('plan_id', 'unknown')}")
