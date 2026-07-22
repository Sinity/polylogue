"""Raw-identity repair commands: missing cursors, quarantined/duplicate/mismatched raws."""

from __future__ import annotations

import json

import click

from polylogue.cli.shared.types import AppEnv
from polylogue.product import raw_authority


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
    config = env.config
    try:
        if plan_ids:
            if not confirmed:
                raise click.ClickException("refusing raw-authority application without --yes")
            if preview_census is None:
                raise click.ClickException("--apply-plan requires --preview-census")
            payload = raw_authority.apply_frontier(
                config,
                preview_census_id=preview_census,
                selected_plan_ids=plan_ids,
            ).to_dict()
        else:
            if preview_census is not None or confirmed:
                raise click.ClickException("apply options require at least one --apply-plan")
            payload = raw_authority.inspect_frontier(config).to_dict()
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
    try:
        payload = raw_authority.read_census(env.config.archive_root, query_handle, limit=limit, offset=offset)
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
    try:
        payload = raw_authority.read_detail(
            env.config.archive_root,
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


@click.command("raw-authority-blockers")
@click.option("--limit", type=click.IntRange(1, 500), default=100, show_default=True)
@click.option("--offset", type=click.IntRange(min=0), default=0, show_default=True)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
)
@click.pass_obj
def raw_authority_blockers_command(env: AppEnv, limit: int, offset: int, output_format: str) -> None:
    """List unresolved raw-authority blockers (read-only operator discovery surface).

    Distinguishes ``stale_plan`` blockers (replan against current evidence),
    ``frontier_judgment`` blockers (require an accepted judgment assertion +
    disposition), and ``frontier_obligation`` blockers (other frontier
    obligation states -- missing bytes, unresolved provenance, corrupt --
    that resolve without a judgment assertion) so an operator can find a
    ``--blocker-id`` for ``raw-authority-blocker-resolve`` without
    page-walking ``raw-authority-census``/``raw-authority-detail`` or writing
    an ad hoc script against the live archive.

    Bounded to ``--limit`` (1-500) per call. If ``truncated`` is true in the
    output, pass ``--offset <next_offset>`` to read the next page.
    """
    try:
        payload = raw_authority.list_blockers(env.config.archive_root, limit=limit, offset=offset)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    blockers = payload["blockers"]
    if not isinstance(blockers, list) or not blockers:
        click.echo("No unresolved raw-authority blockers.")
        return
    for item in blockers:
        if not isinstance(item, dict):
            continue
        click.echo(f"{item['blocker_id']}  kind={item['kind']}  plan={item['plan_id']}  census={item['census_id']}")
        click.echo(f"  reason: {item['reason']}")
    click.echo(f"({payload['returned_count']} of {payload['total_count']} unresolved)")
    if payload.get("truncated"):
        click.echo(f"Truncated: pass --offset {payload['next_offset']} for the next page.")


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
    """Resolve one stale-plan blocker after replanning current evidence.

    Routed through ``OperationExecutor``/``BlockerResolveActuator`` (t46.9
    phase 3): PREPARE previews the exact blocker target, EXECUTE requires a
    confirm-flag-strength authorization bound to that plan's hash, and a
    fresh PREPARE immediately before EXECUTE refuses (``PlanStaleError``) if
    the blocker was concurrently resolved between preview and confirm.
    """
    if not confirmed:
        raise click.ClickException("refusing to resolve a durable blocker without --yes")
    from polylogue.operations.mutation_actuators import BlockerResolveActuator, BlockerResolveArgs
    from polylogue.operations.mutation_transaction import (
        MutationTransactionError,
        OperationExecutor,
    )

    actuator = BlockerResolveActuator()
    executor = OperationExecutor()
    args = BlockerResolveArgs(
        archive_root=env.config.archive_root,
        blocker_id=blocker_id,
        resolution=reason,
        assertion_id=assertion_id,
        judgment_disposition=judgment_disposition,
    )
    try:
        plan = executor.prepare(actuator, args)
        authorization = executor.authorize(
            actuator,
            plan,
            actor="cli",
            role="write",
            capability="raw_authority.resolve_blocker",
            confirmation_strength="confirm_flag",
        )
        result = executor.execute(actuator, plan, authorization, args)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError, MutationTransactionError) as exc:
        raise click.ClickException(str(exc)) from exc
    if result.status != "applied":
        raise click.ClickException(f"blocker {blocker_id!r} not found or already resolved")
    receipt = dict(result.domain_receipt)
    if output_format == "json":
        click.echo(json.dumps(receipt, indent=2, sort_keys=True))
        return
    click.echo(f"Resolved {blocker_id}")
    current_plan = receipt.get("current_plan")
    if isinstance(current_plan, dict):
        click.echo(f"Current plan: {current_plan.get('plan_id', 'unknown')}")
