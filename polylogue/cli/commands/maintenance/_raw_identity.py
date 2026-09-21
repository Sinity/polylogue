"""Raw-authority frontier commands: inspect the frontier and read or acknowledge its blockers."""

from __future__ import annotations

import json

import click

from polylogue.cli.shared.types import AppEnv
from polylogue.maintenance import raw_authority


def _submit(env: AppEnv, payload: dict[str, object]) -> dict[str, object]:
    from polylogue.cli.operation_kernel import OperationKernelError, configured_mutation_operation
    from polylogue.cli.shared.helpers import mutation_refusal

    operation = "mutation.raw-authority-blocker.resolve"
    try:
        return configured_mutation_operation(env.config, operation, payload)
    except OperationKernelError as exc:
        raise mutation_refusal(exc, operation) from exc


@click.command("raw-authority-frontier")
@click.option(
    "--output-format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
)
@click.pass_obj
def raw_authority_frontier_command(
    env: AppEnv,
    output_format: str,
) -> None:
    """Inspect and record the raw-authority frontier without applying plans."""
    try:
        payload = raw_authority.inspect_frontier(env.config).to_dict()
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        if isinstance(exc, click.ClickException):
            raise
        raise click.ClickException(str(exc)) from exc
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    click.echo(
        f"Frontier {payload['pass_id']}: accepted={payload['accepted_head_count']} obligations={payload['plan_count']}"
    )
    click.echo(f"States: {json.dumps(payload['state_counts'], sort_keys=True)}")
    click.echo("Blocking items are published as durable blockers; list them with raw-authority-blockers.")


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

    Reports each row's ``kind`` -- how the resolver reads its stored snapshot,
    not a different effect -- so an operator can find a ``--blocker-id`` for
    ``raw-authority-blocker-resolve`` without writing an ad hoc script against
    the live archive.

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
        click.echo(
            f"{item['blocker_id']}  kind={item['kind']}  plan={item['plan_id']}  observed_in={item['observed_pass_id']}"
        )
        click.echo(f"  reason: {item['reason']}")
    click.echo(f"({payload['returned_count']} of {payload['total_count']} unresolved)")
    if payload.get("truncated"):
        click.echo(f"Truncated: pass --offset {payload['next_offset']} for the next page.")


@click.command("raw-authority-blocker-resolve")
@click.option("--blocker-id", required=True, help="Exact unresolved durable blocker identifier.")
@click.option("--reason", required=True, help="Operator rationale recorded in the immutable resolution receipt.")
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
    confirmed: bool,
    output_format: str,
) -> None:
    """Acknowledge one durable frontier blocker against current evidence.

    Routed through ``OperationExecutor``/``BlockerResolveActuator`` (t46.9
    phase 3): PREPARE previews the exact blocker target, EXECUTE requires a
    confirm-flag-strength authorization bound to that plan's hash, and a
    fresh PREPARE immediately before EXECUTE refuses (``PlanStaleError``) if
    the blocker was concurrently resolved between preview and confirm.

    This acknowledges evidence; it repairs nothing. The obligation the
    blocker named is discharged by ordinary acquisition or derivation, or it
    reappears on the next census pass.
    """
    if not confirmed:
        raise click.ClickException("refusing to resolve a durable blocker without --yes")
    result = _submit(
        env,
        {
            "blocker_id": blocker_id,
            "resolution": reason,
        },
    )
    receipt_result = result.get("result")
    receipt = receipt_result if isinstance(receipt_result, dict) else {}
    affected_value = result.get("affected_count", 0)
    affected = affected_value if isinstance(affected_value, int) else 0
    if output_format == "json":
        click.echo(json.dumps(receipt, indent=2, sort_keys=True))
        if not affected:
            raise SystemExit(1)
        return
    if not affected:
        # The actuator answered ``already_satisfied`` with a zero affected
        # count: nothing was resolved. Printing "Resolved <id>" here reported a
        # no-op as a durable effect -- the defect this branch exists to close.
        raise click.ClickException(f"blocker {blocker_id} not found or already acknowledged; nothing was mutated")
    click.echo(f"Acknowledged {blocker_id}")
    current_plan = receipt.get("current_plan")
    if isinstance(current_plan, dict):
        click.echo(f"Current plan: {current_plan.get('plan_id', 'unknown')}")
