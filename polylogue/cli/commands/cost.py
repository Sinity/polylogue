"""Operational cost command — subscription cycle outlook and quota forecasting (#1138).

Hosts the ``polylogue ops cost`` group. The canonical subcommand is
``ops cost outlook``, which renders the typed
:class:`polylogue.cost.outlook.CycleOutlook` payload produced by the
#1137 engine. JSON output is deterministic and the schema is pinned by
contract tests. Plain output visibly labels subscription-quota math as
non-authoritative and tags estimated USD figures.
"""

from __future__ import annotations

import asyncio
import json

import click

from polylogue.cli.shared.types import AppEnv
from polylogue.cost.outlook import CycleOutlook, ProjectionMethod, QuotaPressure, QuotaPressureMissing
from polylogue.cost.plans import PlanLookupError


@click.group("cost")
def cost_command() -> None:
    """Subscription usage outlook and quota forecasting.

    All subscription quota math is non-authoritative. Provider pricing
    and quotas change without notice; numbers here are estimates from
    the curated seed (or user overrides in ``polylogue.toml``).
    """


@cost_command.command("outlook")
@click.option(
    "--plan",
    "plan_name",
    required=True,
    help="Subscription plan name (e.g. claude-pro, claude-max-5x, chatgpt-plus).",
)
@click.option(
    "--method",
    type=click.Choice([m.value for m in ProjectionMethod]),
    default=ProjectionMethod.linear.value,
    help="Projection method to use for cycle extrapolation.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    help="Output format. JSON is the canonical contract.",
)
@click.pass_obj
def outlook_command(env: AppEnv, plan_name: str, method: str, output_format: str) -> None:
    """Project the current billing cycle for a subscription plan.

    Emits the typed ``CycleOutlook`` payload from #1137: cycle window,
    burn rate, projected total, quota pressure, overage rows, coverage,
    and confidence. Subscription-equivalent and API-equivalent figures
    are never merged into a single unlabelled ``cost`` number.
    """

    projection_method = ProjectionMethod(method)

    async def _fetch() -> CycleOutlook | None:
        from polylogue.api import Polylogue

        async with Polylogue() as poly:
            return await poly.cost_outlook(plan_name, method=projection_method)

    try:
        outlook = asyncio.run(_fetch())
    except PlanLookupError as exc:
        raise click.ClickException(str(exc)) from exc

    if outlook is None:
        message = (
            f"No cycle window for plan {plan_name!r}: the plan does not declare "
            "a 'cycle_anchor_day'. Configure one under [[cost.subscription.plans]] "
            "or use a plan with a fixed monthly anchor."
        )
        if output_format == "json":
            env.ui.console.print(
                json.dumps(
                    {"plan_name": plan_name, "outlook": None, "reason": "no_cycle_anchor"},
                    indent=2,
                )
            )
            return
        env.ui.console.print(f"[yellow]{message}[/yellow]")
        return

    if output_format == "json":
        env.ui.console.print(json.dumps(outlook.model_dump(mode="json"), indent=2, default=str))
        return

    _render_outlook_plain(env, outlook)


def _render_outlook_plain(env: AppEnv, o: CycleOutlook) -> None:
    """Render a CycleOutlook in plain mode with explicit labelling.

    The renderer never merges subscription-equivalent and API-equivalent
    figures into a single unlabelled number. Estimated USD totals carry
    a ``~`` prefix; quota figures are labelled with their basis; absent
    quota declarations are surfaced explicitly.
    """
    c = env.ui.console
    c.print(f"\n[bold]Cycle Outlook — {o.plan_name}[/bold]  [dim](estimate)[/dim]")
    c.print(f"  Cycle: {o.window.start.date().isoformat()} → {o.window.end.date().isoformat()}")
    c.print(f"  Elapsed: {o.window.elapsed_days:.1f}d / {o.window.total_days:.1f}d")
    c.print(f"  Projection method: {o.projection_method.value}")

    if not o.cycle_to_date:
        c.print("  [dim]No usage observed in the current cycle.[/dim]")
    else:
        c.print("  Cycle-to-date:")
        for basis, used in sorted(o.cycle_to_date.items()):
            burn = o.burn_rate_per_day.get(basis, 0.0)
            projected = o.projected_total.get(basis, 0.0)
            label = "USD (API-equivalent)" if basis == "usd" else basis
            unit = "$" if basis == "usd" else ""
            c.print(f"    {label}: ~{unit}{used:.4f} used, ~{unit}{burn:.4f}/day, projected ~{unit}{projected:.4f}")

    if isinstance(o.quota_pressure, QuotaPressureMissing):
        c.print("  [dim]Quota pressure: not configured (plan has no quota).[/dim]")
    else:
        pressure: QuotaPressure = o.quota_pressure
        basis_label = pressure.basis.value
        c.print(
            f"  Quota pressure ({basis_label}): "
            f"~{pressure.used:.2f}/{pressure.quota:.0f} "
            f"({pressure.used_ratio * 100:.1f}% used, "
            f"projected {pressure.projected_ratio * 100:.1f}%)"
        )
        if pressure.breach_day is not None:
            c.print(f"    [yellow]Projected breach: {pressure.breach_day.isoformat()}[/yellow]")

    for row in o.overage_rows:
        cost_part = (
            f", projected cost ~${row.projected_overage_cost_usd:.4f}"
            if row.projected_overage_cost_usd is not None
            else ""
        )
        c.print(
            f"  Overage ({row.basis.value}, {row.overage_rule.value}): "
            f"actual ~{row.actual_overage:.2f}, projected ~{row.projected_overage:.2f}{cost_part}"
        )

    c.print(f"  Coverage: {o.coverage_ratio * 100:.0f}%  Confidence: {o.confidence * 100:.0f}%")
    if o.incomplete_days:
        days = ", ".join(d.isoformat() for d in o.incomplete_days[:7])
        more = "" if len(o.incomplete_days) <= 7 else f" (+{len(o.incomplete_days) - 7} more)"
        c.print(f"    [dim]Incomplete days: {days}{more}[/dim]")
    c.print(
        "  [dim]Subscription quota math is non-authoritative; vendor pricing and quotas change without notice.[/dim]"
    )


__all__ = ["cost_command", "outlook_command"]
