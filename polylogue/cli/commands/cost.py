"""Cost command — subscription cycle outlook and quota forecasting (#1138).

Hosts the ``polylogue cost`` group. The canonical subcommand is
``cost outlook``, which renders the typed
:class:`polylogue.cost.outlook.CycleOutlook` payload produced by the
#1137 engine. JSON output is deterministic and the schema is pinned by
contract tests. Plain output visibly labels subscription-quota math as
non-authoritative and tags estimated USD figures.

The legacy flat ``polylogue cost --plan <name>`` form is preserved via
the ``cost rollup`` subcommand, which keeps the old
``compute_usage_outlook`` rollup view for now and is intentionally
called out as the deprecated surface in its help text.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

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


@cost_command.command("rollup")
@click.option("--plan", default="pro", help="Legacy subscription plan label (pro, max).")
@click.option("--anomaly-threshold", type=float, default=2.0, help="Multiplier for anomaly detection.")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    help="Output format.",
)
@click.pass_obj
def rollup_command(env: AppEnv, plan: str, anomaly_threshold: float, output_format: str) -> None:
    """Flat cost rollup (legacy surface).

    Retained for the existing #803 cost-rollup pipeline. New work
    targets the typed ``cost outlook`` payload instead.
    """
    from polylogue.archive.semantic.outlook import compute_usage_outlook
    from polylogue.archive.semantic.subscription_models import UsageOutlookPayload

    cost_data: list[dict[str, object]] = []
    try:

        async def _fetch() -> None:
            from polylogue.api import Polylogue

            async with Polylogue() as poly:
                summaries = await poly.list_summaries(limit=1000)
                for s in summaries:
                    cost_data.append(
                        {
                            "created_at": s.created_at,
                            "model": getattr(s, "model", "unknown"),
                            "has_cost": getattr(s, "total_cost_usd", 0) > 0,
                            "api_cost_usd": getattr(s, "total_cost_usd", 0) or 0,
                            "cost_is_estimated": getattr(s, "cost_is_estimated", False),
                        }
                    )

        asyncio.run(_fetch())
    except Exception:
        pass

    outlook = compute_usage_outlook(cost_data, plan_name=plan, anomaly_threshold=anomaly_threshold)
    payload = UsageOutlookPayload.model_validate(outlook)

    if output_format == "json":
        env.ui.console.print(json.dumps(payload.model_dump(mode="json"), indent=2, default=str))
        return

    _render_rollup_plain(env, payload)


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


def _render_rollup_plain(env: Any, p: Any) -> None:
    c = env.ui.console
    c.print(f"\n[bold]Subscription Outlook — {p.plan_name}[/bold]")
    c.print(f"  Cycle: {p.cycle_start[:10]} → {p.cycle_end[:10]}")
    c.print(f"  Credits: {p.credits_used:.0f} used / {p.credits_total:.0f} total ({p.credits_remaining:.0f} remaining)")
    if p.burn_rate_credits_per_day > 0:
        c.print(f"  Burn rate: {p.burn_rate_credits_per_day:.1f} credits/day")
    if p.projected_exhaustion_date:
        c.print(f"  Projected exhaustion: [yellow]{p.projected_exhaustion_date[:10]}[/yellow]")
    c.print(f"  Confidence: {p.confidence * 100:.0f}%  Coverage: {p.coverage_pct:.0f}%")
    if p.api_equivalent_usd_total > 0:
        c.print(f"  API-equivalent: ${p.api_equivalent_usd_total:.2f} (estimated)")
    if p.anomalies:
        c.print(f"\n[bold]Anomalies ({len(p.anomalies)}):[/bold]")
        for a in p.anomalies[:5]:
            c.print(f"  {a.date}: ${a.cost_usd:.2f} ({a.explanation})")
    if p.per_model:
        c.print("\n[bold]Per-model:[/bold]")
        for m in p.per_model[:10]:
            c.print(f"  {m.model}: ${m.api_equivalent_usd:.2f} ({m.session_count} sessions)")


__all__ = ["cost_command", "outlook_command", "rollup_command"]
