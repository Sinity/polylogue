"""Shared rendering helpers for cost outlook surfaces."""

from __future__ import annotations

from polylogue.cli.shared.types import AppEnv
from polylogue.cost.outlook import CycleOutlook, QuotaPressure, QuotaPressureMissing


def render_outlook_plain(env: AppEnv, outlook: CycleOutlook) -> None:
    """Render a CycleOutlook in plain mode with explicit labelling."""
    console = env.ui.console
    console.print(f"\n[bold]Cycle Outlook -- {outlook.plan_name}[/bold]  [dim](estimate)[/dim]")
    console.print(f"  Cycle: {outlook.window.start.date().isoformat()} -> {outlook.window.end.date().isoformat()}")
    console.print(f"  Elapsed: {outlook.window.elapsed_days:.1f}d / {outlook.window.total_days:.1f}d")
    console.print(f"  Projection method: {outlook.projection_method.value}")

    if not outlook.cycle_to_date:
        console.print("  [dim]No usage observed in the current cycle.[/dim]")
    else:
        console.print("  Cycle-to-date:")
        for basis, used in sorted(outlook.cycle_to_date.items()):
            burn = outlook.burn_rate_per_day.get(basis, 0.0)
            projected = outlook.projected_total.get(basis, 0.0)
            label = "USD (API-equivalent)" if basis == "usd" else basis
            unit = "$" if basis == "usd" else ""
            console.print(
                f"    {label}: ~{unit}{used:.4f} used, ~{unit}{burn:.4f}/day, projected ~{unit}{projected:.4f}"
            )

    if isinstance(outlook.quota_pressure, QuotaPressureMissing):
        console.print("  [dim]Quota pressure: not configured (plan has no quota).[/dim]")
    else:
        pressure: QuotaPressure = outlook.quota_pressure
        basis_label = pressure.basis.value
        console.print(
            f"  Quota pressure ({basis_label}): "
            f"~{pressure.used:.2f}/{pressure.quota:.0f} "
            f"({pressure.used_ratio * 100:.1f}% used, "
            f"projected {pressure.projected_ratio * 100:.1f}%)"
        )
        if pressure.breach_day is not None:
            console.print(f"    [yellow]Projected breach: {pressure.breach_day.isoformat()}[/yellow]")

    for row in outlook.overage_rows:
        cost_part = (
            f", projected cost ~${row.projected_overage_cost_usd:.4f}"
            if row.projected_overage_cost_usd is not None
            else ""
        )
        console.print(
            f"  Overage ({row.basis.value}, {row.overage_rule.value}): "
            f"actual ~{row.actual_overage:.2f}, projected ~{row.projected_overage:.2f}{cost_part}"
        )

    console.print(f"  Coverage: {outlook.coverage_ratio * 100:.0f}%  Confidence: {outlook.confidence * 100:.0f}%")
    if outlook.incomplete_days:
        days = ", ".join(day.isoformat() for day in outlook.incomplete_days[:7])
        more = "" if len(outlook.incomplete_days) <= 7 else f" (+{len(outlook.incomplete_days) - 7} more)"
        console.print(f"    [dim]Incomplete days: {days}{more}[/dim]")
    console.print(
        "  [dim]Subscription quota math is non-authoritative; vendor pricing and quotas change without notice.[/dim]"
    )


__all__ = ["render_outlook_plain"]
