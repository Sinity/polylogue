"""Cost command — subscription usage outlook and quota forecasting (#870)."""

from __future__ import annotations

import json
from typing import Any

import click

from polylogue.cli.shared.types import AppEnv


@click.command("cost")
@click.option("--plan", default="pro", help="Subscription plan name (pro, max).")
@click.option("--anomaly-threshold", type=float, default=2.0, help="Multiplier for anomaly detection.")
@click.option("--format", "-f", "output_format", type=click.Choice(["plain", "json"]), default="plain", help="Output format.")
@click.pass_obj
def cost_command(env: AppEnv, plan: str, anomaly_threshold: float, output_format: str) -> None:
    """Show subscription usage outlook and quota forecasting.

    All subscription quota math is estimated and not vendor-authoritative.
    Reads cost data from the archive computed by the #803 cost tracking pipeline.
    """
    from polylogue.archive.semantic.outlook import compute_usage_outlook
    from polylogue.archive.semantic.subscription_models import UsageOutlookPayload

    # Collect cost data from archive
    cost_data: list[dict[str, object]] = []
    try:
        import asyncio

        async def _fetch() -> None:
            from polylogue.api import Polylogue

            async with Polylogue() as poly:
                summaries = await poly.repository.list_summaries(limit=1000)
                for s in summaries:
                    cost_data.append({
                        "created_at": s.created_at,
                        "model": getattr(s, "model", "unknown"),
                        "has_cost": getattr(s, "total_cost_usd", 0) > 0,
                        "api_cost_usd": getattr(s, "total_cost_usd", 0) or 0,
                        "cost_is_estimated": getattr(s, "cost_is_estimated", False),
                    })

        asyncio.run(_fetch())
    except Exception:
        pass

    outlook = compute_usage_outlook(cost_data, plan_name=plan, anomaly_threshold=anomaly_threshold)
    payload = UsageOutlookPayload(**outlook)

    if output_format == "json":
        env.ui.console.print(json.dumps(payload.model_dump(mode="json"), indent=2, default=str))
        return

    _render_plain(env, payload)


def _render_plain(env: Any, p: Any) -> None:
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


__all__ = ["cost_command"]
