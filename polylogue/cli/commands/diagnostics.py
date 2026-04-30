"""Temporal diagnostics: session pace, per-turn cost, and tool hotlist."""

from __future__ import annotations

from datetime import timedelta

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.cli.shared.helpers import fail
from polylogue.cli.shared.types import AppEnv
from polylogue.lib.query.spec import ConversationQuerySpec


@click.group("diagnostics", help="Temporal session diagnostics")
def diagnostics_group() -> None:
    pass


@diagnostics_group.command("pace")
@click.argument("conversation_id", required=False)
@click.option("--limit", "-n", type=int, default=10, help="Number of recent conversations to analyze")
@click.option("--threshold", type=int, default=60, help="Gap threshold in seconds for classification")
@click.pass_context
def pace_command(
    ctx: click.Context,
    conversation_id: str | None,
    limit: int,
    threshold: int,
) -> None:
    """Show inter-turn gap analysis for one or more conversations."""
    env: AppEnv = ctx.obj
    run_coroutine_sync(_pace(env, conversation_id, limit, threshold))


async def _pace(env: AppEnv, conversation_id: str | None, limit: int, threshold: int) -> None:
    if conversation_id:
        spec = ConversationQuerySpec(conversation_id=conversation_id, limit=1)
        convs = await spec.list(env.repository)
        if not convs:
            fail("pace", f"No conversation matching '{conversation_id}'")
    else:
        spec = ConversationQuerySpec(sort="date", limit=limit)
        convs = await spec.list(env.repository)

    for conv in convs:
        msgs = conv.messages
        if len(msgs) < 2:
            env.ui.console.print(f"[bold]{conv.display_title}[/bold] — not enough messages")
            continue

        substantive = [m for m in msgs if hasattr(m, "role") and str(m.role) in ("user", "assistant") and m.timestamp]
        if len(substantive) < 2:
            env.ui.console.print(f"[bold]{conv.display_title}[/bold] — not enough timestamped turns")
            continue

        total_active = 0
        total_idle = 0
        gaps: list[tuple[int, int, int, str]] = []  # (from_idx, to_idx, seconds, kind)
        for i in range(len(substantive) - 1):
            t0 = substantive[i].timestamp
            t1 = substantive[i + 1].timestamp
            if t0 is None or t1 is None:
                continue
            delta = int((t1 - t0).total_seconds())
            if delta < 0:
                continue
            if delta <= threshold:
                kind = "active"
                total_active += delta
            elif str(substantive[i + 1].role) == "assistant":
                kind = "model_work"
                total_active += delta
            else:
                kind = "user_idle"
                total_idle += delta
            if delta > 5:
                gaps.append((i, i + 1, delta, kind))

        env.ui.console.print(f"\n[bold]{conv.display_title}[/bold] ({str(conv.id)[:12]})")
        env.ui.console.print(
            f"  Turns: {len(substantive)} | Active: {timedelta(seconds=total_active)} | Idle: {timedelta(seconds=total_idle)}"
        )

        gaps_sorted = sorted(gaps, key=lambda g: g[2], reverse=True)[:5]
        if gaps_sorted:
            env.ui.console.print("  Largest gaps:")
        for from_idx, to_idx, seconds, kind in gaps_sorted:
            env.ui.console.print(f"    {from_idx:3d}→{to_idx:3d}  {timedelta(seconds=seconds)!s:>10s}  [{kind}]")


@diagnostics_group.command("turns")
@click.argument("conversation_id")
@click.option("--limit", "-n", type=int, default=20, help="Max turns to show")
@click.pass_context
def turns_command(ctx: click.Context, conversation_id: str, limit: int) -> None:
    """Show per-turn cost and duration for one conversation."""
    env: AppEnv = ctx.obj
    run_coroutine_sync(_turns(env, conversation_id, limit))


async def _turns(env: AppEnv, conversation_id: str, limit: int) -> None:
    spec = ConversationQuerySpec(conversation_id=conversation_id, limit=1)
    convs = await spec.list(env.repository)
    if not convs:
        fail("turns", f"No conversation matching '{conversation_id}'")
    conv = convs[0]
    msgs = conv.messages

    env.ui.console.print(f"\n[bold]{conv.display_title}[/bold] ({str(conv.id)[:12]})")
    header = f"{'#':>3s}  {'role':12s}  {'duration':>10s}  {'thinking':>10s}  {'tools':>5s}  {'chars':>6s}"
    env.ui.console.print(header)
    env.ui.console.print("-" * len(header))

    for shown, msg in enumerate(msgs):
        if shown >= limit:
            break
        blocks = getattr(msg, "content_blocks", []) or []
        thinking_chars = sum(
            len(str(b.get("text", ""))) for b in blocks if isinstance(b, dict) and b.get("type") == "thinking"
        )
        tool_count = sum(1 for b in blocks if isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result"))
        duration_ms = 0
        meta = getattr(msg, "provider_meta", None) or {}
        if isinstance(meta, dict):
            duration_ms = int(meta.get("durationMs", 0) or meta.get("duration_ms", 0) or 0)

        duration_str = f"{duration_ms / 1000:.1f}s" if duration_ms else ""
        thinking_str = str(thinking_chars) if thinking_chars else ""
        tool_str = str(tool_count) if tool_count else ""
        role = str(getattr(msg, "role", "?"))[:12]
        text_len = len(getattr(msg, "text", "") or "")

        env.ui.console.print(
            f"{shown:3d}  {role:12s}  {duration_str:>10s}  {thinking_str:>10s}  {tool_str:>5s}  {text_len:>6d}"
        )


@diagnostics_group.command("tools")
@click.option("--provider", "-p", help="Filter by provider")
@click.option("--limit", "-n", type=int, default=20, help="Max tools to show")
@click.pass_context
def tools_command(ctx: click.Context, provider: str | None, limit: int) -> None:
    """Show top tools by invocation count across filtered conversations."""
    env: AppEnv = ctx.obj
    run_coroutine_sync(_tools(env, provider, limit))


async def _tools(env: AppEnv, provider: str | None, limit: int) -> None:
    from collections import Counter

    # Get recent conversations and aggregate their action events
    spec = ConversationQuerySpec(sort="date", limit=100)
    if provider:
        from polylogue.types import Provider

        spec = ConversationQuerySpec(sort="date", limit=100, providers=(Provider.from_string(provider),))
    convs = await spec.list_summaries(env.repository)

    tool_counts: Counter[str] = Counter()
    for summary in convs:
        try:
            events = await env.repository.get_action_events(str(summary.id))
            for evt in events:
                name = getattr(evt, "normalized_tool_name", None) or getattr(evt, "tool_name", None)
                if name:
                    tool_counts[str(name)] += 1
        except Exception:
            continue

    if not tool_counts:
        env.ui.console.print("No tool invocations found.")
        return

    env.ui.console.print(f"\n[bold]Top tools by invocation count[/bold] (across {len(convs)} conversations)")
    env.ui.console.print(f"{'tool':40s}  {'count':>6s}")
    env.ui.console.print("-" * 50)
    for name, cnt in tool_counts.most_common(limit):
        env.ui.console.print(f"{name[:40]:40s}  {cnt:>6d}")
