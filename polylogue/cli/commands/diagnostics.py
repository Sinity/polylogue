"""Archive and temporal diagnostics for the ops command surface."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.cli.shared.helpers import fail
from polylogue.cli.shared.types import AppEnv


@click.group("diagnostics", help="Run archive and session diagnostics.")
def diagnostics_group() -> None:
    pass


@diagnostics_group.command("workload")
@click.option("--db", type=click.Path(path_type=Path), default=None, help="Archive SQLite database path.")
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable JSON.")
@click.option("--limit", type=int, default=5, help="Recent attempt limit.")
@click.option(
    "--compare",
    nargs=2,
    type=click.Path(path_type=Path),
    default=None,
    metavar="BEFORE AFTER",
    help="Compare two saved probe reports and report structured deltas.",
)
def workload_command(
    db: Path | None,
    json_output: bool,
    limit: int,
    compare: tuple[Path, Path] | None,
) -> None:
    """Inspect daemon ingest workload, convergence debt, and hot query plans."""
    from devtools.daemon_workload_probe import main as workload_main

    argv: list[str] = []
    if db is not None:
        argv.extend(("--db", str(db)))
    if json_output:
        argv.append("--json")
    argv.extend(("--limit", str(limit)))
    if compare is not None:
        before, after = compare
        argv.extend(("--compare", str(before), str(after)))
    raise click.exceptions.Exit(workload_main(argv))


@diagnostics_group.command("space")
@click.option("--db", type=click.Path(path_type=Path), default=None, help="Archive database path.")
@click.option("--limit", type=int, default=25, help="Largest object rows to include.")
@click.option("--objects", is_flag=True, help="Run the dbstat table/index object scan.")
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable JSON.")
def space_command(db: Path | None, limit: int, objects: bool, json_output: bool) -> None:
    """Report SQLite archive file, page, table, and index space."""
    from devtools.archive_space_report import main as space_main

    argv: list[str] = []
    if db is not None:
        argv.extend(("--db", str(db)))
    argv.extend(("--limit", str(limit)))
    if objects:
        argv.append("--objects")
    if json_output:
        argv.append("--json")
    raise click.exceptions.Exit(space_main(argv))


@diagnostics_group.command("pace")
@click.argument("session_id", required=False)
@click.option("--limit", "-l", "-n", type=int, default=10, help="Number of recent sessions to analyze")
@click.option("--threshold", type=int, default=60, help="Gap threshold in seconds for classification")
@click.pass_context
def pace_command(
    ctx: click.Context,
    session_id: str | None,
    limit: int,
    threshold: int,
) -> None:
    """Show inter-turn gap analysis for one or more sessions."""
    env: AppEnv = ctx.obj
    run_coroutine_sync(_pace(env, session_id, limit, threshold))


async def _pace(env: AppEnv, session_id: str | None, limit: int, threshold: int) -> None:
    if session_id:
        spec = SessionQuerySpec(session_id=session_id, limit=1)
        convs = await spec.list(env.config)
        if not convs:
            fail("pace", f"No session matching '{session_id}'")
    else:
        spec = SessionQuerySpec(sort="date", limit=limit)
        convs = await spec.list(env.config)

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
@click.argument("session_id", required=False)
@click.option("--limit", "-l", "-n", type=int, default=20, help="Max turns to show")
@click.pass_context
def turns_command(ctx: click.Context, session_id: str | None, limit: int) -> None:
    """Show per-turn cost and duration for one session.

    The session id may be omitted when a root filter like ``--latest`` or
    ``--origin`` narrows the archive to a single match (#1642).
    """
    from polylogue.cli.shared.insight_command_contracts import find_root_params
    from polylogue.cli.shared.latest_resolver import resolve_session_id_from_root_params

    env: AppEnv = ctx.obj
    if session_id is None:
        session_id = resolve_session_id_from_root_params(dict(find_root_params(ctx)))
        if not session_id:
            fail("turns", "turns requires a session ID (positional or --latest/--origin)")
    run_coroutine_sync(_turns(env, session_id, limit))


async def _turns(env: AppEnv, session_id: str, limit: int) -> None:
    spec = SessionQuerySpec(session_id=session_id, limit=1)
    convs = await spec.list(env.config)
    if not convs:
        fail("turns", f"No session matching '{session_id}'")
    conv = convs[0]
    msgs = conv.messages

    env.ui.console.print(f"\n[bold]{conv.display_title}[/bold] ({str(conv.id)[:12]})")
    header = f"{'#':>3s}  {'role':12s}  {'duration':>10s}  {'thinking':>10s}  {'tools':>5s}  {'chars':>6s}"
    env.ui.console.print(header)
    env.ui.console.print("-" * len(header))

    for shown, msg in enumerate(msgs):
        if shown >= limit:
            break
        blocks = getattr(msg, "blocks", []) or []
        thinking_chars = sum(
            len(str(b.get("text", ""))) for b in blocks if isinstance(b, dict) and b.get("type") == "thinking"
        )
        tool_count = sum(1 for b in blocks if isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result"))
        duration_ms = int(getattr(msg, "duration_ms", 0) or 0)

        duration_str = f"{duration_ms / 1000:.1f}s" if duration_ms else ""
        thinking_str = str(thinking_chars) if thinking_chars else ""
        tool_str = str(tool_count) if tool_count else ""
        role = str(getattr(msg, "role", "?"))[:12]
        text_len = len(getattr(msg, "text", "") or "")

        env.ui.console.print(
            f"{shown:3d}  {role:12s}  {duration_str:>10s}  {thinking_str:>10s}  {tool_str:>5s}  {text_len:>6d}"
        )


@diagnostics_group.command("tools")
@click.option("--origin", help="Filter by origin")
@click.option("--limit", "-l", "-n", type=int, default=20, help="Max tools to show")
@click.pass_context
def tools_command(ctx: click.Context, origin: str | None, limit: int) -> None:
    """Show top tools by invocation count across filtered sessions."""
    env: AppEnv = ctx.obj
    run_coroutine_sync(_tools(env, origin, limit))


async def _tools(env: AppEnv, origin: str | None, limit: int) -> None:
    from collections import Counter

    # Get recent sessions and aggregate their actions
    spec = SessionQuerySpec(sort="date", limit=100)
    if origin:
        spec = SessionQuerySpec(sort="date", limit=100, origins=(origin,))
    convs = await spec.list_summaries(env.config)

    poly = env.polylogue
    tool_counts: Counter[str] = Counter()
    for summary in convs:
        try:
            actions = await poly.get_actions(str(summary.id))
            for action in actions:
                name = getattr(action, "normalized_tool_name", None) or getattr(action, "tool_name", None)
                if name:
                    tool_counts[str(name)] += 1
        except Exception:
            continue

    if not tool_counts:
        env.ui.console.print("No tool invocations found.")
        return

    env.ui.console.print(f"\n[bold]Top tools by invocation count[/bold] (across {len(convs)} sessions)")
    env.ui.console.print(f"{'tool':40s}  {'count':>6s}")
    env.ui.console.print("-" * 50)
    for name, cnt in tool_counts.most_common(limit):
        env.ui.console.print(f"{name[:40]:40s}  {cnt:>6d}")
