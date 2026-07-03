"""Archive diagnostics plus session-analysis command implementations."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.cli.shared.helpers import fail
from polylogue.cli.shared.types import AppEnv


@click.group("diagnostics", help="Run archive and daemon diagnostics.")
def diagnostics_group() -> None:
    pass


@diagnostics_group.command("workload")
@click.option("--db", type=click.Path(path_type=Path), default=None, help="Archive SQLite database path.")
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable JSON.")
@click.option("--limit", "-l", type=int, default=5, help="Recent attempt limit.")
@click.option(
    "--integrity-check",
    is_flag=True,
    help="Run SQLite quick_check for each archive tier. This can be expensive on large archives.",
)
@click.option(
    "--exact-derived-counts",
    is_flag=True,
    help="Run exact derived-readiness reconciliation counts. This can scan large archive tables.",
)
@click.option(
    "--blob-reference-debt",
    is_flag=True,
    help="Count missing referenced blob files exactly. This can stat many blob paths on large archives.",
)
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
    integrity_check: bool,
    exact_derived_counts: bool,
    blob_reference_debt: bool,
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
    if integrity_check:
        argv.append("--integrity-check")
    if exact_derived_counts:
        argv.append("--exact-derived-counts")
    if blob_reference_debt:
        argv.append("--blob-reference-debt")
    if compare is not None:
        before, after = compare
        argv.extend(("--compare", str(before), str(after)))
    raise click.exceptions.Exit(workload_main(argv))


@diagnostics_group.command("space")
@click.option("--db", type=click.Path(path_type=Path), default=None, help="Archive database path.")
@click.option("--limit", "-l", type=int, default=25, help="Largest object rows to include.")
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


@click.command("pace")
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


@click.command("turns")
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


@click.command("usage")
@click.option("--origin", help="Filter to one archive origin, such as codex-session or claude-code-session.")
@click.option(
    "--limit",
    "-l",
    "sample_limit",
    type=int,
    default=25,
    help="Max diagnostic session IDs to include per sample family.",
)
@click.option(
    "--detail",
    type=click.Choice(["headline", "full"]),
    default="full",
    show_default=True,
    help="Report detail: headline skips expensive provider-event and stale-rollup diagnostics.",
)
@click.option(
    "--json",
    "output_format",
    flag_value="json",
    default=None,
    help="Shortcut for --format json.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format.",
)
@click.pass_context
def usage_command(
    ctx: click.Context,
    origin: str | None,
    sample_limit: int,
    detail: str,
    output_format: str,
) -> None:
    """Audit provider usage accounting without turning it into a cost report.

    Provider event rows, provider cumulative counters, transcript words, and
    model rollups are printed as separate evidence streams so cached tokens,
    reasoning tokens, zero-token events, missing models, multi-model sessions,
    source acquisition debt, and stale rollups stay visible.
    """
    import json

    env: AppEnv = ctx.obj
    report = run_coroutine_sync(env.polylogue.provider_usage_report(origin=origin, limit=sample_limit, detail=detail))
    if output_format == "json":
        click.echo(json.dumps(report.to_dict(), indent=2))
        return
    _render_usage_report(env, report)


def _render_usage_report(env: AppEnv, report: object) -> None:
    origins = tuple(getattr(report, "origins", ()))
    env.ui.console.print(f"[bold]Provider usage accounting[/bold] ({getattr(report, 'archive_root', '')})")
    env.ui.console.print(f"  detail: {getattr(report, 'detail_level', 'full')}")
    for caveat in getattr(report, "caveats", ()):
        env.ui.console.print(f"  [yellow]note[/yellow] {caveat}")
    env.ui.console.print(
        "  all-origin model rollup usage "
        f"({getattr(report, 'model_rollup_grain', 'physical_session')}): "
        f"{_usage_counter_line(getattr(report, 'model_rollup_usage', None))}"
    )
    env.ui.console.print(
        "  all-origin model rollup usage "
        f"({getattr(report, 'logical_model_rollup_grain', 'logical_session_model_high_water')}): "
        f"{_usage_counter_line(getattr(report, 'logical_model_rollup_usage', None))}"
    )
    if not origins:
        env.ui.console.print("  No sessions matched.")
        return
    for row in origins:
        env.ui.console.print(f"\n[bold]{row.origin}[/bold]")
        env.ui.console.print(
            "  coverage: "
            f"provider={row.provider}  declared={row.declared_coverage}  state={row.coverage_state}  "
            f"evidence={row.evidence_stream}"
        )
        if row.coverage_basis:
            env.ui.console.print(f"    basis: {row.coverage_basis}")
        if row.cache_semantics:
            env.ui.console.print(f"    cache: {row.cache_semantics}")
        env.ui.console.print(
            "  sessions: "
            f"{row.session_count}  messages: {row.message_count}  transcript_words: {row.transcript_word_count}"
        )
        env.ui.console.print(
            "  source raw rows: "
            f"acquired={row.raw_session_count}  parse_errors={row.raw_parse_error_count}  "
            f"acquired_not_materialized={row.acquired_not_materialized_count}"
        )
        env.ui.console.print(
            "  provider events: "
            f"{row.provider_event_count} across {row.provider_event_session_count} sessions  "
            f"token_count={row.token_count_event_count}  message_usage={row.message_usage_event_count}  "
            f"zero_token={row.zero_token_event_count}  missing_model={row.missing_model_event_count}"
        )
        env.ui.console.print(
            "  model rollup rows: "
            f"priced={row.priced_model_row_count}  origin_reported={row.origin_reported_model_row_count}  "
            f"estimated={row.estimated_model_row_count}  multi_model_sessions={row.multi_model_session_count}  "
            f"stale_sessions={row.stale_rollup_session_count}"
        )
        env.ui.console.print(f"  provider request usage:    {_usage_counter_line(row.provider_request_usage)}")
        env.ui.console.print(f"  provider cumulative usage: {_usage_counter_line(row.provider_cumulative_usage)}")
        env.ui.console.print(
            f"  model rollup usage ({row.model_rollup_grain}): {_usage_counter_line(row.model_rollup_usage)}"
        )
        env.ui.console.print(
            f"  model rollup usage ({row.logical_model_rollup_grain}): "
            f"{_usage_counter_line(row.logical_model_rollup_usage)}"
        )
        if row.rebuild_guidance:
            env.ui.console.print(f"  rebuild: {row.rebuild_guidance}")
        for caveat in row.caveats:
            env.ui.console.print(f"    [yellow]caveat[/yellow] {caveat}")
        if row.sample_missing_model_sessions:
            env.ui.console.print("    missing-model samples: " + ", ".join(row.sample_missing_model_sessions))
        if row.sample_zero_token_sessions:
            env.ui.console.print("    zero-token samples: " + ", ".join(row.sample_zero_token_sessions))
        if row.sample_acquired_not_materialized_raw_ids:
            env.ui.console.print(
                "    acquired-not-materialized raw samples: " + ", ".join(row.sample_acquired_not_materialized_raw_ids)
            )
        if row.sample_stale_rollup_sessions:
            env.ui.console.print("    stale-rollup samples: " + ", ".join(row.sample_stale_rollup_sessions))


def _usage_counter_line(counters: object) -> str:
    values = counters.to_dict() if hasattr(counters, "to_dict") else {}
    return (
        f"input={int(values.get('input_tokens', 0))} "
        f"output={int(values.get('output_tokens', 0))} "
        f"cached_input={int(values.get('cached_input_tokens', 0))} "
        f"cache_write={int(values.get('cache_write_tokens', 0))} "
        f"reasoning_output={int(values.get('reasoning_output_tokens', 0))} "
        f"total={int(values.get('total_tokens', 0))}"
    )


@click.command("tools")
@click.option("--origin", help="Filter by origin or provider token")
@click.option("--tool", help="Only entries for this exact normalized tool name, e.g. mcp__serena__find_symbol")
@click.option("--mcp-server", help="Only MCP tools with this server prefix, e.g. serena -> mcp__serena__*")
@click.option("--action-kind", help="Only entries for this action kind")
@click.option(
    "--detail-pattern",
    multiple=True,
    help="With --basis actions, match command/path/input detail text. Repeatable.",
)
@click.option(
    "--days",
    type=int,
    default=None,
    help="Restrict to sessions whose sort key is within this many days.",
)
@click.option(
    "--basis",
    type=click.Choice(["tool-use-blocks", "observed-events", "actions"]),
    default="tool-use-blocks",
    show_default=True,
    help=(
        "Underlying projection: tool-use-blocks counts calls; "
        "observed-events counts finished tool outcomes; actions counts canonical action evidence."
    ),
)
@click.option(
    "--compare-family",
    help=(
        "Compare one affordance family across tool-use blocks, observed-event outcomes, "
        "and action/detail evidence without merging the datasets."
    ),
)
@click.option("--limit", "-l", "-n", type=int, default=20, help="Max tools to show")
@click.option(
    "--json",
    "output_format",
    flag_value="json",
    default=None,
    help="Shortcut for --format json.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format.",
)
@click.pass_context
def tools_command(
    ctx: click.Context,
    origin: str | None,
    tool: str | None,
    mcp_server: str | None,
    action_kind: str | None,
    detail_pattern: tuple[str, ...],
    days: int | None,
    basis: str,
    compare_family: str | None,
    limit: int,
    output_format: str,
) -> None:
    """Show tool usage rollups from archive projections."""
    env: AppEnv = ctx.obj
    run_coroutine_sync(
        _tools(
            env,
            origin,
            tool,
            mcp_server,
            action_kind,
            detail_pattern,
            days,
            basis,
            limit,
            output_format,
            compare_family=compare_family,
        )
    )


async def _tools(
    env: AppEnv,
    origin: str | None,
    tool: str | None,
    mcp_server: str | None,
    action_kind: str | None,
    detail_patterns: tuple[str, ...],
    days: int | None,
    basis: str,
    limit: int,
    output_format: str = "text",
    compare_family: str | None = None,
) -> None:
    from typing import cast

    from polylogue.insights.tool_usage import ToolUsageInsightQuery
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.surfaces.payloads import (
        ToolCountBasis,
        ToolCountDetailLevel,
        ToolCountFiltersPayload,
        ToolCountKind,
        ToolCountPayload,
        ToolCountRowPayload,
        ToolFamilyComparisonPayload,
    )

    if compare_family is not None and (tool is not None or mcp_server is not None):
        raise click.UsageError("--compare-family cannot be combined with --tool or --mcp-server")
    if detail_patterns and basis != "actions" and compare_family is None:
        raise click.UsageError("--detail-pattern is only supported with --basis actions")
    if days is not None and days < 1:
        raise click.UsageError("--days must be positive")
    since_ms = None
    if days is not None:
        since_ms = int((datetime.now(UTC) - timedelta(days=days)).timestamp() * 1000)

    def row_payload(row: dict[str, object]) -> ToolCountRowPayload:
        return ToolCountRowPayload(
            source_name=str(row["source_name"]),
            origin=str(row["origin"]),
            normalized_tool_name=str(row["normalized_tool_name"]),
            action_kind=str(row["action_kind"]),
            evidence_kind=str(row["evidence_kind"]) if row.get("evidence_kind") is not None else None,
            matched_by=str(row["matched_by"]) if row.get("matched_by") is not None else None,
            call_count=int(str(row["call_count"])) if row.get("call_count") is not None else None,
            session_count=int(str(row["session_count"])) if row.get("session_count") is not None else None,
            error_count=int(str(row["error_count"])) if row.get("error_count") is not None else None,
            nonzero_exit_count=int(str(row["nonzero_exit_count"]))
            if row.get("nonzero_exit_count") is not None
            else None,
            status=str(row["status"]) if row.get("status") is not None else None,
            event_count=int(str(row["event_count"])) if row.get("event_count") is not None else None,
        )

    def payload_for(
        *,
        rows: list[dict[str, object]],
        payload_basis: ToolCountBasis,
        kind_value: ToolCountKind,
        detail_value: ToolCountDetailLevel,
        payload_tool: str | None,
        payload_mcp_server: str | None,
        payload_detail_patterns: tuple[str, ...],
    ) -> ToolCountPayload:
        return ToolCountPayload(
            kind=kind_value,
            detail_level=detail_value,
            archive_root=str(env.config.archive_root),
            filters=ToolCountFiltersPayload(
                origin=origin,
                tool=payload_tool,
                mcp_server=payload_mcp_server,
                action_kind=action_kind,
                detail_patterns=payload_detail_patterns,
                days=days,
                basis=payload_basis,
                limit=limit,
            ),
            items=tuple(row_payload(row) for row in rows),
        )

    query = ToolUsageInsightQuery(
        provider=origin,
        tool=tool,
        mcp_server=mcp_server,
        action_kind=action_kind,
        since_ms=since_ms,
        limit=limit,
    )
    with ArchiveStore.open_existing(env.config.archive_root) as archive:
        if compare_family:
            family = compare_family.strip().lower()
            if not family:
                raise click.UsageError("--compare-family must not be empty")
            mcp_family = family.replace("-", "_")
            action_patterns = tuple(dict.fromkeys((family, *detail_patterns)))
            mcp_query = ToolUsageInsightQuery(
                provider=origin,
                mcp_server=mcp_family,
                action_kind=action_kind,
                since_ms=since_ms,
                limit=limit,
            )
            action_query = ToolUsageInsightQuery(
                provider=origin,
                action_kind=action_kind,
                since_ms=since_ms,
                limit=limit,
            )
            call_payload = payload_for(
                rows=archive.list_tool_call_count_rows(mcp_query),
                payload_basis="tool-use-blocks",
                kind_value="tool_call_counts",
                detail_value="tool_use_block_call_counts",
                payload_tool=None,
                payload_mcp_server=mcp_family,
                payload_detail_patterns=(),
            )
            event_payload = payload_for(
                rows=archive.list_tool_observed_event_count_rows(mcp_query),
                payload_basis="observed-events",
                kind_value="tool_observed_event_counts",
                detail_value="tool_finished_observed_events",
                payload_tool=None,
                payload_mcp_server=mcp_family,
                payload_detail_patterns=(),
            )
            action_payload = payload_for(
                rows=archive.list_tool_action_evidence_count_rows(
                    action_query,
                    detail_patterns=action_patterns,
                    since_ms=since_ms,
                ),
                payload_basis="actions",
                kind_value="tool_action_evidence_counts",
                detail_value="canonical_action_evidence_counts",
                payload_tool=None,
                payload_mcp_server=None,
                payload_detail_patterns=action_patterns,
            )
            comparison = ToolFamilyComparisonPayload(
                kind="tool_family_evidence_comparison",
                archive_root=str(env.config.archive_root),
                family=family,
                bases=(call_payload, event_payload, action_payload),
                caveats=(
                    "Counts are grouped by evidence basis and must not be summed across bases.",
                    "Observed-event sections count source-derived tool_finished outcomes, not raw tool_use calls.",
                    "Action evidence can include command/path/input detail matches for tools used outside MCP rows.",
                ),
            )
            if output_format == "json":
                click.echo(comparison.to_json(exclude_none=True))
                return
            env.ui.console.print(f"\n[bold]Tool family evidence comparison: {family}[/bold]")
            for basis_payload in comparison.bases:
                env.ui.console.print(f"  {basis_payload.filters.basis}: {len(basis_payload.items)} row(s)")
                for item in basis_payload.items:
                    count = item.event_count if basis_payload.filters.basis == "observed-events" else item.call_count
                    evidence = f" [{item.evidence_kind}]" if item.evidence_kind else ""
                    env.ui.console.print(f"    {item.origin}  {item.normalized_tool_name}{evidence}: {count or 0}")
            env.ui.console.print("  caveat: counts are basis-specific; do not sum them across sections")
            return
        if basis == "observed-events":
            rows = archive.list_tool_observed_event_count_rows(query)
            kind = "tool_observed_event_counts"
            detail = "tool_finished_observed_events"
            count_key = "event_count"
        elif basis == "actions":
            rows = archive.list_tool_action_evidence_count_rows(
                query,
                detail_patterns=detail_patterns,
                since_ms=since_ms,
            )
            kind = "tool_action_evidence_counts"
            detail = "canonical_action_evidence_counts"
            count_key = "call_count"
        else:
            rows = archive.list_tool_call_count_rows(query)
            kind = "tool_call_counts"
            detail = "tool_use_block_call_counts"
            count_key = "call_count"
    if output_format == "json":
        payload = payload_for(
            rows=rows,
            payload_basis=cast(ToolCountBasis, basis),
            kind_value=cast(ToolCountKind, kind),
            detail_value=cast(ToolCountDetailLevel, detail),
            payload_tool=tool,
            payload_mcp_server=mcp_server,
            payload_detail_patterns=tuple(detail_patterns),
        )
        click.echo(payload.to_json(exclude_none=True))
        return

    if not rows:
        env.ui.console.print("No tool invocations found.")
        return

    if basis == "observed-events":
        title = "Tool observed-event counts"
    elif basis == "actions":
        title = "Tool action evidence counts"
    else:
        title = "Tool call counts"
    env.ui.console.print(f"\n[bold]{title}[/bold]")
    if basis == "observed-events":
        env.ui.console.print("  detail: tool_finished observed events; counts tool outcomes, not raw tool_use blocks")
        header = f"{'origin':18s}  {'tool':38s}  {'kind':14s}  {'status':10s}  {'events':>7s}"
    elif basis == "actions":
        env.ui.console.print("  detail: canonical actions evidence; can include command/path/input detail matches")
        header = f"{'origin':18s}  {'tool':38s}  {'kind':14s}  {'evidence':16s}  {'calls':>7s}"
    else:
        env.ui.console.print(
            "  detail: tool_use block call counts; use `analyze insights tool-usage` for exact coverage/detail fields"
        )
        header = f"{'origin':18s}  {'tool':38s}  {'kind':14s}  {'calls':>7s}"
    env.ui.console.print(header)
    env.ui.console.print("-" * len(header))
    for row in rows:
        source_name = str(row["source_name"])
        tool_name = str(row["normalized_tool_name"])
        action_kind_value = str(row["action_kind"])
        count = int(str(row[count_key]))
        if basis == "observed-events":
            status = str(row["status"])
            env.ui.console.print(
                f"{source_name[:18]:18s}  "
                f"{tool_name[:38]:38s}  "
                f"{action_kind_value[:14]:14s}  "
                f"{status[:10]:10s}  "
                f"{count:7d}"
            )
        elif basis == "actions":
            evidence_kind = str(row.get("evidence_kind") or "unknown")
            env.ui.console.print(
                f"{source_name[:18]:18s}  "
                f"{tool_name[:38]:38s}  "
                f"{action_kind_value[:14]:14s}  "
                f"{evidence_kind[:16]:16s}  "
                f"{count:7d}"
            )
        else:
            env.ui.console.print(
                f"{source_name[:18]:18s}  {tool_name[:38]:38s}  {action_kind_value[:14]:14s}  {count:7d}"
            )
